/*!
 *  Copyright (c) 2016 by Contributors
 * \file bound.cc
 * \brief The bound inference logic.
 */
#include <tvm/ir_visitor.h>
#include <tvm/schedule_pass.h>
#include <tvm/operation.h>
#include <unordered_map>
#include <unordered_set>
#include "./graph.h"
#include "./message_passing.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace schedule {

using runtime::ThreadScope;
using runtime::StorageScope;

/*! \brief The graph context used during bound inference. */
struct GraphContext {
  /*! \brief The feed graph */
  FeedGraph feed_graph;
  /*! \brief Attachment path */
  AttachPath attach_path;
  /*! \brief The bind map */
  std::unordered_map<IterVar, IterVar> bind_map;
  /*! \brief map from op to stage */
  std::unordered_map<const Node*, Stage> op2stage_;
};

bool NeedRelax(const IterVar& iv,
               bool found_attach,
               const std::unordered_map<IterVar, IterVar>& bind_map,
               const runtime::StorageScope& scope) {
  auto it = bind_map.find(iv);
  const std::string& tag = (
      it != bind_map.end() ? it->second->thread_tag : iv->thread_tag);
  if (tag.length() == 0 || tag == "pipeline") {
    return !found_attach;
  }
  return scope.rank <= ThreadScope::make(tag).rank;
}

// infer storage scope, if not given
StorageScope InferStorageScope(
    const Stage& stage, const GraphContext& ctx) {
  if (stage->scope.length() != 0) {
    return StorageScope::make(stage->scope);
  }
  int max_rank = 0;
  for (IterVar iv : ctx.attach_path.at(stage->op)) {
    auto it = ctx.bind_map.find(iv);
    const std::string& tag = (
        it != ctx.bind_map.end() ? it->second->thread_tag : iv->thread_tag);
    if (tag != "pipeline" && tag.length() != 0) {
      max_rank = std::max(max_rank, ThreadScope::make(tag).rank + 1);
    }
  }
  StorageScope s; s.rank = max_rank;
  return s;
}


void InferRootBound(const Stage& stage,
                    const GraphContext& ctx,
                    std::unordered_map<IterVar, Range>* rmap) {
  CHECK_NE(stage->attach_type, kInline)
      << "call schedule.normalize before scheduleops";
  if (stage->attach_type == kInlinedAlready) return;
  if (stage->is_output) {
    // verify correctness.
    CHECK_EQ(stage.GetAttachSpec()->attach_type, kGroupRoot)
          << "Output must be attached at root";
  }
  if (stage->is_output || stage->op.as<PlaceholderOpNode>()) {
    for (auto iv :  stage->op->root_iter_vars()) {
      CHECK(iv->dom.defined());
      CHECK(!rmap->count(iv));
      (*rmap)[iv] = iv->dom;
    }
    return;
  }
  // The tensor domain.
  std::unordered_map<Tensor, TensorDom> tmap;
  // The consumers of the op.
  std::unordered_set<Operation> consumers;
  for (int i = 0; i < stage->op->num_outputs(); ++i) {
    Tensor t = stage->op.output(i);
    tmap.emplace(t, TensorDom(static_cast<int>(t.ndim())));
    auto it = ctx.feed_graph.find(t);
    if (it != ctx.feed_graph.end()) {
      for (const Operation& op : it->second) {
        consumers.insert(op);
      }
    } else {
      LOG(INFO) << "not in feed graph consumer = " << stage->op;
    }
  }
  // storage scope.
  runtime::StorageScope scope = InferStorageScope(stage, ctx);
  // Bound prop by other consumers.
  // - Compute bound by relaxation rules: NeedRelax
  //   - For normal index, use relative location of loop nest./
  //   - For thread index, use the thread scope.
  //
  Array<IterVar> stage_attach = ctx.attach_path.at(stage->op);
  // The parent set.
  for (const Operation& op : consumers) {
    std::unordered_map<const Variable*, IntSet> relax_set;
    std::unordered_map<IterVar, IntSet> up_state;
    bool found_attach = false;
    CHECK(ctx.op2stage_.count(op.get()));
    const Stage& op_stage = ctx.op2stage_.at(op.get());
    // Consumer nest
    for (size_t i = op_stage->leaf_iter_vars.size(); i != 0; --i) {
      IterVar iv = op_stage->leaf_iter_vars[i - 1];
      if (stage_attach.size() != 0 && iv == stage_attach[0]) {
        found_attach = true;
      }
      auto it = rmap->find(iv);
      CHECK(it != rmap->end());
      const Range& vrange = it->second;
      if (is_one(vrange->extent)) {
        up_state[iv] = IntSet::single_point(vrange->min);
      } else if (!NeedRelax(iv, found_attach, ctx.bind_map, scope)) {
        CHECK(is_zero(vrange->min))
            << "InferBound requires every leaf iter var's min equals 0, "
            << " call schedule.normalize to achieve this. ";
        if (ctx.bind_map.count(iv)) {
          up_state[iv] = IntSet::single_point(ctx.bind_map.at(iv)->var);
        } else {
          up_state[iv] = IntSet::single_point(iv->var);
        }
      } else {
        up_state[iv] = IntSet::range(vrange);
      }
    }
    // Consumer's attach nest
    for (IterVar iv : ctx.attach_path.at(op)) {
      if (stage_attach.size() != 0 && iv == stage_attach[0]) {
        found_attach = true;
      }
      Range vrange = rmap->at(iv);
      CHECK(is_zero(vrange->min))
          << "InferBound requires every leaf iter var's min equals 0, "
          << "call schedule.normalize to achieve this.";
      if (NeedRelax(iv, found_attach, ctx.bind_map, scope)) {
        relax_set[iv->var.get()] = IntSet::range(vrange);
        if (ctx.bind_map.count(iv)) {
          relax_set[ctx.bind_map.at(iv)->var.get()] = IntSet::range(vrange);
        }
      }
    }
    CHECK(found_attach || stage_attach.size() == 0)
        << "Invalid Schedule, cannot find the producer " << stage->op
        << " along the loop nest specified by compute_at of consumer " << op;
    // Get the domain of the consumer
    PassUpDomain(op_stage, *rmap, &up_state);
    // Relax if needed.
    std::unordered_map<const Variable*, IntSet> dom_map;
    for (auto iv : op->root_iter_vars()) {
      Range r;
      if (up_state.count(iv)) {
        r = up_state.at(iv).cover_range(iv->dom);
      } else {
        r = iv->dom;
      }
      if (relax_set.size() != 0) {
        dom_map[iv->var.get()] = EvalSet(r, relax_set);
      } else {
        dom_map[iv->var.get()] = IntSet::range(r);
      }
    }
    op->PropBoundToInputs(op, dom_map, &tmap);
  }
  stage->op->GatherBound(stage->op, tmap, rmap);
}

Map<IterVar, Range> InferBound(const Schedule& sch) {
  // Prepare context
  GraphContext ctx;
  Array<Operation> roots;
  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  ctx.feed_graph = CreateFeedGraph(CreateReadGraph(roots));

  for (Stage stage : sch->stages) {
    for (auto kv : stage->iter_var_attrs) {
      if (kv.second->bind_thread.defined()) {
        CHECK(!ctx.bind_map.count(kv.first));
        ctx.bind_map[kv.first] = kv.second->bind_thread;
      }
    }
    ctx.op2stage_[stage->op.get()] = stage;
  }
  ctx.attach_path = CreateAttachPath(sch);
  // Run inference.
  std::unordered_map<IterVar, Range> ret;
  for (size_t i = sch->stages.size(); i != 0; --i) {
    const Stage& stage = sch->stages[i - 1];
    InferRootBound(stage, ctx, &ret);
    // pass down to get bound of all iter vars.
    PassDownDomain(stage, &ret);
    for (IterVar iv : stage->env_threads) {
      CHECK(iv->dom.defined());
      ret[iv] = iv->dom;
    }
  }
  return Map<IterVar, Range>(ret.begin(), ret.end());
}

}  // namespace schedule
}  // namespace tvm
