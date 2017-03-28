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

// check if scope
inline bool ScopeRelax(const IterVar& iv, const std::string& scope) {
  using runtime::ThreadScope;
  using runtime::StorageScope;
  if (iv->thread_tag.length() == 0) return false;
  if (scope.length() == 0) return false;

  return StorageScope::make(scope).rank <= ThreadScope::make(iv->thread_tag).rank;
}

void InferRootBound(const Stage& stage,
                    const GraphContext& ctx,
                    const AttachPath& attach_path,
                    std::unordered_map<IterVar, Range>* rmap) {
  CHECK_NE(stage->attach_type, kInline)
      << "call schedule.normalize before scheduleops";
  if (stage->attach_type == kInlinedAlready) return;
  if (stage->is_output || stage->op.as<PlaceholderOpNode>()) {
    for (auto iv :  stage->op->root_iter_vars()) {
      CHECK(iv->dom.defined());
      CHECK(!rmap->count(iv));
      (*rmap)[iv] = iv->dom;
    }
    return;
  }
  // parent stage, if any
  Stage parent;
  if (stage->attach_type == kScope || stage->attach_type == kScanUpdate) {
    parent = stage->attach_stage;
  }
  // The tensor domain.
  std::unordered_map<Tensor, TensorDom> tmap;
  // consumers other than parent
  std::unordered_set<Operation> consumers;
  // initialize the result
  bool direct_consume_by_parent = false;
  for (int i = 0; i < stage->op->num_outputs(); ++i) {
    Tensor t = stage->op.output(i);
    tmap.emplace(t, TensorDom(static_cast<int>(t.ndim())));
    auto it = ctx.feed_graph.find(t);
    if (it != ctx.feed_graph.end()) {
      for (const Operation& op : it->second) {
        if (!parent.defined() || op != parent->op) {
          consumers.insert(op);
        } else {
          direct_consume_by_parent = true;
        }
      }
    } else {
      LOG(INFO) << "not in feed graph consumer = " << stage->op;
    }
  }
  // The relax set
  // Thie specifieds the iteration variables that need to be relaxed
  // from the already inferred bounds.
  std::unordered_map<const Variable*, IntSet> relax_set;
  for (IterVar iv : attach_path.at(stage->op)) {
    if (ScopeRelax(iv, stage->scope)) {
      relax_set[iv->var.get()] = IntSet::range(rmap->at(iv));
    }
  }
  if (direct_consume_by_parent) {
    // parent stage if exist
    Stage parent = stage->attach_stage;
    // Bound inference logics in parent.
    std::unordered_map<IterVar, IntSet> up_state;
    bool fix_value = true;
    for (auto iv : parent->leaf_iter_vars) {
      auto it = rmap->find(iv);
      CHECK(it != rmap->end());
      Range vrange = it->second;
      CHECK(is_zero(vrange->min))
          << "InferBound requires every leaf iter var's min equals 0, "
          << " call schedule.normalize to achieve this. "
          << " stage=" << parent;
      // special optimization to remove trivial loop
      if (is_one(vrange->extent)) {
        up_state[iv] = IntSet::single_point(vrange->min);
      } else if (fix_value && !ScopeRelax(iv, stage->scope)) {
        up_state[iv] = IntSet::single_point(iv->var);
      } else {
        up_state[iv] = IntSet::range(vrange);
      }
      if (stage->attach_ivar == iv) {
        fix_value = false;
      }
    }
    // get the bound of the root IterVars given current location.
    PassUpDomain(parent, *rmap, &up_state);

    std::unordered_map<const Variable*, IntSet> dom_map;
    for (auto iv : parent->op->root_iter_vars()) {
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
    // prop from parent.
    parent->op->PropBoundToInputs(parent->op, dom_map, &tmap);
  }
  // Bound prop by other consumers.
  // To explain the the general logic, consider the example:
  //
  // for (i_outer, 0, 10) {
  //   producer
  //
  //   for (i_inner, 0, 4) {
  //     consumer op
  //   }
  // }
  // - Get domain of each of consumer op, say [i_inner + i_outer*8, extent=4)
  // - We need to relax it since the producer is attached at i_outer
  // - Consumer's path is [i_inner, i_outer], then [i_inner] need to be relaxed
  // - Traverse attach_path, relax until reaching the producer's attachment point.
  for (const Operation& op : consumers) {
    std::unordered_map<const Variable*, IntSet> dom_map;
    bool found = false;
    Array<IterVar> attach = attach_path.at(stage->op);
    for (IterVar iv : attach_path.at(op)) {
      if (attach.size() != 0 && iv == attach[0]) {
        found = true; break;
      }
      Range vrange = rmap->at(iv);
      CHECK(is_zero(vrange->min))
          << "InferBound requires every leaf iter var's min equals 0, "
          << "call schedule.normalize to achieve this.";
      relax_set[iv->var.get()] = IntSet::range(vrange);
    }
    CHECK(found || attach.size() == 0)
        << "Invalid Schedule, cannot find the producer " << stage->op
        << " along the loop nest specified by compute_at of consumer " << op;
    for (auto iv : op->root_iter_vars()) {
      Range r = rmap->at(iv);
      dom_map[iv->var.get()] = EvalSet(r, relax_set);
    }
    op->PropBoundToInputs(op, dom_map, &tmap);
  }
  stage->op->GatherBound(stage->op, ctx, tmap, rmap);
}

Map<IterVar, Range> InferBound(const Schedule& sch) {
  Array<Operation> roots;
  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  GraphContext ctx;
  ctx.feed_graph = CreateFeedGraph(CreateReadGraph(roots));
  AttachPath attach_path = CreateAttachPath(sch);

  std::unordered_map<IterVar, Range> ret;
  for (size_t i = sch->stages.size(); i != 0; --i) {
    const Stage& stage = sch->stages[i - 1];
    InferRootBound(stage, ctx, attach_path, &ret);
    // pass down to get bound of all iter vars.
    PassDownDomain(stage, &ret);
    // setup outer most threads.
    for (IterVar iv : stage->outermost_threads) {
      CHECK(iv->dom.defined());
      ret[iv] = iv->dom;
    }
  }
  return Map<IterVar, Range>(ret.begin(), ret.end());
}

}  // namespace schedule
}  // namespace tvm
