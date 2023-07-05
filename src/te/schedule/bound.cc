/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file bound.cc
 * \brief The bound inference logic.
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>

#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "graph.h"
#include "message_passing.h"

namespace tvm {
namespace te {

using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;

/*! \brief The graph context used during bound inference. */
struct GraphContext {
  /*! \brief The feed graph */
  FeedGraph feed_graph;
  /*! \brief Attachment path */
  AttachPath attach_path;
  /*! \brief The bind map */
  std::unordered_map<IterVar, IterVar> bind_map;
  /*! \brief map from op to stage */
  std::unordered_map<const Object*, Stage> op2stage_;
};

bool NeedRelax(const IterVar& iv, bool found_attach,
               const std::unordered_map<IterVar, IterVar>& bind_map,
               const runtime::StorageScope& scope) {
  auto it = bind_map.find(iv);
  const std::string& tag = (it != bind_map.end() ? it->second->thread_tag : iv->thread_tag);
  if (tag.length() == 0 || tag == "pipeline") {
    return !found_attach;
  }
  ThreadScope ts = ThreadScope::Create(tag);

  // When there is warp memory
  // threadIdx.x must be set to be warp index.
  if (scope.rank == StorageRank::kWarp && ts.rank == 1 && ts.dim_index == 0) {
    return true;
  }
  return static_cast<int>(scope.rank) <= ts.rank;
}

// infer storage scope, if not given
StorageScope InferStorageScope(const Stage& stage, const GraphContext& ctx) {
  if (stage->scope.length() != 0) {
    return StorageScope::Create(stage->scope);
  }
  int max_rank = -1;
  for (IterVar iv : ctx.attach_path.at(stage->op)) {
    auto it = ctx.bind_map.find(iv);
    const std::string& tag = (it != ctx.bind_map.end() ? it->second->thread_tag : iv->thread_tag);
    if (tag != "pipeline" && tag.length() != 0) {
      max_rank = std::max(max_rank, ThreadScope::Create(tag).rank);
    }
  }
  StorageScope s;
  s.rank = runtime::DefaultStorageRank(max_rank);
  return s;
}

void InferRootBound(const Stage& stage, const GraphContext& ctx,
                    std::unordered_map<IterVar, Range>* rmap) {
  ICHECK_NE(stage->attach_type, kInline) << "call schedule.normalize before scheduleops";
  if (stage->attach_type == kInlinedAlready) return;
  if (stage->is_output) {
    // verify correctness.
    ICHECK_EQ(stage.GetAttachSpec()->attach_type, kGroupRoot) << "Output must be attached at root";
  }
  if (stage->is_output || stage->op.as<PlaceholderOpNode>()) {
    for (auto iv : stage->op->root_iter_vars()) {
      ICHECK(iv->dom.defined());
      ICHECK(!rmap->count(iv));
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
    Map<Var, IntSet> relax_set;
    std::unordered_map<IterVar, IntSet> up_state;
    bool found_attach = false;
    ICHECK(ctx.op2stage_.count(op.get()));
    const Stage& op_stage = ctx.op2stage_.at(op.get());
    // Consumer nest
    for (size_t i = op_stage->leaf_iter_vars.size(); i != 0; --i) {
      IterVar iv = op_stage->leaf_iter_vars[i - 1];
      if (stage_attach.size() != 0 && iv == stage_attach[0]) {
        found_attach = true;
      }
      auto it = rmap->find(iv);
      ICHECK(it != rmap->end());
      const Range& vrange = it->second;
      if (is_one(vrange->extent)) {
        up_state[iv] = IntSet::SinglePoint(vrange->min);
      } else if (!NeedRelax(iv, found_attach, ctx.bind_map, scope)) {
        ICHECK(is_zero(vrange->min)) << "InferBound requires every leaf iter var's min equals 0, "
                                     << " call schedule.normalize to achieve this. ";
        if (ctx.bind_map.count(iv)) {
          up_state[iv] = IntSet::SinglePoint(ctx.bind_map.at(iv)->var);
        } else {
          up_state[iv] = IntSet::SinglePoint(iv->var);
        }
      } else {
        up_state[iv] = IntSet::FromRange(vrange);
      }
    }
    // Consumer's attach nest
    for (IterVar iv : ctx.attach_path.at(op)) {
      if (stage_attach.size() != 0 && iv == stage_attach[0]) {
        found_attach = true;
      }
      Range vrange = rmap->at(iv);
      ICHECK(is_zero(vrange->min)) << "InferBound requires every leaf iter var's min equals 0, "
                                   << "call schedule.normalize to achieve this.";
      if (NeedRelax(iv, found_attach, ctx.bind_map, scope)) {
        relax_set.Set(iv->var, IntSet::FromRange(vrange));
        if (ctx.bind_map.count(iv)) {
          relax_set.Set(ctx.bind_map.at(iv)->var, IntSet::FromRange(vrange));
        }
      }
    }
    ICHECK(found_attach || stage_attach.size() == 0)
        << "Invalid Schedule, cannot find the producer " << stage->op
        << " along the loop nest specified by compute_at of consumer " << op;
    // Get the domain of the consumer
    PassUpDomain(op_stage, *rmap, &up_state);
    // Relax if needed.
    std::unordered_map<const VarNode*, IntSet> dom_map;
    arith::Analyzer analyzer;
    for (auto entry : *rmap) {
      analyzer.Bind(entry.first->var, entry.second);
    }
    for (auto iv : op->root_iter_vars()) {
      Range r;
      if (up_state.count(iv)) {
        r = up_state.at(iv).CoverRange(iv->dom);
      } else {
        r = iv->dom;
      }
      if (relax_set.size() != 0) {
        dom_map[iv->var.get()] =
            IntSet::Interval(analyzer.int_set(r->min, relax_set).min(),
                             analyzer.int_set(r->min + r->extent - 1, relax_set).max());
      } else {
        dom_map[iv->var.get()] = IntSet::FromRange(r);
      }
      analyzer.Bind(iv->var, r, true);
    }
    op->PropBoundToInputs(op, &analyzer, dom_map, &tmap);
  }
  stage->op->GatherBound(stage->op, tmap, rmap);
}

Map<IterVar, Range> InferBound(const Schedule& sch) {
  // Prepare context
  GraphContext ctx;
  Array<Operation> roots;
  arith::Analyzer analyzer;

  for (Operation op : sch->outputs) {
    roots.push_back(sch->stage_map[op]->op);
  }
  ctx.feed_graph = CreateFeedGraph(CreateReadGraph(roots));

  for (Stage stage : sch->stages) {
    for (auto kv : stage->iter_var_attrs) {
      if (kv.second->bind_thread.defined()) {
        ICHECK(!ctx.bind_map.count(kv.first));
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

    // bind bound of root iter vars.
    for (auto iv : stage->op->root_iter_vars()) {
      auto it = ret.find(iv);
      if (it != ret.end()) {
        analyzer.Bind(iv->var, it->second);
      }
    }

    // pass down to get bound of all iter vars.
    PassDownDomain(stage, &ret, &analyzer);
    for (IterVar iv : stage->env_threads) {
      ICHECK(iv->dom.defined());
      ret[iv] = iv->dom;
    }
  }
  for (auto it = ret.begin(); it != ret.end(); it++) {
    DataType var_type = it->first->var.dtype();
    it->second = Range::FromMinExtent(
        // The range associated with each itervar must have the same dtype as the var
        analyzer.Simplify(cast(var_type, it->second->min)),
        analyzer.Simplify(cast(var_type, it->second->extent)));
  }
  return Map<IterVar, Range>(ret.begin(), ret.end());
}

TVM_REGISTER_GLOBAL("schedule.InferBound").set_body_typed(InferBound);

}  // namespace te
}  // namespace tvm
