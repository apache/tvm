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
 * \brief Logics related to cross thread reduction, used by ComputeOpNode.
 * \file cross_thread_reduction.cc
 */
#include <tvm/tir/builtin.h>

#include "compute_op.h"
#include "op_utils.h"

namespace tvm {
namespace te {
using namespace tir;

//
// Cross thread reduction transformation.
//
// The input loop nest in generic form (single reduction/thread case)
//
// let m be the reduction extent
// let N be the thread extent
// let input_pred be the predicate on the reduction
//
// B[..] = 0
// for (tid, 0, N)
//   for (i, 0, floordiv(m+N-1, N))
//     if (i + tid * floordiv(m+N-1, N) < m)
//       if (input_pred)
//         B[..] = op(B[..], A[i + tid  * floordiv(m+N-1,N)])
//
// The threaded reduction looks like
//
// (1) normal reductions (leaves)
// for (i, 0, floordiv(m+N-1, N))
//   if (i + tid * floordiv(m+N-1, N) < m)
//     if (input_pred)
//       B_temp[0] = op(B_temp[0], A[i + tid  * floordiv(m+N-1,N)])
//
// (2) threaded reduction does not require predicates as an identity
//     element will be filled if out of bounds.
//
// tvm_thread_allreduce(size, B_temp, (bool)1, tid)
//
// The last step is to write the final reduction variable,
// which should be predicated by the existing input_pred if any
// The consequence is that input_pred should be independent of
// the reduction axis. Otherwise, we need to seperate it into
// dependent part and independent one.
//
// (3) write back
// if (input_pred)
//    B[..] = B_temp[0]
//
// In summary, we are going to need two predicates
//
// * the original input_pred from reduction itself
//
// * the normal reduction axis predicate
//     normal_pred = (i + tid * floordiv(m+N-1,N)) < m
//   this predicate depends on the normal reduction variable.
//
// input_pred will be applied to both normal reduction and
// the writeback step.
//
Stmt MakeCrossThreadReduction(const ComputeOpNode* self, const Stage& stage,
                              const std::unordered_map<IterVar, Range>& dom_map,
                              bool debug_keep_trivial_loop) {
  Array<PrimExpr> args;
  for (IterVar iv : self->axis) {
    args.push_back(iv->var);
  }
  std::unordered_map<IterVar, PrimExpr> value_map;
  auto nest = MakeLoopNest(stage, dom_map, 0, false, std::unordered_set<IterVar>(), &value_map,
                           debug_keep_trivial_loop);

  size_t size = self->body.size();
  ICHECK_GT(size, 0);
  std::vector<const ReduceNode*> reduces(size);
  for (size_t i = 0; i < size; ++i) {
    const ReduceNode* reduce = self->body[i].as<ReduceNode>();
    ICHECK(reduce);
    ICHECK(reduce->init.empty())
        << "Cannot perform cross_thread_reduction for reductions with init";
    reduces[i] = reduce;
  }

  // This computes the bound checking predicates in normal reduction.
  auto normal_preds =
      MakeBoundCheck(stage, dom_map, value_map, false, std::unordered_set<IterVar>());

  // normal_pred = input_pred && normal_pred
  PrimExpr input_pred = reduces[0]->condition;
  normal_preds.push_back(input_pred);
  normal_preds.erase(std::remove_if(normal_preds.begin(), normal_preds.end(),
                                    [](const PrimExpr& e) { return !e.defined(); }),
                     normal_preds.end());

  std::vector<std::vector<Stmt>> common, normal_red;
  for (size_t i = 0, n = stage->leaf_iter_vars.size(); i < n; ++i) {
    IterVar iv = stage->leaf_iter_vars[i];
    IterVarAttr attr;
    auto it = stage->iter_var_attrs.find(iv);
    if (it != stage->iter_var_attrs.end()) {
      attr = (*it).second;
    }
    if (iv->iter_type == kCommReduce) {
      if (attr.defined() && attr->bind_thread.defined()) {
        common.emplace_back(nest[i + 1]);
      } else {
        normal_red.emplace_back(nest[i + 1]);
      }
    } else {
      common.emplace_back(nest[i + 1]);
    }
  }

  // If we load from and then store into the same res_handles in the thread_allreduce intrinsic,
  // something goes wrong, so we use an extra variable here for normal reduction.
  std::vector<Var> normal_res_handles;
  std::vector<Stmt> normal_init, normal_update;
  if (!normal_red.empty()) {
    normal_res_handles.reserve(size);
    normal_init.reserve(size);
    normal_update.resize(size);
    const CommReducerNode* combiner = reduces[0]->combiner.as<CommReducerNode>();
    ICHECK(combiner);
    Array<PrimExpr> lhs;
    for (size_t i = 0; i < size; ++i) {
      DataType t = reduces[i]->dtype;
      normal_res_handles.emplace_back("normal_reduce_temp" + std::to_string(i),
                                      PointerType(PrimType(t), "local"));
      lhs.push_back(Load(t, normal_res_handles[i], 0, const_true(t.lanes())));
    }
    Array<PrimExpr> init_value = combiner->identity_element;
    Array<PrimExpr> update_value = (*combiner)(lhs, reduces[0]->source);
    for (size_t i = 0; i < size; ++i) {
      DataType t = reduces[i]->dtype;
      normal_init.emplace_back(
          Store(normal_res_handles[i], init_value[i], 0, const_true(t.lanes())));
      normal_update.emplace_back(
          Store(normal_res_handles[i], update_value[i], 0, const_true(t.lanes())));
    }
  }

  Array<PrimExpr> freduce_args;
  freduce_args.push_back(make_const(DataType::UInt(32), static_cast<uint32_t>(size)));
  for (size_t i = 0; i < size; ++i) {
    if (!normal_red.empty()) {
      DataType t = reduces[i]->dtype;
      freduce_args.push_back(Load(t, normal_res_handles[i], 0, const_true(t.lanes())));
    } else {
      freduce_args.push_back(reduces[0]->source[i]);
    }
  }

  // No constraints on the thread reduction step. It may have redundent
  // computation for rare cases. TODO(tvm-team): revisit this.
  freduce_args.push_back(const_true(1));
  std::vector<Var> res_handles(size);
  for (size_t idx = 0; idx < size; ++idx) {
    DataType dtype = reduces[idx]->dtype;
    res_handles[idx] =
        Var("reduce_temp" + std::to_string(idx), PointerType(PrimType(dtype), "local"));
    freduce_args.push_back(res_handles[idx]);
  }

  for (IterVar iv : stage->leaf_iter_vars) {
    if (iv->iter_type == kCommReduce) {
      auto it = stage->iter_var_attrs.find(iv);
      if (it != stage->iter_var_attrs.end() && (*it).second->bind_thread.defined()) {
        IterVar tv = (*it).second->bind_thread;
        freduce_args.push_back(tv->var);
      }
    }
  }

  // Checks for the thread.
  std::vector<PrimExpr> output_preds;
  if (stage->store_predicate.defined()) {
    output_preds.emplace_back(stage->store_predicate);
  }

  // Apply the existing input predicate if any.
  output_preds.push_back(input_pred);

  Stmt reduce_body =
      Evaluate(Call(DataType::Handle(), tir::builtin::tvm_thread_allreduce(), freduce_args));
  reduce_body = AttrStmt(reduces[0]->combiner, tir::attr::reduce_scope,
                         make_zero(DataType::Handle()), reduce_body);

  if (!normal_red.empty()) {
    Stmt init_body = SeqStmt::Flatten(normal_init);
    Stmt update_body = SeqStmt::Flatten(normal_update);
    update_body = MergeNest(MakeIfNest(normal_preds), update_body);
    update_body = MergeNest(normal_red, update_body);
    reduce_body = SeqStmt::Flatten(init_body, update_body, reduce_body);
  }

  std::vector<Stmt> assigns(size);
  for (size_t idx = 0; idx < size; ++idx) {
    DataType t = reduces[idx]->dtype;
    assigns[idx] = ProducerStore(stage->op.output(idx),
                                 Load(t, res_handles[idx], 0, const_true(t.lanes())), args);
  }
  Stmt assign_body = SeqStmt::Flatten(assigns);
  assign_body = MergeNest(MakeIfNest(output_preds), assign_body);
  Stmt body = SeqStmt::Flatten(reduce_body, assign_body);
  for (size_t idx = size; idx != 0; --idx) {
    body = Allocate(res_handles[idx - 1], reduces[idx - 1]->dtype, {1}, const_true(), body);
    if (!normal_red.empty()) {
      body =
          Allocate(normal_res_handles[idx - 1], reduces[idx - 1]->dtype, {1}, const_true(), body);
    }
  }
  body = Substitute(body, value_map);
  return MergeNest(common, body);
}
}  // namespace te
}  // namespace tvm
