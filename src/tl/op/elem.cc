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
 * \file tl/op/elem.cc
 *
 * Define elment-wise operators.
 */

#include "elem.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;

Copy::Copy(Array<PrimExpr> args, BufferMap vmap) : args_(args) {
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto expr = args[i];
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto region = RegionOp(call->args, vmap);
    rgs[i] = region.GetRanges();
    bf[i] = region.GetBuffer();
  }
  std::tie(this->src, this->dst) = std::tie(bf[0], bf[1]);
  std::tie(this->src_range, this->dst_range) = std::tie(rgs[0], rgs[1]);
}

Array<IterVar> Copy::MakeIterVars() const {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < src_range.size(); i++) {
    if (is_one(src_range[i]->extent)) continue;
    Var var = Var(std::string{char('i' + idx)});
    idx++;
    loop_vars.push_back({Range(0, src_range[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

// ivs: itervars returned by MakeIterVars()
// src_dst: 0 for src_indices, 1 for dst_indices
Array<PrimExpr> Copy::MakeIndices(const Array<IterVar>& ivs, int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  ICHECK(idx == ivs.size());
  return indices;
}

PrimExpr Copy::MakePredicate(arith::Analyzer* analyzer, const Array<IterVar>& ivs,
                             Array<PrimExpr> extents, int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent)) continue;
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++) cond = And(cond, cond_list[i]);
    return cond;
  }
}

Stmt Copy::Lower(const LowerArgs& T, arith::Analyzer* analyzer) const {
  if (TargetIsHopper(T.target) && src.scope() == "global" &&
      (dst.scope() == "shared.dyn" || dst.scope() == "shared")) {
    // Use the Hopper TMA bulk copy instructions
    return LowerBulkCopy(T, analyzer);
  }
  Layout src_layout, dst_layout;
  Buffer src_tensor = src, dst_tensor = dst;
  if (T.layout_map.count(src)) {
    src_layout = T.layout_map[src];
    src_tensor = T.buffer_remap[src];
  }
  if (T.layout_map.count(dst)) {
    dst_layout = T.layout_map[dst];
    dst_tensor = T.buffer_remap[dst];
  }

  Array<IterVar> loop_vars = MakeIterVars();
  for (const auto& iv : loop_vars) analyzer->Bind(iv->var, iv->dom);
  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);
  // Simplify the indice so that vectorize size can be correctly detected
  auto simplify_fn = [&](PrimExpr e) { return analyzer->Simplify(e); };
  if (src_layout.defined()) src_indices = src_layout->Forward(src_indices).Map(simplify_fn);
  if (dst_layout.defined()) dst_indices = dst_layout->Forward(dst_indices).Map(simplify_fn);

  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  PrimExpr value = BufferLoad(src_tensor, src_indices);
  if (src->dtype != dst->dtype) value = Cast(dst->dtype, value);
  if (src_predicate.defined()) value = if_then_else(src_predicate, value, make_zero(dst->dtype));
  Stmt body = BufferStore(dst_tensor, value, dst_indices);
  if (dst_predicate.defined()) body = IfThenElse(dst_predicate, body);

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent, ForKind::kParallel, body);
  }

  Fragment loop_partition;
  For for_node = Downcast<For>(body);
  if (dst_tensor.scope() == "local") {
    loop_partition = Downcast<Fragment>(dst_layout);
  } else if (src_tensor.scope() == "local") {
    loop_partition = Downcast<Fragment>(src_layout);
  } else {
    int vector_size = GetVectorizeSize(for_node);
    loop_partition = PlanLoopPartition(for_node, T.block_size, vector_size);
  }
  for_node = PartitionLoop(for_node, T.thread_var, analyzer, loop_partition);
  for_node = VectorizeLoop(for_node);
  // Adding predicates
  auto loop_thread_extent = loop_partition->ThreadExtent();
  if (!analyzer->CanProveEqual(loop_thread_extent, static_cast<int>(T.block_size)))
    return IfThenElse(LT(T.thread_var, loop_thread_extent), for_node);
  else
    return for_node;
}

LayoutMap Copy::InferLayout(const LayoutInferArgs& T, InferLevel level) {
  if (level == InferLevel::kCommon) {
  }

  return {};
}

Fill::Fill(Array<PrimExpr> args, BufferMap vmap) {
  dst = vmap[GetVarFromAccessPtr(args[0])];
  if (args[1]->dtype != dst->dtype) {
    value = Cast(dst->dtype, args[1]);
  } else {
    value = args[1];
  }
}

Stmt Fill::Lower(const LowerArgs& T, arith::Analyzer* analyzer) const {
  auto dst_tensor = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;
  int ndim = dst_tensor->shape.size();
  Array<PrimExpr> indices;
  for (int i = 0; i < ndim; i++) {
    Var var = Var(std::string{char('i' + i)});
    indices.push_back(var);
  }
  Stmt body = BufferStore(dst_tensor, this->value, indices);
  if (dst_tensor.scope() == "local") {
    for (int i = ndim - 1; i >= 0; i--) {
      Map<String, ObjectRef> anno;
      anno.Set("pragma_unroll_explicit", Bool(false));
      body = For(Downcast<Var>(indices[i]), 0, dst_tensor->shape[i], ForKind::kUnrolled, body,
                 NullOpt, anno);
    }
    return body;
  } else {
    for (int i = ndim - 1; i >= 0; i--) {
      body = For(Downcast<Var>(indices[i]), 0, dst_tensor->shape[i], ForKind::kParallel, body);
    }
    For for_node = Downcast<For>(body);
    int vector_size = GetVectorizeSize(for_node);
    auto loop_layout = PlanLoopPartition(for_node, T.block_size, vector_size);
    for_node = PartitionLoop(for_node, T.thread_var, analyzer, loop_layout);
    return for_node;
  }
}

TIR_REGISTER_TL_OP(Copy, copy)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_REGISTER_TL_OP(Fill, fill)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace tl
}  // namespace tvm