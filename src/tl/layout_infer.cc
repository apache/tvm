/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
  * \file tl/layout_infer.cc
  * \brief Infer layout from ops and parallel for
  */

#include "layout_infer.h"

#include <tvm/tir/op.h>

#include "arith.h"
#include "auto_vectorize.h"
#include "loop_partition.h"

namespace tvm {
namespace tl {

using namespace tir;

ForNodeLayoutInfer::ForNodeLayoutInfer(const ForNode* root, size_t block_size)
  : root_(root), block_size_(block_size) {
  VisitStmt_(root);
  // Check if the buffer indice matches full range
  for (const auto& [buffer, indices] : indice_map_) {
    Layout layout(loop_vars_, indices);
    ICHECK(StructuralEqual()(buffer->shape, layout->OutputShape()))
      << "Parallel for over fragment does not match full region, " << buffer->shape << " "
      << layout->OutputShape();
  }
}

bool ForNodeLayoutInfer::IsCommonAccessIndice(const Buffer& buffer) const {
  auto common_indice = loop_vars_.Map([](const auto& iv) { return iv->var; });
  return StructuralEqual()(indice_map_[buffer], common_indice);
}

void ForNodeLayoutInfer::VisitStmt_(const ForNode* op) {
  ICHECK(op->kind == ForKind::kParallel);
  loop_vars_.push_back(IterVar(Range(op->min, op->extent), op->loop_var, IterVarType::kDataPar));
  StmtExprVisitor::VisitStmt_(op);
}

void ForNodeLayoutInfer::VisitStmt_(const BufferStoreNode* op) {
  if (op->buffer.scope() == "local.fragment") {
    if (indice_map_.find(op->buffer) != indice_map_.end()) {
      ICHECK(StructuralEqual()(indice_map_.at(op->buffer), op->indices))
        << op->buffer << ": " << op->indices << " and " << indice_map_.at(op->buffer);
    } else {
      indice_map_.Set(op->buffer, op->indices);
    }
    buffer_is_write_.insert(op->buffer);
  }
  StmtExprVisitor::VisitStmt_(op);
}

void ForNodeLayoutInfer::VisitExpr_(const BufferLoadNode* op) {
  if (op->buffer.scope() == "local.fragment") {
    if (indice_map_.find(op->buffer) != indice_map_.end()) {
      ICHECK(StructuralEqual()(indice_map_.at(op->buffer), op->indices))
        << op->buffer << ": " << op->indices << " and " << indice_map_.at(op->buffer);
    } else {
      indice_map_.Set(op->buffer, op->indices);
    }
  }
  StmtExprVisitor::VisitExpr_(op);
}

LayoutMap ForNodeLayoutInfer::Inference(const LayoutMap& layout_map, InferLevel level) {
  if (loop_layout_.defined()) return {};
  if (level >= InferLevel::kStrict) return {};

  // Step 1: try to infer loop's partition from a source fragment
  Buffer source_buffer;
  for (const auto& [buffer, _] : indice_map_) {
    if (layout_map.count(buffer)) {
      auto frag = layout_map[buffer].as<Fragment>().value();
      if (is_one(frag->ReplicateExtent()) || buffer_is_write_.count(buffer))
        source_buffer = buffer;
    }
  }
  if (source_buffer.defined()) {
    auto src_layout = layout_map[source_buffer].as<Fragment>().value();
    if (IsCommonAccessIndice(source_buffer)) {
      loop_layout_ = src_layout;
    } else {
      PrimExpr loop_var_to_thread = src_layout->ForwardThread(indice_map_[source_buffer], {});
      loop_layout_ = Fragment(loop_vars_, {}, loop_var_to_thread, src_layout->thread_replicate_);
    }
  } else if (level == InferLevel::kFree) {
    // TODO: take existing fragment constraint into consideration
    int vector_size = GetVectorizeSize(GetRef<For>(root_));
    loop_layout_ = PlanLoopPartition(root_, block_size_, vector_size);
  } else {
    return {};
  }

  // Make sure the loop_layout_ infered is correct.
  for (const auto& [buffer, _] : indice_map_) {
    if (layout_map.count(buffer))
      ICHECK(FragmentThreadEqual(loop_layout_, layout_map[buffer].as<Fragment>().value()));
  }

  // Step 2: Infer other fragment's layout from the loop's partition
  LayoutMap results;
  for (const auto& [buffer, _] : indice_map_) {
    if (!layout_map.count(buffer))
      results.Set(buffer, CompleteBufferFragment(buffer));
  }
  return results;
}

Fragment ForNodeLayoutInfer::CompleteBufferFragment(const Buffer& buffer) const {
  ICHECK(loop_layout_.defined());
  if (IsCommonAccessIndice(buffer)) return loop_layout_;

  arith::Analyzer analyzer;
  loop_layout_->UpdateAnalyzer(&analyzer);

  PrimExpr rep_b = MakeFlattenedExpression(
    DivideUnusedIterators(indice_map_[buffer], loop_vars_, &analyzer));

  auto bijective_indice = indice_map_[buffer];
  bijective_indice.push_back(rep_b);
  Layout ind_inv = Layout(loop_vars_, bijective_indice)->Inverse();

  PrimExpr indice_rep_extent = ind_inv->InputShape().back();  // this is the size of rep_b
  PrimExpr loop_rep_extent = loop_layout_->ReplicateExtent();
  PrimExpr dest_buffer_rep_extent = indice_rep_extent * loop_rep_extent;
  IterVar rep = IterVar(Range(0, dest_buffer_rep_extent), Var("rep"), IterVarType::kDataPar);

  Array<IterVar> iter_vars;
  Array<PrimExpr> fwd;
  for (size_t i = 0; i + 1 < ind_inv->InputDim(); i++) {
    auto var = Var("i" + std::to_string(i));
    iter_vars.push_back(
      IterVar(Range(0, ind_inv->InputShape()[i]), var, IterVarType::kDataPar));
    fwd.push_back(var);
  }
  fwd.push_back(FloorMod(rep, indice_rep_extent));
  PrimExpr thd_b =
    loop_layout_->ForwardThread(ind_inv->Forward(fwd), FloorDiv(rep, indice_rep_extent));

  return Fragment(iter_vars, {}, thd_b, rep)->CondenseReplicateVar();
}

GemmOpLayoutInfer::GemmOpLayoutInfer(const GemmArgs& gemm_args, size_t block_size) :
  args(gemm_args), block_size_(block_size) {}

LayoutMap GemmOpLayoutInfer::Inference(const LayoutMap& layout_map, InferLevel level) {
  if (completed_) return {};

  LayoutMap results;
  ICHECK(args.C.scope() == "local.fragment");
  auto [warp_m, warp_n] = args.ComputeWarpPartition(block_size_ / 32);
  auto fragment = makeGemmFragmentC(args.M, args.N, args.M / warp_m, args.N / warp_n);
  results.Set(args.C, fragment);

  if (args.A.scope() == "shared" || args.A.scope() == "shared.dyn") {
    results.Set(args.A,
      makeGemmABLayout(*as_const_int(args.A->shape[0]),
        *as_const_int(args.A->shape[1]), args.A->dtype.bits(), args.trans_A ? 1 : 2));
  }
  if (args.B.scope() == "shared" || args.B.scope() == "shared.dyn") {
    results.Set(args.B,
      makeGemmABLayout(*as_const_int(args.B->shape[0]),
        *as_const_int(args.B->shape[1]), args.B->dtype.bits(), args.trans_B ? 2 : 1));
  }
  if (args.A.scope() == "local.fragment") {
    results.Set(args.A, makeGemmFragmentA(args.M, args.N, args.K, args.M / warp_m, args.N / warp_n));
  }
  completed_ = true;
  return results;
}

} // namespace tl
} // namespace tvm
