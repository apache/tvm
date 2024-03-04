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
#include "target_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

ForNodeLayoutInfer::ForNodeLayoutInfer(const ForNode* root, IterVar thread_var)
    : root_(root), thread_var_(thread_var) {
  VisitStmt_(root);
}

bool ForNodeLayoutInfer::IsCommonAccessIndice(const Buffer& buffer) const {
  auto common_indice = loop_vars_.Map([](const auto& iv) { return iv->var; });
  return StructuralEqual()(indice_map_[buffer], common_indice);
}

void ForNodeLayoutInfer::VisitStmt_(const ForNode* op) {
  ICHECK(op->kind == ForKind::kParallel);
  loop_vars_.push_back(IterVar(Range(op->min, op->extent), op->loop_var, IterVarType::kDataPar));
  analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
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
  if (level == InferLevel::kStrict) return {};
  if (level == InferLevel::kFree && root_->annotations.count(attr::kSkipLayoutInfer)) return {};

  // Step 1: try to infer loop's partition from a source fragment
  Buffer source_buffer, read_source_buffer;
  for (const auto& [buffer, _] : indice_map_) {
    if (layout_map.count(buffer)) {
      auto frag = layout_map[buffer].as<Fragment>().value();
      if (buffer_is_write_.count(buffer))
        source_buffer = buffer;
      else
        read_source_buffer = buffer;
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
    if (read_source_buffer.defined()) {
      source_buffer = read_source_buffer;
      auto src_layout = layout_map[source_buffer].as<Fragment>().value();
      if (IsCommonAccessIndice(source_buffer)) {
        loop_layout_ = src_layout;
      } else {
        PrimExpr loop_var_to_thread = src_layout->ForwardThread(indice_map_[source_buffer], {});
        loop_layout_ = Fragment(loop_vars_, {}, loop_var_to_thread, src_layout->thread_replicate_);
      }
      // Loop don't need to be replicated.
      if (!is_one(loop_layout_->ReplicateExtent())) loop_layout_ = loop_layout_->DeReplicate();
      // if still has replication, add a condition
      if (!is_one(loop_layout_->ReplicateExtent())) {
        auto inv = loop_layout_->Inverse();
        Array<PrimExpr> fwd;
        for (size_t i = 0; i < loop_layout_->OutputDim(); i++) fwd.push_back(0);
        fwd.push_back(thread_var_->var);
        auto rep = inv->Forward(fwd).back();
        AddPredicate(EQ(rep, 0));
      }
    } else {
      int vector_size = GetVectorizeSize(GetRef<For>(root_));
      auto num_thread = as_const_int(thread_var_->dom->extent);
      ICHECK(num_thread != nullptr);
      loop_layout_ = PlanLoopPartition(root_, *num_thread, vector_size);
    }
    PrimExpr loop_thread_extent = loop_layout_->ThreadExtent();
    if (!analyzer_.CanProveEqual(loop_thread_extent, thread_var_->dom->extent))
      AddPredicate(LT(thread_var_->var, loop_thread_extent));
  } else {
    return {};
  }
  // Step 2: Check that the loop's partition ccan correctly align with all source fragment
  for (const auto& [buffer, _] : indice_map_) {
    if (layout_map.count(buffer)) {
      auto fragment = layout_map[buffer].as<Fragment>().value();
      // TODO: Add thread checks for replicated cases
      // need to wildcard match the rhs with lhs
      if (!is_one(loop_layout_->ReplicateExtent()) || !is_one(fragment->ReplicateExtent()))
        continue;
      auto vars = loop_vars_.Map([](const IterVar& iv) { return PrimExpr(iv->var); });
      auto lhs = loop_layout_->ForwardThread(vars, NullOpt);
      auto rhs = fragment->ForwardThread(indice_map_[buffer], NullOpt);
      auto diff = analyzer_.Simplify(lhs - rhs);
      ICHECK(is_zero(diff)) << "Layout infer conflict for " << buffer << " " << source_buffer
                            << "\nLHS = " << lhs << "\nRHS = " << rhs;
    }
  }
  // Step 3: Infer other fragment's layout from the loop's partition
  LayoutMap results;
  for (const auto& [buffer, _] : indice_map_) {
    if (!layout_map.count(buffer)) results.Set(buffer, CompleteBufferFragment(buffer));
  }
  return results;
}

Fragment ForNodeLayoutInfer::CompleteBufferFragment(const Buffer& buffer) {
  ICHECK(loop_layout_.defined());
  if (IsCommonAccessIndice(buffer)) return loop_layout_;

  PrimExpr rep_b =
      MakeFlattenedExpression(DivideUnusedIterators(indice_map_[buffer], loop_vars_, &analyzer_));

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
    iter_vars.push_back(IterVar(Range(0, ind_inv->InputShape()[i]), var, IterVarType::kDataPar));
    fwd.push_back(var);
  }
  fwd.push_back(FloorMod(rep, indice_rep_extent));
  PrimExpr thd_b =
      loop_layout_->ForwardThread(ind_inv->Forward(fwd), FloorDiv(rep, indice_rep_extent));

  return Fragment(iter_vars, {}, thd_b, rep)->CondenseReplicateVar();
}

GemmOpLayoutInfer::GemmOpLayoutInfer(const GemmArgs& gemm_args, size_t block_size,
                                     const TargetNode* target)
    : args(gemm_args), block_size_(block_size), target_(target) {}

LayoutMap GemmOpLayoutInfer::Inference(const LayoutMap& layout_map, InferLevel level) {
  if (completed_) return {};

  LayoutMap results;
  ICHECK(args.C.scope() == "local.fragment");
  auto [warp_m, warp_n] = args.ComputeWarpPartition(block_size_ / 32);

  if (TargetIsVolta(target_)) {
    auto fragment = makeGemmVoltaFragmentC(args.M, args.N, args.M / warp_m, args.N / warp_n,
                                           args.C->dtype.bits());
    results.Set(args.C, fragment);
    if (args.A.scope() == "shared" || args.A.scope() == "shared.dyn") {
      results.Set(args.A, makeGemmVoltaABLayout(*as_const_int(args.A->shape[0]),
                                                *as_const_int(args.A->shape[1]), true,
                                                args.trans_A ? 1 : 2));
    } else if (args.A.scope() == "local.fragment") {
      ICHECK(args.trans_A == false);
      results.Set(args.A,
                  makeGemmVoltaFragmentA(args.M, args.N, args.K, args.M / warp_m, args.N / warp_n));
    } else {
      ICHECK(0);
    }

    ICHECK(args.B.scope() == "shared" || args.B.scope() == "shared.dyn");
    results.Set(args.B, makeGemmVoltaABLayout(*as_const_int(args.B->shape[0]),
                                              *as_const_int(args.B->shape[1]), false,
                                              args.trans_B ? 2 : 1));
  } else if (TargetIsAmpere(target_) || TargetIsTuring(target_)) {
    auto fragment =
        makeGemmFragmentC(args.M, args.N, args.M / warp_m, args.N / warp_n, args.C->dtype.bits());
    results.Set(args.C, fragment);

    if (args.A.scope() == "shared" || args.A.scope() == "shared.dyn") {
      results.Set(args.A,
                  makeGemmABLayout(*as_const_int(args.A->shape[0]), *as_const_int(args.A->shape[1]),
                                   args.A->dtype.bits(), args.trans_A ? 1 : 2));
    } else if (args.A.scope() == "local.fragment") {
      ICHECK(args.trans_A == false);
      results.Set(args.A,
                  makeGemmFragmentA(args.M, args.N, args.K, args.M / warp_m, args.N / warp_n));
    } else {
      ICHECK(0);
    }
    if (args.B.scope() == "shared" || args.B.scope() == "shared.dyn") {
      results.Set(args.B,
                  makeGemmABLayout(*as_const_int(args.B->shape[0]), *as_const_int(args.B->shape[1]),
                                   args.B->dtype.bits(), args.trans_B ? 2 : 1));
    } else if (args.B.scope() == "local.fragment") {
      ICHECK(args.trans_B == false);
      results.Set(args.B,
                  makeGemmFragmentB(args.M, args.N, args.K, args.M / warp_m, args.N / warp_n));
    } else {
      ICHECK(0);
    }

  } else {
    ICHECK(0) << "Not supported " << target_->str();
  }
  completed_ = true;
  return results;
}

ReduceOpLayoutInfer::ReduceOpLayoutInfer(const ReduceArgs& reduce_args, size_t block_size)
    : args(reduce_args), block_size_(block_size) {}

LayoutMap ReduceOpLayoutInfer::Inference(const LayoutMap& layout_map, InferLevel level) {
  if (level >= InferLevel::kStrict) return {};
  if (args.src.scope() == "local.fragment" && args.dst.scope() == "local.fragment" &&
      layout_map.count(args.src) && !layout_map.count(args.dst)) {
    auto src_layout = layout_map[args.src].as<Fragment>().value();

    PrimExpr indice_rep_extent = args.src->shape[args.dim];
    PrimExpr src_rep_extent = src_layout->ReplicateExtent();
    PrimExpr dest_buffer_rep_extent = indice_rep_extent * src_rep_extent;
    IterVar rep = IterVar(Range(0, dest_buffer_rep_extent), Var("rep"), IterVarType::kDataPar);

    Array<IterVar> iter_vars;
    Array<PrimExpr> fwd;
    for (size_t i = 0; i < src_layout->InputDim(); i++) {
      if (int(i) == args.dim) {
        fwd.push_back(FloorMod(rep, indice_rep_extent));
      } else {
        auto var = Var("i" + std::to_string(i));
        iter_vars.push_back(
            IterVar(Range(0, src_layout->InputShape()[i]), var, IterVarType::kDataPar));
        fwd.push_back(var);
      }
    }
    auto thd = src_layout->ForwardThread(fwd, FloorDiv(rep, indice_rep_extent));
    Fragment dst_layout = Fragment(iter_vars, {}, thd, rep)->CondenseReplicateVar();
    return {{args.dst, dst_layout}};
  }
  return {};
}

}  // namespace tl
}  // namespace tvm
