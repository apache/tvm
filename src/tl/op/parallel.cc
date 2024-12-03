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
 * \file op/parallel.cc
 * \brief Define Parallel for operator
 */

#include "parallel.h"

#include <tvm/tir/op.h>

#include "../layout/utils.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;

class IfBufferRemapLoopGenerator : public StmtExprMutator {
 public:
  static For run(Stmt stmt, Map<Buffer, Buffer> buffer_remap,
                 Map<Buffer, Layout> layout_map) {
    IfBufferRemapLoopGenerator generator(buffer_remap, layout_map);
    return Downcast<For>(generator(std::move(stmt)));
  }

 private:
  IfBufferRemapLoopGenerator(Map<Buffer, Buffer> buffer_remap, Map<Buffer, Layout> layout_map)
      : buffer_remap_(buffer_remap), layout_map_(layout_map) {}

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));

    if (buffer_remap_.count(load->buffer)) {
      auto new_indices = layout_map_[load->buffer]->Forward(load->indices);
      auto new_buffer = buffer_remap_[load->buffer];

      return BufferLoad(new_buffer, new_indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (buffer_remap_.count(store->buffer)) {
      auto new_indices = layout_map_[store->buffer]->Forward(store->indices);
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, new_indices);
    }
    return store;
  }

  Map<Buffer, Buffer> buffer_remap_;
  Map<Buffer, Layout> layout_map_;
};

void ParallelLoopNestVisitor::VisitStmt_(const ForNode* op) {
  ICHECK(op->kind == ForKind::kParallel);
  p->loop_vars_.push_back(IterVar(Range(op->min, op->extent), op->loop_var, IterVarType::kDataPar));
  p->analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  StmtExprVisitor::VisitStmt_(op);
}

void ParallelLoopNestVisitor::VisitStmt_(const BufferStoreNode* op) {
  if (op->buffer.scope() == "local.fragment") {
    if (p->indice_map_.find(op->buffer) != p->indice_map_.end()) {
      ICHECK(StructuralEqual()(p->indice_map_.at(op->buffer), op->indices))
          << op->buffer << ": " << op->indices << " and " << p->indice_map_.at(op->buffer);
    } else {
      p->indice_map_.Set(op->buffer, op->indices);
    }
    p->buffer_is_write_.insert(op->buffer);
  }
  StmtExprVisitor::VisitStmt_(op);
}

void ParallelLoopNestVisitor::VisitExpr_(const BufferLoadNode* op) {
  if (op->buffer.scope() == "local.fragment") {
    if (p->indice_map_.find(op->buffer) != p->indice_map_.end()) {
      ICHECK(StructuralEqual()(p->indice_map_.at(op->buffer), op->indices))
          << op->buffer << ": " << op->indices << " and " << p->indice_map_.at(op->buffer);
    } else {
      p->indice_map_.Set(op->buffer, op->indices);
    }
  }
  StmtExprVisitor::VisitExpr_(op);
}

ParallelOp::ParallelOp(For root) : root_(root), V(this) { V.VisitStmt(root); }

bool ParallelOp::IsCommonAccessIndice(const Buffer& buffer) const {
  auto common_indice = loop_vars_.Map([](const auto& iv) { return iv->var; });
  return StructuralEqual()(indice_map_[buffer], common_indice);
}

LayoutMap ParallelOp::InferLayout(const LayoutInferArgs& T, InferLevel level) {
  if (loop_layout_.defined()) return {};
  if (level == InferLevel::kStrict) return {};

  // Step 1: try to infer loop's partition from a source fragment
  Buffer source_buffer, read_source_buffer;
  for (const auto& [buffer, _] : indice_map_) {
    if (T.layout_map.count(buffer)) {
      auto frag = T.layout_map[buffer].as<Fragment>().value();
      if (buffer_is_write_.count(buffer))
        source_buffer = buffer;
      else
        read_source_buffer = buffer;
    }
  }
  auto compute_loop_layout_from_buffer = [&](const Buffer& buffer) {
    Fragment src_layout = T.layout_map[buffer].as<Fragment>().value();
    if (IsCommonAccessIndice(buffer)) {
      return src_layout;
    } else {
      Var rep;
      auto rep_iter = IterVar({0, src_layout->ReplicateExtent()}, rep, IterVarType::kDataPar);
      PrimExpr loop_var_to_thread = src_layout->ForwardThread(indice_map_[buffer], rep);
      return Fragment(loop_vars_, {}, loop_var_to_thread, rep_iter);
    }
  };
  if (source_buffer.defined()) {
    loop_layout_ = compute_loop_layout_from_buffer(source_buffer);
  } else if (level == InferLevel::kFree) {
    if (read_source_buffer.defined()) {
      loop_layout_ = compute_loop_layout_from_buffer(read_source_buffer);
      // Loop don't need to be replicated.
      if (!is_one(loop_layout_->ReplicateExtent())) loop_layout_ = loop_layout_->DeReplicate();
      // if still has replication, add a condition
      if (!is_one(loop_layout_->ReplicateExtent())) {
        auto inv = loop_layout_->Inverse();
        Array<PrimExpr> fwd;
        for (size_t i = 0; i < loop_layout_->OutputDim(); i++) fwd.push_back(0);
        fwd.push_back(InputPlaceholder(0));
        auto rep = inv->Forward(fwd).back();
        AddPredicate(EQ(rep, 0));
      }
    } else {
      // Vectorize Size must be aware of the buffer_remap
      // As the pass will do post processing to the layout
      auto maybe_remapped_root_ = IfBufferRemapLoopGenerator::run(root_, T.buffer_remap, T.layout_map);
      int vector_size = GetVectorizeSize(maybe_remapped_root_);

      // Check if coalesced_width is defined
      if (auto coalesced_width = root_->annotations.Get(tir::attr::coalesced_width)) {
        if (const auto* imm = coalesced_width.as<IntImmNode>()) {
          int expected = imm->value;
          // Verify that vector_size is divisible by expected
          if (vector_size % expected != 0) {
            LOG(FATAL) << "Vector size " << vector_size << " is not divisible by coalesced width "
                       << expected;
          }
          vector_size = expected;
        } else {
          LOG(FATAL) << "coalesced_width should be an IntImmNode.";
        }
      }

      loop_layout_ = PlanLoopPartition(root_, T.block_size, vector_size);
    }
    PrimExpr loop_thread_extent = loop_layout_->ThreadExtent();
    if (!analyzer_.CanProveEqual(loop_thread_extent, static_cast<int>(T.block_size)))
      AddPredicate(LT(InputPlaceholder(0), loop_thread_extent));
  } else {
    return {};
  }
  // Step 2: Check that the loop's partition can correctly align with all source fragment
  for (const auto& [buffer, _] : indice_map_) {
    if (T.layout_map.count(buffer)) {
      auto fragment = T.layout_map[buffer].as<Fragment>().value();
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
    if (!T.layout_map.count(buffer)) results.Set(buffer, CompleteBufferFragment(buffer));
  }
  return results;
}

Optional<PrimExpr> ParallelOp::GetPredicate(Var thread_var) const {
  if (predicate_.defined()) {
    return Substitute(predicate_.value(), {{InputPlaceholder(0), thread_var}});
  } else {
    return NullOpt;
  }
}

Fragment ParallelOp::CompleteBufferFragment(const Buffer& buffer) {
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

  Array<PrimExpr> fwd;
  for (size_t i = 0; i < buffer->shape.size(); i++) {
    fwd.push_back(InputPlaceholder(i));
  }
  fwd.push_back(FloorMod(ReplicationPlaceholder(), indice_rep_extent));
  PrimExpr thd_b = loop_layout_->ForwardThread(
      ind_inv->Forward(fwd), FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

  return Fragment(buffer->shape, {}, thd_b, dest_buffer_rep_extent, NullOpt)
      ->CondenseReplicateVar();
}

}  // namespace tl
}  // namespace tvm
