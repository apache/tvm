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
 * \file s_tir/analysis/sblock_access_region_detector.cc
 * \brief Detect sblock read/write regions by visiting its body
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op.h>

#include <unordered_map>
#include <unordered_set>

#include "../../tirx/transform/ir_utils.h"
#include "../transform/ir_utils.h"
#include "conditional_bounds.h"

namespace tvm {
namespace tirx {

/*!
 * \brief Detect which regions of tensors in this block are read or written to. Regions are sorted
 * by order of appearance in the AST. \note This detector can only visit blocks and will not visit
 * child blocks recursively
 */
class BlockReadWriteDetector : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  explicit BlockReadWriteDetector(const ffi::Map<Var, BufferVar>& buffer_var_map)
      : buffer_var_map_(buffer_var_map) {
    for (const auto& item : buffer_var_map) {
      const BufferVar& buffer = item.second;
      buffer_var_map_.Set(buffer.var(), buffer);
    }
  }

  /*! \brief Return read regions of the block */
  ffi::Array<TensorRegion> CollectReads(
      const std::unordered_set<const VarNode*>* excluded_buffers = nullptr);
  /*! \brief Return write regions of the block */
  ffi::Array<TensorRegion> CollectWrites(
      const std::unordered_set<const VarNode*>* excluded_buffers = nullptr);
  /*!
   * \brief Return opaque buffer regions of the block
   * \note The buffer accessed by load/store or call with buffer.data will
   *       be marked as opaque.
   */
  ffi::Array<TensorRegion> CollectOpaques();
  /*! \brief overload operator() to make sure it accepts a block node */
  void operator()(const Stmt& stmt);

 private:
  /*! \brief Iteration range for loop_vars */
  std::unordered_map<const VarNode*, sym::IntSet> dom_map_;
  /*! \brief Extra iteration range hint for free vars */
  std::unordered_map<const VarNode*, sym::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The buffers that the current block reads */
  std::vector<BufferVar> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<BufferVar> writes_buffers_;
  /*! \brief The opaque buffer which is access by buffer.data */
  std::vector<BufferVar> opaque_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> write_regions_;
  /*! \brief The opaque regions of the current block */
  std::vector<std::vector<tvm::sym::IntSet>> opaque_regions_;
  /*! \brief The outside buffer data mapping to its buffer */
  ffi::Map<Var, BufferVar> buffer_var_map_;
  /*! \brief The target buffer var mapping to its matching */
  std::unordered_map<const VarNode*, s_tir::MatchBufferRegion> match_buffers_;
  /*! \brief let bindings inside the block */
  std::unordered_map<const VarNode*, PrimExpr> let_bindings_;
  /*!\ brief Internal analyzer. */
  sym::Analyzer ana_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and region
   * \param buffers The buffers should be updated
   * \param regions The access regions should be updated
   * \param buffer The provided buffer
   * \param region The provided region
   */
  void Update(std::vector<BufferVar>* buffers, std::vector<std::vector<sym::IntSet>>* regions,
              BufferVar buffer, std::vector<sym::IntSet> region);

  /*! \brief Helper function to collect access regions. */
  ffi::Array<TensorRegion> CollectRegions(
      const std::vector<BufferVar>& buffers,
      const std::vector<std::vector<tvm::sym::IntSet>>& regions,
      const std::unordered_set<const VarNode*>* excluded_buffers = nullptr);

  /*! \brief Helper function to convert matched access region to source region. */
  std::vector<sym::IntSet> ConvertMatchedRegion(const s_tir::MatchBufferRegion& match_buffer,
                                                const std::vector<sym::IntSet>& int_sets) const;

  /*! \brief Helper function to update a opaque access. */
  void UpdateOpaque(const Var& buffer_var);

  /*! \brief Helper function to relax the buffer indices */
  sym::IntSet RelaxAccessIndex(const PrimExpr& index);

  // Declared regions carry bounds, not opaque runtime accesses.
  ffi::Optional<VisitInterrupt> Visit_(const TensorRegionNode* op) final {
    if (!op->source.as<BufferVar>()) return StmtExprVisitor::Visit_(op);
    for (const Range& range : op->region) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->min));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(range->extent));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const IfThenElseNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const s_tir::SBlockRealizeNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const DeclBufferNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const BufferStoreNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const BindNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) override;
};

void BlockReadWriteDetector::operator()(const Stmt& stmt) {
  const auto* block = stmt.as<s_tir::SBlockNode>();
  TVM_FFI_ICHECK(block != nullptr)
      << "Only visiting Blocks is allowed, but got " << stmt->GetTypeKey();
  for (const s_tir::MatchBufferRegion& match_buffer : block->match_buffers) {
    const Var target_var = match_buffer->buffer.var();
    const Var source_var = match_buffer->source->source.as_or_throw<tvm::tirx::BufferVar>().var();
    if (buffer_var_map_.find(source_var) != buffer_var_map_.end()) {
      match_buffers_[target_var.get()] = match_buffer;
      buffer_var_map_.Set(target_var, match_buffer->buffer);
    }
  }
  s_tir::StmtExprVisitor::Visit(stmt);
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectReads(
    const std::unordered_set<const VarNode*>* excluded_buffers) {
  return CollectRegions(read_buffers_, read_regions_, excluded_buffers);
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectWrites(
    const std::unordered_set<const VarNode*>* excluded_buffers) {
  return CollectRegions(writes_buffers_, write_regions_, excluded_buffers);
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectOpaques() {
  return CollectRegions(opaque_buffers_, opaque_regions_);
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const VarNode* op) {
  if (def_region_kind() != kTVMFFIDefRegionKindNone) return std::nullopt;
  UpdateOpaque(ffi::GetRef<Var>(op));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const TensorLoadNode* op) {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = let_bindings_.find(var.get()); it != let_bindings_.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  std::vector<sym::IntSet> relaxed_region;
  for (PrimExpr index : op->indices) {
    PrimExpr remapped_index =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute).as_or_throw<PrimExpr>();
    while (!remapped_index.same_as(index)) {
      index = remapped_index;
      remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                           .as_or_throw<PrimExpr>();
    }
    relaxed_region.push_back(sym::EvalSet(sym::IntSet::Vector(remapped_index), dom_map_));
  }
  Update(&read_buffers_, &read_regions_, op->source.as_or_throw<tvm::tirx::BufferVar>(),
         relaxed_region);
  for (const auto& index : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const ForNode* op) {
  Range range = Range::FromMinExtent(op->min, op->extent);
  dom_map_[op->loop_var.get()] = sym::IntSet::FromRange(range);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
  dom_map_.erase(op->loop_var.get());
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const IfThenElseNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->condition));
  {
    // Visit then branch
    With<s_tir::ConditionalBoundsContext> ctx(op->condition, &dom_map_, &hint_map_,
                                              &pending_conditions_);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit(op->then_case));
  }
  if (op->else_case) {
    // Visit else branch
    With<s_tir::ConditionalBoundsContext> ctx(!op->condition, &dom_map_, &hint_map_,
                                              &pending_conditions_);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit(op->else_case.value()));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const DeclBufferNode* op) {
  // A DeclBuffer data expression defines the alias source.  It is not an
  // opaque buffer access by the containing block.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
      WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() { return Visit(op->buffer); }));
  return VisitBufferMetadata(op->buffer);
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const BindNode* op) {
  if (auto value = op->value.as<PrimExpr>()) {
    let_bindings_[op->var.get()] = value.value();
  }
  return s_tir::StmtExprVisitor::Visit_(op);
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const CallNode* op) {
  auto update_masked_access = [this](const BufferVar& buffer, const ffi::Array<PrimExpr>& indices,
                                     std::vector<BufferVar>* buffers,
                                     std::vector<std::vector<sym::IntSet>>* regions) {
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto it = let_bindings_.find(var.get()); it != let_bindings_.end()) {
        return ffi::Any(it->second);
      }
      return ffi::Unchanged();
    };
    std::vector<sym::IntSet> relaxed_region;
    for (PrimExpr index : indices) {
      PrimExpr remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                                    .as_or_throw<PrimExpr>();
      while (!remapped_index.same_as(index)) {
        index = remapped_index;
        remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                             .as_or_throw<PrimExpr>();
      }
      relaxed_region.push_back(sym::EvalSet(sym::IntSet::Vector(remapped_index), dom_map_));
    }
    Update(buffers, regions, buffer, relaxed_region);
  };
  if (op->op.same_as(tirx::builtin::masked_load()) ||
      op->op.same_as(tirx::builtin::masked_store())) {
    bool is_load = op->op.same_as(tirx::builtin::masked_load());
    BufferVar buffer(op->args[0].as_or_throw<Var>());
    ffi::Array<PrimExpr> indices;
    for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
      indices.push_back(op->args[i].as_or_throw<PrimExpr>());
    }
    update_masked_access(buffer, indices, is_load ? &read_buffers_ : &writes_buffers_,
                         is_load ? &read_regions_ : &write_regions_);
    for (size_t i = 1; i < op->args.size(); ++i) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->args[i]));
    }
    return std::nullopt;
  }
  if (op->op.same_as(tirx::builtin::tvm_access_ptr())) {
    const VarNode* buffer_var = op->args[1].as<VarNode>();
    if (const auto* data = op->args[1].as<CallNode>();
        data && data->op.same_as(tirx::builtin::buffer_data())) {
      buffer_var = data->args[0].as<VarNode>();
    }
    const prim::IntImmNode* access_mask = op->args[4].as<prim::IntImmNode>();
    if (buffer_var && access_mask) {
      auto it = buffer_var_map_.find(ffi::GetRef<Var>(buffer_var));
      if (it != buffer_var_map_.end()) {
        const BufferVar& buffer = (*it).second;
        const TensorRegion buffer_region = FullBufferRegion(buffer);
        const ffi::Array<Range>& region = buffer_region->region;
        std::vector<sym::IntSet> int_set;
        int_set.reserve(region.size());
        for (const Range& range : region) {
          int_set.push_back(sym::EvalSet(range, dom_map_));
        }
        // read access, write access or opaque access
        if ((access_mask->value & 1) && (access_mask->value & 2)) {
          Update(&opaque_buffers_, &opaque_regions_, buffer, int_set);
        } else if (access_mask->value & 1) {
          Update(&read_buffers_, &read_regions_, buffer, int_set);
        } else if (access_mask->value & 2) {
          Update(&writes_buffers_, &write_regions_, buffer, int_set);
        }
      }
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }
  if (op->op.same_as(prim::builtin::if_then_else())) {
    PrimExpr condition = op->args[0].as_or_throw<PrimExpr>();
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(condition));
    {
      // Visit then branch
      With<s_tir::ConditionalBoundsContext> ctx(condition, &dom_map_, &hint_map_,
                                                &pending_conditions_);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
          s_tir::StmtExprVisitor::Visit(op->args[1].as_or_throw<PrimExpr>()));
    }
    {
      // Visit else branch
      With<s_tir::ConditionalBoundsContext> ctx(!condition, &dom_map_, &hint_map_,
                                                &pending_conditions_);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
          s_tir::StmtExprVisitor::Visit(op->args[2].as_or_throw<PrimExpr>()));
    }
    return std::nullopt;
  }
  return s_tir::StmtExprVisitor::Visit_(op);
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const BufferStoreNode* op) {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = let_bindings_.find(var.get()); it != let_bindings_.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  std::vector<sym::IntSet> relaxed_region;
  for (PrimExpr index : op->indices) {
    PrimExpr remapped_index =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute).as_or_throw<PrimExpr>();
    while (!remapped_index.same_as(index)) {
      index = remapped_index;
      remapped_index = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute)
                           .as_or_throw<PrimExpr>();
    }
    relaxed_region.push_back(sym::EvalSet(sym::IntSet::Vector(remapped_index), dom_map_));
  }
  Update(&writes_buffers_, &write_regions_, op->buffer, relaxed_region);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
  for (const auto& index : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> BlockReadWriteDetector::Visit_(const s_tir::SBlockRealizeNode* op) {
  /*! \note detector will not visit child block recursively, so it will stop here */
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (size_t i = 0; i < op->block->iter_vars.size(); ++i) {
    vmap[op->block->iter_vars[i]->var.get()] = op->iter_values[i];
  }
  auto f_substitute = [&vmap](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = vmap.find(var.get()); it != vmap.end()) return ffi::Any(it->second);
    return ffi::Unchanged();
  };
  for (const auto& read : op->block->reads) {
    std::vector<sym::IntSet> relaxed_region;
    for (const auto& range : read->region) {
      PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->min, f_substitute)
                         .as_or_throw<PrimExpr>();
      PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                            .as_or_throw<PrimExpr>();
      relaxed_region.push_back(
          sym::EvalSet(sym::IntSet::FromRange(Range::FromMinExtent(min, extent)), dom_map_));
    }
    Update(&read_buffers_, &read_regions_, read->source.as_or_throw<tvm::tirx::BufferVar>(),
           relaxed_region);
  }
  for (const auto& write : op->block->writes) {
    std::vector<sym::IntSet> relaxed_region;
    for (const auto& range : write->region) {
      PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->min, f_substitute)
                         .as_or_throw<PrimExpr>();
      PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                            .as_or_throw<PrimExpr>();
      relaxed_region.push_back(
          sym::EvalSet(sym::IntSet::FromRange(Range::FromMinExtent(min, extent)), dom_map_));
    }
    Update(&writes_buffers_, &write_regions_, write->source.as_or_throw<tvm::tirx::BufferVar>(),
           relaxed_region);
  }
  return std::nullopt;
}

std::vector<sym::IntSet> BlockReadWriteDetector::ConvertMatchedRegion(
    const s_tir::MatchBufferRegion& match_buffer, const std::vector<sym::IntSet>& int_sets) const {
  const BufferVar& buffer = match_buffer->buffer;

  ffi::Array<Range> region;
  region.reserve(int_sets.size());
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), int_sets.size());
  for (size_t i = 0; i < int_sets.size(); ++i) {
    const tvm::sym::IntSet& int_set = int_sets[i];
    region.push_back(int_set.CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
  }

  region = ConvertRegion(match_buffer, region);

  std::vector<sym::IntSet> result;
  result.reserve(region.size());
  for (const Range& range : region) {
    result.push_back(sym::EvalSet(range, dom_map_));
  }
  return result;
}

void BlockReadWriteDetector::Update(std::vector<BufferVar>* buffers,
                                    std::vector<std::vector<sym::IntSet>>* regions,
                                    BufferVar buffer, std::vector<sym::IntSet> region) {
  if (buffer_var_map_.find(buffer.var()) == buffer_var_map_.end()) return;
  // Handle match_buffer remap
  auto it = match_buffers_.find(buffer.get());
  if (it != match_buffers_.end()) {
    const s_tir::MatchBufferRegion& match_buffer = it->second;
    buffer = match_buffer->source->source.as_or_throw<tvm::tirx::BufferVar>();
    region = ConvertMatchedRegion(match_buffer, std::move(region));
  }
  TVM_FFI_ICHECK_EQ(buffers->size(), regions->size())
      << " Expected the buffer and regions to have the same size ";
  for (size_t i = 0; i < regions->size(); ++i) {
    if ((*buffers)[i].same_as(buffer)) {
      TVM_FFI_ICHECK_EQ((*regions)[i].size(), region.size()) << "Inconsistent buffer dimension";
      for (size_t j = 0; j < region.size(); ++j) {
        (*regions)[i][j] = sym::Union({(*regions)[i][j], region[j]});
      }
      return;
    }
  }
  buffers->push_back(std::move(buffer));
  regions->push_back(std::move(region));
}

ffi::Array<TensorRegion> BlockReadWriteDetector::CollectRegions(
    const std::vector<BufferVar>& buffers,
    const std::vector<std::vector<tvm::sym::IntSet>>& regions,
    const std::unordered_set<const VarNode*>* excluded_buffers) {
  TVM_FFI_ICHECK_EQ(buffers.size(), regions.size());
  ffi::Array<TensorRegion> res;
  res.reserve(buffers.size());
  for (size_t i = 0; i < regions.size(); ++i) {
    if (excluded_buffers != nullptr && excluded_buffers->count(buffers[i].get())) {
      continue;
    }
    ffi::Array<Range> region;
    region.reserve(regions[i].size());
    TVM_FFI_ICHECK_EQ(buffers[i]->shape.size(), regions[i].size());
    for (size_t j = 0; j < regions[i].size(); j++) {
      const tvm::sym::IntSet& range = regions[i][j];
      if (range.CanProveSinglePoint(ana_)) {
        PrimExpr min = range.min();
        region.push_back(Range::FromMinExtent(min, prim::MakeConst(min.ty(), 1)));
      } else {
        region.push_back(range.CoverRange(Range::FromMinExtent(0, buffers[i]->shape[j])));
      }
    }
    res.push_back(BufferRegion(buffers[i], region));
  }
  return res;
}

void BlockReadWriteDetector::UpdateOpaque(const Var& buffer_var) {
  auto it = buffer_var_map_.find(buffer_var);
  if (it != buffer_var_map_.end()) {
    const BufferVar& buffer = (*it).second;
    const TensorRegion buffer_region = FullBufferRegion(buffer);
    const ffi::Array<Range>& region = buffer_region->region;
    std::vector<sym::IntSet> int_set;
    int_set.reserve(region.size());
    for (const Range& range : region) {
      int_set.push_back(sym::EvalSet(range, dom_map_));
    }
    Update(&opaque_buffers_, &opaque_regions_, buffer, int_set);
  }
}

ffi::Array<ffi::Array<TensorRegion>> GetSBlockAccessRegion(
    const s_tir::SBlock& block, const ffi::Map<Var, BufferVar>& buffer_var_map) {
  auto detector = ffi::make_object<BlockReadWriteDetector>(buffer_var_map);
  detector->operator()(block);
  ffi::Array<TensorRegion> writes = detector->CollectWrites();
  std::unordered_set<const VarNode*> excluded_buffers;
  // exclude write buffers from read regions for reductions if init block is defined.
  if (block->init.has_value()) {
    for (const TensorRegion& write_access : writes) {
      excluded_buffers.insert(write_access->source.as_or_throw<tvm::tirx::BufferVar>().get());
    }
  }
  ffi::Array<TensorRegion> reads = detector->CollectReads(&excluded_buffers);
  ffi::Array<TensorRegion> opaques = detector->CollectOpaques();
  return {reads, writes, opaques};
}

ffi::Array<ffi::Array<TensorRegion>> GetSBlockReadWriteRegion(
    const s_tir::SBlock& block, const ffi::Map<Var, BufferVar>& buffer_var_map) {
  auto detector = ffi::make_object<BlockReadWriteDetector>(buffer_var_map);
  detector->operator()(block);
  ffi::Array<TensorRegion> opaques = detector->CollectOpaques();
  std::unordered_set<const VarNode*> excluded_buffers;
  for (const TensorRegion& opaque_access : opaques) {
    excluded_buffers.insert(opaque_access->source.as_or_throw<tvm::tirx::BufferVar>().get());
  }
  ffi::Array<TensorRegion> writes = detector->CollectWrites(&excluded_buffers);
  if (block->init.has_value()) {
    for (const TensorRegion& write_access : writes) {
      excluded_buffers.insert(write_access->source.as_or_throw<tvm::tirx::BufferVar>().get());
    }
  }
  ffi::Array<TensorRegion> reads = detector->CollectReads(&excluded_buffers);
  for (const TensorRegion& opaque_access : opaques) {
    reads.push_back(opaque_access);
    writes.push_back(opaque_access);
  }
  return {reads, writes};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.analysis.GetSBlockAccessRegion", GetSBlockAccessRegion)
      .def("s_tir.analysis.GetSBlockReadWriteRegion", GetSBlockReadWriteRegion);
}

}  // namespace tirx
}  // namespace tvm
