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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>

#include "../../tirx/transform/ir_utils.h"
#include "./utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/******** Annotation ********/

SBlock WithAnnotation(const SBlockNode* block, const ffi::String& attr_key,
                      const ffi::ObjectRef& attr_value) {
  ffi::Map<ffi::String, Any> annotations = block->annotations;
  annotations.Set(attr_key, attr_value);
  ffi::ObjectPtr<SBlockNode> new_block = ffi::make_object<SBlockNode>(*block);
  new_block->annotations = std::move(annotations);
  return SBlock(new_block);
}

/******** Buffer Related ********/
BufferVar WithScope(const BufferVar& buffer, const ffi::String& scope) {
  BufferType new_type(scope, buffer->dtype, buffer->shape, buffer->strides, buffer->elem_offset,
                      buffer->data_alignment, buffer->offset_factor, buffer->layout,
                      buffer->allocated_addr);
  return BufferVar(buffer.name() + "_" + scope, new_type, buffer.span());
}

BufferVar WithDType(const BufferVar& buffer, PrimType dtype) {
  BufferType new_type(buffer->storage_scope, dtype, buffer->shape, buffer->strides,
                      buffer->elem_offset, buffer->data_alignment, buffer->offset_factor,
                      buffer->layout, buffer->allocated_addr);
  return BufferVar(buffer.name(), new_type, buffer.span());
}

ffi::Array<TensorRegion> ReplaceBuffer(ffi::Array<TensorRegion> regions, const BufferVar& source,
                                       const BufferVar& target) {
  regions.MutateByApply([&source, &target](TensorRegion region) -> TensorRegion {
    if (region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(source)) {
      ffi::ObjectPtr<TensorRegionNode> n = ffi::make_object<TensorRegionNode>(*region.get());
      n->source = target;
      return TensorRegion(n);
    }
    return region;
  });
  return regions;
}

ffi::Array<TensorRegion> ReplaceBuffer(ffi::Array<TensorRegion> regions,
                                       const ffi::Map<BufferVar, BufferVar>& buffer_map) {
  regions.MutateByApply([&buffer_map](TensorRegion region) -> TensorRegion {
    if (buffer_map.count(region->source.as_or_throw<tvm::tirx::BufferVar>())) {
      ffi::ObjectPtr<TensorRegionNode> n = ffi::make_object<TensorRegionNode>(*region.get());
      n->source = buffer_map[region->source.as_or_throw<tvm::tirx::BufferVar>()];
      return TensorRegion(n);
    }
    return region;
  });
  return regions;
}

ffi::Array<MatchBufferRegion> ReplaceBuffer(ffi::Array<MatchBufferRegion> match_buffers,
                                            const BufferVar& source, const BufferVar& target) {
  match_buffers.MutateByApply(
      [&source, &target](MatchBufferRegion match_buffer) -> MatchBufferRegion {
        if (match_buffer->source->source.as_or_throw<tvm::tirx::BufferVar>().same_as(source)) {
          ffi::ObjectPtr<MatchBufferRegionNode> n =
              ffi::make_object<MatchBufferRegionNode>(*match_buffer.get());
          n->source = BufferRegion(target, n->source->region);
          return MatchBufferRegion(n);
        }
        return match_buffer;
      });
  return match_buffers;
}

ffi::Array<TensorRegion> ReplaceBufferRegion(ffi::Array<TensorRegion> regions,
                                             const BufferVar& source_buffer,
                                             const TensorRegion& target) {
  regions.MutateByApply([&source_buffer, &target](const TensorRegion& region) -> TensorRegion {
    if (region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(source_buffer)) {
      return target;
    }
    return region;
  });
  return regions;
}

ffi::Array<MatchBufferRegion> ReplaceBufferRegion(ffi::Array<MatchBufferRegion> match_buffers,
                                                  const BufferVar& source_buffer,
                                                  const TensorRegion& target) {
  match_buffers.MutateByApply([&source_buffer, &target](
                                  const MatchBufferRegion& match_buffer) -> MatchBufferRegion {
    if (match_buffer->source->source.as_or_throw<tvm::tirx::BufferVar>().same_as(source_buffer)) {
      ffi::ObjectPtr<MatchBufferRegionNode> n =
          ffi::make_object<MatchBufferRegionNode>(*match_buffer.get());
      n->source = target;
      return MatchBufferRegion(n);
    }
    return match_buffer;
  });
  return match_buffers;
}

/******** ReplaceBufferMutator ********/
ReplaceBufferMutator::ReplaceBufferMutator(const BufferVar& old_buffer, BufferVar new_buffer,
                                           ffi::Map<SBlock, SBlock>* block_sref_reuse)
    : block_sref_reuse_(block_sref_reuse) {
  VarRemapSet(old_buffer, new_buffer);
}

ReplaceBufferMutator::ReplaceBufferMutator(const ffi::Map<BufferVar, BufferVar>& buffer_map,
                                           ffi::Map<SBlock, SBlock>* block_sref_reuse)
    : block_sref_reuse_(block_sref_reuse) {
  for (const auto& [old_buffer, new_buffer] : buffer_map) {
    VarRemapSet(old_buffer, new_buffer);
  }
}

UnchangedOr<Expr> ReplaceBufferMutator::Mutate_(const CallNode* op, InplaceMode inplace_mode) {
  auto result = StmtExprMutator::Mutate_(op, inplace_mode);
  if (!result.IsUnchanged()) {
    op = ffi::AnyView(result).as<CallNode>();
    if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
  }
  if (!op->op.same_as(tirx::builtin::buffer_data()) || op->args.size() != 1) return result;
  PointerType type = op->args[0].as_or_throw<BufferVar>().DataPointerType();
  if (ffi::StructuralEqual()(op->ty, type)) return result;
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<CallNode*>(op)->ty = std::move(type);
    return result;
  }
  auto copy = ffi::make_object<CallNode>(*op);
  copy->ty = std::move(type);
  return Expr(std::move(copy));
}

MatchBufferRegion ReplaceBufferMutator::VisitMatchBufferRegion(
    const MatchBufferRegion& match_buffer) {
  if (auto replacement =
          VarRemapGet(match_buffer->source->source.as_or_throw<tvm::tirx::BufferVar>())
              .as<BufferVar>()) {
    return MatchBufferRegion(match_buffer->buffer,
                             BufferRegion(replacement.value(), match_buffer->source->region));
  } else {
    return match_buffer;
  }
}

UnchangedOr<Stmt> ReplaceBufferMutator::Mutate_(const SBlockNode* block, InplaceMode inplace_mode) {
  // To reduce the number of blocks in block sref reuse map, we check whether the block is really
  // mutated (i.e., the old buffer appears in the block). If so, we return the block after
  // mutation. Otherwise we just return the original block.

  auto f_mutate_match_buffer = [this](const MatchBufferRegion& match_buffer) {
    return this->VisitMatchBufferRegion(match_buffer);
  };
  auto f_mutate_read_write_region = [this](const TensorRegion& buffer_region) {
    auto region = MutateArray(buffer_region->region, [this](const Range& range) {
      auto min_result = Mutate(range->min, InplaceMode::kDisallow);
      bool min_unchanged = min_result.UnchangedOrSameAs(range->min);
      PrimExpr min = std::move(min_result).ValueOrUnchanged(range->min);
      auto extent_result = Mutate(range->extent, InplaceMode::kDisallow);
      bool extent_unchanged = extent_result.UnchangedOrSameAs(range->extent);
      PrimExpr extent = std::move(extent_result).ValueOrUnchanged(range->extent);
      if (min_unchanged && extent_unchanged) {
        return range;
      } else {
        return Range::FromMinExtent(min, extent);
      }
    });

    BufferVar buf = VarRemapGet(buffer_region->source.as_or_throw<tvm::tirx::BufferVar>())
                        .as<BufferVar>()
                        .value_or(buffer_region->source.as_or_throw<tvm::tirx::BufferVar>());

    if (buf.same_as(buffer_region->source.as_or_throw<tvm::tirx::BufferVar>()) &&
        region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(buf, region);
    }
  };
  auto f_mutate_alloc_buffers = [this](const BufferVar& buffer) {
    return VarRemapGet(buffer).as<BufferVar>().value_or(buffer);
  };

  // Step 1. Mutate `match_buffers`. If an old buffer appears as a source of MatchBufferRegion,
  ffi::Array<MatchBufferRegion> match_buffers = block->match_buffers.Map(f_mutate_match_buffer);
  // Step 2. Mutate the read/write region.
  ffi::Array<TensorRegion> reads = block->reads.Map(f_mutate_read_write_region);
  ffi::Array<TensorRegion> writes = block->writes.Map(f_mutate_read_write_region);
  // Step 3. Mutate `alloc_buffers` for the old buffer allocated in this block.
  ffi::Array<BufferVar> alloc_buffers = block->alloc_buffers.Map(f_mutate_alloc_buffers);
  // Step 4. Recursively mutate the block.
  SBlock mutated_block = StmtExprMutator::Mutate_(block, inplace_mode)
                             .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                             .as_or_throw<SBlock>();

  if (mutated_block.get() == block && reads.same_as(mutated_block->reads) &&
      writes.same_as(mutated_block->writes) &&
      alloc_buffers.same_as(mutated_block->alloc_buffers) &&
      match_buffers.same_as(mutated_block->match_buffers)) {
    return ffi::Unchanged();
  } else {
    SBlockNode* n = mutated_block.CopyOnWrite();
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->alloc_buffers = std::move(alloc_buffers);
    n->match_buffers = std::move(match_buffers);

    SBlock new_block = std::move(mutated_block);
    if (block_sref_reuse_ != nullptr) {
      block_sref_reuse_->Set(ffi::GetRef<SBlock>(block), new_block);
    }
    return new_block;
  }
}

/******** SBlock Removal ********/

void LeafBlockRemovalPlan(const ScheduleState& self, const StmtSRef& leaf_block_sref,
                          Stmt* src_stmt, Stmt* tgt_stmt) {
  class OnlyLeafError : public ScheduleErrorContextObj {
   public:
    explicit OnlyLeafError(IRModule mod, SBlock leaf_block, SBlock scope_root)
        : mod_(mod), leaf_block_(leaf_block), scope_root_(scope_root) {}

    ffi::String FastErrorString() const final {
      return "ScheduleError: Cannot remove the only leaf in the scope";
    }

    ffi::String DetailRenderTemplate() const final {
      return "Block {0} is the only leaf in the scope {1}, which cannot be removed; Otherwise the "
             "scope will be empty.";
    }

    IRModule mod() const final { return mod_; }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
      return {leaf_block_, scope_root_};
    }

    IRModule mod_;
    SBlock leaf_block_;
    SBlock scope_root_;
  };

  // Go upwards until find an ancestor with more than one child
  const StmtNode* last_stmt = leaf_block_sref->stmt;
  StmtSRefNode* sref = leaf_block_sref->parent;
  for (;; last_stmt = sref->stmt, sref = sref->parent) {
    if (const auto* loop = sref->StmtAs<ForNode>()) {
      if (const auto* seq = loop->body.as<SeqStmtNode>()) {
        if (seq->size() > 1) {
          break;
        }
      }
    } else {
      // Removal is not done beyond scope-level.
      // When encountering a block, i.e. the scope root, we simply stop
      break;
    }
  }
  if (const auto* block = sref->StmtAs<SBlockNode>()) {
    auto body = block->body;
    if (const auto* seq = body.as<SeqStmtNode>()) {
      ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*block);
      auto new_seq = RemoveFromSeqStmt(ffi::GetRef<SeqStmt>(seq), ffi::GetRef<Stmt>(last_stmt));
      n->body = new_seq;
      *src_stmt = ffi::GetRef<Stmt>(block);
      *tgt_stmt = Stmt(std::move(n));
      return;
    }
  }
  if (const auto* loop = sref->StmtAs<ForNode>()) {
    if (const auto* seq = loop->body.as<SeqStmtNode>()) {
      ffi::ObjectPtr<ForNode> n = ffi::make_object<ForNode>(*loop);
      n->body = RemoveFromSeqStmt(ffi::GetRef<SeqStmt>(seq), ffi::GetRef<Stmt>(last_stmt));
      *src_stmt = ffi::GetRef<Stmt>(loop);
      *tgt_stmt = Stmt(std::move(n));
      return;
    }
  }
  TVM_FFI_ICHECK(sref != nullptr && sref->stmt != nullptr);
  const auto* leaf_block = TVM_SREF_TO_SBLOCK(leaf_block_sref);
  const auto* scope_block = TVM_SREF_TO_SBLOCK(sref);
  throw MakeScheduleError<OnlyLeafError>(self->mod, ffi::GetRef<SBlock>(leaf_block),
                                         ffi::GetRef<SBlock>(scope_block));
}

ffi::Optional<LoopRV> TileWithTensorIntrin(const s_tir::Schedule& sch,
                                           const s_tir::SBlockRV& block_rv,
                                           const ffi::String& intrin_name, bool allow_padding) {
  ffi::Optional<TensorizeInfo> opt_tensorize_info =
      GetTensorizeLoopMapping(sch->state(), sch->GetSRef(block_rv),
                              tirx::TensorIntrin::Get(intrin_name).value()->desc, allow_padding);
  if (!opt_tensorize_info) return std::nullopt;
  const TensorizeInfoNode* info = opt_tensorize_info.value().get();
  if (info->block_iter_paddings.has_value()) {
    // We have to track whether each producer or consumer is padded.
    // To do so, we first record all the Block's.
    std::unordered_set<const StmtSRefNode*> original_producers, original_consumers;
    {
      for (const auto& p : GetProducers(sch->state(), sch->GetSRef(block_rv)))
        original_producers.insert(p.get());
      for (const auto& c : GetConsumers(sch->state(), sch->GetSRef(block_rv)))
        original_consumers.insert(c.get());
    }

    // Pad. Maybe we can make PadEinsum return the changes it made, to avoid bookkeeping?
    sch->PadEinsum(block_rv, info->block_iter_paddings.value());

    // Now we need to find out all the padded Block's.
    ffi::Array<SBlockRV> inlined_producers, inlined_consumers;
    for (const auto& producer : sch->GetProducers(block_rv)) {
      // PadEinsum will not modify the producer if it does not need padding.
      if (original_producers.count(sch->GetSRef(producer).get())) {
        // Producer not padded. No inlining.
        continue;
      }
      auto the_original_producers = sch->GetProducers(producer);
      if (the_original_producers.empty()) {
        // The original producer is input.
        continue;
      }
      TVM_FFI_ICHECK_EQ(the_original_producers.size(), 1u);
      auto the_original_producer = the_original_producers[0];
      TVM_FFI_ICHECK(original_producers.count(sch->GetSRef(the_original_producer).get()));
      inlined_producers.push_back(the_original_producer);
    }
    for (const auto& consumer : sch->GetConsumers(block_rv)) {
      // PadEinsum will not modify the consumer if it does not need padding.
      if (original_consumers.count(sch->GetSRef(consumer).get())) {
        // Consumer not padded. No inlining.
        continue;
      }
      auto the_original_consumers = sch->GetConsumers(consumer);
      if (the_original_consumers.empty()) {
        // The original consumer is output.
        continue;
      }
      TVM_FFI_ICHECK_EQ(the_original_consumers.size(), 1u);
      auto the_original_consumer = the_original_consumers[0];
      TVM_FFI_ICHECK(original_consumers.count(sch->GetSRef(the_original_consumer).get()));
      inlined_consumers.push_back(consumer);
    }

    // Inline the producer and consumer padding blocks
    for (const auto& the_original_producer : inlined_producers) {
      // Inline the original producer into the padding block. This ensures that the new producer
      // has the padded shape.
      sch->ComputeInline(the_original_producer);
    }
    for (const auto& consumer : inlined_consumers) {
      sch->ComputeInline(consumer);
    }
  }
  // Construct a mapping from tirx loops back to LoopRVs
  ffi::Map<tirx::StmtSRef, LoopRV> loop2rv;
  {
    ffi::Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    for (const LoopRV& loop_rv : loop_rvs) {
      loop2rv.Set(sch->GetSRef(loop_rv), loop_rv);
    }
  }
  // Split the loops
  arith::Analyzer analyzer;
  std::unordered_set<const tirx::StmtSRefNode*> inner_loops;
  std::vector<LoopRV> reorder_suffix;
  reorder_suffix.resize(info->loop_map.size());
  for (const auto& kv : info->loop_map) {
    // Extract mapping (block_loop => desc_loop)
    const tirx::StmtSRef& block_loop_sref = kv.first;
    const tirx::ForNode* block_loop = block_loop_sref->StmtAs<tirx::ForNode>();
    const tirx::ForNode* desc_loop = kv.second.get();
    TVM_FFI_ICHECK(block_loop != nullptr && desc_loop != nullptr);
    // Extract the loop extent
    PrimExpr block_extent = analyzer->Simplify(block_loop->extent);
    PrimExpr desc_extent = analyzer->Simplify(desc_loop->extent);
    const auto* int_block_extent = block_extent.as<IntImmNode>();
    const auto* int_desc_extent = desc_extent.as<IntImmNode>();
    TVM_FFI_ICHECK(int_block_extent != nullptr && int_desc_extent != nullptr);
    // Check divisibility
    const ffi::BigInt& total = int_block_extent->value;
    const ffi::BigInt& inner = int_desc_extent->value;
    TVM_FFI_ICHECK_EQ(total % inner, 0);
    // Do the split. Leave the outer extent as std::nullopt (unspecified) so that the split factors
    // can be used for different extents (needed during tuning).
    ffi::Array<LoopRV> split =
        sch->Split(loop2rv.at(block_loop_sref), {std::nullopt, IntImm::Int32(inner)});
    TVM_FFI_ICHECK_EQ(split.size(), 2);
    inner_loops.insert(sch->GetSRef(split[1]).operator->());
    // The inner split will be reordered to the loop domain that is tensorized
    int desc_loop_index =
        static_cast<int>(info->desc_loop_indexer.at(ffi::GetRef<tirx::For>(desc_loop)));
    reorder_suffix[desc_loop_index] = split[1];
  }
  // Reorder the loops
  std::vector<LoopRV> reorder_list;
  bool meet = false;
  ffi::Array<LoopRV> all_loops = sch->GetLoops(block_rv);
  for (const LoopRV& loop : all_loops) {
    if (inner_loops.count(sch->GetSRef(loop).operator->())) {
      meet = true;
    } else if (meet) {
      reorder_list.push_back(loop);
    }
  }
  reorder_list.insert(reorder_list.end(), reorder_suffix.begin(), reorder_suffix.end());
  sch->Reorder(reorder_list);
  TVM_FFI_ICHECK(!reorder_suffix.empty());
  return reorder_suffix[0];
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.TileWithTensorIntrin", TileWithTensorIntrin);
}

/******** BlockBufferAccessSimplifier ********/
void BlockBufferAccessSimplifier::SimplifyAccessRegion(
    ffi::Array<TensorRegion>* old_access_regions) {
  auto fmutate = [this](const TensorRegion& buffer_region) {
    ffi::Array<Range> new_buffer_region;
    ffi::Array<PrimExpr> simplified_min;
    for (const auto& range : buffer_region->region) {
      simplified_min.push_back(range->min);
    }
    simplified_min = this->IterMapSimplifyWithContext(simplified_min, true);
    int n = buffer_region->region.size();
    for (int i = 0; i < n; ++i) {
      PrimExpr min = simplified_min[i];
      PrimExpr extent = analyzer_->Simplify(buffer_region->region[i]->extent);
      new_buffer_region.push_back(Range::FromMinExtent(min, extent));
    }
    return BufferRegion(buffer_region->source.as_or_throw<tvm::tirx::BufferVar>(),
                        new_buffer_region);
  };
  (*old_access_regions).MutateByApply(fmutate);
}

void BlockBufferAccessSimplifier::SimplifyBufferIndices(ffi::Array<PrimExpr>* indices) {
  *indices = this->IterMapSimplifyWithContext(*indices, true);
}

UnchangedOr<Stmt> BlockBufferAccessSimplifier::Mutate_(const SBlockNode* op,
                                                       InplaceMode inplace_mode) {
  SBlock block = tirx::IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<SBlock>();
  auto* n = block.CopyOnWrite();
  SimplifyAccessRegion(&n->reads);
  SimplifyAccessRegion(&n->writes);
  return block;
}

UnchangedOr<Stmt> BlockBufferAccessSimplifier::Mutate_(const BufferStoreNode* op,
                                                       InplaceMode inplace_mode) {
  BufferStore node = tirx::IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                         .as_or_throw<BufferStore>();
  SimplifyBufferIndices(&node.CopyOnWrite()->indices);
  return node;
}

UnchangedOr<PrimExpr> BlockBufferAccessSimplifier::Mutate_(const TensorLoadNode* op,
                                                           InplaceMode inplace_mode) {
  TensorLoad node = tirx::IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                        .as_or_throw<TensorLoad>();
  SimplifyBufferIndices(&node.CopyOnWrite()->indices);
  return node;
}

/******** PrimFunc-level analysis and transformation ********/

void GetLeafBlocksHelper(Schedule sch, SBlockRV cur_block_rv, ffi::Array<SBlockRV>* leaf_blocks) {
  ffi::Array<SBlockRV> blocks = sch->GetChildBlocks(cur_block_rv);
  if (blocks.empty()) {
    leaf_blocks->push_back(cur_block_rv);
  } else {
    for (const SBlockRV& block : blocks) {
      GetLeafBlocksHelper(sch, block, leaf_blocks);
    }
  }
}

ffi::Optional<ffi::ObjectRef> NormalizePrimFunc(Schedule sch) {
  SBlockRV root_block = sch->GetSBlock("root");
  ffi::Array<SBlockRV> leaf_blocks;
  GetLeafBlocksHelper(sch, root_block, &leaf_blocks);
  for (const SBlockRV& block : leaf_blocks) {
    StmtSRef block_sref = sch->GetSRef(block);
    ffi::Array<StmtSRef> loops = GetLoops(block_sref);
    ffi::Array<PrimExpr> binds = GetSBlockRealize(sch->state(), block_sref)->iter_values;
    if (loops.size() == 0) continue;
    if (loops.size() != binds.size()) {
      return std::nullopt;
    }
    for (int i = 0, n = loops.size(); i < n; ++i) {
      const ForNode* loop = TVM_SREF_TO_FOR(loops[i]);
      if (binds[i].get() != loop->loop_var.get()) {
        return std::nullopt;
      }
      if (!is_zero(loop->min)) {
        return std::nullopt;
      }
    }
  }

  ffi::Array<ffi::Array<LoopRV>> block_loops;
  ffi::Array<ffi::Array<IterVar>> block_iters;
  ffi::Array<IntImm> block_is_reduction;
  for (const SBlockRV& block : leaf_blocks) {
    ffi::Array<IterVar> iters = sch->Get(block)->iter_vars;
    bool has_spatial_iter = false;
    ffi::Array<PrimVar> index_map_inputs;
    ffi::Array<PrimExpr> index_map_outputs;
    for (const IterVar& iter : sch->Get(block)->iter_vars) {
      PrimVar var = iter->var.CopyWithSuffix("");
      index_map_inputs.push_back(var);
      if (!is_one(iter->dom->extent)) {
        index_map_outputs.push_back(var);
        if (iter->iter_type == IterVarType::kDataPar) {
          has_spatial_iter = true;
        }
      }
    }
    if (index_map_outputs.empty() || !has_spatial_iter) {
      index_map_outputs.insert(index_map_outputs.begin(), IntImm::Int64(0));
    }
    try {
      sch->TransformBlockLayout(block, IndexMap(index_map_inputs, index_map_outputs));
    } catch (tvm::ffi::Error& e) {
      // Skip layout transformation when not transformable.
    }
    block_loops.push_back(sch->GetLoops(block));
    block_iters.push_back(sch->Get(block)->iter_vars);
    bool is_reduction = IsReductionBlock(sch->state(),         //
                                         sch->GetSRef(block),  //
                                         sch->GetSRef(root_block));
    block_is_reduction.push_back(IntImm::Bool(is_reduction));
  }
  return ffi::Array<ffi::ObjectRef>{leaf_blocks, block_loops, block_iters, block_is_reduction};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.NormalizePrimFunc", NormalizePrimFunc);
}

}  // namespace s_tir
}  // namespace tvm
