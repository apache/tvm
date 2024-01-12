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

#include "../transforms/ir_utils.h"
#include "./utils.h"

namespace tvm {
namespace tir {

/******** Annotation ********/

Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = block->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
  new_block->annotations = std::move(annotations);
  return Block(new_block);
}

/******** Buffer Related ********/
Buffer WithScope(const Buffer& buffer, const String& scope) {
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());
  ObjectPtr<VarNode> new_var = make_object<VarNode>(*buffer->data.get());
  const auto* ptr_type = TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  new_var->type_annotation = PointerType(ptr_type->element_type, scope);
  new_buffer->data = Var(new_var->name_hint + "_" + scope, new_var->type_annotation);
  new_buffer->name = buffer->name + "_" + scope;
  return Buffer(new_buffer);
}

Buffer WithDType(const Buffer& buffer, const DataType& dtype) {
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());
  new_buffer->dtype = dtype;
  const auto* ptr_type = TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  new_buffer->data =
      Var(buffer->data->name_hint, PointerType(PrimType(dtype), ptr_type->storage_scope));
  new_buffer->name = buffer->name;
  return Buffer(new_buffer);
}

Array<BufferRegion> ReplaceBuffer(Array<BufferRegion> regions, const Buffer& source,
                                  const Buffer& target) {
  regions.MutateByApply([&source, &target](BufferRegion region) -> BufferRegion {
    if (region->buffer.same_as(source)) {
      ObjectPtr<BufferRegionNode> n = make_object<BufferRegionNode>(*region.get());
      n->buffer = target;
      return BufferRegion(n);
    }
    return region;
  });
  return regions;
}

Array<BufferRegion> ReplaceBuffer(Array<BufferRegion> regions,
                                  const Map<Buffer, Buffer>& buffer_map) {
  regions.MutateByApply([&buffer_map](BufferRegion region) -> BufferRegion {
    if (buffer_map.count(region->buffer)) {
      ObjectPtr<BufferRegionNode> n = make_object<BufferRegionNode>(*region.get());
      n->buffer = buffer_map[region->buffer];
      return BufferRegion(n);
    }
    return region;
  });
  return regions;
}

Array<MatchBufferRegion> ReplaceBuffer(Array<MatchBufferRegion> match_buffers, const Buffer& source,
                                       const Buffer& target) {
  match_buffers.MutateByApply([&source,
                               &target](MatchBufferRegion match_buffer) -> MatchBufferRegion {
    if (match_buffer->source->buffer.same_as(source)) {
      ObjectPtr<MatchBufferRegionNode> n = make_object<MatchBufferRegionNode>(*match_buffer.get());
      n->source = BufferRegion(target, n->source->region);
      return MatchBufferRegion(n);
    }
    return match_buffer;
  });
  return match_buffers;
}

Array<BufferRegion> ReplaceBufferRegion(Array<BufferRegion> regions, const Buffer& source_buffer,
                                        const BufferRegion& target) {
  regions.MutateByApply([&source_buffer, &target](const BufferRegion& region) -> BufferRegion {
    if (region->buffer.same_as(source_buffer)) {
      return target;
    }
    return region;
  });
  return regions;
}

Array<MatchBufferRegion> ReplaceBufferRegion(Array<MatchBufferRegion> match_buffers,
                                             const Buffer& source_buffer,
                                             const BufferRegion& target) {
  match_buffers.MutateByApply([&source_buffer, &target](
                                  const MatchBufferRegion& match_buffer) -> MatchBufferRegion {
    if (match_buffer->source->buffer.same_as(source_buffer)) {
      ObjectPtr<MatchBufferRegionNode> n = make_object<MatchBufferRegionNode>(*match_buffer.get());
      n->source = target;
      return MatchBufferRegion(n);
    }
    return match_buffer;
  });
  return match_buffers;
}

/******** ReplaceBufferMutator ********/
ReplaceBufferMutator::ReplaceBufferMutator(const Buffer& old_buffer, Buffer new_buffer,
                                           Map<Block, Block>* block_sref_reuse)
    : block_sref_reuse_(block_sref_reuse) {
  buffer_var_map_[old_buffer->data.get()] = std::move(new_buffer);
}

ReplaceBufferMutator::ReplaceBufferMutator(const Map<Buffer, Buffer>& buffer_map,
                                           Map<Block, Block>* block_sref_reuse)
    : block_sref_reuse_(block_sref_reuse) {
  for (const auto& [old_buffer, new_buffer] : buffer_map) {
    buffer_var_map_[old_buffer->data.get()] = new_buffer;
  }
}

PrimExpr ReplaceBufferMutator::VisitExpr_(const VarNode* var) {
  auto it = buffer_var_map_.find(var);
  return it != buffer_var_map_.end() ? it->second->data : GetRef<Var>(var);
}

Stmt ReplaceBufferMutator::VisitStmt_(const BufferStoreNode* op) {
  auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
  return VisitBufferAccess(std::move(node));
}

PrimExpr ReplaceBufferMutator::VisitExpr_(const BufferLoadNode* op) {
  auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
  return VisitBufferAccess(std::move(node));
}

MatchBufferRegion ReplaceBufferMutator::VisitMatchBufferRegion(
    const MatchBufferRegion& match_buffer) {
  auto it = buffer_var_map_.find(match_buffer->source->buffer->data.get());
  if (it != buffer_var_map_.end()) {
    return MatchBufferRegion(match_buffer->buffer,
                             BufferRegion(it->second, match_buffer->source->region));
  } else {
    return match_buffer;
  }
}

Stmt ReplaceBufferMutator::VisitStmt_(const BlockNode* block) {
  // To reduce the number of blocks in block sref reuse map, we check whether the block is really
  // mutated (i.e., the old buffer appears in the block). If so, we return the block after
  // mutation. Otherwise we just return the original block.

  auto f_mutate_match_buffer = [this](const MatchBufferRegion& match_buffer) {
    return this->VisitMatchBufferRegion(match_buffer);
  };
  auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
    auto region = MutateArray(buffer_region->region, [this](const Range& range) {
      PrimExpr min = VisitExpr(range->min);
      PrimExpr extent = VisitExpr(range->extent);
      if (min.same_as(range->min) && extent.same_as(range->extent)) {
        return range;
      } else {
        return Range::FromMinExtent(min, extent);
      }
    });

    Buffer buf = [&]() {
      auto it = buffer_var_map_.find(buffer_region->buffer->data.get());
      if (it == buffer_var_map_.end()) {
        return buffer_region->buffer;
      } else {
        return it->second;
      }
    }();

    if (buf.same_as(buffer_region->buffer) && region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(buf, region);
    }
  };
  auto f_mutate_alloc_buffers = [this](const Buffer& buffer) {
    auto it = buffer_var_map_.find(buffer->data.get());
    return it == buffer_var_map_.end() ? buffer : it->second;
  };

  // Step 1. Mutate `match_buffers`. If an old buffer appears as a source of MatchBufferRegion,
  Array<MatchBufferRegion> match_buffers = block->match_buffers.Map(f_mutate_match_buffer);
  // Step 2. Mutate the read/write region.
  Array<BufferRegion> reads = block->reads.Map(f_mutate_read_write_region);
  Array<BufferRegion> writes = block->writes.Map(f_mutate_read_write_region);
  // Step 3. Mutate `alloc_buffers` for the old buffer allocated in this block.
  Array<Buffer> alloc_buffers = block->alloc_buffers.Map(f_mutate_alloc_buffers);
  // Step 4. Recursively mutate the block.
  Block mutated_block = Downcast<Block>(StmtMutator::VisitStmt_(block));

  if (mutated_block.get() == block && reads.same_as(mutated_block->reads) &&
      writes.same_as(mutated_block->writes) &&
      alloc_buffers.same_as(mutated_block->alloc_buffers) &&
      match_buffers.same_as(mutated_block->match_buffers)) {
    return GetRef<Block>(block);
  } else {
    ObjectPtr<BlockNode> n = CopyOnWrite(mutated_block.get());
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->alloc_buffers = std::move(alloc_buffers);
    n->match_buffers = std::move(match_buffers);

    Block new_block(n);
    if (block_sref_reuse_ != nullptr) {
      block_sref_reuse_->Set(GetRef<Block>(block), new_block);
    }
    return std::move(new_block);
  }
}

/******** Block Removal ********/

void LeafBlockRemovalPlan(const ScheduleState& self, const StmtSRef& leaf_block_sref,
                          Stmt* src_stmt, Stmt* tgt_stmt) {
  class OnlyLeafError : public ScheduleError {
   public:
    explicit OnlyLeafError(IRModule mod, Block leaf_block, Block scope_root)
        : mod_(mod), leaf_block_(leaf_block), scope_root_(scope_root) {}

    String FastErrorString() const final {
      return "ScheduleError: Cannot remove the only leaf in the scope";
    }

    String DetailRenderTemplate() const final {
      return "Block {0} is the only leaf in the scope {1}, which cannot be removed; Otherwise the "
             "scope will be empty.";
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {leaf_block_, scope_root_}; }

    IRModule mod_;
    Block leaf_block_;
    Block scope_root_;
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
  if (const auto* block = sref->StmtAs<BlockNode>()) {
    auto body = block->body;
    // Peel off AllocateConst nodes at the beginning of the block body.
    std::vector<Stmt> allocs;
    while (true) {
      if (auto opt = body.as<AllocateConst>()) {
        auto alloc = opt.value();
        body = alloc->body;
        alloc.CopyOnWrite()->body = Evaluate(0);
        allocs.push_back(alloc);
      } else if (auto opt = body.as<DeclBuffer>()) {
        auto decl_buffer = opt.value();
        body = decl_buffer->body;
        decl_buffer.CopyOnWrite()->body = Evaluate(0);
        allocs.push_back(decl_buffer);
      } else {
        break;
      }
    }

    if (const auto* seq = body.as<SeqStmtNode>()) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*block);
      auto new_seq = RemoveFromSeqStmt(GetRef<SeqStmt>(seq), GetRef<Stmt>(last_stmt));
      // Re-attach AllocateConst nodes
      auto new_body = MergeNest(allocs, new_seq);
      n->body = new_body;
      *src_stmt = GetRef<Stmt>(block);
      *tgt_stmt = Stmt(std::move(n));
      return;
    }
  }
  if (const auto* loop = sref->StmtAs<ForNode>()) {
    if (const auto* seq = loop->body.as<SeqStmtNode>()) {
      ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
      n->body = RemoveFromSeqStmt(GetRef<SeqStmt>(seq), GetRef<Stmt>(last_stmt));
      *src_stmt = GetRef<Stmt>(loop);
      *tgt_stmt = Stmt(std::move(n));
      return;
    }
  }
  ICHECK(sref != nullptr && sref->stmt != nullptr);
  const auto* leaf_block = TVM_SREF_TO_BLOCK(leaf_block_sref);
  const auto* scope_block = TVM_SREF_TO_BLOCK(sref);
  throw OnlyLeafError(self->mod, GetRef<Block>(leaf_block), GetRef<Block>(scope_block));
}

Optional<LoopRV> TileWithTensorIntrin(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                      const String& intrin_name, bool allow_padding) {
  Optional<tir::TensorizeInfo> opt_tensorize_info =
      GetTensorizeLoopMapping(sch->state(), sch->GetSRef(block_rv),
                              tir::TensorIntrin::Get(intrin_name).value()->desc, allow_padding);
  if (!opt_tensorize_info) return NullOpt;
  const tir::TensorizeInfoNode* info = opt_tensorize_info.value().get();
  if (info->block_iter_paddings.defined()) {
    sch->PadEinsum(block_rv, info->block_iter_paddings.value());
    // Inline the producer and consumer padding blocks
    auto producers = sch->GetProducers(block_rv);
    for (const auto& producer : producers) {
      auto original_producers = sch->GetProducers(producer);
      // NOTICE: there may not all producers padded.
      // Inline the original producer into the padding block. This ensures that the new producer
      // has the padded shape.
      if (original_producers.size() == 1u) {
        sch->ComputeInline(original_producers[0]);
      }
    }
    auto consumers = sch->GetConsumers(block_rv);
    for (const auto& consumer : consumers) {
      sch->ComputeInline(consumer);
    }
  }
  // Construct a mapping from tir loops back to LoopRVs
  Map<tir::StmtSRef, LoopRV> loop2rv;
  {
    Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    for (const LoopRV& loop_rv : loop_rvs) {
      loop2rv.Set(sch->GetSRef(loop_rv), loop_rv);
    }
  }
  // Split the loops
  arith::Analyzer analyzer;
  std::unordered_set<const tir::StmtSRefNode*> inner_loops;
  std::vector<LoopRV> reorder_suffix;
  reorder_suffix.resize(info->loop_map.size());
  for (const auto& kv : info->loop_map) {
    // Extract mapping (block_loop => desc_loop)
    const tir::StmtSRef& block_loop_sref = kv.first;
    const tir::ForNode* block_loop = block_loop_sref->StmtAs<tir::ForNode>();
    const tir::ForNode* desc_loop = kv.second.get();
    ICHECK(block_loop != nullptr && desc_loop != nullptr);
    // Extract the loop extent
    PrimExpr block_extent = analyzer.Simplify(block_loop->extent);
    PrimExpr desc_extent = analyzer.Simplify(desc_loop->extent);
    const auto* int_block_extent = block_extent.as<IntImmNode>();
    const auto* int_desc_extent = desc_extent.as<IntImmNode>();
    ICHECK(int_block_extent != nullptr && int_desc_extent != nullptr);
    // Check divisibility
    int64_t total = int_block_extent->value;
    int64_t inner = int_desc_extent->value;
    ICHECK_EQ(total % inner, 0);
    // Do the split. Leave the outer extent as NullOpt (unspecified) so that the split factors
    // can be used for different extents (needed during tuning).
    Array<LoopRV> split = sch->Split(loop2rv.at(block_loop_sref), {NullOpt, Integer(inner)});
    ICHECK_EQ(split.size(), 2);
    inner_loops.insert(sch->GetSRef(split[1]).operator->());
    // The inner split will be reordered to the loop domain that is tensorized
    int desc_loop_index = info->desc_loop_indexer.at(GetRef<tir::For>(desc_loop)).IntValue();
    reorder_suffix[desc_loop_index] = split[1];
  }
  // Reorder the loops
  std::vector<LoopRV> reorder_list;
  bool meet = false;
  Array<LoopRV> all_loops = sch->GetLoops(block_rv);
  for (const LoopRV& loop : all_loops) {
    if (inner_loops.count(sch->GetSRef(loop).operator->())) {
      meet = true;
    } else if (meet) {
      reorder_list.push_back(loop);
    }
  }
  reorder_list.insert(reorder_list.end(), reorder_suffix.begin(), reorder_suffix.end());
  sch->Reorder(reorder_list);
  ICHECK(!reorder_suffix.empty());
  return reorder_suffix[0];
}

TVM_REGISTER_GLOBAL("tir.schedule.TileWithTensorIntrin").set_body_typed(TileWithTensorIntrin);

/******** BlockBufferAccessSimplifier ********/
void BlockBufferAccessSimplifier::SimplifyAccessRegion(Array<BufferRegion>* old_access_regions) {
  auto fmutate = [this](const BufferRegion& buffer_region) {
    Array<Range> new_buffer_region;
    Array<PrimExpr> simplified_min;
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
    return BufferRegion(buffer_region->buffer, new_buffer_region);
  };
  (*old_access_regions).MutateByApply(fmutate);
}

void BlockBufferAccessSimplifier::SimplifyBufferIndices(Array<PrimExpr>* indices) {
  *indices = this->IterMapSimplifyWithContext(*indices, true);
}

Stmt BlockBufferAccessSimplifier::VisitStmt_(const BlockNode* op) {
  Block block = Downcast<Block>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
  auto* n = block.CopyOnWrite();
  SimplifyAccessRegion(&n->reads);
  SimplifyAccessRegion(&n->writes);
  return std::move(block);
}

Stmt BlockBufferAccessSimplifier::VisitStmt_(const BufferStoreNode* op) {
  BufferStore node = Downcast<BufferStore>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
  SimplifyBufferIndices(&node.CopyOnWrite()->indices);
  return std::move(node);
}

PrimExpr BlockBufferAccessSimplifier::VisitExpr_(const BufferLoadNode* op) {
  BufferLoad node = Downcast<BufferLoad>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));
  SimplifyBufferIndices(&node.CopyOnWrite()->indices);
  return std::move(node);
}

/******** PrimFunc-level analysis and transformation ********/

void GetLeafBlocksHelper(Schedule sch, BlockRV cur_block_rv, Array<BlockRV>* leaf_blocks) {
  Array<BlockRV> blocks = sch->GetChildBlocks(cur_block_rv);
  if (blocks.empty()) {
    leaf_blocks->push_back(cur_block_rv);
  } else {
    for (const BlockRV& block : blocks) {
      GetLeafBlocksHelper(sch, block, leaf_blocks);
    }
  }
}

Optional<ObjectRef> NormalizePrimFunc(Schedule sch) {
  BlockRV root_block = sch->GetBlock("root");
  Array<BlockRV> leaf_blocks;
  GetLeafBlocksHelper(sch, root_block, &leaf_blocks);
  for (const BlockRV& block : leaf_blocks) {
    StmtSRef block_sref = sch->GetSRef(block);
    Array<StmtSRef> loops = GetLoops(block_sref);
    Array<PrimExpr> binds = GetBlockRealize(sch->state(), block_sref)->iter_values;
    if (loops.size() == 0) continue;
    if (loops.size() != binds.size()) {
      return NullOpt;
    }
    for (int i = 0, n = loops.size(); i < n; ++i) {
      const ForNode* loop = TVM_SREF_TO_FOR(loops[i]);
      if (binds[i].get() != loop->loop_var.get()) {
        return NullOpt;
      }
      if (!is_zero(loop->min)) {
        return NullOpt;
      }
    }
  }

  Array<Array<LoopRV>> block_loops;
  Array<Array<IterVar>> block_iters;
  Array<IntImm> block_is_reduction;
  for (const BlockRV& block : leaf_blocks) {
    Array<IterVar> iters = sch->Get(block)->iter_vars;
    bool has_spatial_iter = false;
    Array<Var> index_map_inputs;
    Array<PrimExpr> index_map_outputs;
    for (const IterVar& iter : sch->Get(block)->iter_vars) {
      Var var = iter->var.copy_with_suffix("");
      index_map_inputs.push_back(var);
      if (!is_one(iter->dom->extent)) {
        index_map_outputs.push_back(var);
        if (iter->iter_type == IterVarType::kDataPar) {
          has_spatial_iter = true;
        }
      }
    }
    if (index_map_outputs.empty() || !has_spatial_iter) {
      index_map_outputs.insert(index_map_outputs.begin(), tir::make_const(DataType::Int(64), 0));
    }
    try {
      sch->TransformBlockLayout(block, IndexMap(index_map_inputs, index_map_outputs));
    } catch (tvm::runtime::Error& e) {
      // Skip layout transformation when not transformable.
    }
    block_loops.push_back(sch->GetLoops(block));
    block_iters.push_back(sch->Get(block)->iter_vars);
    bool is_reduction = IsReductionBlock(sch->state(),         //
                                         sch->GetSRef(block),  //
                                         sch->GetSRef(root_block));
    block_is_reduction.push_back(Bool(is_reduction));
  }
  return Array<ObjectRef>{leaf_blocks, block_loops, block_iters, block_is_reduction};
}

TVM_REGISTER_GLOBAL("tir.schedule.NormalizePrimFunc").set_body_typed(NormalizePrimFunc);

}  // namespace tir
}  // namespace tvm
