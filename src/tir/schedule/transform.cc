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
  const auto* ptr_type = TVM_TYPE_AS(ptr_type, buffer->data->type_annotation, PointerTypeNode);
  new_var->type_annotation = PointerType(ptr_type->element_type, scope);
  new_buffer->data = Var(new_var->name_hint + "_" + scope, new_var->type_annotation);
  new_buffer->name = buffer->name + "_" + scope;
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
    if (const auto* seq = block->body.as<SeqStmtNode>()) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*block);
      n->body = RemoveFromSeqStmt(GetRef<SeqStmt>(seq), GetRef<Stmt>(last_stmt));
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
  const auto* leaf_block = TVM_SREF_TO_BLOCK(leaf_block, leaf_block_sref);
  const auto* scope_block = TVM_SREF_TO_BLOCK(scope_block, sref);
  throw OnlyLeafError(self->mod, GetRef<Block>(leaf_block), GetRef<Block>(scope_block));
}

Optional<LoopRV> TileWithTensorIntrin(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                      const String& intrin_name) {
  Optional<tir::TensorizeInfo> opt_tensorize_info = GetTensorizeLoopMapping(
      sch->state(), sch->GetSRef(block_rv), tir::TensorIntrin::Get(intrin_name)->desc);
  if (!opt_tensorize_info) return NullOpt;
  const tir::TensorizeInfoNode* info = opt_tensorize_info.value().get();
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
    int64_t outer = int_block_extent->value / int_desc_extent->value;
    // Do the split
    Array<LoopRV> split = sch->Split(loop2rv.at(block_loop_sref), {Integer(outer), Integer(inner)});
    ICHECK_EQ(split.size(), 2);
    inner_loops.insert(sch->GetSRef(split[1]).operator->());
    // The inner split will be reordered to the loop domain that is tensorized
    int desc_loop_index = info->desc_loop_indexer.at(GetRef<tir::For>(desc_loop));
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

}  // namespace tir
}  // namespace tvm
