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
 * \file manifest_shared_memroy_local_stage.cc
 * \brief Add the explicit local stage for the shared memory access on GPU.
 *
 * This pass finds the cache_read stage on the shared memory, and create another intermediate stage
 * to store the data into local memory first, and then copy the data from local memory to the shared
 * memory. This is similar to the schedule primitive cache_read, but it bypasses the limitation
 * of requiring buffer access to be contiguous in each dimension.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "../schedule/transform.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tir {

/*! \brief Rewriter for the block storing to the target buffer. Create an intermediate cache stage
 * to store the result. Rewrite the original block to load from the intermediate buffer.
 */
class IntermediateStageRewriter {
 public:
  explicit IntermediateStageRewriter(const std::vector<Stmt>& ancestor_loop_or_blocks)
      : ancestor_loop_or_blocks_(ancestor_loop_or_blocks) {}

  std::tuple<Buffer, Buffer, Block, Stmt> Rewrite(const BlockNode* block) {
    const BufferStoreNode* store = block->body.as<BufferStoreNode>();
    CHECK(store != nullptr && runtime::StorageScope::Create(store->buffer.scope()).rank ==
                                  runtime::StorageRank::kShared)
        << "ValueError: Expect the body of the block to be BufferStore to shared memory.";

    const Buffer& target_buffer = store->buffer;

    // Step 0: Collect relaxed loops
    std::vector<const ForNode*> relaxed_loops = CollectRelaxedOuterLoops(block, target_buffer);

    // Step 1: Create buffer for the local stage
    auto [new_buffer, buffer_indices] = CreateIntermediateBuffer(relaxed_loops, target_buffer);

    // Step 2: Create the local stage block
    Stmt local_stage = MakeLocalStage(block, new_buffer, buffer_indices, relaxed_loops, store);

    // Step 3: Create BufferLoad from the intermediate buffer
    BufferLoad new_buffer_load = BufferLoad(new_buffer, buffer_indices);
    BufferStore new_buffer_store = Downcast<BufferStore>(block->body);
    new_buffer_store.CopyOnWrite()->value = new_buffer_load;
    Block new_block = GetRef<Block>(block);
    new_block.CopyOnWrite()->body = std::move(new_buffer_store);

    return {target_buffer, new_buffer, new_block, local_stage};
  }

 private:
  /*! \brief Collect relaxed outer loops from innermost to outermost */
  std::vector<const ForNode*> CollectRelaxedOuterLoops(const BlockNode* block,
                                                       const Buffer& target_buffer) {
    std::vector<const ForNode*> relaxed_loops;
    for (int n = static_cast<int>(ancestor_loop_or_blocks_.size()) - 1, i = n - 1; i >= 0; --i) {
      const Stmt& ancestor = ancestor_loop_or_blocks_[i];
      if (const ForNode* ancestor_loop = ancestor.as<ForNode>()) {
        CHECK(ancestor_loop->kind == ForKind::kSerial ||
              ancestor_loop->kind == ForKind::kVectorized)
            << "ValueError: Expect the ancestor loops to be serial or vectorized, got "
            << ancestor_loop->kind;
        relaxed_loops.push_back(ancestor.as<ForNode>());

        if (i < n - 1) {
          CHECK(ancestor_loop->body.same_as(ancestor_loop_or_blocks_[i + 1]))
              << "ValueError: Expect the ancestor loops to have a single child.";
        } else {
          const BlockRealizeNode* block_realize = ancestor_loop->body.as<BlockRealizeNode>();
          ICHECK(block_realize != nullptr);
          CHECK(block_realize != nullptr && block_realize->block.get() == block)
              << "ValueError: Expect the ancestor loops to have a single child.";
        }
      } else {
        const BlockRealizeNode* ancestor_block_realize = ancestor.as<BlockRealizeNode>();
        ICHECK(ancestor_block_realize != nullptr);
        const BlockNode* ancestor_block = ancestor_block_realize->block.get();
        auto it = std::find_if(
            ancestor_block->alloc_buffers.begin(), ancestor_block->alloc_buffers.end(),
            [&target_buffer](const Buffer& buffer) { return buffer.same_as(target_buffer); });
        CHECK(it != ancestor_block->alloc_buffers.end())
            << "ValueError: Expect the shared memory allocation to be in the parent block.";
        break;
      }
    }
    return relaxed_loops;
  }

  /*! \brief Create the intermediate stage. */
  Stmt MakeLocalStage(const BlockNode* block, const Buffer& new_buffer,
                      Array<PrimExpr> local_stage_indices,
                      std::vector<const ForNode*> relaxed_loops, const BufferStoreNode* store) {
    // Step 0: Create the body of the local stage, which is BufferStore to the intermediate buffer.
    Stmt local_stage = BufferStore(new_buffer, store->value, local_stage_indices);

    // Step 1: Make block and block realize
    BufferRegion write_buffer_region = BufferRegion::FromPoint(new_buffer, local_stage_indices);
    local_stage =
        Block(/*iter_vars=*/{}, /*reads=*/block->reads, /*writes=*/{write_buffer_region}, "",
              /*body=*/std::move(local_stage));
    local_stage = BlockRealize(
        /*iter_values=*/{},
        /*predicate=*/ancestor_loop_or_blocks_.back().as<BlockRealizeNode>()->predicate,
        Downcast<Block>(local_stage));

    // Step 2: Add outer loops
    Map<Var, Var> subst_map;
    for (const ForNode* relaxed_loop : relaxed_loops) {
      ObjectPtr<ForNode> for_node = make_object<ForNode>(*relaxed_loop);
      for_node->loop_var = for_node->loop_var.copy_with_suffix("");
      for_node->body = std::move(local_stage);
      local_stage = For(for_node);
      subst_map.Set(relaxed_loop->loop_var, for_node->loop_var);
    }
    local_stage = Substitute(local_stage, subst_map);
    return local_stage;
  }

  /*! \brief Create the intermediate buffer with the extents of the relaxed outer loops. */
  std::pair<Buffer, Array<PrimExpr>> CreateIntermediateBuffer(
      const std::vector<const ForNode*> relaxed_loops, const Buffer& buffer) const {
    Array<PrimExpr> buffer_indices;
    Array<PrimExpr> new_buffer_shape;

    // Create the intermediate buffer for the local stage. The shape of the new buffer is the
    // extents of the relaxed outer loops.

    for (auto it = relaxed_loops.rbegin(); it != relaxed_loops.rend(); ++it) {
      const ForNode* relaxed_loop = *it;
      buffer_indices.push_back(relaxed_loop->min + relaxed_loop->loop_var);
      new_buffer_shape.push_back(relaxed_loop->extent);
    }
    Buffer new_buffer = WithScope(buffer, "local");
    new_buffer.CopyOnWrite()->shape = new_buffer_shape;
    return {new_buffer, buffer_indices};
  }

  const std::vector<Stmt>& ancestor_loop_or_blocks_;
};

class SharedMemoryLocalStageInserter : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    ancestor_loop_or_blocks_.push_back(GetRef<Stmt>(op));
    Stmt new_stmt = StmtMutator::VisitStmt_(op);
    ancestor_loop_or_blocks_.pop_back();
    return new_stmt;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    ancestor_loop_or_blocks_.push_back(GetRef<Stmt>(op));
    Stmt new_stmt = StmtMutator::VisitStmt_(op);
    ancestor_loop_or_blocks_.pop_back();
    return new_stmt;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    if (op->annotations.count(attr::manifest_shared_memory_local_stage)) {
      // Rewrite the shared memory access to load from the intermediate buffer.
      // The annotated block must be a leaf block (will be checked during rewriting). No need to
      // visit its body recursively.

      IntermediateStageRewriter rewriter(ancestor_loop_or_blocks_);
      auto [target_buffer, new_buffer, new_block, local_stage] = rewriter.Rewrite(op);
      buffer_remap_.Set(target_buffer, new_buffer);

      new_block.CopyOnWrite()->annotations.erase(attr::manifest_shared_memory_local_stage);
      buffer_local_stage_.Set(target_buffer, local_stage);
      target_buffers_.push_back(target_buffer);

      return std::move(new_block);
    }

    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocated_buffers(
        op->alloc_buffers.begin(), op->alloc_buffers.end());

    // Visit children and insert local stages (if any) to the proper location.
    Array<Buffer> new_alloc_buffers;
    Array<Stmt> new_seq;

    // Helper function to check if the subtree (body of the block) contains any target buffers.
    // If so, the allocated intermediate buffer and the local stage should be lifted to the current
    // block.
    auto f_check_subtree = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        const Buffer& buffer = target_buffers_[i];
        if (allocated_buffers.count(buffer)) {
          new_seq.push_back(buffer_local_stage_.at(buffer));
          new_alloc_buffers.push_back(buffer_remap_.at(buffer));
        }
      }
    };

    if (const SeqStmtNode* seq = op->body.as<SeqStmtNode>()) {
      // Visit each element of the SeqStmt. Create a new SeqStmt if any of the children is modified.
      bool changed = false;  // whether the SeqStmt has been changed
      for (int i = 0, n = seq->seq.size(); i < n; ++i) {
        int subtree_start = target_buffers_.size();
        Stmt new_seq_elem = VisitStmt(seq->seq[i]);
        int subtree_end = target_buffers_.size();
        f_check_subtree(subtree_start, subtree_end);
        new_seq.push_back(new_seq_elem);
        if (!new_seq_elem.same_as(seq->seq[i])) {
          changed = true;
        }
      }
      if (!changed) {
        return GetRef<Stmt>(op);
      }
    } else {
      int subtree_start = target_buffers_.size();
      Stmt body = VisitStmt(op->body);
      int subtree_end = target_buffers_.size();
      f_check_subtree(subtree_start, subtree_end);
      if (body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      }
      new_seq.push_back(body);
    }

    Block new_block = GetRef<Block>(op);
    BlockNode* new_block_node = new_block.CopyOnWrite();
    // Add new buffer allocations if any.
    if (new_alloc_buffers.size() > 0) {
      new_block_node->alloc_buffers = Concat(new_block_node->alloc_buffers, new_alloc_buffers);
    }
    new_block_node->body = new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
    return std::move(new_block);
  }

  std::vector<Stmt> ancestor_loop_or_blocks_;  // ancestor loops or block realize
  Map<Buffer, Buffer> buffer_remap_;  // mapping from the target buffer to the intermediate buffer
  Map<Buffer, Stmt> buffer_local_stage_;  // mapping from the target buffer to the local stage
  Array<Buffer> target_buffers_;          // the target buffers for rewriting
};

namespace transform {

Pass ManifestSharedMemoryLocalStage() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = SharedMemoryLocalStageInserter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ManifestSharedMemoryLocalStage", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ManifestSharedMemoryLocalStage")
    .set_body_typed(ManifestSharedMemoryLocalStage);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
