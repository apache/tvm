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
 * \file remove_weight_layout_rewrite_block.cc
 * \brief Remove weight layout rewrite block before benchmark
 */

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class WeightLayoutRewriteBlockRemover : public StmtMutator {
 public:
  static PrimFunc Remove(PrimFunc f) {
    WeightLayoutRewriteBlockRemover remover;
    PrimFuncNode* n = f.CopyOnWrite();
    n->body = remover(std::move(n->body));
    Map<tir::Var, Buffer> buffer_map;
    for (const auto& kv : f->buffer_map) {
      Var param = kv.first;
      Buffer buffer = kv.second;
      auto it = remover.buf_map_.find(buffer);
      if (it != remover.buf_map_.end()) {
        buffer_map.Set(param, (*it).second);
      } else {
        buffer_map.Set(param, buffer);
      }
    }
    n->buffer_map = std::move(buffer_map);
    return f;
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));

    auto it = block->annotations.find(attr::meta_schedule_layout_rewrite_preproc);
    if (it == block->annotations.end() || !is_one(Downcast<PrimExpr>((*it).second))) {
      // The block is not a weight layout block
      // Remove allocates if needed
      Array<Buffer> alloc_buffers;
      for (const Buffer& buffer : block->alloc_buffers) {
        if (!rewritten_buffers_.count(buffer)) {
          alloc_buffers.push_back(buffer);
        }
      }
      if (alloc_buffers.size() < block->alloc_buffers.size()) {
        auto n = CopyOnWrite(block.get());
        n->alloc_buffers = std::move(alloc_buffers);
        return Stmt(n);
      } else {
        return std::move(block);
      }
    }

    // Step 0. Checking block attrs
    ICHECK(block->alloc_buffers.empty());
    ICHECK(block->match_buffers.empty());

    // Step 1. Checking the body is a BufferStore
    const auto* store = block->body.as<BufferStoreNode>();
    ICHECK(store);

    // Step 2. Checking the rhs of buffer store is a BufferLoad
    const auto* load = store->value.as<BufferLoadNode>();
    ICHECK(load);

    // Step 3. Update Buffer
    buf_map_.Set(load->buffer, store->buffer);
    rewritten_buffers_.insert(store->buffer);

    // Step 4. Set block body as no_op
    auto n = CopyOnWrite(block.get());
    n->body = std::move(Evaluate(0));
    n->reads = {};
    n->writes = {};
    return Stmt(n);
  }

 private:
  /*! \brief The buffer map from original layout buffer to rewritten buffer */
  Map<Buffer, Buffer> buf_map_;
  /*! \brief The buffer map from original layout buffer to rewritten buffer */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> rewritten_buffers_;
};
namespace transform {

Pass RemoveWeightLayoutRewriteBlock() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return WeightLayoutRewriteBlockRemover::Remove(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveWeightLayoutRewriteBlock", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveWeightLayoutRewriteBlock")
    .set_body_typed(RemoveWeightLayoutRewriteBlock);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
