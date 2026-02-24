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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

namespace tvm {
namespace s_tir {
using namespace tvm::tir;

class RemoveLayoutRewriteBlock : public StmtMutator {
 public:
  static std::tuple<PrimFunc, ffi::Map<Buffer, Buffer>,
                    std::unordered_map<const VarNode*, IndexMap>,
                    std::unordered_map<const VarNode*, ffi::Array<PrimExpr>>>
  Rewrite(PrimFunc f) {
    RemoveLayoutRewriteBlock rewriter;

    PrimFuncNode* n = f.CopyOnWrite();
    n->body = rewriter(std::move(n->body));
    return std::make_tuple(f, rewriter.buf_map_, rewriter.buffer_var_to_index_map_,
                           rewriter.buffer_var_to_rewritten_shape_);
  }

 private:
  Stmt VisitStmt_(const SBlockNode* op) final {
    SBlock block = Downcast<SBlock>(StmtMutator::VisitStmt_(op));

    auto it = block->annotations.find(s_tir::attr::meta_schedule_layout_rewrite_preproc);
    if (it == block->annotations.end() || !is_one(Downcast<PrimExpr>((*it).second))) {
      // The block is not a weight layout block
      // Remove allocates if needed
      ffi::Array<Buffer> alloc_buffers;
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
        return block;
      }
    }

    // Step 0. Checking block attrs
    TVM_FFI_ICHECK(block->alloc_buffers.empty());
    TVM_FFI_ICHECK(block->match_buffers.empty());

    // Step 1. Checking the body is a BufferStore
    const auto* store = block->body.as<BufferStoreNode>();
    TVM_FFI_ICHECK(store);

    // Step 2. Checking the rhs of buffer store is a BufferLoad
    const auto* load = store->value.as<BufferLoadNode>();
    TVM_FFI_ICHECK(load);

    // Step 3. Update Buffer
    buf_map_.Set(load->buffer, store->buffer);
    rewritten_buffers_.insert(store->buffer);

    // Step 4. Set block body as no_op
    auto n = CopyOnWrite(block.get());
    n->body = std::move(Evaluate(0));
    n->reads = {};
    n->writes = {};

    ffi::Array<Var> load_indices;
    for (auto ind : load->indices) {
      TVM_FFI_ICHECK(ind->IsInstance<VarNode>());
      load_indices.push_back(Downcast<Var>(ind));
    }
    buffer_var_to_index_map_[load->buffer->data.get()] = IndexMap(load_indices, store->indices);

    buffer_var_to_rewritten_shape_[load->buffer->data.get()] = store->buffer->shape;

    return Stmt(n);
  }

 private:
  /*! \brief The buffer map from original layout buffer to rewritten buffer */
  ffi::Map<Buffer, Buffer> buf_map_;
  /*! \brief The buffer map from original layout buffer to rewritten buffer */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> rewritten_buffers_;
  /*! \brief Maps a buffer load to an index map associated with the load / store
    in a layout rewrite block. */
  std::unordered_map<const VarNode*, IndexMap> buffer_var_to_index_map_;
  /*! \brief Maps a buffer load to the shape of the corresponding rewritten buffer. */
  std::unordered_map<const VarNode*, ffi::Array<PrimExpr>> buffer_var_to_rewritten_shape_;
};

class WeightLayoutRewriteBlockRemover : public StmtMutator {
 public:
  static PrimFunc Remove(PrimFunc f, bool skip_tensor_rewrite) {
    auto [f_, buf_map, buffer_var_to_index_map, buffer_var_to_rewritten_shape] =
        RemoveLayoutRewriteBlock().Rewrite(f);

    PrimFuncNode* n = f_.CopyOnWrite();

    ffi::Map<tir::Var, Buffer> buffer_map;
    for (const auto& [param, buffer] : f_->buffer_map) {
      auto it = buf_map.find(buffer);
      if (it != buf_map.end()) {
        buffer_map.Set(param, (*it).second);
      } else {
        buffer_map.Set(param, buffer);
      }
    }
    n->buffer_map = std::move(buffer_map);
    return f_;
  }
};

namespace transform {

Pass RemoveWeightLayoutRewriteBlock(bool skip_tensor_rewrite) {
  auto pass_func = [skip_tensor_rewrite](PrimFunc f, IRModule m, PassContext ctx) {
    return WeightLayoutRewriteBlockRemover::Remove(std::move(f), skip_tensor_rewrite);
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.RemoveWeightLayoutRewriteBlock", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.RemoveWeightLayoutRewriteBlock",
                        RemoveWeightLayoutRewriteBlock);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
