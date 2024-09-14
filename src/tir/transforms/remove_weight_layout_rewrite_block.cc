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

#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

namespace tvm {
namespace tir {

class RemoveLayoutRewriteBlock : public StmtMutator {
 public:
  static std::tuple<PrimFunc, Map<Buffer, Buffer>, std::unordered_map<const VarNode*, IndexMap>,
                    std::unordered_map<const VarNode*, Array<PrimExpr>>>
  Rewrite(PrimFunc f) {
    RemoveLayoutRewriteBlock rewriter;

    PrimFuncNode* n = f.CopyOnWrite();
    n->body = rewriter(std::move(n->body));
    return std::make_tuple(f, rewriter.buf_map_, rewriter.buffer_var_to_index_map_,
                           rewriter.buffer_var_to_rewritten_shape_);
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

    Array<Var> load_indices;
    for (auto ind : load->indices) {
      ICHECK(ind->IsInstance<VarNode>());
      load_indices.push_back(Downcast<Var>(ind));
    }
    buffer_var_to_index_map_[load->buffer->data.get()] = IndexMap(load_indices, store->indices);

    buffer_var_to_rewritten_shape_[load->buffer->data.get()] = store->buffer->shape;

    return Stmt(n);
  }

 private:
  /*! \brief The buffer map from original layout buffer to rewritten buffer */
  Map<Buffer, Buffer> buf_map_;
  /*! \brief The buffer map from original layout buffer to rewritten buffer */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> rewritten_buffers_;
  /*! \brief Maps a buffer load to an index map associated with the load / store
    in a layout rewrite block. */
  std::unordered_map<const VarNode*, IndexMap> buffer_var_to_index_map_;
  /*! \brief Maps a buffer load to the shape of the corresponding rewritten buffer. */
  std::unordered_map<const VarNode*, Array<PrimExpr>> buffer_var_to_rewritten_shape_;
};

// After RemoveLayoutRewriteBlock, the body of a compute update block references a
// non-existant buffer. For example, fused_constant_2_global below is originally a
// cache_read buffer, whose allocation is removed by RemoveLayoutRewriteBlock:
//
// constant fused_constant_2[float32 * 3 * 3 * 64 * 64]
// conv2d_nhwc[nn, yy, xx, ff] += ... * fused_constant_2_global[ry,
//                                                              floordiv(rc, 32),
//                                                              floordiv(ff, 16),
//                                                              rx,
//                                                              floormod(rc, 32),
//                                                              floormod(ff, 16)]))
//
// When cache_read is reading from AllocateConstant, we need to replace the reference
// to fused_constant_2_global with the corresponding transformed AllocateConstant.
// To do that, we manually rewrite the original constant using the associated index map,
// and let the body of the compute block to load from the rewritten constant.
//
// After this transformation, the example above looks like:
//
// constant fused_constant_2[float32 * 3 * 2 * 4 * 3 * 32 * 16]
// conv2d_nhwc[nn, yy, xx, ff] += ... * fused_constant_2[ry,
//                                                       floordiv(rc, 32),
//                                                       floordiv(ff, 16),
//                                                       rx,
//                                                       floormod(rc, 32),
//                                                       floormod(ff, 16)]))

using BufferVarMap = std::unordered_map<const tir::VarNode*, const tir::VarNode*>;

class AllocateConstRewrite : public StmtExprMutator {
 public:
  AllocateConstRewrite(
      const BufferVarMap& buffer_var_map,
      const std::unordered_map<const VarNode*, IndexMap>& buffer_var_to_index_map,
      const std::unordered_map<const VarNode*, Array<PrimExpr>>& buffer_var_to_rewritten_shape,
      bool skip_ndarray_rewrite)
      : buffer_var_map_(buffer_var_map),
        buffer_var_to_index_map_(buffer_var_to_index_map),
        buffer_var_to_rewritten_shape_(buffer_var_to_rewritten_shape),
        skip_ndarray_rewrite_(skip_ndarray_rewrite) {}

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
    auto n = CopyOnWrite(block.get());
    Array<BufferRegion> new_reads;
    for (auto read_region : op->reads) {
      if (auto it = new_load_buf_.find(read_region->buffer->data.get());
          it != new_load_buf_.end()) {
        new_reads.push_back(BufferRegion(it->second, read_region->region));
      } else {
        new_reads.push_back(read_region);
      }
    }
    n->reads = new_reads;
    return Stmt(n);
  }

  Stmt VisitStmt_(const AllocateConstNode* alloc) final {
    if (auto it = buffer_var_to_index_map_.find(alloc->buffer_var.get());
        it != buffer_var_to_index_map_.end()) {
      ICHECK(buffer_var_to_rewritten_shape_.count(alloc->buffer_var.get()));
      auto new_body = StmtMutator::VisitStmt(alloc->body);
      auto rewritten_ndarray = RewriteNDArray(
          alloc->data.value(), it->second, buffer_var_to_rewritten_shape_[alloc->buffer_var.get()]);
      Array<PrimExpr> rewritten_extents;
      for (auto s : rewritten_ndarray.Shape()) {
        rewritten_extents.push_back(PrimExpr(static_cast<int>(s)));
      }
      return AllocateConst(alloc->buffer_var, alloc->dtype, rewritten_extents, rewritten_ndarray,
                           new_body, alloc->annotations, alloc->span);
    }
    return StmtMutator::VisitStmt_(alloc);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (auto it = buffer_var_map_.find(op->buffer->data.get()); it != buffer_var_map_.end()) {
      auto new_buffer =
          Buffer(GetRef<Var>(it->second), op->buffer->dtype, op->buffer->shape, op->buffer->strides,
                 op->buffer->elem_offset, it->second->name_hint, op->buffer->data_alignment,
                 op->buffer->offset_factor, op->buffer->buffer_type);
      new_load_buf_[op->buffer->data.get()] = new_buffer;
      return BufferLoad(new_buffer, op->indices, op->predicate);
    }
    return ExprMutator::VisitExpr_(op);
  }

  runtime::NDArray RewriteNDArray(runtime::NDArray src, const IndexMap& index_map,
                                  const Array<PrimExpr>& dst_shape) {
    if (skip_ndarray_rewrite_) {
      // Only the shape of the destination array needs to be correct.
      std::vector<int64_t> dst_shape_int;
      for (auto s : dst_shape) {
        ICHECK(s->IsInstance<IntImmNode>());
        dst_shape_int.push_back(s.as<IntImmNode>()->value);
      }
      return src.CreateView(dst_shape_int, src.DataType());
    } else {
      return index_map->MapNDArray(src);
    }
  }

  /*! \brief Maps a buffer store to a load in a layout rewrite block */
  BufferVarMap buffer_var_map_;
  /*! \brief Maps a buffer load to an index map associated with the load / store
    in a layout rewrite block. */
  std::unordered_map<const VarNode*, IndexMap> buffer_var_to_index_map_;
  /*! \brief Maps a buffer load to the shape of the corresponding rewritten buffer. */
  std::unordered_map<const VarNode*, Array<PrimExpr>> buffer_var_to_rewritten_shape_;
  /*! \brief Maps load buffer variables to newly created buffers */
  std::unordered_map<const VarNode*, Buffer> new_load_buf_;
  /*! \brief Whether or not to skip rewriting of NDArray contents */
  bool skip_ndarray_rewrite_;
};

class CollectAllocateConstBufferVars : public StmtVisitor {
 public:
  void VisitStmt_(const AllocateConstNode* alloc) final {
    StmtVisitor::VisitStmt_(alloc);
    constant_buf_var.insert(alloc->buffer_var.get());
  }

  std::unordered_set<const VarNode*> constant_buf_var;
};

class WeightLayoutRewriteBlockRemover : public StmtMutator {
 public:
  static PrimFunc Remove(PrimFunc f, bool skip_ndarray_rewrite) {
    CollectAllocateConstBufferVars collector;
    collector(f->body);

    auto [f_, buf_map, buffer_var_to_index_map, buffer_var_to_rewritten_shape] =
        RemoveLayoutRewriteBlock().Rewrite(f);

    BufferVarMap buffer_var_map;
    for (const auto& [load_buf, store_buf] : buf_map) {
      if (collector.constant_buf_var.find(load_buf->data.get()) !=
          collector.constant_buf_var.end()) {
        buffer_var_map[store_buf->data.get()] = load_buf->data.get();
      }
    }

    PrimFuncNode* n = f_.CopyOnWrite();

    AllocateConstRewrite rewriter(buffer_var_map, buffer_var_to_index_map,
                                  buffer_var_to_rewritten_shape, skip_ndarray_rewrite);
    n->body = rewriter(std::move(n->body));

    Map<tir::Var, Buffer> buffer_map;
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

Pass RemoveWeightLayoutRewriteBlock(bool skip_ndarray_rewrite) {
  auto pass_func = [skip_ndarray_rewrite](PrimFunc f, IRModule m, PassContext ctx) {
    return WeightLayoutRewriteBlockRemover::Remove(std::move(f), skip_ndarray_rewrite);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveWeightLayoutRewriteBlock", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveWeightLayoutRewriteBlock")
    .set_body_typed(RemoveWeightLayoutRewriteBlock);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
