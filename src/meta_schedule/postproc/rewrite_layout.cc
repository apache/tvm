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
#include <optional>
#include <unordered_set>

#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Collect the block and index where the buffer is read.
 * \note The buffer is expected to be read by only one BufferLoad
 */
class BufferReadPosCollector : public StmtExprVisitor {
 public:
  explicit BufferReadPosCollector(const Buffer& buffer) : buffer_(buffer.get()) {}

  const std::pair<Block, int>& GetBufferLocation() const { return buffer_loc_; }

  const Optional<IndexMap> GetBufferIndexMap() const { return buffer_index_map_; }

 private:
  void VisitStmt_(const ForNode* op) final {
    loop_stack_.push_back(GetRef<For>(op));
    StmtVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize outer_block_realize = GetRef<BlockRealize>(op);
    std::swap(outer_block_realize, cur_realize_);
    StmtVisitor::VisitStmt_(op);
    std::swap(cur_realize_, outer_block_realize);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    CHECK(cur_realize_.defined()) << "BufferLoad occurred outside of any block";

    const Buffer& buffer = op->buffer;
    if (buffer_ == buffer.get()) {
      Map<Var, PrimExpr> subst_map;
      for (size_t i = 0; i < cur_realize_->iter_values.size(); i++) {
        const Var& var = cur_realize_->block->iter_vars[i]->var;
        const PrimExpr& value = cur_realize_->iter_values[i];
        subst_map.Set(var, value);
      }
      Array<PrimExpr> subst_indices;
      for (const PrimExpr& e : op->indices) {
        subst_indices.push_back(Substitute(e, subst_map));
      }
      buffer_index_map_ = SuggestIndexMap(/*buffer=*/buffer,                      //
                                          /*indices=*/subst_indices,              //
                                          /*loops=*/loop_stack_,                  //
                                          /*predicate=*/cur_realize_->predicate,  //
                                          /*analyzer=*/&analyzer_);
      int buffer_index = GetReadBufferIndex(cur_realize_->block, buffer);
      ICHECK(buffer_index != -1);
      buffer_loc_ = std::make_pair(cur_realize_->block, buffer_index);
    }
  }

  static int GetReadBufferIndex(const Block& block, const Buffer& buffer) {
    for (size_t i = 0; i < block->reads.size(); i++) {
      if (block->reads[i]->buffer.same_as(buffer)) {
        return i;
      }
    }
    return -1;
  }

 private:
  /*! \brief The buffer of interest. */
  const BufferNode* buffer_;
  /*! \brief The block that consumes the buffer and the corresponding read index. */
  std::pair<Block, int> buffer_loc_;
  /*! \brief The proposed IndexMap. */
  Optional<IndexMap> buffer_index_map_;

  /*! \brief Loop stack for calculating IndexMap. */
  Array<For> loop_stack_;
  /*! \brief Arithmetic analyzer. */
  arith::Analyzer analyzer_;
  /*! \brief Current BlockRealize scope, used in recursive visit */
  BlockRealize cur_realize_;
};

class LayoutFreeBufferCollector : public StmtVisitor {
 public:
  void VisitStmt_(const BlockNode* block) final {
    StmtVisitor::VisitStmt_(block);
    if (Optional<ObjectRef> ann = block->annotations.Get("layout_free_placeholders")) {
      for (Buffer buffer : Downcast<Array<Buffer>>(ann)) {
        buffers.insert(buffer);
      }
    }
  }

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffers;
};

Array<Buffer> CollectLayoutFreeBuffers(const PrimFuncNode* func) {
  // Only rewrite PrimFuncs with attr "layout_free_buffers"
  Array<Integer> layout_free_buffer_index =
      func->GetAttr(attr::layout_free_buffers, Array<Integer>()).value();

  Array<Buffer> layout_free_buffers;
  for (const Integer& index : layout_free_buffer_index) {
    ICHECK(static_cast<size_t>(index->value) < func->params.size());
    const Var& param = func->params[index->value];
    layout_free_buffers.push_back(func->buffer_map.at(param));
  }

  LayoutFreeBufferCollector collector;
  collector(func->body);

  for (auto buf : collector.buffers) {
    layout_free_buffers.push_back(buf);
  }
  return layout_free_buffers;
}

std::optional<std::tuple<Block, int, IndexMap>> GetSuggestedIndexMap(
    Buffer buffer, const PrimFuncNode* prim_func) {
  BufferReadPosCollector collector(buffer);
  collector(prim_func->body);

  const auto& index_map = collector.GetBufferIndexMap();

  if (!index_map.defined() || !index_map) {
    return std::nullopt;
  }

  const auto& [anchor_block, buffer_index] = collector.GetBufferLocation();

  return std::make_tuple(anchor_block, buffer_index, index_map.value());
}

/*! \brief Get a chain of cache-read blocks, starting from the one consuming buf. */
std::vector<std::string> GetCacheReadChain(const Buffer& buf, const PrimFuncNode* prim_func) {
  class BufferReadChainCollector : public StmtVisitor {
   public:
    explicit BufferReadChainCollector(const Buffer& buffer) : cur_buffer_(buffer.get()) {}

    void VisitStmt_(const BlockNode* op) final {
      // Check if this block is doing cache_read or a similar operation that consumes cur_buffer_.
      if (!op->init && op->reads.size() == 1 && op->writes.size() == 1 &&
          op->reads[0]->buffer.get() == cur_buffer_) {
        cache_read_chain.push_back(op->name_hint);
        cur_buffer_ = op->writes[0]->buffer.get();
      }
      StmtVisitor::VisitStmt_(op);
    }

    std::vector<std::string> cache_read_chain;

   private:
    const BufferNode* cur_buffer_;
  };

  BufferReadChainCollector collector(buf);
  collector(prim_func->body);
  return collector.cache_read_chain;
}

bool RewriteLayout(const Schedule& sch) {
  std::vector<std::pair<StmtSRef, String>> results;
  auto add_layout_rewrite_block = [&sch](BlockRV consumer_block_rv, int buffer_index) {
    BlockRV rewrite_block_rv = sch->CacheRead(consumer_block_rv, buffer_index, "global");
    sch->Annotate(rewrite_block_rv, attr::meta_schedule_layout_rewrite_preproc, const_true());
  };

  for (const auto& [g_var, base_func] : sch->mod()->functions) {
    const String& func_name = g_var->name_hint;
    const auto* prim_func = base_func.as<PrimFuncNode>();
    // Only consider PrimFunc
    if (prim_func == nullptr) {
      continue;
    }

    for (auto buffer : CollectLayoutFreeBuffers(prim_func)) {
      const auto cache_read_chain = GetCacheReadChain(buffer, prim_func);
      if (cache_read_chain.empty()) {
        // The common case, where the layout-free buffer is directly consumed by an anchor op such
        // as conv2d or dense.
        auto tup_opt = GetSuggestedIndexMap(buffer, prim_func);
        if (tup_opt == std::nullopt) continue;

        auto [anchor_block, buffer_index, index_map] = *tup_opt;
        auto anchor_block_rv = sch->GetBlock(anchor_block->name_hint, func_name);
        add_layout_rewrite_block(anchor_block_rv, buffer_index);
        sch->TransformLayout(anchor_block_rv, buffer_index, BufferIndexType::kRead, index_map,
                             NullOpt);
      } else {
        // When the layout-free buffer is consumed by cache_read, we need to find the index map
        // for a cache-read buffer that is directly consumed by an anchor op. The last buffer
        // in cache_read_chain corresponds to that buffer.
        Block cache_read_block = sch->Get(sch->GetBlock(cache_read_chain.back(), func_name));
        ICHECK_EQ(cache_read_block->writes.size(), 1);
        auto tup_opt = GetSuggestedIndexMap(cache_read_block->writes[0]->buffer, prim_func);
        if (tup_opt == std::nullopt) continue;

        auto [anchor_block, buffer_index, index_map] = *tup_opt;
        // Transform the layout of the last cache-read buffer.
        sch->TransformLayout(sch->GetBlock(anchor_block->name_hint, func_name), buffer_index,
                             BufferIndexType::kRead, index_map, NullOpt);

        // Propagate the layout transformation over cache_read_chain, starting from
        // the next-to-last cache-read buffer.
        for (int i = static_cast<int>(cache_read_chain.size()) - 1; i >= 0; --i) {
          BlockRV cache_read_block_rv = sch->GetBlock(cache_read_chain[i], func_name);
          if (i == 0) {
            // Before the first cache_read that consumes the layout-free buffer, insert
            // a layout-rewrite block. Another cache-read buffer is added, and its layout is
            // transformed by TransformLayout below.
            add_layout_rewrite_block(cache_read_block_rv, 0);
          }
          sch->TransformLayout(cache_read_block_rv, 0, BufferIndexType::kRead, index_map, NullOpt);
        }
      }
    }
  }
  return true;
}

}  // namespace tir

namespace meta_schedule {
/*! \brief Layout Rewrite. */
class RewriteLayoutNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final { return tir::RewriteLayout(sch); }

  Postproc Clone() const {
    ObjectPtr<RewriteLayoutNode> n = make_object<RewriteLayoutNode>(*this);
    return Postproc(n);
  }

  static constexpr const char* _type_key = "meta_schedule.RewriteLayout";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteLayoutNode, PostprocNode);
};

Postproc Postproc::RewriteLayout() {
  auto n = make_object<RewriteLayoutNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteLayoutNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteLayout").set_body_typed(Postproc::RewriteLayout);

}  // namespace meta_schedule
}  // namespace tvm
