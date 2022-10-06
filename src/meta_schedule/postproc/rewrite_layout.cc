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
#include <unordered_set>

#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Collect the block and index where the buffer is read.
 * \note The buffers are expected to be read by only one BufferLoad
 */
class BufferReadPosCollector : public StmtExprVisitor {
 public:
  explicit BufferReadPosCollector(const Array<Buffer>& buffers) {
    for (const Buffer& buf : buffers) {
      buffers_.insert(buf.get());
    }
  }

  const std::unordered_map<const BufferNode*, std::pair<Block, int>>& GetBufferLocations() const {
    return buffer_locs_;
  }

  const std::unordered_map<const BufferNode*, Optional<IndexMap>>& GetBufferIndexMap() const {
    return buffer_index_maps_;
  }

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
    if (buffers_.count(buffer.get())) {
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
      buffer_index_maps_[buffer.get()] = SuggestIndexMap(/*buffer=*/buffer,                      //
                                                         /*indices=*/subst_indices,              //
                                                         /*loops=*/loop_stack_,                  //
                                                         /*predicate=*/cur_realize_->predicate,  //
                                                         /*analyzer=*/&analyzer_);
      int buffer_index = GetReadBufferIndex(cur_realize_->block, buffer);
      ICHECK(buffer_index != -1);
      buffer_locs_[buffer.get()] = std::make_pair(cur_realize_->block, buffer_index);
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
  /*! \brief All interested buffer. */
  std::unordered_set<const BufferNode*> buffers_;
  /*! \brief The result mapping from buffer to its inner-most block and read index. */
  std::unordered_map<const BufferNode*, std::pair<Block, int>> buffer_locs_;
  /*! \brief The result mapping from buffer to its IndexMap. */
  std::unordered_map<const BufferNode*, Optional<IndexMap>> buffer_index_maps_;

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

bool RewriteLayout(const Schedule& sch) {
  std::vector<std::pair<StmtSRef, String>> results;
  for (const auto& [g_var, base_func] : sch->mod()->functions) {
    const String& func_name = g_var->name_hint;
    const auto* prim_func = base_func.as<PrimFuncNode>();
    // Only consider PrimFunc
    if (prim_func == nullptr) {
      continue;
    }

    Array<Buffer> layout_free_buffers = CollectLayoutFreeBuffers(prim_func);

    // Collect Buffer read positions
    BufferReadPosCollector collector(layout_free_buffers);
    collector(prim_func->body);
    const auto& locations = collector.GetBufferLocations();
    const auto& index_maps = collector.GetBufferIndexMap();
    // Check all buffers are collected
    if (locations.size() != layout_free_buffers.size() ||
        index_maps.size() != layout_free_buffers.size()) {
      return false;
    }

    for (const auto& kv : locations) {
      const Buffer& buffer = GetRef<Buffer>(kv.first);
      const Block& block = kv.second.first;
      int buffer_index = kv.second.second;

      // Get IndexMap
      const Optional<IndexMap> index_map = index_maps.at(buffer.get());
      if (!index_map.defined()) {
        continue;
      }

      // Apply schedule
      BlockRV block_rv = sch->GetBlock(block->name_hint, func_name);
      BlockRV cached_block_rv = sch->CacheRead(block_rv, buffer_index, "global");
      sch->TransformLayout(block_rv, buffer_index, BufferIndexType::kRead, index_map.value(),
                           NullOpt);
      sch->Annotate(cached_block_rv, attr::meta_schedule_layout_rewrite_preproc, const_true());
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
