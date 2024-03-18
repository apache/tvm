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
#include "../../transforms/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief Mutator for BufferIndex. */
class BufferIndicesRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add stages of writting precomputed index
   * \param scope_sref The parent scope of this mutation
   * \param info The index information
   * \return The new AST rooting at the original parent scope
   */
  BufferIndicesRewriter(Map<Buffer, Array<PrimExpr>> buffer_indices_map)
      : _buffer_indices_map(buffer_indices_map) {}

 private:
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    const Buffer& buffer = store->buffer;
    if (_buffer_indices_map.count(buffer)) {
      Array<PrimExpr> indices = _buffer_indices_map[buffer];
      auto n = make_object<BufferStoreNode>(*store.get());
      n->buffer = buffer;
      n->indices = indices;
      return BufferStore(n);
    }
    return store;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    const Buffer& buffer = load->buffer;
    if (_buffer_indices_map.count(buffer)) {
      Array<PrimExpr> indices = _buffer_indices_map[buffer];
      auto n = make_object<BufferLoadNode>(*load.get());
      n->buffer = buffer;
      n->indices = indices;
      return BufferLoad(n);
    }
    return load;
  }

  Map<Buffer, Array<PrimExpr>> _buffer_indices_map;
};

/******** Implementation ********/

void UnsafeRewriteBufferAccess(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                               BufferIndexType buffer_index_type, const Array<PrimExpr>& indices) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);

  /* Step 0: Collect new buffer access regions. */

  Array<BufferRegion> reads, writes;
  Map<Buffer, Array<PrimExpr>> buffer_indices_map;

  if (buffer_index_type == BufferIndexType::kRead) {
    for (size_t i = 0; i < block->reads.size(); ++i) {
      if (i != (size_t)buffer_index) {
        reads.push_back(block->reads[i]);
      } else {
        BufferRegion region = block->reads[i];
        Array<Range> new_regions;
        for (const auto& indice : indices) {
          new_regions.push_back(Range::FromMinExtent(indice, 1));
        }
        Buffer buffer = region->buffer;
        BufferRegion new_region = BufferRegion(buffer, std::move(new_regions));
        buffer_indices_map.Set(buffer, indices);
        reads.push_back(new_region);
      }
    }
    writes = block->writes;
  } else if (buffer_index_type == BufferIndexType::kWrite) {
    for (size_t i = 0; i < block->writes.size(); ++i) {
      if (i != (size_t)buffer_index) {
        writes.push_back(block->writes[i]);
      } else {
        BufferRegion region = block->writes[i];
        Array<Range> new_regions;
        for (const auto& indice : indices) {
          new_regions.push_back(Range::FromMinExtent(indice, 1));
        }
        Buffer buffer = region->buffer;
        BufferRegion new_region = BufferRegion(buffer, std::move(new_regions));
        buffer_indices_map.Set(buffer, indices);
        writes.push_back(new_region);
      }
    }
    reads = block->reads;
  } else {
    CHECK(false) << "Unrecognized buffer type " << BufferIndexType2Str(buffer_index_type)
                 << ", only support read/write";
  }

  /* Step 1: Replace old block with the new block */
  auto n = make_object<BlockNode>(*block);
  n->reads = reads;
  n->writes = writes;
  Block new_block = Block(n);
  BufferIndicesRewriter rewriter(buffer_indices_map);
  Stmt stmt = rewriter(new_block);
  new_block = Downcast<Block>(stmt);
  Map<Block, Block> blk_map;
  blk_map.Set(GetRef<Block>(block), new_block);
  self->Replace(block_sref, new_block, blk_map);
}

struct UnsafeRewriteBufferAccessTraits
    : public UnpackedInstTraits<UnsafeRewriteBufferAccessTraits> {
  static constexpr const char* kName = "UnsafeRewriteBufferAccess";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 4;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer buffer_index,
                                      Integer buffer_index_type, Array<PrimExpr> indices) {
    sch->UnsafeRewriteBufferAccess(block, buffer_index.IntValue(),
                                   static_cast<BufferIndexType>(buffer_index_type->value), indices);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer buffer_index,
                                 Integer buffer_index_type, Array<PrimExpr> indices) {
    PythonAPICall py("unsafe_rewrite_buffer_access");
    py.Input("block", block_rv);
    std::ostringstream os;
    os << "(\"" << BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value))
       << "\", " << buffer_index << ")";
    py.Input("buffer", os.str());
    py.Input("indices", indices);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(UnsafeRewriteBufferAccessTraits);

}  // namespace tir
}  // namespace tvm
