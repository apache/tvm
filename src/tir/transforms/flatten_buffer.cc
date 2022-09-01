/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file flatten_buffer.cc
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into device-supported dimension
 *        for the TIR not contains opaque block.
 */
class BufferFlattener : public StmtExprMutator {
 public:
  static PrimFunc Flatten(PrimFunc func) {
    Map<Var, Buffer> preflattened_buffer_map =
        Merge(func->buffer_map, func->preflattened_buffer_map);
    auto pass = BufferFlattener(func->buffer_map);
    auto writer = func.CopyOnWrite();
    writer->body = pass.VisitStmt(func->body);
    writer->preflattened_buffer_map = preflattened_buffer_map;
    writer->buffer_map = pass.updated_extern_buffer_map_;
    return func;
  }

 private:
  explicit BufferFlattener(const Map<Var, Buffer>& extern_buffer_map) {
    for (const auto& kv : extern_buffer_map) {
      updated_extern_buffer_map_.Set(kv.first, GetFlattenedBuffer(kv.second));
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    ICHECK_EQ(op->match_buffers.size(), 0)
        << "Unexpected MatchBufferRegion found during tir.transform.FlattenBuffer.  "
        << "All MatchBufferRegion should be removed in tir.transform.LowerMatchBuffer.";

    Block block = GetRef<Block>(op);

    Array<Buffer> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply([this](Buffer buf) { return GetFlattenedBuffer(buf); });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    }

    Array<BufferRegion> reads = op->reads;
    reads.MutateByApply([this](BufferRegion region) { return MutateBufferRegion(region); });
    if (!reads.same_as(op->reads)) {
      block.CopyOnWrite()->reads = reads;
    }

    Array<BufferRegion> writes = op->writes;
    writes.MutateByApply([this](BufferRegion region) { return MutateBufferRegion(region); });
    if (!writes.same_as(op->writes)) {
      block.CopyOnWrite()->writes = writes;
    }

    return StmtExprMutator::VisitStmt_(block.get());
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Allocate alloc = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (alloc->dtype == DataType::Bool()) {
      auto writer = alloc.CopyOnWrite();
      writer->dtype = DataType::Int(8);
    }

    if (alloc->extents.size() == 1) {
      // No flattening required for buffers that are already flat
      return std::move(alloc);
    }

    if (auto* decl_buffer = alloc->body.as<DeclBufferNode>();
        decl_buffer && decl_buffer->buffer->data.same_as(alloc->buffer_var)) {
      // N-d buffer, use the DeclBuffer inside to determine how it
      // should be flattened.
      auto& buffer = decl_buffer->buffer;
      bool matching_buffer = [&]() {
        if (alloc->dtype != buffer->dtype) {
          return false;
        }
        if (alloc->extents.size() != buffer->shape.size()) {
          return false;
        }
        ExprDeepEqual expr_equal;
        for (size_t i = 0; i < alloc->extents.size(); i++) {
          if (!expr_equal(alloc->extents[i], buffer->shape[i])) {
            return false;
          }
        }
        return true;
      }();

      if (matching_buffer) {
        Buffer flattened = GetFlattenedBuffer(buffer);

        auto n = alloc.CopyOnWrite();
        // TODO(rfc-70): Update the DeclBuffer node instead of
        // stripping it out.  Stripping it out in the current
        // implementation as not all lowering passes support
        // DeclBuffer.
        //
        // n->body = DeclBuffer(flattened, std::move(decl_buffer->body));
        n->body = std::move(decl_buffer->body);
        n->extents = flattened->shape;
        return std::move(alloc);
      } else {
        ICHECK(decl_buffer->buffer->axis_separators.empty())
            << "DeclBuffer node doesn't match Allocate extents, but also shouldn't be "
               "flattened to 1-d physical memory";
      }
    }

    // Fallback, this is an allocation without a matching DeclBuffer
    PrimExpr flat_extent = 1;
    for (const auto& dim : alloc->extents) {
      flat_extent *= dim;
    }

    auto n = alloc.CopyOnWrite();
    n->extents = {flat_extent};
    return std::move(alloc);
  }

  Buffer GetFlattenedBuffer(Buffer buf) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    auto flattened = buf.GetFlattenedBuffer();

    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (flattened->dtype == DataType::Bool()) {
      auto writer = flattened.CopyOnWrite();
      writer->dtype = DataType::Int(8);
    }

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool store_returns_bool = (op->value.dtype() == DataType::Bool());
    store = VisitBufferAccess(store);

    // Handle casts from the value's dtype to the dtype of the
    // backing array.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (store_returns_bool) {
      ICHECK_EQ(store->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(DataType::Int(8), store->value);
      return std::move(store);
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    bool load_returns_bool = (op->dtype == DataType::Bool());
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    load = VisitBufferAccess(load);
    // Handle casts from dtype of the backing array to value's dtype.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (load_returns_bool) {
      ICHECK_EQ(load->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->dtype = DataType::Int(8);
      return tvm::cast(DataType::Bool(), load);
    } else {
      return std::move(load);
    }
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    ICHECK(node->buffer.defined());
    auto flattened_indices = node->buffer->ElemOffset(node->indices);
    Buffer flattened_buffer = GetFlattenedBuffer(node->buffer);

    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  BufferRegion MutateBufferRegion(BufferRegion region) {
    Buffer orig_buf = region->buffer;
    Buffer flattened_buf = GetFlattenedBuffer(orig_buf);
    if (flattened_buf.same_as(orig_buf)) {
      return region;
    }

    Array<PrimExpr> min_values;
    Array<PrimExpr> max_values;
    for (const auto& range : region->region) {
      min_values.push_back(range->min);
      max_values.push_back(range->min + range->extent - 1);
    }

    Array<PrimExpr> flattened_min = orig_buf->ElemOffset(min_values);
    Array<PrimExpr> flattened_max = orig_buf->ElemOffset(max_values);

    Array<Range> flattened_ranges;
    ICHECK_EQ(flattened_min.size(), flattened_max.size());
    for (size_t i = 0; i < flattened_min.size(); i++) {
      flattened_ranges.push_back(Range(flattened_min[i], flattened_max[i] + 1));
    }

    return BufferRegion(flattened_buf, flattened_ranges);
  }

  /*! \brief Map of buffers being remapped. */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;

  /*! \brief The updated external buffer map. */
  Map<Var, Buffer> updated_extern_buffer_map_;
};

PrimFunc FlattenBuffer(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    return BufferFlattener::Flatten(f);
  } else {
    return f;
  }
}

namespace transform {

Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FlattenBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FlattenBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FlattenBuffer").set_body_typed(FlattenBuffer);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
