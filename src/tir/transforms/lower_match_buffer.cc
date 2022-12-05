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
 * \file lower_match_buffer.cc
 * \brief The pass for lowering match_buffer.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../ir/functor_common.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {
class MatchBufferLower : public StmtExprMutator {
 public:
  explicit MatchBufferLower(const PrimFunc& func) {
    for (const Var& param : func->params) {
      // Mark input var as const variable.
      if (!param.dtype().is_handle()) var_map_.Set(param, param);
    }
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    for (const MatchBufferRegion& match_buffer : op->match_buffers) {
      CheckAndUpdateVarMap(match_buffer);
    }

    Stmt stmt = StmtExprMutator ::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);
    Array<BufferRegion> reads =
        op->reads.Map(std::bind(&MatchBufferLower::VisitBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes = op->writes.Map(
        std::bind(&MatchBufferLower::VisitBufferRegion, this, std::placeholders::_1));

    if (reads.same_as(op->reads) && writes.same_as(op->writes) && op->match_buffers.empty()) {
      return stmt;
    } else {
      auto n = CopyOnWrite(op);
      n->match_buffers = {};
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    auto it = var_map_.find(v);
    if (it != var_map_.end()) {
      return (*it).second;
    } else {
      return std::move(v);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);

    auto it = match_buffers_.find(op->buffer);
    if (it == match_buffers_.end()) {
      return stmt;
    } else {
      const Buffer& buffer = (*it).first;
      const BufferRegion& source = (*it).second;

      auto n = CopyOnWrite(op);
      n->indices = ConvertIndices(MatchBufferRegion(buffer, source), op->indices);
      n->buffer = source->buffer;
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op != nullptr);

    auto it = match_buffers_.find(op->buffer);
    if (it == match_buffers_.end()) {
      return expr;
    } else {
      const Buffer& buffer = (*it).first;
      const BufferRegion& source = (*it).second;
      Array<PrimExpr> indices = ConvertIndices(MatchBufferRegion(buffer, source), op->indices);
      return BufferLoad(source->buffer, indices);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    CHECK(var_map_.find(op->buffer_var) == var_map_.end())
        << "Load from buffer created by match_buffer is not allowed, but got: " << expr;
    return expr;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    CHECK(var_map_.find(op->buffer_var) == var_map_.end())
        << "Store from buffer created by match_buffer is not allowed, but got: " << stmt;
    return stmt;
  }

  BufferRegion VisitBufferRegion(const BufferRegion& buffer_region) {
    const Buffer& buffer = buffer_region->buffer;
    auto it = match_buffers_.find(buffer);
    if (it == match_buffers_.end()) {
      return buffer_region;
    } else {
      const BufferRegion& source = (*it).second;
      Region region = ConvertRegion(MatchBufferRegion(buffer, source), buffer_region->region);
      return BufferRegion(source->buffer, std::move(region));
    }
  }

 private:
  void CheckAndUpdateVarMap(const MatchBufferRegion& match_buffer) {
    // Step.1. Check
    const Buffer& buffer = match_buffer->buffer;
    const BufferRegion& source = VisitBufferRegion(match_buffer->source);
    const Buffer& source_buffer = source->buffer;

    // Step.1.1. Check scope & dtype
    ICHECK_EQ(buffer.scope(), source_buffer.scope())
        << "MatchBuffer " << buffer << " scope mismatch:" << buffer.scope() << "vs."
        << source_buffer.scope();
    ICHECK_EQ(buffer->dtype, source_buffer->dtype)
        << "MatchBuffer " << buffer << " data type mismatch:" << buffer->dtype << "vs."
        << source_buffer->dtype;

    // Step.1.2. Check data alignment
    if (source_buffer->data_alignment % buffer->data_alignment != 0) {
      LOG(WARNING) << "Trying to bind buffer to another one with lower alignment requirement "
                   << " required_alignment=" << buffer->data_alignment
                   << ", provided_alignment=" << source_buffer->data_alignment;
    }
    if (is_zero(buffer->elem_offset)) {
      ICHECK(is_zero(source_buffer->elem_offset))
          << "Trying to bind a Buffer with offset into one without offset "
          << " required elem_offset=" << buffer->elem_offset
          << ", provided elem_offset=" << source_buffer->elem_offset;
    }

    // Step.2. Update
    match_buffers_.Set(buffer, source);
    // Step.2.1. Update buffer data
    Bind(buffer->data, source_buffer->data, buffer->name + ".data");

    // Step.2.2. Update element offset
    // We use the ElemOffset method to avoid duplicating the index calculation.
    {
      Array<PrimExpr> indices;
      indices.reserve(source->region.size());
      for (const Range& range : source->region) {
        indices.push_back(range->min);
      }

      Array<PrimExpr> buffer_start_indices = source_buffer->ElemOffset(indices);
      if (buffer_start_indices.size() == 1) {
        Bind(buffer->elem_offset, buffer_start_indices[0], buffer->name + ".elem_offset");
        CHECK(analyzer_.CanProve(truncmod(buffer->elem_offset, buffer->offset_factor) == 0))
            << "The source elem_offset " << buffer_start_indices[0]
            << " does not satisfy the offset_factor " << buffer->offset_factor << ".";
      } else {
        // Non-zero elem_offset is ill-defined for non-flat memory.
        // If needed in the future, will require `Array<PrimExpr>
        // elem_offsets`, with one offset for each flattened index.
        Bind(buffer->elem_offset, make_const(buffer->elem_offset.dtype(), 0));
      }
    }

    // Step 2.3. Check and update strides
    // Check if target buffer strides are defined
    ICHECK(source->region.size() >= buffer->shape.size());
    int offset = source->region.size() - buffer->shape.size();
    if (!buffer->strides.empty()) {
      ICHECK_EQ(buffer->strides.size(), buffer->shape.size());
      if (source_buffer->strides.empty()) {
        PrimExpr stride = make_const(buffer->strides.back().dtype(), 1);
        for (size_t i = buffer->shape.size(); i > 0; --i) {
          const PrimExpr& shape = source_buffer->shape[i - 1 + offset];
          Bind(buffer->strides[i - 1], stride, buffer->name + ".strides_" + std::to_string(i - 1));
          stride *= shape;
        }
      } else {
        ICHECK_EQ(buffer->shape.size() + offset, source_buffer->strides.size());
        for (size_t i = buffer->shape.size(); i > 0; --i) {
          const PrimExpr& stride = source_buffer->strides[i - 1 + offset];
          Bind(buffer->strides[i - 1], stride, buffer->name + ".strides_" + std::to_string(i - 1));
        }
      }
    }

    // Step 2.4. Check and update shape
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      const Range& range = source->region[i + offset];
      Bind(buffer->shape[i], range->extent, buffer->name + ".shape_" + std::to_string(i));
    }
  }

  void Bind(const PrimExpr& arg, PrimExpr value, const std::string& arg_name = "argument") {
    CHECK_EQ(arg.dtype(), value.dtype())
        << "The data type mismatched: " << arg->dtype << " vs. " << value->dtype;
    // Handle recursive case
    value = Substitute(std::move(value), var_map_);
    if (arg->IsInstance<VarNode>()) {
      Var v = Downcast<Var>(arg);
      auto it = var_map_.find(v);
      if (it == var_map_.end()) {
        var_map_.Set(v, value);
        analyzer_.Bind(v, value);
      } else {
        AssertBinding((*it).second, value, arg_name);
      }
    } else {
      AssertBinding(arg, value, arg_name);
    }
  }

  void AssertBinding(const PrimExpr& lhs, const PrimExpr& rhs,
                     const std::string& arg_name = "argument") {
    CHECK(analyzer_.CanProve(lhs == rhs)) << "The buffer match constraint for " << arg_name
                                          << " unmet: " << lhs << "==" << rhs << ".";
  }

 private:
  /*! \brief Buffer region mapping. */
  Map<Buffer, BufferRegion> match_buffers_;
  /*! \brief Var mapping for buffer signature (data, strides, element_offset, etc.) */
  Map<Var, PrimExpr> var_map_;
  /*! \brief The analyzer */
  arith::Analyzer analyzer_;
};

PrimFunc LowerMatchBuffer(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  fptr->body = MatchBufferLower(func)(std::move(fptr->body));
  return func;
}

namespace transform {

Pass LowerMatchBuffer() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerMatchBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerMatchBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerMatchBuffer").set_body_typed(LowerMatchBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
