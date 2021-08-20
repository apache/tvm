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
 * \file src/tir/ir/specialize.cc
 * \brief Specialize parameters of PrimFunc.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>

#include "functor_common.h"

namespace tvm {
namespace tir {

using VarMap = std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

/**************** Helper functions ****************/

/*! \brief Helper function to check whether the given var is in function parameter list. */
inline bool IsParam(const PrimFunc& func, const Var& param) {
  return std::any_of(func->params.begin(), func->params.end(),
                     [&](const Var& var) { return var.same_as(param); });
}

/**************** Specializer ****************/

// Try fold constants if op's child get specialized to constant.
#define DEFINE_SPECIALIZER_BINARY_OP_MUTATE(BinaryNode, BinaryFunc) \
  PrimExpr VisitExpr_(const BinaryNode* op) final {                 \
    PrimExpr a = VisitExpr(op->a);                                  \
    PrimExpr b = VisitExpr(op->b);                                  \
    if (a.same_as(op->a) && b.same_as(op->b)) {                     \
      return GetRef<PrimExpr>(op);                                  \
    } else {                                                        \
      return BinaryFunc(a, b);                                      \
    }                                                               \
  }
#define DEFINE_SPECIALIZER_UNARY_OP_MUTATE(UnaryNode, UnaryFunc) \
  PrimExpr VisitExpr_(const UnaryNode* op) final {               \
    PrimExpr a = VisitExpr(op->a);                               \
    if (a.same_as(op->a)) {                                      \
      return GetRef<PrimExpr>(op);                               \
    } else {                                                     \
      return UnaryFunc(a);                                       \
    }                                                            \
  }

/*! \brief Mutator to specialize function and remove const parameters */
class PrimFuncSpecializer : public StmtExprMutator {
 public:
  explicit PrimFuncSpecializer(const VarMap& var_map) : var_map_(var_map) {}

  static PrimFunc Specialize(PrimFunc f, const VarMap& var_map) {
    PrimFuncSpecializer specializer(var_map);
    // Updating Buffer map
    Map<Var, Buffer> buffer_map;
    bool buffer_map_updated = false;
    for (const auto& it : f->buffer_map) {
      const Var& var = it.first;
      const Buffer& buffer = it.second;
      Buffer new_buffer = specializer.MutateBuffer(buffer);
      buffer_map.Set(var, new_buffer);
      if (!new_buffer.same_as(buffer)) {
        buffer_map_updated = true;
        specializer.buffer_map_[buffer] = new_buffer;
      }
    }

    // Updating parmeters
    Array<Var> params;
    bool param_updated = false;
    for (const auto& var : f->params) {
      // Remove parmeters which has been specialized.
      if (var_map.find(var) == var_map.end()) {
        params.push_back(var);
      } else {
        param_updated = true;
      }
    }

    // Updating function body
    Stmt body = specializer(f->body);

    if (param_updated || buffer_map_updated || !f->body.same_as(body)) {
      PrimFuncNode* f_ptr = f.CopyOnWrite();
      f_ptr->params = std::move(params);
      f_ptr->buffer_map = std::move(buffer_map);
      f_ptr->body = std::move(body);
    }
    return f;
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    // Step.0. Define buffer mappings which is allocated inside the block
    Array<Buffer> alloc_buffers = MutateArray(
        op->alloc_buffers,
        std::bind(&PrimFuncSpecializer::MutateAllocBuffer, this, std::placeholders::_1));

    // Step.1. Recursively visit block body
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    Array<BufferRegion> reads = MutateArray(
        op->reads,
        std::bind(&PrimFuncSpecializer::MutateBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes = MutateArray(
        op->writes,
        std::bind(&PrimFuncSpecializer::MutateBufferRegion, this, std::placeholders::_1));

    if (alloc_buffers.same_as(op->alloc_buffers) && reads.same_as(op->reads)) {
      return GetRef<Block>(op);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(op);
      n->alloc_buffers = std::move(alloc_buffers);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);
    auto it = buffer_map_.find(op->buffer);
    if (it == buffer_map_.end()) {
      return GetRef<BufferStore>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op != nullptr);
    auto it = buffer_map_.find(op->buffer);
    if (it == buffer_map_.end()) {
      return GetRef<BufferLoad>(op);
    } else {
      auto n = make_object<BufferLoadNode>(*op);
      n->buffer = it->second;
      return PrimExpr(n);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(GetRef<Var>(op));
    if (it == var_map_.end()) {
      return GetRef<PrimExpr>(op);
    } else {
      return it->second;
    }
  }

  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(AddNode, add);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(SubNode, sub);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(MulNode, mul);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(DivNode, div);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(ModNode, truncmod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(FloorDivNode, floordiv);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(FloorModNode, floormod);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(MaxNode, max);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(MinNode, min);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(EQNode, equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(NENode, not_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(LTNode, less);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(LENode, less_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(GTNode, greater);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(GENode, greater_equal);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(AndNode, logical_and);
  DEFINE_SPECIALIZER_BINARY_OP_MUTATE(OrNode, logical_or);
  DEFINE_SPECIALIZER_UNARY_OP_MUTATE(NotNode, logical_not);

 private:
  Buffer MutateBuffer(const Buffer& buffer) {
    Array<PrimExpr> shape =
        MutateArray(buffer->shape, [this](const PrimExpr& e) { return VisitExpr(e); });
    Array<PrimExpr> strides =
        MutateArray(buffer->strides, [this](const PrimExpr& e) { return VisitExpr(e); });

    PrimExpr elem_offset = Substitute(buffer->elem_offset, var_map_);

    if (buffer->elem_offset.same_as(elem_offset) && buffer->shape.same_as(shape) &&
        buffer->strides.same_as(strides)) {
      return buffer;
    } else {
      auto n = make_object<BufferNode>(*buffer.get());
      n->elem_offset = std::move(elem_offset);
      n->shape = std::move(shape);
      n->strides = std::move(strides);
      return Buffer(n);
    }
  }

  Range MutateRange(const Range& range) {
    PrimExpr min = this->VisitExpr(range->min);
    PrimExpr extent = this->VisitExpr(range->extent);
    if (min.same_as(range->min) && extent.same_as(range->extent)) {
      return range;
    } else {
      return Range::FromMinExtent(std::move(min), std::move(extent));
    }
  }

  Buffer MutateAllocBuffer(const Buffer& alloc_buf) {
    Buffer buf = MutateBuffer(alloc_buf);
    if (buf.same_as(alloc_buf)) {
      return alloc_buf;
    } else {
      ICHECK(buffer_map_.find(alloc_buf) == buffer_map_.end());
      buffer_map_[alloc_buf] = buf;
      return buf;
    }
  }

  BufferRegion MutateBufferRegion(const BufferRegion& buffer_region) {
    auto it = buffer_map_.find(buffer_region->buffer);
    Array<Range> region =
        MutateArray(buffer_region->region,
                    std::bind(&PrimFuncSpecializer::MutateRange, this, std::placeholders::_1));
    if (it == buffer_map_.end() && region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(it->second, std::move(region));
    }
  }

 private:
  /*! \brief The vars to be substitute and their values */
  const VarMap& var_map_;
  /*! \brief map from old buffer to mutated buffer */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
};

/*!
 * \brief Update Specialize var map with buffer matching.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_buf The matching buffer.
 * \param var_map The var mapping to be updated.
 * \note This function will match target buffer's shape, strides and element_offset
 *   For example, we define a buffer in PrimFunc:
 *   A = tir.match_buffer(a, [m, n])
 *
 *   Then we match it with a buffer B =  tir.decl_buffer((8, 16))
 *
 *   It means we have two var mappings here: m = 8 and n = 16
 *
 *   If the buffer signature is not a Var, the mapping will fail.
 *   e.g. A = tir.match_buffer(a, [m * 2, n + 1])
 */
void UpdateSpecializeVarMap(const PrimFunc& func, const Var& param, const Buffer& specific_buf,
                            VarMap* var_map) {
  // preliminaries
  tir::ExprDeepEqual equal;

  auto it = func->buffer_map.find(param);
  CHECK(it != func->buffer_map.end())
      << "ValueError: specialize expects param to be in PrimFunc's buffer_map";
  const Buffer& buf_to_specialize = (*it).second;

  // build var mapping using specific_buf's parameters
  auto build_var_mapping = [&](const PrimExpr& new_expr, const PrimExpr& old_expr) {
    if (!equal(new_expr, old_expr)) {
      CHECK(old_expr->IsInstance<VarNode>())
          << "TypeError: The signature of target buffer exprected an independent Var, but got "
          << old_expr << ".";
      const Var& var = Downcast<Var>(old_expr);
      auto it = var_map->find(var);
      if (it != var_map->end()) {
        CHECK(equal(it->second, new_expr))
            << "ValueError: The assigned value of var " << var << " mismatched. " << it->second
            << " vs. " << new_expr << ".";
      } else {
        (*var_map)[var] = new_expr;
      }
    }
  };

  // Check buffer dimensions
  CHECK(specific_buf->shape.size() == buf_to_specialize->shape.size())
      << "ValueError: The buffer dimensions mismatched" << buf_to_specialize->shape.size()
      << " vs. " << specific_buf->shape.size() << ".";

  CHECK(specific_buf->strides.size() == buf_to_specialize->strides.size())
      << "ValueError: The buffer strides dimensions mismatched" << buf_to_specialize->strides.size()
      << " vs. " << specific_buf->strides.size() << ".";

  // Updating var mapping using specific_expr
  for (size_t i = 0; i < specific_buf->shape.size(); ++i) {
    build_var_mapping(specific_buf->shape[i], buf_to_specialize->shape[i]);
  }
  for (size_t i = 0; i < specific_buf->strides.size(); ++i) {
    build_var_mapping(specific_buf->strides[i], buf_to_specialize->strides[i]);
  }
  build_var_mapping(specific_buf->elem_offset, buf_to_specialize->elem_offset);

  // Check data_alignment and offset_factor.
  // These two signatures are int, so we do not need map them.
  CHECK_EQ(specific_buf->data_alignment, buf_to_specialize->data_alignment)
      << "ValueError: The buffer data_alignment mismatched" << buf_to_specialize->data_alignment
      << " vs. " << specific_buf->data_alignment << ".";

  CHECK_EQ(specific_buf->offset_factor, buf_to_specialize->offset_factor)
      << "ValueError: The buffer offset_factor mismatched" << buf_to_specialize->offset_factor
      << " vs. " << specific_buf->offset_factor << ".";
}

/*!
 * \brief Update Specialize var map with parameter value.
 * \param func The function to be specialized.
 * \param param The given function parameter
 * \param specific_expr The parameter value.
 * \param var_map The var mapping to be updated.
 */
void UpdateSpecializeVarMap(const PrimFunc& func, const Var& param, const PrimExpr& specific_expr,
                            VarMap* var_map) {
  // check param is in PrimFunc's parameters
  CHECK(IsParam(func, param)) << "ValueError: Specialize expects param to be in PrimFunc's params";
  // specialize a param not in buffer_map
  CHECK_EQ(func->buffer_map.count(param), 0)
      << "ValueError: Specialize expects param to not be in PrimFunc's buffer_map";
  // build var mapping using specific_expr
  (*var_map)[param] = specific_expr;
}

/**************** Implementation ****************/

PrimFunc Specialize(PrimFunc func, const Map<Var, ObjectRef>& param_map) {
  VarMap var_map;
  for (const auto& kv : param_map) {
    const Var& param = kv.first;
    const ObjectRef& instance = kv.second;
    if (instance->IsInstance<BufferNode>()) {
      UpdateSpecializeVarMap(func, param, Downcast<Buffer>(instance), &var_map);
    } else if (instance->IsInstance<PrimExprNode>()) {
      UpdateSpecializeVarMap(func, param, Downcast<PrimExpr>(instance), &var_map);
    } else {
      LOG(FATAL) << "TypeError: specialize expected instance to be Buffer or PrimExpr, but got "
                 << instance->GetTypeKey();
    }
  }
  return PrimFuncSpecializer::Specialize(func, std::move(var_map));
}

/**************** FFI ****************/

TVM_REGISTER_GLOBAL("tir.Specialize").set_body_typed(Specialize);

}  // namespace tir
}  // namespace tvm
