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

/*! \brief Mutator to specialize function and remove const parameters */
class PrimFuncSpecializer : public StmtExprMutator {
 public:
  explicit PrimFuncSpecializer(VarMap var_map) : var_map_(var_map) {}

  static PrimFunc Specialize(PrimFunc f, const VarMap& var_map) {
    PrimFuncSpecializer specializer(var_map);
    // Updating Buffer map
    Map<Var, Buffer> buffer_map;
    for (const auto& it : f->buffer_map) {
      const Var& var = it.first;
      const Buffer& buffer = it.second;
      Buffer new_buffer = specializer.MutateBuffer(buffer);
      buffer_map.Set(var, new_buffer);
      if (!new_buffer.same_as(buffer)) {
        specializer.buffer_map_[buffer] = new_buffer;
      }
    }

    // Updating parmeters
    Array<Var> params;
    for (const auto& var : f->params) {
      // Remove parmeters which has been specialized.
      if (var_map.find(var) == var_map.end()) {
        params.push_back(var);
      }
    }

    PrimFuncNode* f_ptr = f.CopyOnWrite();
    f_ptr->params = std::move(params);
    f_ptr->buffer_map = std::move(buffer_map);
    f_ptr->body = specializer(std::move(f_ptr->body));

    // Updating attrs
    if (f->attrs.defined()) {
      auto& attr_dict = f_ptr->attrs.CopyOnWrite()->dict;
      for (const auto& kv : attr_dict) {
        const String& key = kv.first;
        const ObjectRef& value = kv.second;
        if (value->IsInstance<PrimExprNode>()) {
          attr_dict.Set(key, Substitute(Downcast<PrimExpr>(value), var_map));
        }
      }
    }
    return f;
  }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Array<Buffer> alloc_buffers = MutateArray(
        op->alloc_buffers,
        std::bind(&PrimFuncSpecializer::MutateAllocBuffer, this, std::placeholders::_1));
    Array<BufferRegion> reads = MutateArray(
        op->reads,
        std::bind(&PrimFuncSpecializer::MutateBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes = MutateArray(
        op->writes,
        std::bind(&PrimFuncSpecializer::MutateBufferRegion, this, std::placeholders::_1));
    Array<IterVar> block_vars = MutateArray(
        op->iter_vars, std::bind(&PrimFuncSpecializer::MutateIterVar, this, std::placeholders::_1));
    Optional<Stmt> init = NullOpt;
    if (op->init.defined()) {
      init = VisitStmt(op->init.value());
    }
    Stmt body = VisitStmt(op->body);

    if (alloc_buffers.same_as(op->alloc_buffers) && reads.same_as(op->reads) &&
        writes.same_as(op->writes) && block_vars.same_as(op->iter_vars) && body.same_as(op->body) &&
        init.same_as(op->init)) {
      return GetRef<Block>(op);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(op);
      n->alloc_buffers = std::move(alloc_buffers);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->iter_vars = std::move(block_vars);
      n->body = std::move(body);
      n->init = std::move(init);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto it = buffer_map_.find(op->buffer);
    if (it == buffer_map_.end()) {
      return GetRef<BufferStore>(op);
    }

    PrimExpr value = VisitExpr(op->value);
    Array<PrimExpr> indices =
        MutateArray(op->indices, [this](const PrimExpr& e) { return this->VisitExpr(e); });

    auto n = CopyOnWrite(op);
    n->buffer = it->second;
    n->value = std::move(value);
    n->indices = std::move(indices);
    return Stmt(n);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = buffer_map_.find(op->buffer);
    if (it == buffer_map_.end()) {
      return GetRef<BufferLoad>(op);
    }

    Array<PrimExpr> indices =
        MutateArray(op->indices, [this](const PrimExpr& e) { return this->VisitExpr(e); });

    auto n = CopyOnWrite(op);
    n->buffer = it->second;
    n->indices = std::move(indices);
    return PrimExpr(n);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(GetRef<Var>(op));
    if (it == var_map_.end()) {
      return GetRef<PrimExpr>(op);
    } else {
      return it->second;
    }
  }

 private:
  Buffer MutateBuffer(Buffer buffer) const {
    BufferNode* buffer_ptr = buffer.CopyOnWrite();
    Array<PrimExpr> new_shape, new_stride;
    new_shape.reserve(buffer_ptr->shape.size());
    new_shape.reserve(buffer_ptr->strides.size());
    for (const auto& dim : buffer_ptr->shape) {
      new_shape.push_back(Substitute(dim, var_map_));
    }
    for (const auto& stride : buffer_ptr->strides) {
      new_shape.push_back(Substitute(stride, var_map_));
    }
    buffer_ptr->elem_offset = Substitute(buffer_ptr->elem_offset, var_map_);
    buffer_ptr->shape = std::move(new_shape);
    buffer_ptr->strides = std::move(new_stride);
    return buffer;
  }

  Range MutateRange(const Range& range) {
    PrimExpr min = this->VisitExpr(range->min);
    PrimExpr extent = this->VisitExpr(range->extent);
    if (min.same_as(range->min) && extent.same_as(range->extent)) {
      return range;
    } else {
      ObjectPtr<RangeNode> n = CopyOnWrite(range.get());
      n->min = std::move(min);
      n->extent = std::move(extent);
      return Range(n);
    }
  }

  IterVar MutateIterVar(const IterVar& iter_var) {
    Range range = MutateRange(iter_var->dom);
    if (range.same_as(iter_var->dom)) {
      return iter_var;
    } else {
      auto n = CopyOnWrite(iter_var.get());
      n->dom = std::move(range);
      return IterVar(n);
    }
  }

  Buffer MutateAllocBuffer(const Buffer& alloc_buf) {
    Buffer buf = MutateBuffer(alloc_buf);
    if (buf.same_as(alloc_buf)) {
      return alloc_buf;
    } else {
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
      auto n = CopyOnWrite(buffer_region.get());
      n->buffer = it->second;
      n->region = std::move(region);
      return BufferRegion(n);
    }
  }

 private:
  /*! \brief The vars to be substitute and their values */
  VarMap var_map_;
  /*! \brief map from old buffer to mutated buffer */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
};

/**************** Implementation ****************/

PrimFunc Specialize(PrimFunc func, const Var& param, const Buffer& specific_buf) {
  // preliminaries
  tir::ExprDeepEqual equal;
  VarMap var_map;

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
      auto it = var_map.find(var);
      if (it != var_map.end()) {
        CHECK(equal(it->second, new_expr))
            << "ValueError: The assigned value of var " << var << " mismatched. " << it->second
            << " vs. " << new_expr << ".";
      } else {
        var_map[var] = new_expr;
      }
    }
  };

  // Check buffer dimensions
  CHECK(specific_buf->shape.size() == buf_to_specialize->shape.size() &&
        specific_buf->strides.size() == buf_to_specialize->strides.size())
      << "ValueError: The buffer dimensions mismatched" << buf_to_specialize->shape.size()
      << " vs. " << specific_buf->shape.size() << ".";

  // Updating var mapping using specific_expr
  for (size_t i = 0; i < specific_buf->shape.size(); ++i) {
    build_var_mapping(specific_buf->shape[i], buf_to_specialize->shape[i]);
  }
  for (size_t i = 0; i < specific_buf->strides.size(); ++i) {
    build_var_mapping(specific_buf->strides[i], buf_to_specialize->strides[i]);
  }
  build_var_mapping(specific_buf->elem_offset, buf_to_specialize->elem_offset);
  // Specialize function with var mapping
  return PrimFuncSpecializer::Specialize(func, var_map);
}

PrimFunc Specialize(PrimFunc func, const Var& param, const PrimExpr& specific_expr) {
  // preliminaries
  VarMap var_map;
  // check param is in PrimFunc's parameters
  CHECK(IsParam(func, param)) << "ValueError: Specialize expects param to be in PrimFunc's params";
  // specialize a param not in buffer_map
  CHECK_EQ(func->buffer_map.count(param), 0)
      << "ValueError: Specialize expects param to not be in PrimFunc's buffer_map";
  // build var mapping using specific_expr
  var_map[param] = specific_expr;
  // Specialize function with var mapping
  return PrimFuncSpecializer::Specialize(std::move(func), var_map);
}

/**************** FFI ****************/

TVM_REGISTER_GLOBAL("tir.Specialize")
    .set_body_typed<PrimFunc(PrimFunc, Map<Var, ObjectRef>)>([](PrimFunc func,
                                                                Map<Var, ObjectRef> param_map) {
      for (const auto& kv : param_map) {
        const Var& param = kv.first;
        const ObjectRef& instance = kv.second;
        if (instance->IsInstance<BufferNode>()) {
          func = Specialize(std::move(func), param, Downcast<Buffer>(instance));
        } else if (instance->IsInstance<PrimExprNode>()) {
          func = Specialize(std::move(func), param, Downcast<PrimExpr>(instance));
        } else {
          LOG(FATAL) << "TypeError: specialize expected instance to be Buffer or PrimExpr, but got "
                     << instance->GetTypeKey();
        }
      }
      return func;
    });

}  // namespace tir
}  // namespace tvm
