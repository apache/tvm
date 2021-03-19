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
 * \file texture_flatten.cc
 * \brief Flattens texture from multi-dimensional array to 2D buffer access
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <stack>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../runtime/texture.h"
#include "../ir/buffer_common.h"
#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {
namespace {

using runtime::IsTextureStorage;
using runtime::DefaultTextureLayoutSeparator;
using runtime::ApplyTexture2DFlattening;

inline PrimExpr SimplifyOffset(const Array<PrimExpr>& shape, const Array<PrimExpr>& index) {
  PrimExpr base = make_const(DataType::Int(32), 0);
  ICHECK_EQ(shape.size(), index.size());
  arith::Analyzer ana;
  if (index.size() > 0) {
    PrimExpr offset = index[0];
    for (size_t i = 1; i < index.size(); ++i) {
      offset = MergeMulMod(&ana, offset * shape[i] + index[i]);
    }
    base = base + offset;
  }
  return base;
}
}

class TextureLoweringBase : public StmtExprMutator {
 public:
  explicit TextureLoweringBase(const Map<Var, Buffer>& extern_buffer_map) {
    for (auto kv : extern_buffer_map) {
      extern_buf_.insert(kv.second);
    }
  }

  virtual Stmt VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == attr::realize_scope) {
      std::string realize_scope = op->value.as<StringImmNode>()->value;
      // If realize_scope for external buffer is unset, infer from buffer scope
      if (realize_scope == "" && op->body->IsInstance<BufferRealizeNode>()) {
        const auto* realize = Downcast<BufferRealize>(op->body).get();
        if (extern_buf_.count(realize->buffer)) {
          realize_scope = realize->buffer->scope;
        }
      }
      storage_scope_[op->node.get()] = realize_scope;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

 protected:

  std::string GetStorageScope(const Buffer& buffer) {
    std::string storage_scope;
    auto it = storage_scope_.find(buffer.get());
    // If buffer has a realize_scope attr return it
    if (it != storage_scope_.end()) {
      storage_scope = it->second;
    } else {
      storage_scope = buffer->scope;
    }
    return storage_scope;
  }

  // External buffer
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> extern_buf_;
  // Storage scope
  std::unordered_map<const Object*, std::string> storage_scope_;
};

class TextureFlattener : public TextureLoweringBase {
 public:
  explicit TextureFlattener(const Map<Var, Buffer>& extern_buffer_map,
                            const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>& extern_buffer_binds_)
    : TextureLoweringBase(extern_buffer_map), buffer_binds_(extern_buffer_binds_) {;}

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    if (extern_buf_.count(op->buffer)) {
      return this->VisitStmt(op->body);
    }

    Var buffer_var(op->buffer->data->name_hint, TextureType(op->buffer->dtype));
    let_binding_.insert({op->buffer->data, buffer_var});

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferRealizeNode>();
    Stmt body = this->VisitStmt(op->body);

    std::string storage_scope = GetStorageScope(op->buffer);
    if (IsTextureStorage(storage_scope)) {
      body = this->VisitStmt(op->body);
      ICHECK(op->bounds.size() >= 3) << "Only 2d RGBA texture is currently supported";
      int vec_length = static_cast<int>(op->bounds.back()->extent.as<IntImmNode>()->value);
      ICHECK(vec_length == 4 || vec_length == 1) << "FCD of texture must be vector of length 1 or 4 (RGBA)";

      struct Shape {
        const Array<Range>& bounds;
        PrimExpr operator[](size_t i) const { return bounds[i]->extent; }
      };
      size_t axis = DefaultTextureLayoutSeparator(op->bounds.size(), storage_scope);
      auto texture = ApplyTexture2DFlattening<PrimExpr>(Shape{op->bounds}, op->bounds.size(), axis);
      Array<PrimExpr> args = {texture.width, texture.height};
      stmt = LetStmt(buffer_var, Call(buffer_var.dtype(), builtin::text2d_alloca(), args), body);
    }

    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    std::string storage_scope = GetStorageScope(op->buffer);
    // Lower to two dimensional access
    if (IsTextureStorage(storage_scope)) {
      Array<PrimExpr> args = GetTextureAccessArgs(op, op->buffer);
      args.push_back(op->value);
      stmt = Evaluate(Call(args[0]->dtype, builtin::text2d_store(), args));
    }

    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    // Replace with identitcal external buffer if one exists
    auto buffer = op->buffer;
    if (buffer_binds_.count(op->buffer)) {
      buffer = buffer_binds_[op->buffer];
    }
    // Lower to two dimensional access
    std::string storage_scope = GetStorageScope(buffer);
    if (IsTextureStorage(storage_scope)) {
      Array<PrimExpr> args = GetTextureAccessArgs(op, buffer);
      args.push_back(op->indices.back());
      expr = Call(op->buffer->dtype, builtin::text2d_load(), args);
    }

    return expr;
  }

 protected:

  template<typename T>
  Array<PrimExpr> GetTextureAccessArgs(const T* op, const Buffer& buffer) {
    Array<PrimExpr> args;
    if (let_binding_.count(op->buffer->data)) {
      args.push_back(let_binding_[op->buffer->data]);
    } else {
      args.push_back(buffer->data);
    }
    Array<PrimExpr> row_dims, row_indices, col_dims, col_indices;
    for (size_t i = 0; i < op->buffer->shape.size()-1; i++) {
      if (i < DefaultTextureLayoutSeparator(op->buffer->shape.size(), GetStorageScope(buffer))) {
        col_dims.push_back(op->buffer->shape[i]);
        col_indices.push_back(op->indices[i]);
      } else {
        row_dims.push_back(op->buffer->shape[i]);
        row_indices.push_back(op->indices[i]);
      }
    }
    PrimExpr row_offset = SimplifyOffset(row_dims, row_indices);
    PrimExpr col_offset = SimplifyOffset(col_dims, col_indices);
    args.push_back(row_offset);
    args.push_back(col_offset);
    return args;
  }

  // Let binding
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_binds_;
};


class ExternalBufferForwarding : public TextureLoweringBase {
 public:
  explicit ExternalBufferForwarding(const Map<Var, Buffer>& extern_buffer_map)
    : TextureLoweringBase(extern_buffer_map) {;}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt stmt = TextureLoweringBase::VisitStmt_(op);
    if (op->attr_key == attr::realize_scope) {
      if (op->body->IsInstance<BufferRealizeNode>()) {
        const auto* realize = Downcast<BufferRealize>(op->body).get();
        std::string realize_scope = GetStorageScope(realize->buffer);
        if (IsTextureStorage(realize_scope) && extern_buffer_copy_.count(realize->buffer)) {
          return realize_attrs_.back();
        } else {
          if (realize_attrs_.size()) {
            realize_attrs_.pop_back();
          }
          realize_attrs_.push_back(stmt);
        }
        return stmt;
      }
    }

    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    ICHECK_EQ(external_loads_.size(), 0) << "Found external loads bound to a different store";
    if (auto* call_node = op->value.as<CallNode>()) {
      // Path to supporting external cache_read canceling when padding has induced
      // a conditional load into the cache_read buffer. We may be able to elide the
      // conditional completely due to hardware support for returning 0 when OOB
      if (call_node->op.same_as(builtin::if_then_else())) {
        external_loads_.emplace_back();
      }
    }
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    auto check_identity = [this](const BufferStoreNode* store, const BufferLoad& load) {
      if (extern_buf_.count(load->buffer)) {
        // If the buffer to load and the buffer to store to are both texture
        // check for identical access
        if (IsTextureStorage(GetStorageScope(load->buffer)) &&
            IsTextureStorage(GetStorageScope(store->buffer))) {
          auto store_index = SimplifyOffset(store->buffer->shape, store->indices);
          auto load_index = SimplifyOffset(load->buffer->shape, load->indices);
          if (arith::Analyzer().CanProve(store_index == load_index)) {
            extern_buffer_copy_.insert(store->buffer);
            buffer_map_.insert({store->buffer, load->buffer});
          }
        }
      }
    };

    if (auto load_node = op->value.as<BufferLoadNode>()) {
      check_identity(op, GetRef<BufferLoad>(load_node));
    } else if (external_loads_.size()) {
      // Stored value is not a load, check for external loads collected
      // when visiting the store node's value, e.g. from if_then_else
      for (auto& expr : external_loads_.back()) {
        check_identity(op, Downcast<BufferLoad>(expr));
      }
      external_loads_.pop_back();
    }
    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    if (external_loads_.size() && extern_buf_.count(op->buffer)) {
      external_loads_.back().push_back(expr);
    }
    return expr;
  }

  const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>& GetForwardedBuffers() {
    return buffer_map_;
  }

 private:
  std::deque<Stmt> realize_attrs_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> extern_buffer_copy_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  std::vector<std::vector<PrimExpr>> external_loads_;
};


PrimFunc TextureFlatten(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  ExternalBufferForwarding forward(fptr->buffer_map);
  fptr->body = forward(std::move(fptr->body));
  fptr->body = TextureFlattener(fptr->buffer_map, forward.GetForwardedBuffers())(std::move(fptr->body));
  return func;
}

namespace transform {

Pass TextureFlatten() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TextureFlatten(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.TextureFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.TextureFlatten").set_body_typed(TextureFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
