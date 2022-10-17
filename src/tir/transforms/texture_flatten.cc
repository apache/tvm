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
 * \brief Flattens texture storage from multi-dimensional array
 * to 2D (width, height) buffer access
 */

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/texture.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace tir {
using arith::IRVisitorWithAnalyzer;
using runtime::ApplyTexture2DFlattening;
using runtime::DefaultTextureLayoutSeparator;
using runtime::IsTextureStorage;

class TextureLoweringBase : public StmtExprMutator {
 public:
  explicit TextureLoweringBase(const Map<Var, Buffer>& extern_buffer_map,
                               IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_{bound_analyzer} {
    for (auto kv : extern_buffer_map) {
      extern_buf_.insert(kv.second);
    }
  }

  inline PrimExpr SimplifyOffset(const Array<PrimExpr>& shape, const Array<PrimExpr>& index) const {
    PrimExpr base = make_const(DataType::Int(32), 0);
    ICHECK_EQ(shape.size(), index.size());
    if (index.size() > 0) {
      PrimExpr offset = index[0];
      for (size_t i = 1; i < index.size(); ++i) {
        offset = bound_analyzer_->Simplify(offset * shape[i] + index[i]);
      }
      base = base + offset;
    }
    return base;
  }

 protected:
  std::string GetStorageScope(const Buffer& buffer) {
    auto* ptr = buffer->data->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr) << "Buffer Var's type annotation must be of PointerType";
    return ptr->storage_scope;
  }

  // Set of all external input and output buffers
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> extern_buf_;
  // Bound analzer
  IRVisitorWithAnalyzer* bound_analyzer_;
};

// Lower Nd storage access to 2d texture access using lowering convention
// specified by the buffers storage scope.
class TextureFlattener : public TextureLoweringBase {
 public:
  using StmtExprMutator::VisitStmt_;
  explicit TextureFlattener(const Map<Var, Buffer>& extern_buffer_map,
                            IRVisitorWithAnalyzer* bound_analyzer)
      : TextureLoweringBase(extern_buffer_map, bound_analyzer) {}

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    if (extern_buf_.count(op->buffer)) {
      return this->VisitStmt(op->body);
    }

    std::string storage_scope = GetStorageScope(op->buffer);
    Var buffer_var(op->buffer->data->name_hint,
                   PointerType(PrimType(op->buffer->dtype), String(storage_scope)));
    let_binding_.insert({op->buffer->data, buffer_var});

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferRealizeNode>();

    // Rewrite any buffer realizations with storage scope to 2d texture allocations
    if (IsTextureStorage(storage_scope)) {
      Stmt body = this->VisitStmt(op->body);
      ICHECK(op->bounds.size() >= 3) << "Only 2d RGBA texture is currently supported";
      int vec_length = static_cast<int>(op->bounds.back()->extent.as<IntImmNode>()->value);
      ICHECK(vec_length == 4 || vec_length == 1)
          << "Inner dimension of texture must be vector of length 1 or 4 (RGBA), was: "
          << vec_length;

      struct ShapeFromRange {
        const Array<Range>& bounds;
        PrimExpr operator[](size_t i) const { return bounds[i]->extent; }
      };
      size_t axis = DefaultTextureLayoutSeparator(op->bounds.size(), storage_scope);
      auto texture =
          ApplyTexture2DFlattening<PrimExpr>(ShapeFromRange{op->bounds}, op->bounds.size(), axis);
      Array<PrimExpr> args;
      args.push_back(StringImm(storage_scope));
      args.push_back(IntImm(DataType::Int(64), 2));  // 2d
      args.push_back(Call(DataType::Handle(), builtin::tvm_stack_make_shape(),
                          {texture.width, texture.height}));
      stmt = LetStmt(buffer_var, Call(buffer_var.dtype(), builtin::nd_mem_alloc_with_scope(), args),
                     body);
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
      stmt = Evaluate(Call(args[0]->dtype, builtin::texture2d_store(), args));
    }

    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    // Lower to two dimensional access
    std::string storage_scope = GetStorageScope(op->buffer);
    if (IsTextureStorage(storage_scope)) {
      Array<PrimExpr> args = GetTextureAccessArgs(op, op->buffer);
      args.push_back(op->indices.back());
      expr = Call(op->buffer->dtype, builtin::texture2d_load(), args);
    }

    return expr;
  }

 protected:
  template <typename T>
  Array<PrimExpr> GetTextureAccessArgs(const T* op, const Buffer& buffer) {
    Array<PrimExpr> args;
    if (let_binding_.count(op->buffer->data)) {
      args.push_back(let_binding_[op->buffer->data]);
    } else {
      args.push_back(buffer->data);
    }
    Array<PrimExpr> row_dims, row_indices, col_dims, col_indices;
    for (size_t i = 0; i < op->buffer->shape.size() - 1; i++) {
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

  // Bindings to new texture vars with texture pointer scope
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> let_binding_;
};

PrimFunc TextureFlatten(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  IRVisitorWithAnalyzer bound_analyzer;
  bound_analyzer(fptr->body);
  fptr->body = TextureFlattener(fptr->buffer_map, &bound_analyzer)(std::move(fptr->body));
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
