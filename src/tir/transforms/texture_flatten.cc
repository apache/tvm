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
 * \file storage_flatten.cc
 * \brief Flattens storage from multi-dimensional array to 1D buffer access
 */
// The pass definition originates from Halide pipeline.

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/te/operation.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;

class TextureFlattener : public StmtExprMutator {
 public:
  explicit TextureFlattener(const Map<Var, Buffer>& extern_buffer_map, int cache_line_size,
                            bool create_bound_attributes, IRVisitorWithAnalyzer* bound_analyzer) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImmNode>()->value;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferRealizeNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      Stmt body = this->VisitStmt(op->body);
      Array<PrimExpr> shape;
      for (auto r : op->bounds) {
        shape.push_back(r->extent);
      }
      ICHECK_EQ(shape.size(), 3) << "Only 2d RGBA texture is currently supported";
      ICHECK_EQ(static_cast<int>(shape[2].as<IntImmNode>()->value), 4) << "FCD of texture must be vector of length 4 (RGBA)";

      // TODO(csullivan): Consider check on float only
      StringImm dtype = StringImm(runtime::DLDataType2String(op->buffer->dtype));
      Array<PrimExpr> args = {dtype, shape[0], shape[1]};
      stmt = LetStmt(op->buffer->data, Call(op->buffer->data.dtype(), builtin::text2d_alloca(), args), body);
      stmt = AttrStmt(op->buffer->data, attr::storage_scope, StringImm(storage_scope), stmt);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      // TODO(csullivan): Implement texture intrinsic as builtin
      // stmt = Evaluate(Call(op->buffer->dtype, builtin::isnan(), {op->value}));
    }
    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    std::string storage_scope;
    auto it = storage_scope_.find(op->buffer.get());
    if (it != storage_scope_.end())
    {
      storage_scope = it->second;
    }
    else
    {
      storage_scope = op->buffer->scope;
    }
    if (storage_scope == "texture")
    {
      // TODO(csullivan): Implement texture intrinsic as builtin
      // expr = Call(op->buffer->dtype, builtin::isnan(), {expr});
    }
    return expr;
  }
 private:
  // Storage scope
  std::unordered_map<const Object*, std::string> storage_scope_;
};

PrimFunc TextureFlatten(PrimFunc func, int cache_line_size, bool create_bound_attributes) {
  // std::cout << "Before TextureFlattening: " << func << std::endl;
  auto fptr = func.CopyOnWrite();

  IRVisitorWithAnalyzer bound_analyzer;
  bound_analyzer(fptr->body);
  fptr->body = TextureFlattener(fptr->buffer_map, cache_line_size, create_bound_attributes,
                                &bound_analyzer)(std::move(fptr->body));
  // std::cout << "After TextureFlattening: " << func << std::endl;
  return func;
}

namespace transform {

Pass TextureFlatten(int cache_line_size, bool create_bound_attributes) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return TextureFlatten(std::move(f), cache_line_size, create_bound_attributes);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.TextureFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.TextureFlatten").set_body_typed(TextureFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
