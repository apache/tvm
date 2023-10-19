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
 * \file storage_rewrite.cc
 * \brief Memory access pattern analysis and optimization.
 *  Re-write data access to enable memory sharing when possible.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class ParamsCollector : public StmtExprVisitor {
 public:
  static std::vector<tir::Buffer> Collect(const Map<tir::Buffer, runtime::NDArray>& constant_map,
                                          tir::Stmt body) {
    auto pass = ParamsCollector(constant_map);
    pass.VisitStmt(body);
    return pass.used_constants_;
  }

 private:
  explicit ParamsCollector(const Map<tir::Buffer, runtime::NDArray>& constant_map) {
    for (const auto& it : constant_map) {
      const auto& buf = it.first;
      unused_constants_[buf->data.get()] = buf;
    }
  }

  void VisitExpr_(const BufferLoadNode* node) {
    HandleAccess(node->buffer->data.get());
    StmtExprVisitor::VisitExpr_(node);
  }

  void VisitExpr_(const VarNode* node) {
    HandleAccess(node);
    StmtExprVisitor::VisitExpr_(node);
  }

  void HandleAccess(const VarNode* buffer_var) {
    auto it = unused_constants_.find(buffer_var);
    if (it != unused_constants_.end()) {
      used_constants_.push_back(it->second);
      unused_constants_.erase(it);
    }
  }

 private:
  std::vector<tir::Buffer> used_constants_;
  std::unordered_map<const tir::VarNode*, tir::Buffer> unused_constants_;
};

PrimFunc BindParams(PrimFunc func, const Array<runtime::NDArray>& constants) {
  // Remove constants from the primfunc signature
  size_t first_constant = func->params.size() - constants.size();
  Array<tir::Var> params;
  Map<tir::Var, tir::Buffer> buffer_map;
  Map<tir::Buffer, runtime::NDArray> constant_map;
  for (unsigned i = 0; i < func->params.size(); i++) {
    Var param = func->params[i];
    if (i < first_constant) {
      params.push_back(func->params[i]);
      if (auto opt = func->buffer_map.Get(param)) {
        buffer_map.Set(param, opt.value());
      }
    } else {
      auto opt = func->buffer_map.Get(param);
      ICHECK(opt.defined()) << "Attempted to bind constant NDArray to parameter " << param
                            << ", but " << param << " is a scalar parameter";
      constant_map.Set(opt.value(), constants[i - first_constant]);
    }
  }

  auto constant_list = ParamsCollector::Collect(constant_map, func->body);

  // Unwrap the root BlockRealize/Block, if present
  Stmt body = func->body;
  if (auto* block_realize = body.as<BlockRealizeNode>()) {
    body = block_realize->block->body;
  }

  // Allocate constants within the primfunc
  for (auto buf : constant_list) {
    const auto& ndarray = constant_map[buf];
    int ndim = ndarray->ndim;

    Array<PrimExpr> extents;
    for (int i = 0; i < ndim; i++) {
      int shape = ndarray->shape[i];
      extents.push_back(make_const(DataType::Int(32), shape));
    }

    DataType dtype = DataType(ndarray->dtype);
    body = tir::DeclBuffer(buf, body);
    body = tir::AllocateConst(buf->data, dtype, extents, ndarray, body);
  }

  // Re-wrap the root BlockRealize/Block, if present
  if (auto opt = func->body.as<BlockRealize>()) {
    auto block_realize = opt.value();
    auto block = block_realize->block;
    block.CopyOnWrite()->body = body;
    block_realize.CopyOnWrite()->block = block;
    body = block_realize;
  }

  auto* write_ptr = func.CopyOnWrite();
  write_ptr->params = params;
  write_ptr->buffer_map = buffer_map;
  write_ptr->body = body;

  return func;
}

namespace transform {

Pass BindParams(const Array<runtime::NDArray>& constants) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return BindParams(f, constants);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BindParams", {});
}
}  // namespace transform

}  // namespace tir
}  // namespace tvm
