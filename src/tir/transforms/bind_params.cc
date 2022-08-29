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
  explicit ParamsCollector(const Map<tir::Var, runtime::NDArray>& constant_map)
      : constant_map_(constant_map) {}
  std::vector<const tir::VarNode*> CollectParams(tir::Stmt body) {
    this->VisitStmt(body);
    return constant_list_;
  }

  void VisitExpr_(const BufferLoadNode* ln) {
    if (constant_map_.find(ln->buffer->data) != constant_map_.end()) {
      auto it = std::find(constant_list_.begin(), constant_list_.end(), ln->buffer->data.get());
      if (it == constant_list_.end()) {
        constant_list_.push_back(ln->buffer->data.get());
      }
    }
    StmtExprVisitor::VisitExpr_(ln);
  }

  void VisitExpr_(const CallNode* cn) {
    if (cn->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(cn->args.size(), 5U);
      const Var& var = Downcast<Var>(cn->args[1]);
      const VarNode* buffer = cn->args[1].as<VarNode>();
      auto it = constant_map_.find(var);
      if (it != constant_map_.end()) {
        auto it = std::find(constant_list_.begin(), constant_list_.end(), buffer);
        if (it == constant_list_.end()) {
          constant_list_.push_back(buffer);
        }
      }
    }
    StmtExprVisitor::VisitExpr_(cn);
  }

 private:
  std::vector<const tir::VarNode*> constant_list_;
  Map<tir::Var, runtime::NDArray> constant_map_;
};

PrimFunc BindParams(PrimFunc f, const Array<runtime::NDArray>& constants) {
  Map<tir::Var, runtime::NDArray> constant_map;

  // Remove constants from the primfunc signature
  size_t num_constants = constants.size();
  size_t start = f->params.size() - num_constants;
  Array<tir::Var> params;
  for (unsigned i = 0; i < start; i++) {
    params.push_back(f->params[i]);
  }

  auto* n = f.CopyOnWrite();
  for (unsigned i = start; i < f->params.size(); i++) {
    tir::Var p = n->params[i];
    tir::Var b = n->buffer_map[p]->data;
    n->buffer_map.erase(p);
    constant_map.Set(b, constants[i - start]);
  }
  n->params = params;
  auto constant_list = ParamsCollector(constant_map).CollectParams(n->body);

  // Allocate constants within the primfunc
  for (auto i : constant_list) {
    auto var = GetRef<Var>(i);
    int ndim = constant_map[var]->ndim;
    Array<PrimExpr> extents;

    for (int i = 0; i < ndim; i++) {
      int shape = constant_map[var]->shape[i];
      extents.push_back(make_const(DataType::Int(32), shape));
    }
    DataType dtype = DataType(constant_map[var]->dtype);

    if (n->body->IsInstance<BlockRealizeNode>()) {
      auto* block_realize = n->body.as<BlockRealizeNode>();
      auto block = block_realize->block;
      block.CopyOnWrite()->body =
          tir::AllocateConst(var, dtype, extents, constant_map[var], block->body);
      n->body = BlockRealize(block_realize->iter_values, block_realize->predicate, block);
    } else {
      n->body = tir::AllocateConst(var, dtype, extents, constant_map[var], n->body);
    }
  }
  return f;
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
