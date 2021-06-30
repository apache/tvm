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

using ConstantMap = std::unordered_map<tir::Var, const relay::ConstantNode*, runtime::ObjectPtrHash,
                                       runtime::ObjectPtrEqual>;

class ParamsCollector : public StmtExprVisitor {
 public:
  std::vector<const tir::VarNode*> CollectParams(tir::Stmt body, const ConstantMap& constant_map) {
    constant_map_ = constant_map;
    this->VisitStmt(body);
    return constant_list_;
  }

  void VisitExpr_(const LoadNode* ln) {
    if (constant_map_.find(ln->buffer_var) != constant_map_.end()) {
      auto it =
          std::find(constant_list_.begin(), constant_list_.end(), ln->buffer_var.operator->());
      if (it == constant_list_.end()) {
        constant_list_.push_back(ln->buffer_var.operator->());
      }
    }
    StmtExprVisitor::VisitExpr_(ln);
  }

 private:
  std::vector<const tir::VarNode*> constant_list_;
  ConstantMap constant_map_;
  bool first_for_ = true;
};

namespace transform {

Pass BindParams(const std::vector<const relay::ConstantNode*>& constants) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ConstantMap constant_map;

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
      constant_map[b] = constants[i - start];
    }
    n->params = params;
    auto constant_list = ParamsCollector().CollectParams(n->body, constant_map);

    // Allocate constants within the primfunc
    for (auto i : constant_list) {
      auto var = GetRef<Var>(i);
      int ndim = constant_map[var]->data->ndim;
      Array<PrimExpr> extents;

      for (int i = 0; i < ndim; i++) {
        int shape = constant_map[var]->data->shape[i];
        extents.push_back(make_const(DataType::Int(32), shape));
      }
      DataType dtype = DataType(constant_map[var]->data->dtype);
      n->body = tir::AllocateConst(var, constant_map[var]->data, dtype, extents, n->body);
    }

    return std::move(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BindParams", {});
}
}  // namespace transform

}  // namespace tir
}  // namespace tvm
