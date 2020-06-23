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

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include <functional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pattern_util.h"

namespace tvm {
namespace relay {

class DefuseOpsMutator : public ExprMutator {
 public:
  class FuncBodyMutator : public ExprMutator {
   public:
    Array<Expr> args_;

    explicit FuncBodyMutator(const Array<Expr>& args) : ExprMutator() { args_ = args; }

    Expr VisitExpr_(const VarNode* n) {
      const std::string& name = n->name_hint();
      CHECK_EQ(name[0], 'p');
      std::string id_str = name.substr(1);
      int id = atoi(id_str.c_str());
      CHECK(id >= 0 && size_t(id) < args_.size());
      return args_[id];
    }
  };

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);

    const auto* call = new_n.as<CallNode>();
    if (call) {
      const auto* func = call->op.as<FunctionNode>();
      if (func) {
        const auto& func_call = func->body.as<CallNode>();
        if (func_call) {
          return FuncBodyMutator(call->args).Mutate(func->body);
        }
      }
    }
    return new_n;
  }
};

Expr DeFuseOps(const Expr& expr) { return DefuseOpsMutator().Mutate(expr); }

namespace transform {

Pass DeFuseOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::DeFuseOps(f));
      };
  return CreateFunctionPass(pass_func, 3, "DeFuseOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.DeFuseOps").set_body_typed(DeFuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
