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
 *
 * \file src/relay/transforms/defuse_ops.cc
 * \brief This is an inverse operation of fusion pass. It transforms a fused
 * program returned by relay::transform::FuseOps into the program before FuseOps.
 * (i.e., x == DefuseOps(FuseOps(x)))
 */

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

class DefuseOpsMutator : public ExprMutator {
 public:
  class FuncBodyMutator : public ExprMutator {
   public:
    explicit FuncBodyMutator(const Array<Expr>& args) : ExprMutator() { args_ = args; }

    Expr VisitExpr_(const VarNode* n) {
      const std::string& name = n->name_hint();
      ICHECK(!name.empty() && (name[0] == 'p'));
      std::string id_str = name.substr(1);
      int id = std::stoi(id_str);
      ICHECK(id >= 0 && size_t(id) < args_.size());
      return args_[id];
    }

   private:
    Array<Expr> args_;
  };

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);

    if (const auto* call = new_n.as<CallNode>()) {
      if (const auto* func = call->op.as<FunctionNode>()) {
        if (func->body->IsInstance<CallNode>()) {
          return FuncBodyMutator(call->args).Mutate(func->body);
        }
      }
    }
    return new_n;
  }
};

Expr DefuseOps(const Expr& expr) { return DefuseOpsMutator().Mutate(expr); }

namespace transform {

Pass DefuseOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(DefuseOps(f)); };
  return CreateFunctionPass(pass_func, 3, "DefuseOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.DefuseOps").set_body_typed(DefuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
