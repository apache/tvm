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
 * \file src/relax/transform/remove_purity_checking.cc
 * \brief Apply kForcePure in all pure functions and unwrap all calls to pure overrides
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

namespace tvm {
namespace relax {

class PurityRemover : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Function RemovePurity(Function func) {
    bool purity = func->is_pure;
    auto ret = func;
    if (purity) {
      ret = std::move(WithAttr<Function>(func, relax::attr::kForcePure, Bool(true)));
    }
    auto new_body = VisitExpr(ret->body);
    if (!new_body.same_as(ret->body)) {
      return Function(ret->params, new_body, ret->ret_struct_info, ret->is_pure, ret->attrs,
                      ret->span);
    }
    return ret;
  }

  Expr VisitExpr_(const CallNode* call) override {
    if (call->op == call_pure_packed_op_) {
      auto ret = Call(call->args[0], Array<Expr>(call->args.begin() + 1, call->args.end()),
                      call->attrs, call->sinfo_args);
      return VisitExpr(ret);
    }
    if (call->op == call_inplace_packed_op_) {
      // call_inplace_packed has its own attrs so we don't pass those down
      auto ret = Call(call->args[0], Array<Expr>(call->args.begin() + 1, call->args.end()),
                      tvm::Attrs(), call->sinfo_args);
      return VisitExpr(ret);
    }
    if (call->op == invoke_pure_closure_op_) {
      auto ret = Call(invoke_closure_op_, call->args, call->attrs, call->sinfo_args);
      return VisitExpr(ret);
    }
    return ExprMutator::VisitExpr_(call);
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    // handling inner functions: we will remove purity annotations from them too
    return RemovePurity(GetRef<Function>(func));
  }

 private:
  const Op& call_pure_packed_op_ = Op::Get("relax.call_pure_packed");
  const Op& call_inplace_packed_op_ = Op::Get("relax.call_inplace_packed");
  const Op& invoke_pure_closure_op_ = Op::Get("relax.invoke_pure_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
};

Function RemovePurityChecking(const Function& f) { return PurityRemover().RemovePurity(f); }

namespace transform {

Pass RemovePurityChecking() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](const Function& f, IRModule mod, PassContext pc) {
        return relax::RemovePurityChecking(f);
      };
  return CreateFunctionPass(pass_func, 0, "RemovePurityChecking", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RemovePurityChecking").set_body_typed(RemovePurityChecking);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
