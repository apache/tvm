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
 * \file src/relax/transform/remove_redundant_assignments.cc
 * \brief This pass removes redundant assignment statements. These stmts are result of other pass
 * like hint_on_device processed by RealizeVDevice may leave them. The subsequent pass like
 * fuse_ops fail to fuse in this case.
 */

#include <tvm/node/serialization.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include <tuple>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

class RemoveRedundantAssignments : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  IRModule Run(IRModule& mod) {
    mod_ = mod;
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<relax::FunctionNode>()) {
        const auto& base_func = mod_->Lookup(gv);
        // Only non primitive relax functions
        if (base_func->HasNonzeroAttr(attr::kPrimitive)) {
          continue;
        }
        relax::Function update_func = Downcast<Function>(VisitExpr(func));
        updates_->Add(gv, update_func);
      }
    }
    mod_.CopyOnWrite()->Update(updates_);
    return mod_;
  }

  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) final {
    redundant_map.Set(GetRef<Expr>(binding->var.get()), GetRef<Expr>(var));
  }

  void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* val) override {
    redundant_map.Set(GetRef<Expr>(binding->var.get()), GetRef<Expr>(val));
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    Tuple args;

    if (call->op == call_tir_op) {
      args = Downcast<Tuple>(call->args[1]);
    } else {
      args = Tuple(call->args);
    }
    Array<Expr> new_args;

    for (auto& arg : args->fields) {
      if (redundant_map.find(arg) != redundant_map.end()) {
        new_args.push_back(redundant_map[arg]);
      } else {
        new_args.push_back(arg);
      }
    }
    if (call->op == call_tir_op) {
      return Call(call_tir_op, {call->args[0], Tuple(new_args)}, call->attrs,
                  {call->sinfo_args[0]});
    } else {
      if (call->sinfo_args.size() > 0) {
        return Call(call->op, new_args, call->attrs, {call->sinfo_args[0]});
      } else {
        return Call(call->op, new_args, call->attrs);
      }
    }
  }

 private:
  Map<Expr, Expr> redundant_map;
  IRModule updates_;
  IRModule mod_;
};

namespace transform {

Pass RemoveRedundantAssignments() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return relax::RemoveRedundantAssignments().Run(mod); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"RemoveRedundantAssignments",
                          /*required=*/{});
}
TVM_REGISTER_GLOBAL("relax.transform.RemoveRedundantAssignments")
    .set_body_typed(RemoveRedundantAssignments);
}  // namespace transform
}  // namespace relax
}  // namespace tvm
