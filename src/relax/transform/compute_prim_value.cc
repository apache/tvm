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

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

namespace {

class PrimValueComputeInjector : public ExprMutator {
 public:
  IRModule Finalize() const { return builder_->Finalize(); }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const PrimValueNode* op) override {
    auto node = Downcast<PrimValue>(ExprMutator::VisitExpr_(op));

    if (node->value->IsInstance<tir::IntImmNode>() || node->value->IsInstance<tir::VarNode>()) {
      return node;
    }

    auto ret_dtype = node->value->dtype;
    auto param_vars = tir::UndefinedVars(node->value);
    tir::Stmt body = tir::Evaluate(tir::Call(ret_dtype, tir::builtin::ret(), {node->value}));

    tir::PrimFunc func(param_vars, body, PrimType(ret_dtype), {},
                       DictAttrs({{tir::attr::kIsHostFunc, Bool(true)}}));
    func = tir::RenewDefs(func);

    auto callee = builder_->AddFunction(func, "compute_symbolic_expr");

    return relax::Call(callee, param_vars.Map([](const tir::Var& tir_var) -> relax::Expr {
      return relax::PrimValue(tir_var);
    }));
  }
};

}  // namespace

namespace transform {

Pass ComputePrimValue() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) -> IRModule {
    PrimValueComputeInjector mutator;

    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto updated = Downcast<Function>(mutator(func.value()));
        if (!updates.same_as(base_func)) {
          updates->Add(gvar, updated);
        }
      }
    }

    if (updates->functions.size()) {
      auto write_ptr = mod.CopyOnWrite();
      write_ptr->Update(updates);
      write_ptr->Update(mutator.Finalize());
    }

    return mod;
  };
  return CreateModulePass(pass_func, 0, "ComputePrimValue", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ComputePrimValue").set_body_typed(ComputePrimValue);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
