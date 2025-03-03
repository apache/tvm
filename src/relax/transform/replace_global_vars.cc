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
 * \file src/relax/transform/replace_global_vars.cc
 *
 * \brief GlobalVar replacement across IR types
 */

#include <tvm/ir/analysis.h>
#include <tvm/ir/replace_global_vars.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
namespace relax {

namespace {
using tvm::transform::GlobalVarReplacer;

struct Mutator : ExprMutator {
  Map<GlobalVar, GlobalVar> replacements;
  explicit Mutator(Map<GlobalVar, GlobalVar> replacements) : replacements(replacements) {}

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const GlobalVarNode* node) override {
    auto gvar = GetRef<GlobalVar>(node);
    return replacements.Get(gvar).value_or(gvar);
  }
};

}  // namespace

TVM_STATIC_IR_FUNCTOR(GlobalVarReplacer, vtable)
    .set_dispatch<relax::FunctionNode>([](const ObjectRef& func,
                                          Map<GlobalVar, GlobalVar> replacements) -> BaseFunc {
      Mutator mutator(replacements);
      auto new_func = Downcast<Function>(mutator(Downcast<Function>(func)));

      // If the function is externally exposed, and is being replaced
      // by a GlobalVar with a new name, then the function's
      // kGlobalSymbol must be updated to match.
      if (auto opt = new_func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
        auto name = opt.value();
        for (const auto& [before, after] : replacements) {
          if (before->name_hint == name) {
            if (after->name_hint != name) {
              new_func = WithAttr(new_func, tvm::attr::kGlobalSymbol, after->name_hint);
            }
            break;
          }
        }
      }

      return new_func;
    });

TVM_STATIC_IR_FUNCTOR(GlobalVarReplacer, vtable)
    .set_dispatch<relax::ExternFuncNode>([](const ObjectRef& func,
                                            Map<GlobalVar, GlobalVar>) -> BaseFunc {
      return Downcast<ExternFunc>(func);
    });

}  // namespace relax
}  // namespace tvm
