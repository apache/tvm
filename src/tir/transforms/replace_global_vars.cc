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
 * \file src/tir/transforms/replace_global_vars.cc
 *
 * \brief GlobalVar replacement across IR types
 */

#include <tvm/ir/replace_global_vars.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

namespace {
using tvm::transform::GlobalVarReplacer;

struct Mutator : StmtExprMutator {
  Map<GlobalVar, GlobalVar> replacements;
  explicit Mutator(Map<GlobalVar, GlobalVar> replacements) : replacements(replacements) {}

  PrimExpr VisitExpr_(const CallNode* node) override {
    auto call = Downcast<Call>(StmtExprMutator::VisitExpr_(node));
    if (auto old_gvar = call->op.as<GlobalVar>()) {
      if (auto new_gvar = replacements.Get(old_gvar.value())) {
        call.CopyOnWrite()->op = new_gvar.value();
      }
    }
    return call;
  }
};

}  // namespace

TVM_STATIC_IR_FUNCTOR(GlobalVarReplacer, vtable)
    .set_dispatch<tir::PrimFuncNode>([](const ObjectRef& obj,
                                        Map<GlobalVar, GlobalVar> replacements) -> BaseFunc {
      Mutator mutator(replacements);
      auto func = Downcast<PrimFunc>(obj);
      auto new_body = mutator(func->body);

      if (!new_body.same_as(func->body)) {
        func.CopyOnWrite()->body = new_body;
      }

      // If the function is externally exposed, and is being replaced
      // by a GlobalVar with a new name, then the function's
      // kGlobalSymbol must be updated to match.
      if (auto opt = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
        auto name = opt.value();
        for (const auto& [before, after] : replacements) {
          if (before->name_hint == name) {
            if (after->name_hint != name) {
              func = WithAttr(func, tvm::attr::kGlobalSymbol, after->name_hint);
            }
            break;
          }
        }
      }

      return func;
    });

}  // namespace tir
}  // namespace tvm
