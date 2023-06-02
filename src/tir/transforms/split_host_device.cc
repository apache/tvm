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
 * \file split_host_device.cc
 * \brief Split device function from host.
 */
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../runtime/thread_storage_scope.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class HostDeviceSplitter : public StmtMutator {
 public:
  explicit HostDeviceSplitter(IRModule* device_mod, std::string name_prefix)
      : device_mod_(device_mod), name_prefix_(name_prefix) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      auto device_target = op->node.as<Target>().value().WithoutHost();
      return SplitDeviceFunc(op->body, device_target);
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  Stmt SplitDeviceFunc(Stmt body, Target device_target) {
    Array<Var> params = [&]() {
      VarUseDefAnalyzer use_def(/*defined_vars=*/{}, /*visit_thread_extent=*/false);
      use_def(body);

      // Sort first by variable typ, then by variable name
      std::vector<Var> params{use_def.undefined_.begin(), use_def.undefined_.end()};
      std::sort(params.begin(), params.end(), [](const Var& a, const Var& b) {
        auto sort_key = [](const Var& var) {
          return std::tuple{
              !var->dtype.is_handle(),
              var->name_hint,
          };
        };
        return sort_key(a) < sort_key(b);
      });
      return params;
    }();

    GlobalVar kernel_symbol_global = [&]() {
      std::stringstream name;
      name << name_prefix_ << "_kernel";
      GlobalVarSupply global_var_supply = GlobalVarSupply(*device_mod_);
      return global_var_supply->FreshGlobal(name.str(), false);
    }();

    PrimFunc device_func(params, body);
    device_func = WithAttrs(std::move(device_func), {{tvm::attr::kTarget, device_target},
                                                     {tir::attr::kNoAlias, Bool(true)},
                                                     {tir::attr::kIsGlobalFunc, Bool(true)}});

    (*device_mod_)->Add(kernel_symbol_global, device_func);
    Array<PrimExpr> args = params.Map([](const Var& var) -> PrimExpr { return var; });

    return Evaluate(Call(DataType::Void(), kernel_symbol_global, args));
  }

  // target ir module
  IRModule* device_mod_;
  // function name hint
  std::string name_prefix_;
};

PrimFunc SplitHostDevice(PrimFunc func, IRModule* device_mod, const GlobalVar& gvar) {
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  auto name_prefix = global_symbol.value_or(gvar->name_hint);

  HostDeviceSplitter splitter(device_mod, name_prefix);

  if (auto body = splitter(func->body); !body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  }

  return func;
}

namespace transform {

Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    IRModule device_mod = IRModule(Map<GlobalVar, BaseFunc>({}));
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        PrimFunc func = opt.value();
        func = SplitHostDevice(std::move(func), &device_mod, gvar);
        if (!func.same_as(base_func)) {
          updates->Add(gvar, func);
        }
      }
    }

    mod->Update(updates);
    mod->Update(device_mod);
    return ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.SplitHostDevice", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SplitHostDevice").set_body_typed(SplitHostDevice);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
