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

#include "../analysis/var_use_def_analysis.h"

namespace tvm {
namespace tir {

class HostDeviceSplitter : public StmtMutator {
 public:
  explicit HostDeviceSplitter(IRModule* device_mod, Target device_target, std::string name_prefix)
      : device_mod_(device_mod), device_target_(device_target), name_prefix_(name_prefix) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    handle_data_type_[op->buffer_var.get()] = make_const(op->dtype, 0);
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::pipeline_exec_scope ||
        op->attr_key == attr::device_scope) {
      return SplitDeviceFunc(GetRef<Stmt>(op));
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  Stmt SplitDeviceFunc(Stmt body) {
    std::ostringstream os;
    os << name_prefix_ << "_kernel" << device_func_counter_++;
    std::string kernel_symbol = os.str();
    // isolate the device function.
    VarUseDefAnalyzer m({}, false);
    body = m(std::move(body));

    Array<Var> params;
    Array<PrimExpr> arguments;
    Map<tir::Var, PrimExpr> remap_vars;

    // Strictly order the arguments: Var pointers, positional arguments.
    for (Var var : m.undefined_) {
      if (var.dtype().is_handle()) {
        // Create a new version of v.
        auto it = handle_data_type_.find(var.get());
        if (it != handle_data_type_.end()) {
          String storage_scope;
          if (auto* ptr_type = var->type_annotation.as<PointerTypeNode>()) {
            storage_scope = ptr_type->storage_scope;
          }
          tir::Var new_var(var->name_hint,
                           PointerType(PrimType((*it).second->dtype), storage_scope));
          params.push_back(new_var);
          remap_vars.Set(var, new_var);
        } else {
          params.push_back(var);
        }
        arguments.push_back(var);
      }
    }
    // positional arguments
    for (Var var : m.undefined_) {
      if (!var.dtype().is_handle()) {
        params.push_back(var);
        arguments.push_back(var);
      }
    }
    GlobalVarSupply global_var_supply = GlobalVarSupply(*device_mod_);
    GlobalVar kernel_symbol_global = global_var_supply->FreshGlobal(kernel_symbol, false);

    PrimFunc device_func(params, Substitute(body, remap_vars));
    device_func = WithAttr(std::move(device_func), tir::attr::kDeviceThreadAxis, m.thread_axis_);
    device_func = WithAttr(std::move(device_func), tvm::attr::kCallingConv,
                           Integer(CallingConv::kDeviceKernelLaunch));
    device_func = WithAttr(std::move(device_func), tvm::attr::kGlobalSymbol,
                           runtime::String(kernel_symbol_global->name_hint));
    device_func = WithAttr(std::move(device_func), tir::attr::kNoAlias, Integer(1));
    device_func = WithAttr(std::move(device_func), tvm::attr::kTarget, device_target_);
    device_func = WithAttr(std::move(device_func), tir::attr::kIsGlobalFunc, Integer(1));
    if (m.use_dyn_shmem_) {
      device_func =
          WithAttr(std::move(device_func), tir::attr::kDeviceUseDynSharedMemory, Integer(1));
    }
    (*device_mod_)->Add(kernel_symbol_global, device_func);

    // generate calls to the device function
    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(kernel_symbol_global->name_hint));
    for (PrimExpr arg : arguments) {
      call_args.push_back(arg);
    }
    for (PrimExpr ext : m.thread_extent_) {
      call_args.push_back(ext);
    }
    if (m.use_dyn_shmem_) {
      call_args.push_back(m.dyn_shmem_size_);
    }
    return Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), call_args));
  }

  // target ir module
  IRModule* device_mod_;
  // Device target
  Target device_target_;
  // function name hint
  std::string name_prefix_;
  // Number of device functions.
  int device_func_counter_{0};
  std::unordered_map<const VarNode*, PrimExpr> handle_data_type_;
};

PrimFunc SplitHostDevice(PrimFunc&& func, IRModule* device_mod) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "SplitHostDevice: Require the target attribute";
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "SplitHostDevice: Expect PrimFunc to have the global_symbol attribute";

  HostDeviceSplitter splitter(device_mod, target.value(),
                              static_cast<std::string>(global_symbol.value()));

  auto* n = func.CopyOnWrite();
  n->body = splitter(std::move(n->body));
  // set the host target to None.
  func = WithAttr(std::move(func), tvm::attr::kTarget, Target(nullptr));
  return std::move(func);
}

namespace transform {

Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    IRModuleNode* mod_ptr = mod.CopyOnWrite();
    auto* func_dict = mod_ptr->functions.CopyOnWrite();
    IRModule device_mod = IRModule(Map<GlobalVar, BaseFunc>({}));

    for (auto& kv : *func_dict) {
      if (kv.second->IsInstance<PrimFuncNode>()) {
        PrimFunc func = Downcast<PrimFunc>(std::move(kv.second));
        ICHECK(device_mod.defined()) << "The device module must be defined.";
        kv.second = SplitHostDevice(std::move(func), &device_mod);
      }
    }
    mod->Update(device_mod);
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.SplitHostDevice", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SplitHostDevice").set_body_typed(SplitHostDevice);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
