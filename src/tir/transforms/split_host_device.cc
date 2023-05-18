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

/*!
 * \brief Visitor class to collect device-side program information.
 */
class DeviceInfoCollector : public StmtVisitor {
 public:
  Array<IterVar> thread_axis_;
  Array<PrimExpr> thread_extent_;
  PrimExpr dyn_shmem_size_{0};
  bool use_dyn_shmem_{false};

  Array<String> GetLaunchParams() const {
    Array<String> output;
    for (const auto& axis : thread_axis_) {
      output.push_back(axis->thread_tag);
    }
    if (use_dyn_shmem_) {
      output.push_back(runtime::launch_param::kUseDynamicSharedMemoryTag);
    }
    return output;
  }

 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      // thread_extent can appear multiple times
      // use the first appearance as def.
      if (!defined_thread.count(iv.get())) {
        defined_thread.insert(iv.get());
        thread_axis_.push_back(iv);
        thread_extent_.push_back(op->value);
      }

      this->VisitExpr(op->value);
      this->VisitStmt(op->body);
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const AllocateNode* op) final {
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      ICHECK_EQ(use_dyn_shmem_, false) << "Only one dynamic shared memory allocation is allowed.";
      ICHECK_GT(op->extents.size(), 0);
      dyn_shmem_size_ = op->extents[0];
      for (size_t i = 1; i < op->extents.size(); ++i) {
        dyn_shmem_size_ *= op->extents[i];
      }
      dyn_shmem_size_ = dyn_shmem_size_ * (op->dtype.bytes());
      use_dyn_shmem_ = true;
    }
    StmtVisitor::VisitStmt_(op);
  }

  // recording what thread axis have been visited.
  std::unordered_set<const IterVarNode*> defined_thread;
};

/*!
 * \brief Mutator class to remove unrefenced let stmt/expressions.
 * \param use_count The pre-computed variable to use count map.
 */
class UnreferencedLetRemover : public StmtExprMutator {
 public:
  explicit UnreferencedLetRemover(const std::unordered_map<const VarNode*, int>& use_count)
      : use_count_(use_count) {}

 private:
  Stmt VisitStmt_(const LetStmtNode* op) final {
    Stmt body = this->VisitStmt(op->body);
    // eliminate unreferenced let
    if (use_count_.at(op->var.get()) == 0 && SideEffect(op->value) <= CallEffectKind::kReadState) {
      return body;
    } else {
      PrimExpr value = this->VisitExpr(op->value);
      if (body.same_as(op->body) && value.same_as(op->value)) {
        return GetRef<Stmt>(op);
      } else {
        return LetStmt(op->var, value, body);
      }
    }
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    PrimExpr body = this->VisitExpr(op->body);
    PrimExpr value = this->VisitExpr(op->value);
    if (use_count_.at(op->var.get()) == 0 && SideEffect(op->value) <= CallEffectKind::kReadState) {
      return body;
    } else {
      if (body.same_as(op->body) && value.same_as(op->value)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Let(op->var, value, body);
      }
    }
  }

  // pre-computed variable to use count map.
  const std::unordered_map<const VarNode*, int>& use_count_;
};

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
    VarUseDefAnalyzer use_def(/*defined_vars=*/{}, /*visit_thread_extent=*/false);
    use_def(body);
    DeviceInfoCollector dev_info;
    dev_info(body);
    UnreferencedLetRemover let_remover(use_def.use_count_);
    body = let_remover(std::move(body));

    Array<Var> params;
    Array<PrimExpr> arguments;
    Map<tir::Var, tir::Var> remap_vars;

    // Strictly order the arguments: Var pointers, positional arguments.
    for (Var var : use_def.undefined_) {
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
    for (Var var : use_def.undefined_) {
      if (!var.dtype().is_handle()) {
        params.push_back(var);
        arguments.push_back(var);
      }
    }
    GlobalVarSupply global_var_supply = GlobalVarSupply(*device_mod_);
    GlobalVar kernel_symbol_global = global_var_supply->FreshGlobal(kernel_symbol, false);

    PrimFunc device_func(params, Substitute(body, remap_vars));
    device_func = WithAttr(std::move(device_func), tir::attr::kKernelLaunchParams,
                           dev_info.GetLaunchParams());

    device_func = WithAttr(std::move(device_func), tvm::attr::kCallingConv,
                           Integer(CallingConv::kDeviceKernelLaunch));
    device_func = WithAttr(std::move(device_func), tvm::attr::kGlobalSymbol,
                           runtime::String(kernel_symbol_global->name_hint));
    device_func = WithAttr(std::move(device_func), tir::attr::kNoAlias, Integer(1));
    device_func = WithAttr(std::move(device_func), tvm::attr::kTarget, device_target_);
    device_func = WithAttr(std::move(device_func), tir::attr::kIsGlobalFunc, Integer(1));

    (*device_mod_)->Add(kernel_symbol_global, device_func);

    // generate calls to the device function
    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(kernel_symbol_global->name_hint));
    for (PrimExpr arg : arguments) {
      call_args.push_back(arg);
    }
    for (PrimExpr ext : dev_info.thread_extent_) {
      call_args.push_back(ext);
    }
    if (dev_info.use_dyn_shmem_) {
      call_args.push_back(dev_info.dyn_shmem_size_);
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
    return ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.SplitHostDevice", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SplitHostDevice").set_body_typed(SplitHostDevice);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
