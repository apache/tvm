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
#include "ir_utils.h"

namespace tvm {
namespace tir {

// use/def analysis, also delete unreferenced lets
class VarUseDefAnalysis : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      // thread_extent can appear multiple times
      // use the first appearance as def.
      if (!use_count_.count(iv->var.get())) {
        this->HandleDef(iv->var.get());
        thread_axis_.push_back(iv);
        thread_extent_.push_back(op->value);
      }

      PrimExpr value = op->value;
      if (visit_thread_extent_) {
        value = this->VisitExpr(value);
      }
      Stmt body = this->VisitStmt(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      }
      return AttrStmt(op->node, op->attr_key, value, body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    this->HandleDef(op->var.get());
    Stmt body = this->VisitStmt(op->body);
    // eliminate unreferenced let
    if (use_count_.at(op->var.get()) == 0 && SideEffect(op->value) <= CallEffectKind::kReadState &&
        simplify_let_) {
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

  Stmt VisitStmt_(const ForNode* op) final {
    this->HandleDef(op->loop_var.get());
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    this->HandleDef(op->buffer_var.get());
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
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    this->HandleUse(op->buffer_var);
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    // Weaker SSA condition
    // A single var can be binded in multiple lets
    // but they have to bind to the same value.
    // This is used to allow cases when we reuse a single let
    // expression to construct a nested expr.
    // (let x = 1 in x + 1) * (let x = 1 in x + 1)
    auto it = let_binding_.find(op->var);
    PrimExpr value = this->VisitExpr(op->value);
    if (it != let_binding_.end()) {
      ICHECK(deep_equal_(it->second->value, value))
          << "Let cannot bind the same var to two different values";
      return GetRef<PrimExpr>(it->second);
    } else {
      this->HandleDef(op->var.get());
      let_binding_[op->var] = op;
    }
    PrimExpr body = this->VisitExpr(op->body);
    // eliminate unreferenced let
    if (use_count_.at(op->var.get()) == 0 && SideEffect(op->value) <= CallEffectKind::kReadState &&
        simplify_let_) {
      return body;
    } else {
      if (body.same_as(op->body) && value.same_as(op->value)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Let(op->var, value, body);
      }
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    this->HandleUse(GetRef<PrimExpr>(op));
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const ReduceNode* op) final {
    for (const auto& iv : op->axis) {
      this->HandleDef(iv->var.get());
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    this->HandleUse(op->buffer_var);
    return StmtExprMutator::VisitExpr_(op);
  }

  void HandleDef(const VarNode* v) {
    ICHECK(!def_count_.count(v)) << "variable " << v->name_hint
                                 << " has already been defined, the Stmt is not SSA";
    ICHECK(!use_count_.count(v)) << "variable " << v->name_hint
                                 << " has been used before definition!";
    use_count_[v] = 0;
    def_count_[v] = 1;
  }

  void HandleUse(const PrimExpr& v) {
    ICHECK(v.as<VarNode>());
    Var var = Downcast<Var>(v);
    auto it = use_count_.find(var.get());
    if (it != use_count_.end()) {
      if (it->second >= 0) {
        ++it->second;
      }
    } else {
      undefined_.push_back(var);
      use_count_[var.get()] = -1;
    }
  }

  // The fields are publically readible to
  // be accessible to the users.
  bool visit_thread_extent_{true};
  bool simplify_let_{true};
  Array<Var> undefined_;
  Array<IterVar> thread_axis_;
  Array<PrimExpr> thread_extent_;
  PrimExpr dyn_shmem_size_{0};
  bool use_dyn_shmem_{false};
  std::unordered_map<const VarNode*, int> use_count_;
  std::unordered_map<const VarNode*, int> def_count_;

 private:
  ExprDeepEqual deep_equal_;
  std::unordered_map<Var, const LetNode*, ObjectPtrHash, ObjectPtrEqual> let_binding_;
};

Array<Var> UndefinedVars(const Stmt& stmt, const Array<Var>& args) {
  VarUseDefAnalysis m;
  m.simplify_let_ = false;
  for (Var arg : args) {
    m.use_count_[arg.get()] = 0;
  }
  m(stmt);
  return m.undefined_;
}

Array<Var> UndefinedVars(const PrimExpr& expr) {
  VarUseDefAnalysis m;
  m.simplify_let_ = false;
  m(expr);
  return m.undefined_;
}

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
    VarUseDefAnalysis m;
    m.visit_thread_extent_ = false;
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
          tir::Var new_var(var->name_hint, PointerType(PrimType((*it).second->dtype)));
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
    PrimFunc device_func(params, Substitute(body, remap_vars));
    device_func = WithAttr(std::move(device_func), tir::attr::kDeviceThreadAxis, m.thread_axis_);
    device_func = WithAttr(std::move(device_func), tvm::attr::kCallingConv,
                           Integer(CallingConv::kDeviceKernelLaunch));
    device_func =
        WithAttr(std::move(device_func), tvm::attr::kGlobalSymbol, runtime::String(kernel_symbol));
    device_func = WithAttr(std::move(device_func), tir::attr::kNoAlias, Integer(1));
    device_func = WithAttr(std::move(device_func), tvm::attr::kTarget, device_target_);
    device_func = WithAttr(std::move(device_func), tir::attr::kIsGlobalFunc, Integer(1));
    if (m.use_dyn_shmem_) {
      device_func =
          WithAttr(std::move(device_func), tir::attr::kDeviceUseDynSharedMemory, Integer(1));
    }
    (*device_mod_)->Add(GlobalVar(kernel_symbol), device_func);

    // generate calls to the device function
    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(kernel_symbol));
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
