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
 * \file lower_device_kernel_launch.cc
 * \brief Split device function from host.
 */
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

namespace {
struct KernelInfo {
  // The device on which the PrimFunc runs
  Target target;

  // The externally visible symbol which may refer to the PrimFunc
  // when launching a device kernel.
  String global_symbol;

  // The parameters accepted by the PrimFunc.  Used to rewrite
  // `launch_args` to be in terms of the calling scope.
  Array<Var> params;

  // The launch parameters that should annotate the PrimFunc, if the
  // kernel is ever called from the host.
  Array<String> launch_params;

  // Additional arguments which must be provided to the host-side
  // PackedFunc.  These may be in terms of the function's parameters
  // (e.g. a function that computes the average of `N` elements, and
  // which must be launched with `N` CUDA threads).
  Array<PrimExpr> launch_args;
};

/*!
 * \brief Visitor class to collect device-side program information.
 */
class DeviceInfoCollector : public StmtVisitor {
 public:
  static KernelInfo Collect(const GlobalVar& gvar, const PrimFunc& func) {
    DeviceInfoCollector collector;
    collector.info_.target = func->GetAttr<Target>(tvm::attr::kTarget).value().WithoutHost();
    collector.info_.params = func->params;

    collector(func->body);

    // The dynamic shared memory is required to be the last of the
    // kernel launch parameters
    if (collector.dyn_shmem_size) {
      collector.info_.launch_params.push_back(
          tvm::runtime::launch_param::kUseDynamicSharedMemoryTag);
    }

    collector.info_.global_symbol =
        func->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);

    collector.info_.launch_args = collector.info_.launch_params.Map(
        [&](const auto& param) { return collector.GetArgument(param); });

    return collector.info_;
  }

 private:
  PrimExpr GetArgument(const String& launch_param) const {
    if (launch_param == tvm::runtime::launch_param::kUseDynamicSharedMemoryTag) {
      CHECK(dyn_shmem_size.defined())
          << "Compute kernel requires launch parameter \"" << launch_param
          << "\", but PrimFunc did not contain Allocate node with shared dynamic scope.";
      return dyn_shmem_size.value();
    }

    auto extent = thread_extent.Get(launch_param);
    CHECK(extent) << "Compute kernel requires launch parameter \"" << launch_param
                  << "\", but PrimFunc does not contain AttrStmt \"" << attr::thread_extent
                  << "\" defining this thread extent";
    return extent.value();
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      // thread_extent can appear multiple times
      // use the first appearance as def.
      if (!defined_thread.count(iv.get())) {
        defined_thread.insert(iv.get());
        info_.launch_params.push_back(iv->thread_tag);
        thread_extent.Set(iv->thread_tag, op->value);
      }
    }

    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      ICHECK(!dyn_shmem_size.defined()) << "Only one dynamic shared memory allocation is allowed.";
      ICHECK_GT(op->extents.size(), 0);

      PrimExpr dyn_size = Integer(1);
      for (const auto& extent : op->extents) {
        dyn_size *= extent;
      }
      dyn_size *= op->dtype.bytes();

      dyn_shmem_size = dyn_size;
    }
    StmtVisitor::VisitStmt_(op);
  }

  // The collected results
  KernelInfo info_;
  // recording what thread axis have been visited.
  std::unordered_set<const IterVarNode*> defined_thread;
  // The extent of each thread
  Map<String, PrimExpr> thread_extent;
  // The amount of dynamic shared memory used
  Optional<PrimExpr> dyn_shmem_size{NullOpt};
};

class ReturnRemover : public StmtExprMutator {
 public:
  static Stmt Apply(const Stmt& stmt) {
    ReturnRemover mutator;
    return mutator(stmt);
  }

 private:
  using Parent = StmtExprMutator;
  Stmt VisitStmt_(const EvaluateNode* op) override {
    if (auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::ret())) {
        ICHECK_EQ(call->args.size(), 1);
        auto as_int = call->args[0].as<IntImmNode>();
        ICHECK(as_int && as_int->value == 0)
            << "Device kernel may only contain successful return, T.ret(0)";
        return Evaluate(0);
      }
    }
    return Parent::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::ret())) {
      LOG(FATAL) << "Call to builtin::ret() should only appear within an Evaluate node";
    }
    return Parent::VisitExpr_(op);
  }
};
}  // namespace

class DeviceKernelMutator : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  explicit DeviceKernelMutator(std::unordered_map<const GlobalVarNode*, KernelInfo> device_info_map)
      : device_info_map_(std::move(device_info_map)) {}

  PrimFunc RewriteKernelLaunchSite(const GlobalVar& gvar, PrimFunc func) {
    ICHECK(!current_target_.defined());
    auto it = device_info_map_.find(gvar.get());
    ICHECK(it != device_info_map_.end());
    current_target_ = it->second.target;

    auto body = VisitStmt(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = body;
    }

    current_target_ = NullOpt;
    return func;
  }

  PrimFunc UpdateKernelAttributes(const GlobalVar& gvar, PrimFunc func) const {
    bool is_kernel_launch = device_kernel_launch_.count(gvar.get());
    bool is_call_extern = extern_function_call_.count(gvar.get());
    CHECK(!is_kernel_launch || !is_call_extern)
        << "Function " << gvar << " has multiple callees, "
        << "and would need to be lowered into a call_extern at some call sites, "
        << "and a device kernel launch at others.  "
        << "This case is not yet supported.";

    if (is_kernel_launch || is_call_extern) {
      func = WithAttr(std::move(func), tvm::tir::attr::kIsGlobalFunc, Bool(true));
    }

    if (is_kernel_launch) {
      const auto& info = device_info_map_.at(gvar.get());

      // Kernel launches provide an int32 error code to the caller,
      // but do not accept any return type from the callee.
      {
        auto write_ptr = func.CopyOnWrite();
        write_ptr->ret_type = VoidType();
        write_ptr->body = ReturnRemover::Apply(write_ptr->body);
      }

      func = WithAttrs(std::move(func),
                       {{tvm::attr::kCallingConv, Integer(tvm::CallingConv::kDeviceKernelLaunch)},
                        {tvm::tir::attr::kKernelLaunchParams, info.launch_params},
                        {tvm::attr::kGlobalSymbol, info.global_symbol}});

    } else if (is_call_extern && !func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      func = WithAttr(func, tvm::attr::kGlobalSymbol, gvar->name_hint);
    }

    return func;
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(Parent::VisitExpr_(op));

    auto* gvar = op->op.as<GlobalVarNode>();
    if (!gvar) return std::move(node);

    auto it = device_info_map_.find(gvar);
    ICHECK(it != device_info_map_.end())
        << "CallNode attempted subroutine call to " << gvar->name_hint << ", but "
        << gvar->name_hint << " did not appear within the IRModule";
    const KernelInfo& dev_info = it->second;

    auto caller_target = current_target_.value();
    auto callee_target = dev_info.target;

    bool same_target = caller_target->str() == callee_target->str();
    if (same_target) {
      // Calls within the same target may be handled at codegen time
      // as internal subroutine calls.
      return std::move(node);
    }

    bool same_device_type =
        caller_target->GetTargetDeviceType() == callee_target->GetTargetDeviceType();
    if (same_device_type) {
      // Calls to another target using the same device (e.g. LLVM
      // calling a custom TIRToRuntime target) do not require a kernel
      // launch, but need to be replaced with call_extern.
      extern_function_call_.insert(gvar);
      Array<PrimExpr> args;
      args.push_back(StringImm(gvar->name_hint));
      for (const auto& arg : node->args) {
        args.push_back(arg);
      }
      return Call(node->dtype, builtin::call_extern(), args);
    }

    ICHECK(dev_info.launch_params.defined())
        << "CallNode attempted kernel launch to " << gvar->name_hint << " on target "
        << dev_info.target << ", but subroutine " << gvar->name_hint
        << " did not have the tir::attr::kKernelLaunchParams attribute "
        << "required for cross-target kernel launch";

    // Collected kernel information may be in terms of the callee's
    // arguments, but we need expressions for them in terms of the
    // caller's parameters.  The param_map allows substitution of
    // parameter values into the thread extents, to generate
    // expressions that are valid within the caller.
    Map<Var, PrimExpr> param_map = [&]() {
      Map<Var, PrimExpr> param_map;
      CHECK_EQ(node->args.size(), dev_info.params.size())
          << "Function " << gvar->name_hint << " accepts " << dev_info.params.size()
          << " arguments as input, but is called using " << node->args.size() << " arguments";
      for (size_t i = 0; i < node->args.size(); i++) {
        param_map.Set(dev_info.params[i], node->args[i]);
      }
      return param_map;
    }();

    device_kernel_launch_.insert(gvar);

    Array<PrimExpr> call_args;
    call_args.push_back(StringImm(dev_info.global_symbol));
    for (PrimExpr arg : node->args) {
      call_args.push_back(arg);
    }
    for (const auto& launch_arg : dev_info.launch_args) {
      call_args.push_back(Substitute(launch_arg, param_map));
    }

    auto dtype = node->dtype.is_void() ? DataType::Int(32) : node->dtype;

    return Call(dtype, builtin::tvm_call_packed(), call_args);
  }

  Optional<Target> current_target_;
  std::unordered_map<const GlobalVarNode*, KernelInfo> device_info_map_;
  std::unordered_set<const GlobalVarNode*> device_kernel_launch_;
  std::unordered_set<const GlobalVarNode*> extern_function_call_;
};

namespace transform {

Pass LowerDeviceKernelLaunch() {
  auto pass_func = [](IRModule mod, PassContext ctx) -> IRModule {
    auto mutator = [&mod]() {
      std::unordered_map<const GlobalVarNode*, KernelInfo> device_info_map;
      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto prim_func = base_func.as<PrimFunc>()) {
          device_info_map[gvar.get()] = DeviceInfoCollector::Collect(gvar, prim_func.value());
        }
      }
      return DeviceKernelMutator(std::move(device_info_map));
    }();

    {
      IRModule updates;
      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto* ptr = base_func.as<PrimFuncNode>()) {
          auto prim_func = mutator.RewriteKernelLaunchSite(gvar, GetRef<PrimFunc>(ptr));
          if (!prim_func.same_as(base_func)) {
            updates->Add(gvar, prim_func);
          }
        }
      }

      if (updates->functions.size()) {
        mod.CopyOnWrite()->Update(updates);
      }
    }

    {
      IRModule updates;
      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto* ptr = base_func.as<PrimFuncNode>()) {
          auto prim_func = mutator.UpdateKernelAttributes(gvar, GetRef<PrimFunc>(ptr));
          if (!prim_func.same_as(base_func)) {
            updates->Add(gvar, prim_func);
          }
        }
      }

      if (updates->functions.size()) {
        mod.CopyOnWrite()->Update(updates);
      }
    }

    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.LowerDeviceKernelLaunch", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerDeviceKernelLaunch")
    .set_body_typed(LowerDeviceKernelLaunch);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
