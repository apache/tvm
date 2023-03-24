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
struct DeviceInfo {
  Target target;
  Array<Var> params;
  Optional<Array<String>> launch_params;
  Map<String, PrimExpr> thread_extent;
  Optional<PrimExpr> dyn_shmem_size{NullOpt};

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
};

/*!
 * \brief Visitor class to collect device-side program information.
 */
class DeviceInfoCollector : public StmtVisitor {
 public:
  static DeviceInfo Collect(const PrimFuncNode* func) {
    DeviceInfoCollector collector;
    collector.info_.target = [&]() -> Target {
      auto target_attr = func->GetAttr<Target>(tvm::attr::kTarget).value();
      bool is_host_func =
          func->GetAttr<Bool>(tvm::tir::attr::kIsHostFunc).value_or(Bool(false))->value;
      if (is_host_func) {
        return target_attr->GetHost().value();
      } else {
        return target_attr.WithoutHost();
      }
    }();
    collector.info_.params = func->params;
    collector.info_.launch_params = func->GetAttr<Array<String>>(tir::attr::kKernelLaunchParams);
    collector(func->body);
    return collector.info_;
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
        info_.thread_extent.Set(iv->thread_tag, op->value);
      }
    }

    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      ICHECK(!info_.dyn_shmem_size.defined())
          << "Only one dynamic shared memory allocation is allowed.";
      ICHECK_GT(op->extents.size(), 0);

      PrimExpr dyn_size = Integer(1);
      for (const auto& extent : op->extents) {
        dyn_size *= extent;
      }
      dyn_size *= op->dtype.bytes();

      info_.dyn_shmem_size = dyn_size;
    }
    StmtVisitor::VisitStmt_(op);
  }

  DeviceInfo info_;
  // recording what thread axis have been visited.
  std::unordered_set<const IterVarNode*> defined_thread;
};
}  // namespace

class DeviceKernelMutator : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  explicit DeviceKernelMutator(std::unordered_map<const GlobalVarNode*, DeviceInfo> device_info_map)
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
    if (device_kernel_launch_.count(gvar.get())) {
      Map<String, ObjectRef> new_attrs;
      new_attrs.Set(tvm::attr::kCallingConv, Integer(tvm::CallingConv::kDeviceKernelLaunch));
      new_attrs.Set(tvm::tir::attr::kIsGlobalFunc, Bool(true));

      if (!func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
        new_attrs.Set(tvm::attr::kGlobalSymbol, gvar->name_hint);
      }

      func = WithAttrs(std::move(func), new_attrs);
    }

    return func;
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) {
    auto node = Downcast<Call>(Parent::VisitExpr_(op));

    auto* gvar = op->op.as<GlobalVarNode>();
    if (!gvar) return std::move(node);

    auto it = device_info_map_.find(gvar);
    ICHECK(it != device_info_map_.end())
        << "CallNode attempted subroutine call to " << gvar->name_hint << ", but "
        << gvar->name_hint << " did not appear within the IRModule";
    const DeviceInfo& dev_info = it->second;

    auto caller_device_type = current_target_.value()->GetTargetDeviceType();
    auto callee_device_type = dev_info.target->GetTargetDeviceType();
    if (caller_device_type == callee_device_type) {
      return std::move(node);
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
    call_args.push_back(StringImm(gvar->name_hint));
    for (PrimExpr arg : node->args) {
      call_args.push_back(arg);
    }
    for (const auto& launch_param : dev_info.launch_params.value()) {
      call_args.push_back(Substitute(dev_info.GetArgument(launch_param), param_map));
    }

    auto dtype = node->dtype.is_void() ? DataType::Int(32) : node->dtype;

    return Call(dtype, builtin::tvm_call_packed(), call_args);
  }

  Optional<Target> current_target_;
  std::unordered_map<const GlobalVarNode*, DeviceInfo> device_info_map_;
  std::unordered_set<const GlobalVarNode*> device_kernel_launch_;
};

namespace transform {

Pass LowerDeviceKernelLaunch() {
  auto pass_func = [](IRModule mod, PassContext ctx) -> IRModule {
    auto mutator = [&mod]() {
      std::unordered_map<const GlobalVarNode*, DeviceInfo> device_info_map;
      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto* prim_func = base_func.as<PrimFuncNode>()) {
          device_info_map[gvar.get()] = DeviceInfoCollector::Collect(prim_func);
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
