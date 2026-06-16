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
 * \brief Annotate and split device functions from host, then lower kernel launches.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>

#include "../../runtime/thread_storage_scope.h"
#include "../analysis/var_use_def_analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {

// Device-region annotation

class DeviceRegionAnnotater : public StmtMutator {
 public:
  explicit DeviceRegionAnnotater(Target device_target) : device_target_(device_target) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      // If a target attribute already exists, use it as-is.
      return ffi::GetRef<Stmt>(op);
    } else if (op->attr_key == attr::thread_extent || op->attr_key == attr::device_scope) {
      // These attributes are only allowed in device-side code, so
      // they should be annotated with the function's default target.
      Stmt body = ffi::GetRef<Stmt>(op);
      return AttrStmt(device_target_, tvm::attr::kTarget, 0, body);
    } else {
      // All other annotations are ignored.
      return StmtMutator::VisitStmt_(op);
    }
  }

 private:
  Target device_target_;
};

PrimFunc AnnotateDeviceRegionsForSplit(PrimFunc func) {
  auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
  TVM_FFI_ICHECK(opt_target) << "SplitHostDevice: Require the target attribute";
  Target target = opt_target.value();

  if (target->GetHost()) {
    DeviceRegionAnnotater mutator(target.WithoutHost());
    auto body = mutator(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = body;
    }
  }
  return func;
}

// Host/device function extraction

class LaunchBoundsAttrExtractor : public StmtMutator {
 public:
  Stmt Extract(Stmt stmt) {
    min_blocks_per_sm_.reset();
    return operator()(std::move(stmt));
  }

  std::optional<int64_t> min_blocks_per_sm() const { return min_blocks_per_sm_; }

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::kLaunchBoundsMinBlocksPerSM) {
      const auto* min_blocks_per_sm = op->value.as<IntImmNode>();
      TVM_FFI_ICHECK(min_blocks_per_sm)
          << tirx::attr::kLaunchBoundsMinBlocksPerSM << " expects an integer value";
      TVM_FFI_ICHECK_GT(min_blocks_per_sm->value, 0)
          << tirx::attr::kLaunchBoundsMinBlocksPerSM << " must be positive";
      if (min_blocks_per_sm_.has_value()) {
        TVM_FFI_ICHECK_EQ(min_blocks_per_sm_.value(), min_blocks_per_sm->value)
            << "Conflicting " << tirx::attr::kLaunchBoundsMinBlocksPerSM << " values";
      }
      min_blocks_per_sm_ = min_blocks_per_sm->value;
      return VisitStmt(op->body);
    }
    return StmtMutator::VisitStmt_(op);
  }

  std::optional<int64_t> min_blocks_per_sm_;
};

class HostDeviceSplitter : public StmtMutator {
 public:
  explicit HostDeviceSplitter(IRModule* device_mod, std::function<GlobalVar()> var_supply,
                              PrimFunc cur_func)
      : device_mod_(device_mod), var_supply_(var_supply), cur_func_(cur_func) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      auto device_target = op->node.as<Target>().value().WithoutHost();
      return SplitDeviceFunc(op->body, device_target);
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  Stmt SplitDeviceFunc(Stmt body, Target device_target) {
    auto [params, buffers_to_declare] = [&]() -> std::tuple<ffi::Array<Var>, ffi::Array<Buffer>> {
      VarUseDefAnalyzer use_def(/*defined_vars=*/{}, /*visit_thread_extent=*/true);
      use_def(body);

      // Sort first by variable type, then by variable name
      std::vector<Var> params{use_def.undefined_.begin(), use_def.undefined_.end()};
      if (device_target->kind->name != "trn") {
        std::sort(params.begin(), params.end(), [](const Var& a, const Var& b) {
          auto sort_key = [](const Var& var) {
            return std::tuple{
                !var->dtype.is_handle(),
                var->name_hint,
            };
          };
          return sort_key(a) < sort_key(b);
        });
      } else {
        std::unordered_map<Var, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> param_order;
        for (size_t i = 0; i < cur_func_->params.size(); ++i) {
          param_order[cur_func_->buffer_map[cur_func_->params[i]]->data] = i;
        }
        // sort by original order
        std::sort(params.begin(), params.end(),
                  [&](const Var& a, const Var& b) { return param_order[a] < param_order[b]; });
      }
      return {params, use_def.undefined_buffers_};
    }();

    // CodeGenCPU is used for some device-side targets, such as
    // "ext_dev", and expects to be able to return a int32_t status
    // code.

    bool can_propagate_errors = [&]() {
      auto kind = device_target->GetTargetDeviceType();
      return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
    }();
    IntImm success(DataType::Int(32), 0);
    Type kernel_ret_type;
    if (can_propagate_errors) {
      kernel_ret_type = PrimType(DataType::Int(32));
      body = SeqStmt::Flatten(body, Evaluate(ret(success)));
    } else {
      kernel_ret_type = VoidType();
    }

    for (Buffer buf : buffers_to_declare) {
      body = SeqStmt::Flatten(DeclBuffer(buf), std::move(body));
    }
    LaunchBoundsAttrExtractor launch_bounds_attr;
    body = launch_bounds_attr.Extract(std::move(body));
    PrimFunc device_func(params, body, kernel_ret_type);
    device_func = WithAttrs(std::move(device_func), {{tvm::attr::kTarget, device_target},
                                                     {tirx::attr::kNoAlias, true},
                                                     {tirx::attr::kIsGlobalFunc, true}});
    bool is_stir = cur_func_->attrs->dict.count(tvm::attr::kSTir);
    if (is_stir) {
      device_func = WithAttr(std::move(device_func), tvm::attr::kSTir, true);
    }
    if (device_target->kind->name == "cuda" && launch_bounds_attr.min_blocks_per_sm().has_value()) {
      device_func = WithAttr(std::move(device_func), tirx::attr::kLaunchBoundsMinBlocksPerSM,
                             launch_bounds_attr.min_blocks_per_sm().value());
    }
    auto num_inputs = cur_func_->GetAttr<int64_t>(tvm::attr::kNumInputs);
    if (num_inputs.has_value()) {
      device_func = WithAttr(std::move(device_func), tvm::attr::kNumInputs, num_inputs);
    }
    GlobalVar kernel_symbol_global = var_supply_();
    (*device_mod_)->Add(kernel_symbol_global, device_func);
    ffi::Array<PrimExpr> args = params.Map([](const Var& var) -> PrimExpr { return var; });

    if (can_propagate_errors) {
      Var kernel_error_code("kernel_error_code", success->dtype);
      Call kernel_call(success->dtype, kernel_symbol_global, args);
      AssertStmt assert_success(kernel_error_code == success, StringImm("RuntimeError"),
                                {StringImm("Error executing compute kernel")});
      return SeqStmt({Bind(kernel_error_code, kernel_call), assert_success});

    } else {
      return Evaluate(Call(DataType::Void(), kernel_symbol_global, args));
    }
  }

  // target ir module
  IRModule* device_mod_;
  // Generate new GlobalVar for the kernel
  std::function<GlobalVar()> var_supply_;
  // Current function being split
  PrimFunc cur_func_;
};

PrimFunc SplitHostDevice(PrimFunc func, IRModule* device_mod,
                         std::function<GlobalVar()> var_supply) {
  HostDeviceSplitter splitter(device_mod, var_supply, func);

  if (auto body = splitter(func->body); !body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  }

  return func;
}

// Device kernel launch lowering

namespace {

struct KernelInfo {
  // The device on which the PrimFunc runs.
  Target target;

  // The externally visible symbol which may refer to the PrimFunc
  // when launching a device kernel.
  ffi::String global_symbol;

  // The parameters accepted by the PrimFunc.  Used to rewrite
  // `launch_args` to be in terms of the calling scope.
  ffi::Array<Var> params;

  // The launch parameters that should annotate the PrimFunc, if the
  // kernel is ever called from the host.
  ffi::Array<ffi::String> launch_params;

  // Additional arguments which must be provided to the host-side
  // ffi::Function.  These may be in terms of the function's parameters
  // (e.g. a function that computes the average of `N` elements, and
  // which must be launched with `N` CUDA threads).
  ffi::Array<PrimExpr> launch_args;
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
    // kernel launch parameters.
    if (collector.dyn_shmem_size) {
      collector.info_.launch_params.push_back(
          tvm::runtime::launch_param::kUseDynamicSharedMemoryTag);
    }

    collector.info_.global_symbol =
        func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);

    collector.info_.launch_args = collector.info_.launch_params.Map(
        [&](const auto& param) { return collector.GetArgument(param); });

    return collector.info_;
  }

 private:
  PrimExpr GetArgument(const ffi::String& launch_param) const {
    if (launch_param == tvm::runtime::launch_param::kUseDynamicSharedMemoryTag) {
      TVM_FFI_ICHECK(dyn_shmem_size.defined())
          << "Compute kernel requires launch parameter \"" << launch_param
          << "\", but PrimFunc did not contain AllocBuffer node with shared dynamic scope.";
      return dyn_shmem_size.value();
    }

    auto extent = thread_extent.Get(launch_param);
    TVM_FFI_ICHECK(extent) << "Compute kernel requires launch parameter \"" << launch_param
                           << "\", but PrimFunc does not contain AttrStmt \"" << attr::thread_extent
                           << "\" defining this thread extent";
    return extent.value();
  }

  void VisitStmt_(const BindNode* op) final {
    // Track Bind definitions so that thread_extent values and
    // dyn_shmem_size expressions that reference locally-bound
    // variables (e.g. CSE variables) can be inlined back to
    // expressions over function parameters.  Substitute earlier
    // bindings into the value to handle chains (cse_v2 = f(cse_v1)).
    PrimExpr value = bind_map_.size() ? Substitute(op->value, bind_map_) : op->value;
    bind_map_.Set(op->var, value);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      ffi::String thread_tag;
      if (auto iv = op->node.as<IterVar>()) {
        thread_tag = iv.value()->thread_tag;
        TVM_FFI_ICHECK_NE(thread_tag.length(), 0U);
      } else if (auto var = op->node.as<Var>()) {
        thread_tag = var.value()->name_hint;
      } else {
        TVM_FFI_THROW(TypeError) << "thread_extent node must be an IterVar or Var, but was "
                                 << op->node.GetTypeKey();
      }
      // thread_extent can appear multiple times
      // use the first appearance as def.
      std::string thread_key = thread_tag;
      if (!defined_thread.count(thread_key)) {
        defined_thread.insert(thread_key);
        info_.launch_params.push_back(thread_tag);
        // Inline any locally-bound variables (e.g. from CSE) so
        // that the extent is expressible in terms of function params.
        PrimExpr value = bind_map_.size() ? Substitute(op->value, bind_map_) : op->value;
        thread_extent.Set(thread_tag, value);
      }
    }

    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode* op) final {
    auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      TVM_FFI_ICHECK(!dyn_shmem_size.defined())
          << "Only one dynamic shared memory allocation is allowed.";
      TVM_FFI_ICHECK_GT(op->buffer->shape.size(), 0);

      PrimExpr dyn_size = IntImm(DataType::Int(32), 1);
      for (const auto& extent : op->buffer->shape) {
        dyn_size *= extent;
      }
      dyn_size *= op->buffer->dtype.bytes();

      // Inline any locally-bound variables (e.g. from CSE).
      if (bind_map_.size()) {
        dyn_size = Substitute(dyn_size, bind_map_);
      }
      dyn_shmem_size = dyn_size;
    }
    StmtVisitor::VisitStmt_(op);
  }

  // The collected results.
  KernelInfo info_;
  // Recording what thread axis have been visited.
  std::unordered_set<std::string> defined_thread;
  // The extent of each thread.
  ffi::Map<ffi::String, PrimExpr> thread_extent;
  // The amount of dynamic shared memory used.
  ffi::Optional<PrimExpr> dyn_shmem_size{std::nullopt};
  // Accumulated Bind definitions for inlining into extent/size expressions.
  ffi::Map<Var, PrimExpr> bind_map_;
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
        TVM_FFI_ICHECK_EQ(call->args.size(), 1);
        auto as_int = call->args[0].as<IntImmNode>();
        TVM_FFI_ICHECK(as_int && as_int->value == 0)
            << "Device kernel may only contain successful return, T.ret(0)";
        return Evaluate(0);
      }
    }
    return Parent::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::ret())) {
      TVM_FFI_THROW(InternalError)
          << "Call to builtin::ret() should only appear within an Evaluate node";
    }
    return Parent::VisitExpr_(op);
  }
};

class GlobalVarCallCollector : public StmtExprVisitor {
 public:
  static std::unordered_set<const GlobalVarNode*> Collect(const IRModule& mod) {
    GlobalVarCallCollector collector;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto prim_func = base_func.as<PrimFunc>()) {
        collector(prim_func.value()->body);
      }
    }
    return collector.called_gvars_;
  }

 private:
  using Parent = StmtExprVisitor;

  void VisitExpr_(const CallNode* op) final {
    if (auto* gvar = op->op.as<GlobalVarNode>()) {
      called_gvars_.insert(gvar);
    }
    Parent::VisitExpr_(op);
  }

  std::unordered_set<const GlobalVarNode*> called_gvars_;
};

}  // namespace

class DeviceKernelMutator : public StmtExprMutator {
 public:
  using Parent = StmtExprMutator;

  explicit DeviceKernelMutator(std::unordered_map<const GlobalVarNode*, KernelInfo> device_info_map)
      : device_info_map_(std::move(device_info_map)) {}

  PrimFunc RewriteKernelLaunchSite(const GlobalVar& gvar, PrimFunc func) {
    TVM_FFI_ICHECK(!current_target_.defined());
    // Track whether the caller is a host function (i.e. its target
    // still has a host attached) and capture its host target.  The
    // same-target shortcut at the call site is only safe when caller
    // and callee are both device-resident; a host caller must take
    // the kernel-launch path even if Target::WithoutHost() makes the
    // strings match.  Conversely, a host caller invoking another host
    // helper (e.g. a same-target subroutine that SplitHostDevice
    // emitted on the host side) should compare against the host
    // target, not the device target stripped by WithoutHost().
    auto full_target = func->GetAttr<Target>(tvm::attr::kTarget).value();
    current_target_ = full_target.WithoutHost();
    if (full_target->GetHost().defined()) {
      current_caller_host_target_ = full_target->GetHost().value();
    } else {
      current_caller_host_target_ = std::nullopt;
    }

    auto body = VisitStmt(func->body);
    if (!body.same_as(func->body)) {
      func.CopyOnWrite()->body = body;
    }

    current_target_ = std::nullopt;
    current_caller_host_target_ = std::nullopt;
    return func;
  }

  PrimFunc UpdateKernelAttributes(const GlobalVar& gvar, PrimFunc func) const {
    bool is_kernel_launch = device_kernel_launch_.count(gvar.get());
    bool is_call_extern = extern_function_call_.count(gvar.get());
    TVM_FFI_ICHECK(!is_kernel_launch || !is_call_extern)
        << "Function " << gvar << " has multiple callees, "
        << "and would need to be lowered into a call_extern at some call sites, "
        << "and a device kernel launch at others.  "
        << "This case is not yet supported.";

    if (is_kernel_launch || is_call_extern) {
      func = WithAttr(std::move(func), tvm::tirx::attr::kIsGlobalFunc, true);
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
                       {{tvm::attr::kCallingConv, tvm::CallingConv::kDeviceKernelLaunch},
                        {tvm::tirx::attr::kKernelLaunchParams, info.launch_params},
                        {tvm::attr::kGlobalSymbol, info.global_symbol}});

    } else if (is_call_extern && !func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      func = WithAttr(func, tvm::attr::kGlobalSymbol, gvar->name_hint);
    }

    return func;
  }

 private:
  PrimExpr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(Parent::VisitExpr_(op));

    auto* gvar = op->op.as<GlobalVarNode>();
    if (!gvar) return node;

    auto it = device_info_map_.find(gvar);
    TVM_FFI_ICHECK(it != device_info_map_.end())
        << "CallNode attempted subroutine call to " << gvar->name_hint << ", but "
        << gvar->name_hint << " did not appear within the IRModule";
    const KernelInfo& dev_info = it->second;

    auto callee_target = dev_info.target;

    // A callee with non-empty launch_params has thread_extent
    // bindings in its body, i.e. it is a real device kernel that
    // must be invoked via a kernel-launch ABI.  Conversely a callee
    // with empty launch_params is a plain subroutine (host helper
    // or intra-device helper) and is never invoked via kernel launch.
    bool callee_is_kernel = dev_info.launch_params.size() > 0;
    bool caller_is_host = current_caller_host_target_.has_value();

    // For host callers, comparisons against the callee target must
    // use the caller's *host* target, not the device target stripped
    // by WithoutHost().  This handles two cases that the device-side
    // comparison gets wrong:
    //   1. A host caller invoking a real device kernel whose
    //      WithoutHost() target happens to match (e.g. kernel target
    //      "cuda" matches "cuda+host=c" after stripping host).  Must
    //      go through kernel launch, not the same-target shortcut.
    //   2. A host caller invoking another host helper with a
    //      different host target (e.g. SplitHostDevice emits an
    //      "add_host" with target "c" while the host body still
    //      carries "cuda+host=c").  Must go through call_extern (or
    //      same-target subroutine), not kernel launch.
    auto caller_target =
        caller_is_host ? current_caller_host_target_.value() : current_target_.value();

    // A host caller invoking a real device kernel must always go
    // through the kernel-launch ABI, regardless of any same-target /
    // same-device-type coincidence.
    bool force_kernel_launch = callee_is_kernel && caller_is_host;

    if (!force_kernel_launch) {
      bool same_target = caller_target->str() == callee_target->str();
      if (same_target) {
        // Calls within the same target may be handled at codegen time
        // as internal subroutine calls.
        return node;
      }

      bool same_device_type =
          caller_target->GetTargetDeviceType() == callee_target->GetTargetDeviceType();
      if (same_device_type) {
        // Calls to another target using the same device (e.g. LLVM
        // calling a custom TIRToRuntime target) do not require a kernel
        // launch, but need to be replaced with call_extern.
        extern_function_call_.insert(gvar);
        ffi::Array<PrimExpr> args;
        args.push_back(StringImm(gvar->name_hint));
        for (const auto& arg : node->args) {
          args.push_back(arg);
        }
        return Call(node->dtype, builtin::call_extern(), args);
      }
    }

    TVM_FFI_ICHECK(dev_info.launch_params.defined())
        << "CallNode attempted kernel launch to " << gvar->name_hint << " on target "
        << dev_info.target << ", but subroutine " << gvar->name_hint
        << " did not have the tirx::attr::kKernelLaunchParams attribute "
        << "required for cross-target kernel launch";

    // Collected kernel information may be in terms of the callee's
    // arguments, but we need expressions for them in terms of the
    // caller's parameters.  The param_map allows substitution of
    // parameter values into the thread extents, to generate
    // expressions that are valid within the caller.
    ffi::Map<Var, PrimExpr> param_map = [&]() {
      ffi::Map<Var, PrimExpr> param_map;
      TVM_FFI_ICHECK_EQ(node->args.size(), dev_info.params.size())
          << "Function " << gvar->name_hint << " accepts " << dev_info.params.size()
          << " arguments as input, but is called using " << node->args.size() << " arguments";
      for (size_t i = 0; i < node->args.size(); i++) {
        param_map.Set(dev_info.params[i], node->args[i]);
      }
      return param_map;
    }();

    device_kernel_launch_.insert(gvar);

    ffi::Array<PrimExpr> call_args;
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

  ffi::Optional<Target> current_target_;
  // The host target of the caller currently being rewritten, if the
  // caller is a host function (its kTarget has a host attached).
  // Used both to detect that the caller is a host function and to
  // compare against the callee target on the host side, so that
  // host-to-host subroutine calls are not misrouted through the
  // device kernel-launch ABI.
  ffi::Optional<Target> current_caller_host_target_;
  std::unordered_map<const GlobalVarNode*, KernelInfo> device_info_map_;
  std::unordered_set<const GlobalVarNode*> device_kernel_launch_;
  std::unordered_set<const GlobalVarNode*> extern_function_call_;
};

IRModule LowerDeviceKernelLaunches(IRModule mod) {
  auto mutator = [&mod]() {
    std::unordered_set<const GlobalVarNode*> called_gvars = GlobalVarCallCollector::Collect(mod);
    std::unordered_map<const GlobalVarNode*, KernelInfo> device_info_map;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (called_gvars.count(gvar.get())) {
        if (auto prim_func = base_func.as<PrimFunc>()) {
          device_info_map[gvar.get()] = DeviceInfoCollector::Collect(gvar, prim_func.value());
        }
      }
    }
    return DeviceKernelMutator(std::move(device_info_map));
  }();

  {
    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* ptr = base_func.as<PrimFuncNode>()) {
        auto prim_func = mutator.RewriteKernelLaunchSite(gvar, ffi::GetRef<PrimFunc>(ptr));
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
        auto prim_func = mutator.UpdateKernelAttributes(gvar, ffi::GetRef<PrimFunc>(ptr));
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
}

namespace transform {

Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    UniqueNameSupply global_names(mod->functions.begin(), mod->functions.end(),
                                  [](const auto& kv) { return kv.first->name_hint; });

    IRModule device_mod = IRModule(ffi::Map<GlobalVar, BaseFunc>({}));
    IRModule updates = IRModule(ffi::Map<GlobalVar, BaseFunc>({}));

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        PrimFunc func = opt.value();
        func = AnnotateDeviceRegionsForSplit(std::move(func));

        auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
        auto name_prefix = global_symbol.value_or(gvar->name_hint);
        auto kernel_name = name_prefix + "_kernel";
        auto var_supply = [&global_names, &kernel_name]() -> GlobalVar {
          return GlobalVar(global_names->FreshName(kernel_name, false));
        };

        func = SplitHostDevice(std::move(func), &device_mod, var_supply);
        if (!func.same_as(base_func)) {
          updates->Add(gvar, func);
        }
      }
    }

    mod->Update(updates);
    mod->Update(device_mod);
    mod = ConvertSSA()(mod);
    return LowerDeviceKernelLaunches(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tirx.SplitHostDevice", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.SplitHostDevice", SplitHostDevice);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
