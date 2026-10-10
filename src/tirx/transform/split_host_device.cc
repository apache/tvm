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
#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/ir/transform.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {

namespace {
static ffi::Array<Var> RegionNoBodyParams(const CallNode*) { return {}; }

void ValidateDeviceScopeRegion(const RegionStmtNode* region) {
  TVM_FFI_CHECK(region->result_vars.empty(), ValueError)
      << region->op->name << " expects no results";
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.device_scope", "Internal host/device splitting boundary.")
      .signature(sig::var_args<Expr>("launch_values"), sig::call_attrs<DictAttrsNode>())
      .set_attr<FRegionGetBodyParams>(tvm::op_attr::kRegionGetBodyParams,
                                      FRegionGetBodyParams::FromNative<&RegionNoBodyParams>())
      .set_attr<FRegionValidate>(tvm::op_attr::kRegionValidate,
                                 FRegionValidate::FromNative<&ValidateDeviceScopeRegion>())
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"));
}

// Device-region annotation

class DeviceRegionAnnotater : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) override {
    if (input.as<ExprNode>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(input, inplace_mode);
  }
  explicit DeviceRegionAnnotater(Target device_target) : device_target_(device_target) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    static const Op device_scope = Op::Get("tirx.device_scope");
    if (op->op.same_as(device_scope)) {
      if (op->attrs->dict.count(tvm::attr::kTarget)) return ffi::Unchanged();
      auto attrs = op->attrs->dict;
      attrs.Set(tvm::attr::kTarget, device_target_);
      return RegionStmt(op->op, op->args, op->body_params, DictAttrs(attrs), op->body,
                        op->result_vars, op->loc);
    }
    if (op->op.same_as(tirx::launch_thread_op())) {
      return RegionStmt(device_scope, {}, {}, DictAttrs({{tvm::attr::kTarget, device_target_}}),
                        ffi::GetRef<Stmt>(op));
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

 private:
  Target device_target_;
};

Function AnnotateDeviceRegionsForSplit(Function func) {
  auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
  TVM_FFI_ICHECK(opt_target) << "SplitHostDevice: Require the target attribute";
  Target target = opt_target.value();

  if (target->GetHost()) {
    auto mutator = ffi::make_object<DeviceRegionAnnotater>(target.WithoutHost());
    auto body_result =
        mutator->Mutate(func->body, func.unique() ? InplaceMode::kAllow : InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);
    if (!body_unchanged) {
      func.CopyOnWrite()->body = body;
    }
  }
  return func;
}

// Launch operands belong to the host region, independently of device captures.
Stmt MakeCudaKernelLaunch(const GlobalVar& symbol, Function* func, ffi::Array<Expr> args,
                          const RegionStmtNode* region);

// Host/device function extraction

class HostDeviceSplitter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) override {
    if (input.as<ExprNode>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(input, inplace_mode);
  }
  explicit HostDeviceSplitter(IRModule* device_mod, std::function<GlobalVar()> var_supply,
                              Function cur_func)
      : device_mod_(device_mod), var_supply_(var_supply), cur_func_(cur_func) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    static const Op device_scope = Op::Get("tirx.device_scope");
    if (op->op.same_as(device_scope)) {
      auto target = op->attrs->dict.Get(tvm::attr::kTarget);
      if (!target) return Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
      Target device_target = target.value().as_or_throw<Target>();
      return SplitDeviceFunc(op->body, device_target.WithoutHost(), op);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

 private:
  class KernelBodyRewriter : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate_;
    UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
      static const Op device_scope = Op::Get("tirx.device_scope");
      if (op->op.same_as(device_scope)) {
        return Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
      }
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  };

  Stmt SplitDeviceFunc(Stmt body, Target device_target, const RegionStmtNode* region) {
    auto [params,
          buffers_to_declare] = [&]() -> std::tuple<ffi::Array<Var>, ffi::Array<TensorVar>> {
      ffi::Array<Var> undefined = UndefinedVars(body);
      ffi::Array<TensorVar> buffers;
      for (const Var& var : undefined) {
        if (auto buffer = var.as<TensorVar>()) buffers.push_back(buffer.value());
      }

      // Sort first by variable type, then by variable name
      std::vector<Var> params{undefined.begin(), undefined.end()};
      if (device_target->kind->name != "trn") {
        std::sort(params.begin(), params.end(), [](const Var& a, const Var& b) {
          auto sort_key = [](const Var& var) {
            bool is_handle =
                var->ty.as<PointerTypeNode>() != nullptr || var->ty.as<TensorTypeNode>() != nullptr;
            return std::tuple{
                !is_handle,
                var->name,
            };
          };
          return sort_key(a) < sort_key(b);
        });
      } else {
        std::unordered_map<Var, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> param_order;
        for (size_t i = 0; i < cur_func_->params.size(); ++i) {
          param_order[cur_func_->params[i].as_or_throw<tvm::tirx::TensorVar>().var()] = i;
        }
        // sort by original order
        std::sort(params.begin(), params.end(),
                  [&](const Var& a, const Var& b) { return param_order[a] < param_order[b]; });
      }
      return {params, buffers};
    }();

    // Buffer Vars are compiler-side values, not ABI values.  Thread their
    // physical pointer projection through the kernel call and recover the
    // typed buffer at the kernel entry with an explicit DeclTensor source.
    ffi::Array<Var> kernel_params;
    ffi::Array<Expr> call_args;
    ffi::Map<Var, Var> buffer_data_params;
    auto kernel_rewriter = ffi::make_object<KernelBodyRewriter>();
    for (const Var& param : params) {
      if (param->ty.as<TensorTypeNode>()) {
        TensorVar buffer = param.as_or_throw<TensorVar>();
        TensorVar kernel_buffer(buffer.name(), buffer.type(), buffer.loc());
        Var data_param(buffer.name() + "_ptr", buffer.type()->DataPointerType());
        kernel_params.push_back(data_param);
        call_args.push_back(buffer.data());
        buffer_data_params.Set(param, data_param);
        kernel_rewriter->VarRemapSet(param, kernel_buffer);
      } else {
        kernel_params.push_back(param);
        call_args.push_back(param);
      }
    }
    body = kernel_rewriter->Mutate(body).ValueOrUnchanged(body);

    // CodeGenCPU is used for some device-side targets, such as
    // "ext_dev", and expects to be able to return a int32_t status
    // code.

    bool can_propagate_errors = [&]() {
      auto kind = device_target->GetTargetDeviceType();
      return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
    }();
    IntImm success(PrimType::Int(32), 0);
    Type kernel_ret_type = Type::Missing();
    if (can_propagate_errors) {
      kernel_ret_type = PrimType::Int(32);
      body = SeqStmt({body, Return(success)});
    } else {
      kernel_ret_type = VoidType();
    }

    for (TensorVar buf : buffers_to_declare) {
      auto data_param = buffer_data_params.Get(buf.var());
      auto kernel_buffer = kernel_rewriter->VarRemapGet(buf);
      TVM_FFI_ICHECK(data_param.has_value())
          << "Undefined buffer " << buf.name() << " was not captured as a kernel parameter";
      TVM_FFI_ICHECK(kernel_buffer != nullptr);
      body = SeqStmt(
          {Bind(kernel_buffer.as_or_throw<TensorVar>(),
                Call(kernel_buffer.as_or_throw<TensorVar>().type(), decl_tensor_op(),
                     {data_param.value(), tvm::Tuple(kernel_buffer.as_or_throw<TensorVar>()->shape),
                      DataTypeImm(kernel_buffer.as_or_throw<TensorVar>()->dtype->dtype),
                      StringImm(kernel_buffer.as_or_throw<TensorVar>().scope())},
                     {})),
           std::move(body)});
    }
    Function device_func(kernel_params, SeqStmt(body), kernel_ret_type);
    device_func = WithAttrs(std::move(device_func), {{tvm::attr::kTarget, device_target},
                                                     {tvm::tirx::attr::kNoAlias, true},
                                                     {tvm::tirx::attr::kIsGlobalFunc, true}});
    bool is_stir = cur_func_->attrs->dict.count(tvm::attr::kSTir);
    if (is_stir) {
      device_func = WithAttr(std::move(device_func), tvm::attr::kSTir, true);
    }
    if (auto launch_params =
            cur_func_->GetAttr<ffi::Array<ffi::String>>(tvm::tirx::attr::kKernelLaunchParams)) {
      device_func = WithAttr(std::move(device_func), tvm::tirx::attr::kKernelLaunchParams,
                             launch_params.value());
    }
    auto num_inputs = cur_func_->GetAttr<int64_t>(tvm::attr::kNumInputs);
    if (num_inputs.has_value()) {
      device_func = WithAttr(std::move(device_func), tvm::attr::kNumInputs, num_inputs);
    }
    GlobalVar kernel_symbol_global = var_supply_();
    if (region->attrs->dict.count(tvm::backend::cuda::attr::kLaunchFields)) {
      Stmt launch = MakeCudaKernelLaunch(kernel_symbol_global, &device_func, call_args, region);
      (*device_mod_)->Add(kernel_symbol_global, device_func);
      return launch;
    }
    (*device_mod_)->Add(kernel_symbol_global, device_func);
    if (can_propagate_errors) {
      Var kernel_error_code("kernel_error_code", success.ty());
      Call kernel_call(success.ty(), kernel_symbol_global, call_args);
      AssertStmt assert_success(kernel_error_code.as_or_throw<PrimExpr>() == success,
                                StringImm("RuntimeError"),
                                {StringImm("Error executing compute kernel")});
      return SeqStmt(ffi::Array<Stmt>{Bind(kernel_error_code, kernel_call.as_or_throw<PrimExpr>()),
                                      assert_success});

    } else {
      return Evaluate(Call(kernel_ret_type, kernel_symbol_global, call_args));
    }
  }

  // target ir module
  IRModule* device_mod_;
  // Generate new GlobalVar for the kernel
  std::function<GlobalVar()> var_supply_;
  // Current function being split
  Function cur_func_;
};

Function SplitHostDevice(Function func, IRModule* device_mod,
                         std::function<GlobalVar()> var_supply) {
  auto splitter = ffi::make_object<HostDeviceSplitter>(device_mod, var_supply, func);

  auto body_result =
      splitter->Mutate(func->body, func.unique() ? InplaceMode::kAllow : InplaceMode::kDisallow);
  if (!body_result.UnchangedOrSameAs(func->body)) {
    func.CopyOnWrite()->body = std::move(body_result).ValueUnchecked();
  }

  return func;
}

// Device kernel launch lowering

namespace {

struct KernelInfo {
  explicit KernelInfo(Target target) : target(std::move(target)) {}
  // The device on which the Function runs.
  Target target;

  // The externally visible symbol which may refer to the Function
  // when launching a device kernel.
  ffi::String global_symbol;

  // The parameters accepted by the Function.  Used to rewrite
  // `launch_args` to be in terms of the calling scope.
  ffi::Array<Var> params;

  // The launch parameters that should annotate the Function, if the
  // kernel is ever called from the host.
  ffi::Array<ffi::String> launch_params;

  // Additional arguments which must be provided to the host-side
  // ffi::Function.  These may be in terms of the function's parameters
  // (e.g. a function that computes the average of `N` elements, and
  // which must be launched with `N` CUDA threads).
  ffi::Array<PrimExpr> launch_args;
  ffi::Optional<PrimExpr> dynamic_smem_requirement;
};

/*!
 * \brief Visitor class to collect device-side program information.
 */
class DeviceInfoCollector : public StmtExprVisitor {
 public:
  explicit DeviceInfoCollector(Target target) : info_(std::move(target)) {}

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprNode>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }
  static KernelInfo Collect(const GlobalVar& gvar, const Function& func,
                            bool allow_placeholder = false) {
    auto collector = ffi::make_object<DeviceInfoCollector>(
        func->GetAttr<Target>(tvm::attr::kTarget).value().WithoutHost());
    collector->info_.params = func->params;
    if (func->GetAttr<ffi::Array<ffi::String>>(tvm::backend::cuda::attr::kLaunchFields)) {
      collector->info_.global_symbol =
          func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);
      return collector->info_;
    }

    if (auto requested =
            func->GetAttr<ffi::Array<ffi::String>>(tvm::tirx::attr::kKernelLaunchParams)) {
      for (const ffi::String& tag : requested.value()) {
        if (tag == tvm::runtime::launch_param::kUseProgramaticDependentLaunch) {
          collector->use_programmatic_dependent_launch_ = true;
        } else if (tag == tvm::runtime::launch_param::kUseCooperativeLaunch) {
          collector->use_cooperative_launch_ = true;
        }
      }
    }

    collector->Visit(func->body);

    if (collector->use_programmatic_dependent_launch_) {
      collector->info_.launch_params.push_back(
          tvm::runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (collector->use_cooperative_launch_) {
      collector->info_.launch_params.push_back(tvm::runtime::launch_param::kUseCooperativeLaunch);
    }
    // Dynamic shared memory remains the final legacy launch operand.
    if (!collector->dyn_shmem_size.has_value() && collector->inferred_shmem_size_.has_value()) {
      const auto* inferred = collector->inferred_shmem_size_.value().as<IntImmNode>();
      TVM_FFI_ICHECK(allow_placeholder || !(inferred && inferred->value == 0))
          << "Function " << gvar->name_hint
          << " allocates dynamic shared memory with a placeholder extent but does not declare "
             "its size; use LaunchConfig.dynamic_smem_bytes or SMEMPool.commit().";
      collector->dyn_shmem_size = collector->inferred_shmem_size_;
    }
    if (collector->dyn_shmem_size) {
      collector->info_.launch_params.push_back(
          tvm::runtime::launch_param::kUseDynamicSharedMemoryTag);
    }

    collector->info_.global_symbol =
        func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value_or(gvar->name_hint);

    for (const ffi::String& param : collector->info_.launch_params) {
      if (param == tvm::runtime::launch_param::kUseProgramaticDependentLaunch ||
          param == tvm::runtime::launch_param::kUseCooperativeLaunch ||
          param == tvm::runtime::launch_param::kUseRequiredBlockDimension) {
        continue;
      }
      collector->info_.launch_args.push_back(collector->GetArgument(param));
    }

    collector->info_.dynamic_smem_requirement = collector->dyn_shmem_size;
    return collector->info_;
  }

 private:
  PrimExpr GetArgument(const ffi::String& launch_param) const {
    if (launch_param == tvm::runtime::launch_param::kUseDynamicSharedMemoryTag) {
      TVM_FFI_ICHECK(dyn_shmem_size.has_value())
          << "Compute kernel requires launch parameter \"" << launch_param
          << "\", but Function has no dynamic shared memory requirement.";
      return dyn_shmem_size.value();
    }

    auto extent = thread_extent.Get(launch_param);
    TVM_FFI_ICHECK(extent)
        << "Compute kernel requires launch parameter \"" << launch_param
        << "\", but Function does not contain a launch region defining this axis";
    return extent.value();
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindNode* op) final {
    if (const auto* call = op->value.as<CallNode>(); call && call->op.same_as(alloc_tensor_op()))
      return DispatchAllocTensor(op, call);
    // Track Bind definitions so that launch extents and
    // dyn_shmem_size expressions that reference locally-bound
    // variables (e.g. CSE variables) can be inlined back to
    // expressions over function parameters.  Substitute earlier
    // bindings into the value to handle chains (cse_v2 = f(cse_v1)).
    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return StmtExprVisitor::Visit_(op);
    }
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = bind_map_.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    PrimExpr value = bind_map_.size() ? ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                                            prim_value.value(), f_substitute)
                                            .as_or_throw<PrimExpr>()
                                      : prim_value.value();
    bind_map_.Set(op->var, value);
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op) final {
    if (op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) != 0) {
      ffi::String thread_tag = op->args[0].as_or_throw<StringImm>()->value;
      auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto repl = bind_map_.Get(var)) return ffi::Any(*std::move(repl));
        return ffi::Unchanged();
      };
      PrimExpr value = op->args[1].as_or_throw<PrimExpr>();
      if (bind_map_.size()) {
        value = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(value, f_substitute)
                    .as_or_throw<PrimExpr>();
      }
      if (auto previous = thread_extent.Get(thread_tag)) {
        sym::Analyzer analyzer;
        TVM_FFI_CHECK(analyzer->CanProveEqual(previous.value(), value), ValueError)
            << "Incompatible launch extents for " << thread_tag;
      } else {
        info_.launch_params.push_back(thread_tag);
        thread_extent.Set(thread_tag, value);
      }
    }
    auto outer_bindings = bind_map_;
    auto result = StmtExprVisitor::Visit_(op);
    bind_map_ = std::move(outer_bindings);
    return result;
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindNode* op, const CallNode* call) {
    ffi::String scope = call->args[2].as_or_throw<StringImm>()->value;
    auto storage_scope = runtime::StorageScope::Create(scope);
    if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
      TVM_FFI_ICHECK(!saw_dyn_shared_alloc_)
          << "Only one dynamic shared memory allocation is allowed.";
      saw_dyn_shared_alloc_ = true;

      // A zero extent is an extern placeholder; native launch regions supply
      // its resource requirement independently of the backing allocation.
      tvm::Tuple shape = call->args[0].as_or_throw<tvm::Tuple>();
      DLDataType dtype = call->args[1].as_or_throw<DataTypeImm>()->value;
      PrimType element_type(dtype);
      TVM_FFI_ICHECK_GT(shape->fields.size(), 0);
      PrimExpr dyn_size = IntImm::Int32(1);
      for (const auto& extent : shape->fields) {
        dyn_size *= extent.as_or_throw<PrimExpr>();
      }
      dyn_size *= IntImm::Int64(static_cast<int64_t>(element_type.StorageBytes()));
      if (bind_map_.size()) {
        auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto repl = bind_map_.Get(var)) return ffi::Any(*std::move(repl));
          return ffi::Unchanged();
        };
        dyn_size = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(dyn_size, f_substitute)
                       .as_or_throw<PrimExpr>();
      }
      inferred_shmem_size_ = dyn_size;
    }
    return StmtExprVisitor::Visit_(op);
  }

  // The collected results.
  KernelInfo info_;
  // The extent of each thread.
  ffi::Map<ffi::String, PrimExpr> thread_extent;
  // The amount of dynamic shared memory used.
  ffi::Optional<PrimExpr> dyn_shmem_size{std::nullopt};
  // Whether a shared.dyn allocation was seen.
  bool saw_dyn_shared_alloc_{false};
  // Launch size inferred from the allocation extent.
  ffi::Optional<PrimExpr> inferred_shmem_size_{std::nullopt};
  // Flag-only launch attributes requested by the original Function.
  bool use_programmatic_dependent_launch_{false};
  bool use_cooperative_launch_{false};
  // Accumulated Bind definitions for inlining into extent/size expressions.
  ffi::Map<Var, PrimExpr> bind_map_;
};

class ReturnRemover : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  static Stmt Apply(const Stmt& stmt, bool remove) {
    auto mutator = ffi::make_object<ReturnRemover>(remove);
    return mutator->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
  }

 public:
  explicit ReturnRemover(bool remove) : remove_(remove) {}

 private:
  UnchangedOr<Stmt> Mutate_(const ReturnNode* op, InplaceMode inplace_mode) override {
    auto as_int = op->value.as<IntImmNode>();
    TVM_FFI_ICHECK(as_int && as_int->value == 0)
        << "Device kernel may only contain a successful return, return 0";
    return remove_ ? Evaluate(0) : ffi::GetRef<Stmt>(op);
  }

  bool remove_;
};

class GlobalVarCallCollector : public StmtExprVisitor {
 public:
  static std::unordered_set<const GlobalVarNode*> Collect(const IRModule& mod) {
    auto collector = ffi::make_object<GlobalVarCallCollector>();
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto function = base_func.as<Function>()) {
        collector->Visit(function.value()->body);
      }
    }
    return collector->called_gvars_;
  }

 private:
  using Parent = StmtExprVisitor;

  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    if (auto* gvar = op->op.as<GlobalVarNode>()) {
      called_gvars_.insert(gvar);
    }
    return Parent::Visit_(op);
  }

  std::unordered_set<const GlobalVarNode*> called_gvars_;
};

}  // namespace

Stmt MakeCudaKernelLaunch(const GlobalVar& symbol, Function* func, ffi::Array<Expr> args,
                          const RegionStmtNode* region) {
  auto fields = region->attrs->dict.at(tvm::backend::cuda::attr::kLaunchFields)
                    .as_or_throw<ffi::Array<ffi::String>>();
  ffi::Array<Expr> values = region->args;
  auto kernel_attrs = region->attrs->dict.at(tvm::backend::cuda::attr::kKernelAttrs)
                          .as_or_throw<ffi::Map<ffi::String, int64_t>>();
  auto info = DeviceInfoCollector::Collect(symbol, *func, true);
  ffi::Array<Stmt> host_stmts;
  auto required_bytes = info.dynamic_smem_requirement;
  if (auto pool_bytes = region->attrs->dict.Get(tvm::backend::cuda::attr::kSmemRequired)) {
    PrimExpr pool = IntImm::Int64(pool_bytes->cast<int64_t>());
    required_bytes = required_bytes ? prim::Max(required_bytes.value(), pool) : pool;
  }
  if (auto required = required_bytes) {
    sym::Analyzer analyzer;
    auto bytes = analyzer->Simplify(required.value());
    int index = -1;
    for (size_t i = 0; i < fields.size(); ++i) {
      if (fields[i] == "dynamic_smem_bytes") index = i;
    }
    if (index < 0) {
      TVM_FFI_CHECK(!prim::IsZero(bytes) ||
                        region->attrs->dict.count(tvm::backend::cuda::attr::kSmemRequired),
                    ValueError)
          << "A shared.dyn placeholder requires LaunchConfig.dynamic_smem_bytes or "
             "SMEMPool.commit()";
      fields.push_back("dynamic_smem_bytes");
      values.push_back(bytes);
    } else {
      auto available = values[index].as_or_throw<PrimExpr>();
      auto condition = analyzer->Simplify(available >= bytes);
      TVM_FFI_CHECK(!prim::IsZero(condition), ValueError)
          << "LaunchConfig.dynamic_smem_bytes is smaller than the kernel allocation";
      if (!prim::IsOne(condition)) {
        host_stmts.push_back(AssertStmt(
            condition, StringImm("ValueError"),
            {StringImm("LaunchConfig.dynamic_smem_bytes is smaller than the kernel allocation")}));
      }
    }
  }
  // Only constants needed for code generation enter the device function's metadata.
  // Dynamic geometry, streams and event handles remain host-side call operands.
  ffi::Array<PrimExpr> dimensions;
  for (const char* prefix : {"block.", "cluster."}) {
    for (char axis : {'x', 'y', 'z'}) {
      PrimExpr dimension = IntImm::Int32(1);
      for (size_t i = 0; i < fields.size(); ++i) {
        if (fields[i] == std::string(prefix) + axis) {
          sym::Analyzer analyzer;
          auto value = analyzer->Simplify(values[i].as_or_throw<PrimExpr>());
          // Zero denotes a dynamic dimension; never emit static launch bounds for it.
          dimension = value.as<IntImmNode>() ? value : IntImm::Int32(0);
        }
      }
      dimensions.push_back(dimension);
    }
  }
  *func =
      WithAttrs(std::move(*func), {{tvm::attr::kCallingConv, tvm::CallingConv::kDeviceKernelLaunch},
                                   {tvm::attr::kGlobalSymbol, symbol->name_hint},
                                   {tvm::backend::cuda::attr::kLaunchFields, fields},
                                   {tvm::backend::cuda::attr::kKernelAttrs, kernel_attrs},
                                   {tvm::backend::cuda::attr::kLaunchDimensions, dimensions}});
  auto attrs = ffi::make_object<CallFFIKernelAttr>();
  attrs->launch_fields = fields;
  attrs->kernel_attrs = kernel_attrs;
  attrs->num_kernel_args = args.size();
  ffi::Array<Expr> call_args{StringImm(symbol->name_hint)};
  call_args.insert(call_args.end(), args.begin(), args.end());
  call_args.insert(call_args.end(), values.begin(), values.end());
  host_stmts.push_back(
      Evaluate(Call(PrimType::Int(32), call_ffi_kernel_op(), call_args, Attrs(attrs))));
  return SeqStmt(host_stmts);
}

class DeviceKernelMutator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  using Parent = StmtExprMutator;

  explicit DeviceKernelMutator(std::unordered_map<const GlobalVarNode*, KernelInfo> device_info_map)
      : device_info_map_(std::move(device_info_map)) {}

  Function RewriteKernelLaunchSite(const GlobalVar& gvar, Function func) {
    TVM_FFI_ICHECK(!current_target_.has_value());
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
    if (full_target->GetHost().has_value()) {
      current_caller_host_target_ = full_target->GetHost().value();
    } else {
      current_caller_host_target_ = std::nullopt;
    }

    auto body_result = Mutate(func->body, InplaceMode::kDisallow);
    bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
    auto body = std::move(body_result).ValueOrUnchanged(func->body);
    if (!body_unchanged) {
      func.CopyOnWrite()->body = body;
    }

    current_target_ = std::nullopt;
    current_caller_host_target_ = std::nullopt;
    return func;
  }

  Function UpdateKernelAttributes(const GlobalVar& gvar, Function func) const {
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
        Target target = func->GetAttr<Target>(tvm::attr::kTarget).value();
        bool preserve_early_returns = target->kind->name == "cuda";
        write_ptr->body = ReturnRemover::Apply(write_ptr->body.value(), !preserve_early_returns);
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
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override {
    auto node = Parent::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();

    auto* gvar = op->op.as<GlobalVarNode>();
    if (!gvar) return node;

    auto it = device_info_map_.find(gvar);
    TVM_FFI_ICHECK(it != device_info_map_.end())
        << "CallNode attempted subroutine call to " << gvar->name_hint << ", but "
        << gvar->name_hint << " did not appear within the IRModule";
    const KernelInfo& dev_info = it->second;

    auto callee_target = dev_info.target;

    // A callee with non-empty launch_params has launch regions
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
        ffi::Array<Expr> args;
        args.push_back(StringImm(gvar->name_hint));
        for (const Expr& arg : node->args) {
          args.push_back(arg);
        }
        Type ret_ty = IsVoidType(node->ty) ? PrimType::Void() : node->ty;
        return Call(ret_ty, call_extern_op(), args);
      }
    }

    TVM_FFI_ICHECK(dev_info.launch_params.defined())
        << "CallNode attempted kernel launch to " << gvar->name_hint << " on target "
        << dev_info.target << ", but subroutine " << gvar->name_hint
        << " did not have the tvm::tirx::attr::kKernelLaunchParams attribute "
        << "required for cross-target kernel launch";

    // Collected kernel information may be in terms of the callee's
    // arguments, but we need expressions for them in terms of the
    // caller's parameters.  The param_map allows substitution of
    // parameter values into the thread extents, to generate
    // expressions that are valid within the caller.
    const ffi::Array<Expr>& args = node->args;
    ffi::Map<Var, PrimExpr> param_map = [&]() {
      ffi::Map<Var, PrimExpr> param_map;
      TVM_FFI_ICHECK_EQ(args.size(), dev_info.params.size())
          << "Function " << gvar->name_hint << " accepts " << dev_info.params.size()
          << " arguments as input, but is called using " << args.size() << " arguments";
      for (size_t i = 0; i < args.size(); i++) {
        if (auto prim_arg = args[i].as<PrimExpr>()) {
          param_map.Set(dev_info.params[i], prim_arg.value());
        }
      }
      return param_map;
    }();

    device_kernel_launch_.insert(gvar);

    ffi::Array<Expr> call_args;
    call_args.push_back(StringImm(dev_info.global_symbol));
    for (const Expr& arg : args) {
      call_args.push_back(arg);
    }
    auto f_substitute = [&param_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = param_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    for (const auto& launch_arg : dev_info.launch_args) {
      call_args.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(launch_arg, f_substitute)
                              .as_or_throw<PrimExpr>());
    }

    PrimType node_ty = IsVoidType(node->ty) ? PrimType::Void() : node->ty.as_or_throw<PrimType>();
    PrimType ret_ty = node_ty.IsVoid() ? PrimType::Int(32) : node_ty;

    auto attrs = ffi::make_object<CallFFIKernelAttr>();
    attrs->launch_params = dev_info.launch_params;
    return Call(ret_ty, call_ffi_kernel_op(), call_args, Attrs(attrs)).as_or_throw<PrimExpr>();
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
        if (auto function = base_func.as<Function>()) {
          device_info_map.emplace(gvar.get(), DeviceInfoCollector::Collect(gvar, function.value()));
        }
      }
    }
    return ffi::make_object<DeviceKernelMutator>(std::move(device_info_map));
  }();

  {
    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto* ptr = base_func.as<FunctionNode>()) {
        auto function = mutator->RewriteKernelLaunchSite(gvar, ffi::GetRef<Function>(ptr));
        if (!function.same_as(base_func)) {
          updates->Add(gvar, function);
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
      if (auto* ptr = base_func.as<FunctionNode>()) {
        auto function = mutator->UpdateKernelAttributes(gvar, ffi::GetRef<Function>(ptr));
        if (!function.same_as(base_func)) {
          updates->Add(gvar, function);
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
      if (auto opt = base_func.as<Function>()) {
        Function func = opt.value();
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

  return tvm::transform::CreateModulePass(pass_func, 0, "tirx.SplitHostDevice");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.SplitHostDevice", SplitHostDevice);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
