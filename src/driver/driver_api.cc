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
 *  Compile executable modules.
 * \file driver_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/transform.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <mutex>
#include <stack>

namespace tvm {

// Register build pipeline related options
TVM_REGISTER_PASS_CONFIG_OPTION("tir.noalias", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.detect_global_barrier", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.instrument_bound_checkers", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.disable_assert", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.disable_vectorize", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.enable_buffer_level_predication", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.disable_cse_tir", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.enable_debug", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.enable_equiv_terms_in_cse_tir", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.disable_storage_rewrite", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.is_entry_func", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.add_lower_pass", Array<Array<ObjectRef>>);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.debug_keep_trivial_loop", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.use_async_copy", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.merge_static_smem", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.instrument_lwp", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.vtcm_capacity", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.ptx_ldg32", Bool);

// WARNING: May cause coherency issues resulting data miscompares
// Experimental feature that, when enabled by the runtime, bypasses the cache when using DMA. When
// bypassing the cache TVM must manage cache coherency in software. Software managed cache coherency
// can be tricky e.g. it is yet to be proven out in the Hexagon runtime. Hence the warning above and
// the "experimental" notation for this feature.
TVM_REGISTER_PASS_CONFIG_OPTION("tir.experimental_dma_bypass_cache", Bool);

using tvm::Array;
using tvm::transform::Pass;

bool LLVMEnabled() {
  const runtime::PackedFunc* pf = runtime::Registry::Get("target.build.llvm");
  return pf != nullptr;
}

/*! \return The default host target for a given device target */
Target DefaultTargetHost(Target target) {
  if (target.defined() && target->GetTargetDeviceType() == kDLCPU) {
    return target;
  } else {
    if (LLVMEnabled()) {
      return Target("llvm");
    } else {
      return Target("stackvm");
    }
  }
}

void GetBinds(const Array<ObjectRef>& args, bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list) {
  *out_binds = binds;

  for (const ObjectRef& x : args) {
    if (auto tensor_node = x.as<te::Tensor>()) {
      te::Tensor x_ref = tensor_node.value();
      if (out_binds->find(x_ref) == out_binds->end()) {
        tir::Buffer buf = tir::BufferWithOffsetAlignment(x_ref->shape, x_ref->dtype,
                                                         x_ref->op->name, -1, 0, compact);
        out_binds->Set(x_ref, buf);
        out_arg_list->push_back(buf);
      } else {
        out_arg_list->push_back((*out_binds)[x_ref]);
      }
    } else if (x.as<te::BufferNode>() || x.as<tir::VarNode>()) {
      out_arg_list->push_back(x);
    } else {
      LOG(FATAL)
          << "Expected type of the elements of args to be te::Tensor, te::Buffer or tir::Var, "
          << "but got a " << x->GetTypeKey();
    }
  }
}

void GetBinds(const Array<te::Tensor>& args, bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list) {
  Array<ObjectRef> ref_args;
  for (ObjectRef x : args) {
    ref_args.push_back(x);
  }
  GetBinds(ref_args, compact, binds, out_binds, out_arg_list);
}

TVM_REGISTER_GLOBAL("driver.get_binds")
    .set_body_typed([](const Array<ObjectRef>& args, bool compact,
                       const Map<te::Tensor, tir::Buffer>& binds) {
      std::unordered_map<te::Tensor, tir::Buffer> c_binds;
      // Check to make sure binds is not null before doing the conversion;
      if (binds.get() != nullptr) {
        for (auto kv : binds) {
          c_binds.insert({kv.first, kv.second});
        }
      }
      Map<te::Tensor, tir::Buffer> out_binds;
      Array<ObjectRef> out_arg_list;
      GetBinds(args, compact, c_binds, &out_binds, &out_arg_list);

      // TVM object system doesn't have a pair object, so we'll put both ret values in an array
      // and return that.
      Array<ObjectRef> out_arr = {out_binds, out_arg_list};
      return out_arr;
    });

Array<tvm::transform::Pass> CreatePassList(bool disable_loop_partition) {
  transform::PassContext pass_ctx = transform::PassContext::Current();

  bool disable_vectorize = pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
  bool disable_storage_rewrite =
      pass_ctx->GetConfig<Bool>("tir.disable_storage_rewrite", Bool(false)).value();
  bool instrument_bound_checkers =
      pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();
  bool disable_cse_tir = pass_ctx->GetConfig<Bool>("tir.disable_cse_tir", Bool(false)).value();
  bool enable_equiv_terms_in_cse_tir =
      pass_ctx->GetConfig<Bool>("tir.enable_equiv_terms_in_cse_tir", Bool(false)).value();

  bool ptx_ldg32 = pass_ctx->GetConfig<Bool>("tir.ptx_ldg32", Bool(false)).value();

  // Get any user-added passes
  Array<Array<ObjectRef>> add_lower_pass =
      pass_ctx->GetConfig<Array<Array<ObjectRef>>>("tir.add_lower_pass", Array<Array<ObjectRef>>())
          .value();

  bool instrument_lwp = pass_ctx->GetConfig<Bool>("tir.instrument_lwp", Bool(false)).value();

  Array<transform::Pass> user_lower_phase0 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase1 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase2 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase3 = Array<transform::Pass>();

  // phase passes is of the form
  // [[phase_number, pass], [phase_number, pass]... ]
  for (Array<ObjectRef> phase_pass : add_lower_pass) {
    auto phase_num = phase_pass[0].as<runtime::Int::ContainerType>();
    ICHECK(phase_num)
        << "Expected the first entry in the inner Array of tir.add_lower_pass to be an integer, "
        << "but instead received " << phase_pass[0] << " with type " << phase_pass[0]->GetTypeKey();
    int phase_num_val = phase_num->value;

    CHECK_GE(phase_num_val, 0);

    auto pass = Downcast<tvm::transform::Pass>(phase_pass[1]);
    // Copy the pass into the correct phase
    if (phase_num_val == 0) {
      user_lower_phase0.push_back(pass);
    } else if (phase_num_val == 1) {
      user_lower_phase1.push_back(pass);
    } else if (phase_num_val == 2) {
      user_lower_phase2.push_back(pass);
    } else if (phase_num_val >= 3) {
      user_lower_phase3.push_back(pass);
    }
  }

  // Construct the pass list, inserting the user provided passes at the end of the phase

  // PHASE 0
  Array<tvm::transform::Pass> pass_list = user_lower_phase0;

  // PHASE 1
  pass_list.push_back(tir::transform::InjectPrefetch());
  pass_list.push_back(tir::transform::TextureFlatten());
  pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
  pass_list.push_back(tir::transform::LowerCrossThreadReduction());
  pass_list.push_back(tir::transform::LowerInitBlock());
  pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
  pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(tir::transform::LiftThreadBinding());
  pass_list.push_back(tir::transform::ManifestSharedMemoryLocalStage());
  pass_list.push_back(tir::transform::CompactBufferAllocation());
  pass_list.push_back(tir::transform::LowerAutoCopy());
  pass_list.push_back(tir::transform::UnifyThreadBinding());
  pass_list.push_back(tir::transform::LowerMatchBuffer());
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::InjectPermutedLayout());
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::InjectSoftwarePipeline());
  pass_list.push_back(tir::transform::TransformMmaBufferLayout());
  pass_list.push_back(tir::transform::LowerOpaqueBlock());
  pass_list.push_back(tir::transform::FlattenBuffer());
  pass_list.push_back(tir::transform::BF16ComputeLegalize());
  pass_list.push_back(tir::transform::NarrowDataType(32));
  pass_list.push_back(tir::transform::Simplify());

  // Add user-defined phase-1 passes
  pass_list.insert(pass_list.end(), user_lower_phase1.begin(), user_lower_phase1.end());

  // PHASE 2
  if (!disable_loop_partition) {
    pass_list.push_back(tir::transform::LoopPartition());
  }

  pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
  pass_list.push_back(tir::transform::InjectVirtualThread());
  pass_list.push_back(tir::transform::InjectDoubleBuffer());
  if (!disable_storage_rewrite) {
    pass_list.push_back(tir::transform::StorageRewrite());
  }
  bool use_async_copy = pass_ctx->GetConfig<Bool>("tir.use_async_copy", Bool(false)).value();

  if (use_async_copy) {
    pass_list.push_back(tir::transform::LowerAsyncDMA());
  }
  // HoistIfThenElse must be applied before UnrollLoop
  // because HoistIfThenElse could utilize for loop structure
  // which might be unrolled in UnrollLoop
  pass_list.push_back(tir::transform::HoistIfThenElse());
  pass_list.push_back(tir::transform::UnrollLoop());

  // Add user-defined phase-2 passes
  pass_list.insert(pass_list.end(), user_lower_phase2.begin(), user_lower_phase2.end());

  // PHASE 3
  pass_list.push_back(tir::transform::RenormalizeSplitPattern());
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::RemoveNoOp());
  pass_list.push_back(tir::transform::RewriteUnsafeSelect());

  // Add user-defined phase-3 passes
  pass_list.insert(pass_list.end(), user_lower_phase3.begin(), user_lower_phase3.end());

  if (instrument_bound_checkers) {
    pass_list.push_back(tir::transform::InstrumentBoundCheckers());
  }

  if (ptx_ldg32) {
    pass_list.push_back(tir::transform::InjectPTXLDG32(true));
  }

  pass_list.push_back(
      tir::transform::CommonSubexprElimTIR(!disable_cse_tir, enable_equiv_terms_in_cse_tir));

  // This pass instruments the loops with the profile builtin calls to capture the runtime
  // performance data (only enabled for Hexagon at the moment). To ensure that no other
  // optimizations are performed on the instrumented code, this pass must be added at the end
  // of the list.
  if (instrument_lwp) {
    pass_list.push_back(tir::transform::InstrumentProfileIntrinsics());
  }

  return pass_list;
}

// Convert te schedule to IRModule
IRModule ScheduleToModule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                          const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                          GlobalVarSupply global_var_supply) {
  sch = sch.normalize();

  transform::PassContext pass_ctx = transform::PassContext::Current();
  bool debug_keep_trivial_loop =
      pass_ctx->GetConfig<Bool>("tir.debug_keep_trivial_loop", Bool(false)).value();

  // Before TIR transformation.
  tir::Stmt stmt = te::ScheduleOps(sch, te::InferBound(sch), debug_keep_trivial_loop);
  bool compact = te::VerifyCompactBuffer(stmt);

  Map<te::Tensor, tir::Buffer> out_binds;
  Array<ObjectRef> out_arg_list;
  GetBinds(args, compact, binds, &out_binds, &out_arg_list);

  // Build the function, converting from te::Tensor to tir::Buffer
  tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

  // Mark this schedule as being converted from an TE schedule. Makes sure that
  // the correct TE passes are run.
  f = WithAttr(std::move(f), "from_legacy_te_schedule", Bool(true));

  bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();

  if (noalias) {
    f = WithAttr(std::move(f), "tir.noalias", Bool(true));
  }
  GlobalVar global_var = global_var_supply->UniqueGlobalFor(name, false);
  return IRModule(Map<GlobalVar, BaseFunc>({{global_var, f}}));
}

TVM_REGISTER_GLOBAL("driver.schedule_to_module")
    .set_body_typed([](te::Schedule sch, const Array<ObjectRef>& args, const String& name,
                       const Map<te::Tensor, tir::Buffer>& binds) {
      std::unordered_map<te::Tensor, tir::Buffer> c_binds;
      // Check to make sure binds is not null before doing the conversion;
      if (binds.defined()) {
        for (auto kv : binds) {
          c_binds.insert({kv.first, kv.second});
        }
      }
      IRModule mod = ScheduleToModule(std::move(sch), args, name, c_binds, GlobalVarSupply());
      return mod;
    });

IRModule LowerModule(IRModule mod, bool simple_mode) {
  Array<transform::Pass> pass_list = CreatePassList(simple_mode);
  tvm::transform::Sequential optimize(pass_list, "tvm.lower");
  return optimize(std::move(mod));
}

TVM_REGISTER_GLOBAL("driver.lower_module").set_body_typed([](IRModule mod, bool simple_mode) {
  return LowerModule(std::move(mod), simple_mode);
});

IRModule LowerPrimFunc(tir::PrimFunc func, const std::string& name, bool simple_mode) {
  transform::PassContext pass_ctx = transform::PassContext::Current();
  tir::PrimFunc f = WithAttr(std::move(func), "global_symbol", runtime::String(name));

  bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();

  if (noalias) {
    f = WithAttr(std::move(f), "tir.noalias", Bool(true));
  }
  IRModule mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(name), f}}));
  return LowerModule(mod, simple_mode);
}

TVM_REGISTER_GLOBAL("driver.lower_primfunc")
    .set_body_typed([](te::PrimFunc func, const String& name, bool simple_mode) {
      return LowerPrimFunc(std::move(func), name, simple_mode);
    });

IRModule LowerSchedule(te::Schedule sch, const Array<te::Tensor>& args, const std::string& name,
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                       GlobalVarSupply global_var_supply, bool simple_mode) {
  Array<ObjectRef> ref_args;
  for (ObjectRef x : args) {
    ref_args.push_back(x);
  }
  return LowerSchedule(std::move(sch), ref_args, name, binds, global_var_supply, simple_mode);
}

IRModule LowerSchedule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                       GlobalVarSupply global_var_supply, bool simple_mode) {
  IRModule mod = ScheduleToModule(std::move(sch), args, name, binds, global_var_supply);
  return LowerModule(mod, simple_mode);
}

TVM_REGISTER_GLOBAL("driver.lower_schedule")
    .set_body_typed([](te::Schedule sch, const Array<ObjectRef>& args, const String& name,
                       const Map<te::Tensor, tir::Buffer>& binds, bool simple_mode) {
      std::unordered_map<te::Tensor, tir::Buffer> c_binds;
      // Check to make sure binds is not null before doing the conversion;
      if (binds.get() != nullptr) {
        for (auto kv : binds) {
          c_binds.insert({kv.first, kv.second});
        }
      }
      return LowerSchedule(std::move(sch), args, name, c_binds, GlobalVarSupply(), simple_mode);
    });

IRModule MergeModules(const Map<Target, IRModule>& inputs) {
  if (inputs.size() == 1) {
    auto [target, mod] = *inputs.begin();
    return tir::transform::BindTarget(target)(mod);
  }

  // Take the attrs from the first module so the eventual modules have them.
  IRModule first_module = (*inputs.begin()).second;
  IRModule merged = IRModule(Map<GlobalVar, BaseFunc>(), {}, {}, {}, first_module->attrs);

  for (auto [target, mod] : inputs) {
    mod = tir::transform::BindTarget(target)(mod);
    merged->Update(mod);
  }

  return merged;
}

Map<Target, IRModule> SplitModule(const IRModule& module) {
  Map<String, IRModule> split;

  for (auto [gvar, base_func] : module->functions) {
    auto target_str = base_func->GetAttr<Target>(tvm::attr::kTarget).value()->str();
    if (auto it = split.find(target_str); it != split.end()) {
      (*it).second->Add(gvar, base_func);
    } else {
      split.Set(target_str, IRModule({{gvar, base_func}}, {}, {}, {}, module->attrs));
    }
  }

  Map<Target, IRModule> out;
  for (auto [str, mod] : split) {
    out.Set(Target(str), mod);
  }

  return out;
}

/*!
 * \brief Check and update host field of the given legacy heterogeneous targets and
 *  target host.Note that this function is for legacy target api compatibility issue only,
 *  not recommended for other use.
 * \param ir_modules The pointer to a Map objects with keys being Target objects
 * \param host The Target typed object for target host to be updated
 */
void CheckAndUpdateHostConsistency(Map<Target, IRModule>* targets, Target* host) {
  Map<Target, IRModule> new_targets;
  for (auto& it : *targets) {
    auto target = it.first;
    CheckAndUpdateHostConsistency(&target, host);
    new_targets.Set(target, it.second);
  }
  *targets = new_targets;
}

runtime::Module TIRToRuntime(const Map<Target, IRModule>& inputs_arg,
                             const Target& target_host_arg) {
  CHECK(inputs_arg.size()) << "TIRToRuntime expects at least one IRModule as input.";
  std::vector<runtime::Module> device_modules;
  Map<Target, IRModule> inputs = inputs_arg;
  Target target_host = target_host_arg;

  // Fetch previous defined target host in targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  if (!target_host.defined()) {
    for (const auto& it : inputs) {
      if (it.first->GetTargetDeviceType() == kDLCPU ||
          it.first->GetTargetDeviceType() == kDLMicroDev) {
        target_host = it.first;
        break;
      }
    }
  }

  if (!target_host.defined()) {
    target_host = DefaultTargetHost(target_host);
  }

  // Update target host for all targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  auto has_gpu_function = [](const IRModule& mod) -> bool {
    for (const auto& [gvar, func] : mod->functions) {
      if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
        if (target.value()->HasKey("gpu")) {
          return true;
        }
      }
    }
    return false;
  };

  IRModule merged = MergeModules(inputs);

  bool contains_gpu_function_pre = has_gpu_function(merged);
  merged = MixedModulePassManager(merged)(merged);
  bool contains_gpu_function_post = has_gpu_function(merged);
  if (contains_gpu_function_pre && !contains_gpu_function_post) {
    DLOG(WARNING) << "Specified GPU targets, "
                  << "but cannot find device code. Did you forget to bind?";
  }

  Map<Target, IRModule> split = SplitModule(merged);

  Map<Target, runtime::Module> built;
  for (const auto& [target, mod] : split) {
    built.Set(target, codegen::Build(mod, target));
  }

  auto host_target = [&]() -> Target {
    // All targets that contain a kIsEntryFunc=True function
    Array<Target> targets_with_entry_func;

    // All targets that can run on the CPU and contain at least one
    // function without kIsEntryFunc=False.
    Array<Target> cpu_targets;
    for (const auto& [target, mod] : split) {
      bool contains_entry_func = false;
      bool may_contain_entry_func = false;
      for (const auto& [gvar, func] : mod->functions) {
        Optional<Bool> is_entry_func = func->attrs.GetAttr<Bool>(tvm::tir::attr::kIsEntryFunc);
        if (is_entry_func.defined() && is_entry_func.value()->value) {
          contains_entry_func = true;
        } else if (!is_entry_func.defined()) {
          may_contain_entry_func = true;
        }
      }

      if (contains_entry_func) {
        targets_with_entry_func.push_back(target);
      }

      if (may_contain_entry_func && target->HasKey("cpu")) {
        cpu_targets.push_back(target);
      }
    }

    if (targets_with_entry_func.size()) {
      ICHECK_EQ(targets_with_entry_func.size(), 1)
          << "Expected at most one function "
          << "annotated with tvm::tir::attr::kIsEntryFunc "
          << "(\"" << tvm::tir::attr::kIsEntryFunc << "\"), "
          << "but found: " << targets_with_entry_func;
      return targets_with_entry_func[0];
    } else if (cpu_targets.size() == 1) {
      return cpu_targets[0];
    } else {
      LOG(FATAL) << "Could not determine which target is the host.  "
                 << "No function was annotated with tvm::tir::attr::kIsEntryFunc (\""
                 << tvm::tir::attr::kIsEntryFunc << "\"), "
                 << "and " << cpu_targets.size() << " targets have the 'cpu' key";
    }
  }();

  auto runtime_module = built[host_target];
  for (const auto& [target, mod] : built) {
    if (!mod.same_as(runtime_module)) {
      runtime_module.Import(mod);
    }
  }
  return runtime_module;
}

TVM_REGISTER_GLOBAL("driver.tir_to_runtime")
    .set_body_typed([](const Map<Target, IRModule>& inputs_arg, Target host_target) {
      return TIRToRuntime(inputs_arg, host_target);
    });

// Build for heterogeneous execution when targets are specified as
// objects.  This wrapper around the internal API is maintained for
// backwards compatibility.
runtime::Module build(const Map<Target, IRModule>& input, const Target& target_host) {
  return TIRToRuntime(input, target_host);
}

// Build for heterogeneous execution when target is a string.
runtime::Module build(const Map<String, IRModule>& inputs_arg, const Target& target_host_arg) {
  Map<Target, IRModule> updated_inputs;
  Target target_host = target_host_arg;
  for (const auto& it : inputs_arg) {
    Target target = Target(it.first);
    CheckAndUpdateHostConsistency(&target, &target_host);
    Optional<String> device = target->GetAttr<String>("device");
    if (device.defined() && device.value() == "vta") {
      target = Target("ext_dev");
    }
    updated_inputs.Set(target, it.second);
  }
  return TIRToRuntime(updated_inputs, target_host);
}

// Build for homogeneous execution.
runtime::Module build(const IRModule& funcs, const Target& target_arg,
                      const Target& target_host_arg) {
  auto target = target_arg, target_host = target_host_arg;
  CheckAndUpdateHostConsistency(&target, &target_host);
  // More maps of target and target host
  Map<Target, IRModule> inputs = {{target, funcs}};
  return TIRToRuntime(inputs, target_host);
}

transform::Sequential MixedModulePassManager(IRModule mixed_mod, Optional<Target> target) {
  transform::PassContext pass_ctx = transform::PassContext::Current();

  Array<Pass> mixed_pass_list;

  // FPComputeLegalize uses the target attrs added by BindTarget, so
  // BindTarget must come first.
  if (target) {
    mixed_pass_list.push_back(tir::transform::BindTarget(target.value()));
  }
  mixed_pass_list.push_back(tir::transform::FP8ComputeLegalize());

  // VerifyVTCMLimit must occur before LowerVtcmAlloc
  mixed_pass_list.push_back(tir::transform::VerifyVTCMLimit(target));
  // LowerVtcmAlloc must occur after any transformations that modify memory allocation locations
  mixed_pass_list.push_back(tir::transform::LowerVtcmAlloc());

  mixed_pass_list.push_back(tir::transform::VerifyMemory());

  mixed_pass_list.push_back(tir::transform::AnnotateEntryFunc());

  bool detect_global_barrier =
      pass_ctx->GetConfig<Bool>("tir.detect_global_barrier", Bool(false)).value();
  if (detect_global_barrier) {
    mixed_pass_list.push_back(tir::transform::ThreadSync("global"));
  }

  mixed_pass_list.push_back(tir::transform::ThreadSync("shared"));
  mixed_pass_list.push_back(tir::transform::ThreadSync("shared.dyn"));
  mixed_pass_list.push_back(tir::transform::ThreadSync("warp"));
  mixed_pass_list.push_back(tir::transform::InferFragment());
  mixed_pass_list.push_back(tir::transform::LowerThreadAllreduce());

  bool use_async_copy = pass_ctx->GetConfig<Bool>("tir.use_async_copy", Bool(false)).value();

  if (use_async_copy) {
    mixed_pass_list.push_back(tir::transform::InjectPTXAsyncCopy());
  }

  bool ptx_ldg32 = pass_ctx->GetConfig<Bool>("tir.ptx_ldg32", Bool(false)).value();
  if (ptx_ldg32) {
    mixed_pass_list.push_back(tir::transform::InjectPTXLDG32());
  }

  mixed_pass_list.push_back(tir::transform::AnnotateDeviceRegions());
  mixed_pass_list.push_back(tir::transform::SplitHostDevice());
  // MergeSharedMemoryAllocations must be applied after SplitHostDevice
  // because the merged allocation site is at the beginning of each device function
  mixed_pass_list.push_back(tir::transform::MergeSharedMemoryAllocations());

  bool unpacked_api = mixed_mod->GetAttr<relay::Executor>(tvm::attr::kExecutor)
                          .value_or(relay::Executor::Create("graph", {}))
                          ->GetAttr<Bool>("unpacked-api")
                          .value_or(Bool(false));
  if (unpacked_api) {
    mixed_pass_list.push_back(tir::transform::MakeUnpackedAPI());
  } else {
    mixed_pass_list.push_back(tir::transform::MakePackedAPI());
  }
  mixed_pass_list.push_back(tir::transform::FP8StorageLegalize());
  mixed_pass_list.push_back(tir::transform::BF16StorageLegalize());

  mixed_pass_list.push_back(tir::transform::LowerDeviceKernelLaunch());

  // After the device kernels have been split into host/device
  // sections, the host section can be inlined.
  mixed_pass_list.push_back(tir::transform::InlinePrivateFunctions());

  // Only applies to the device functions, identified by inspection of
  // each function's tvm::attr::kTarget attribute.
  mixed_pass_list.push_back(tir::transform::LowerWarpMemory());

  // Only applies to the host functions, identified by inspection of
  // each function's tvm::attr::kTarget attribute.
  mixed_pass_list.push_back(tir::transform::LowerTVMBuiltin());

  // Apply to both host and device functions
  mixed_pass_list.push_back(tir::transform::Simplify());
  mixed_pass_list.push_back(tir::transform::LowerCustomDatatypes());
  mixed_pass_list.push_back(tir::transform::LowerIntrin());
  mixed_pass_list.push_back(tir::transform::LowerDeviceStorageAccessInfo());

  // Only applies to the host functions, identified by inspection of
  // each function's tvm::attr::kTarget attribute.
  mixed_pass_list.push_back(tir::transform::CombineContextCall());
  if (pass_ctx->GetConfig<Bool>("tir.enable_debug", Bool(false)).value()) {
    mixed_pass_list.push_back(tir::transform::InstallDebugSpans());
  }

  return transform::Sequential(mixed_pass_list, "tvm.build");
}

TVM_REGISTER_GLOBAL("driver.mixed_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target) {
      return MixedModulePassManager(mixed_mod, target);
    });

transform::Sequential HostModulePassManager(IRModule mixed_mod, Target target_host) {
  LOG(WARNING) << "Use of driver.host_mod_passes is deprecated.  "
               << "All lowering passes are now included "
               << "as part of driver.mixed_mod_passes.";

  transform::PassContext pass_ctx = transform::PassContext::Current();
  bool enable_debug = pass_ctx->GetConfig<Bool>("tir.enable_debug", Bool(false)).value();

  Array<tvm::transform::Pass> host_pass_list;

  runtime::TypedPackedFunc<bool(tir::PrimFunc)> fcond = [](const tir::PrimFunc& f) {
    return f->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) !=
           CallingConv::kDeviceKernelLaunch;
  };
  host_pass_list.push_back(tir::transform::Filter(fcond));

  ICHECK(mixed_mod.defined()) << "This module must be defined";

  host_pass_list.push_back(tir::transform::BindTarget(target_host));

  host_pass_list.push_back(tir::transform::LowerTVMBuiltin());
  host_pass_list.push_back(tir::transform::LowerCustomDatatypes());
  host_pass_list.push_back(tir::transform::LowerIntrin());
  host_pass_list.push_back(tir::transform::LowerDeviceStorageAccessInfo());
  host_pass_list.push_back(tir::transform::CombineContextCall());

  if (enable_debug) {
    host_pass_list.push_back(tir::transform::InstallDebugSpans());
  }

  return transform::Sequential(host_pass_list, "tir.host_mod_passes");
}

TVM_REGISTER_GLOBAL("driver.host_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target_host) {
      return HostModulePassManager(mixed_mod, target_host);
    });

transform::Sequential DeviceModulePassManager(IRModule mixed_mod, Target target) {
  LOG(WARNING) << "Use of driver.device_mod_passes is deprecated.  "
               << "All lowering passes are now included "
               << "as part of driver.mixed_mod_passes.";

  Array<Pass> device_pass_list;
  runtime::TypedPackedFunc<bool(tir::PrimFunc)> fcond = [](const tir::PrimFunc& f) {
    return f->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) ==
           CallingConv::kDeviceKernelLaunch;
  };
  device_pass_list.push_back(tir::transform::Filter(fcond));

  device_pass_list.push_back(tir::transform::BindTarget(target));

  device_pass_list.push_back(tir::transform::LowerWarpMemory());
  device_pass_list.push_back(tir::transform::Simplify());
  device_pass_list.push_back(tir::transform::LowerCustomDatatypes());
  device_pass_list.push_back(tir::transform::LowerDeviceStorageAccessInfo());
  device_pass_list.push_back(tir::transform::LowerIntrin());

  return transform::Sequential(device_pass_list, "tir.device_mod_passes");
}

TVM_REGISTER_GLOBAL("driver.device_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target_host) {
      return DeviceModulePassManager(mixed_mod, target_host);
    });

}  // namespace tvm
