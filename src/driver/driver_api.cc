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
    const IntImmNode* phase_num = phase_pass[0].as<IntImmNode>();
    ICHECK(phase_num)
        << "Expected the first entry in the inner Array of tir.add_lower_pass to be an integer";
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
  pass_list.push_back(tir::transform::FP8ComputeLegalize());
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
  pass_list.push_back(tir::transform::UnrollLoop());

  // Add user-defined phase-2 passes
  pass_list.insert(pass_list.end(), user_lower_phase2.begin(), user_lower_phase2.end());

  // PHASE 3
  pass_list.push_back(tir::transform::RenormalizeSplitPattern());
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::RemoveNoOp());
  pass_list.push_back(tir::transform::RewriteUnsafeSelect());
  pass_list.push_back(tir::transform::HoistIfThenElse());

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

IRModule LowerWithPassList(IRModule mod, Array<tvm::transform::Pass> pass_list) {
  auto optimize = tvm::transform::Sequential(pass_list);
  mod = optimize(std::move(mod));
  return mod;
}

IRModule ApplyPasses(IRModule mod, transform::Sequential seq) {
  mod = seq(std::move(mod));
  return mod;
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
      IRModule mod =
          ScheduleToModule(std::move(sch), args, name, c_binds, GlobalVarSupply(NameSupply("")));
      return mod;
    });

IRModule LowerModule(IRModule mod, bool simple_mode) {
  Array<transform::Pass> pass_list = CreatePassList(simple_mode);
  return LowerWithPassList(std::move(mod), pass_list);
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

  // Get the pass list
  Array<transform::Pass> pass_list = CreatePassList(simple_mode);
  return LowerWithPassList(std::move(mod), pass_list);
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
  // Get the legacy TE pass list
  Array<transform::Pass> pass_list = CreatePassList(simple_mode);
  return LowerWithPassList(mod, pass_list);
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
      return LowerSchedule(std::move(sch), args, name, c_binds, GlobalVarSupply(NameSupply("")),
                           simple_mode);
    });

/**
 * This function takes the input module that contains both the device and host opts.
 * Then, it applies transformation on the original module before splitting into separate modules for
 * device and host. Then it also applies transformations on the new splitted modules.
 */
std::pair<IRModule, IRModule> SplitMixedModule(IRModule mod_mixed, const Target& target_arg,
                                               const Target& target_host_arg) {
  Target target = target_arg, target_host = target_host_arg;
  CheckAndUpdateHostConsistency(&target, &target_host);

  ICHECK(mod_mixed.defined()) << "This module must be defined";

  mod_mixed = ApplyPasses(mod_mixed, MixedModulePassManager(mod_mixed, target));

  IRModule host_mod = ApplyPasses(mod_mixed, HostModulePassManager(mod_mixed, target_host));

  IRModule device_mod = ApplyPasses(mod_mixed, DeviceModulePassManager(mod_mixed, target));

  auto keys = target->GetKeys();

  CheckAndUpdateHostConsistency(&target, &target_host);

  bool target_is_gpu = std::find(keys.begin(), keys.end(), "gpu") != keys.end();
  if (target_is_gpu && device_mod->functions.size() == 0) {
    DLOG(WARNING) << "Specified target " << target->str()
                  << " but cannot find device code. Did you forget to bind?";
  }

  return {host_mod, device_mod};
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

  // Take the attrs from the first module so the eventual modules have them.
  // Ideally this would just be one unified module all the way through;
  IRModule first_module = (*inputs.begin()).second;
  IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>(), {}, {}, {}, first_module->attrs);

  ICHECK(mhost_all.defined()) << "The host module must be defined";

  for (const auto& it : inputs) {
    if (it.second.defined()) {
      const Target& target = it.first;
      const IRModule& ir_module = it.second;
      auto pair = SplitMixedModule(ir_module, target, target_host);
      auto& host_mod = pair.first;
      auto& device_mod = pair.second;

      ICHECK(host_mod.defined()) << "The split host module must be defined";

      ICHECK(mhost_all.defined()) << "The host module must be defined";

      // We don't want library modules going back into host codegen
      // unless they're supposed to. Here if we overrode the target host
      // to allow lowering previously we check that it's meant to be placed
      // back into the host Module.
      bool overrides_host_target =
          target->GetTargetDeviceType() == target_host->GetTargetDeviceType();
      bool non_host_target_kind = target->kind != target_host->kind;
      if (overrides_host_target && non_host_target_kind) {
        device_modules.push_back(codegen::Build(host_mod, it.first));
      } else {
        mhost_all->Update(host_mod);
      }

      if (device_mod->functions.size() != 0) {
        device_modules.push_back(codegen::Build(device_mod, it.first));
      }
    }
  }

  runtime::Module mhost = codegen::Build(mhost_all, target_host);
  for (const auto& it : device_modules) {
    if (it.operator->()) {
      mhost.Import(it);
    }
  }

  return mhost;
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

transform::Sequential MixedModulePassManager(IRModule mixed_mod, Target target) {
  transform::PassContext pass_ctx = transform::PassContext::Current();

  Array<Pass> mixed_pass_list;

  // VerifyVTCMLimit must occur before LowerVtcmAlloc
  mixed_pass_list.push_back(tir::transform::VerifyVTCMLimit(target));
  // LowerVtcmAlloc must occur after any transformations that modify memory allocation locations
  mixed_pass_list.push_back(tir::transform::LowerVtcmAlloc());

  mixed_pass_list.push_back(tir::transform::BindTarget(target));

  mixed_pass_list.push_back(tir::transform::VerifyMemory());

  mixed_pass_list.push_back(tir::transform::AnnotateEntryFunc());

  bool detect_global_barrier =
      pass_ctx->GetConfig<Bool>("tir.detect_global_barrier", Bool(false)).value();
  if (detect_global_barrier) {
    mixed_pass_list.push_back(tir::transform::ThreadSync("global"));
  }

  mixed_pass_list.push_back(tir::transform::ThreadSync("shared"));
  mixed_pass_list.push_back(tir::transform::ThreadSync("shared.dyn"));
  mixed_pass_list.push_back(tir::transform::MergeSharedMemoryAllocations());
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

  return transform::Sequential(mixed_pass_list);
}

TVM_REGISTER_GLOBAL("driver.mixed_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target) {
      return MixedModulePassManager(mixed_mod, target);
    });

transform::Sequential HostModulePassManager(IRModule mixed_mod, Target target_host) {
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

  return transform::Sequential(host_pass_list);
}

TVM_REGISTER_GLOBAL("driver.host_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target_host) {
      return HostModulePassManager(mixed_mod, target_host);
    });

transform::Sequential DeviceModulePassManager(IRModule mixed_mod, Target target) {
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

  return transform::Sequential(device_pass_list);
}

TVM_REGISTER_GLOBAL("driver.device_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target_host) {
      return DeviceModulePassManager(mixed_mod, target_host);
    });

}  // namespace tvm
