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
TVM_REGISTER_PASS_CONFIG_OPTION("tir.is_entry_func", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.add_lower_pass", Array<Array<ObjectRef>>);

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;
using tvm::Array;
using tvm::transform::Pass;

bool LLVMEnabled() {
  const runtime::PackedFunc* pf = runtime::Registry::Get("target.build.llvm");
  return pf != nullptr;
}

/*! \return The default host target for a given device target */
Target DefaultTargetHost(Target target) {
  if (target.defined() && target->kind->device_type == kDLCPU) {
    return target;
  } else {
    if (LLVMEnabled()) {
      return Target("llvm");
    } else {
      return Target("stackvm");
    }
  }
}

tir::Buffer BufferWithOffsetAlignment(Array<PrimExpr> shape, DataType dtype, std::string name,
                                      int data_alignment, int offset_factor, bool compact) {
  DataType storage_dtype = (dtype == DataType::Bool() ? DataType::Int(8) : dtype);
  auto data = tir::Var(name, PointerType(PrimType(storage_dtype)));
  bool has_any = false;
  if (!compact) {
    for (const auto& it : shape) {
      if (it.as<tir::VarNode>()) {
        has_any = true;
        break;
      }
    }
  }
  tir::BufferType buffer_type = has_any ? tir::kAutoBroadcast : tir::kDefault;

  PrimExpr elem_offset;
  if (offset_factor != 0) {
    elem_offset = tir::Var(name + "_elem_offset", shape[0].dtype());
  } else {
    elem_offset = PrimExpr();
  }

  return tir::Buffer(data, dtype, shape, Array<PrimExpr>(), elem_offset, name, data_alignment,
                     offset_factor, buffer_type);
}

void GetBinds(const Array<ObjectRef>& args, bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list) {
  *out_binds = binds;

  for (const ObjectRef& x : args) {
    if (const te::TensorNode* tensor_node = x.as<te::TensorNode>()) {
      te::Tensor x_ref = GetRef<te::Tensor>(tensor_node);
      if (out_binds->find(x_ref) == out_binds->end()) {
        tir::Buffer buf =
            BufferWithOffsetAlignment(x_ref->shape, x_ref->dtype, x_ref->op->name, -1, 0, compact);
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

transform::Pass BindTarget(Target target) {
  auto fpass = [target](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    return WithAttr(std::move(f), tvm::attr::kTarget, target);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "BindTarget", {});
}

static transform::Pass AnnotateEntryFunc(bool b) {
  auto fpass = [b](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    return WithAttr(std::move(f), tir::attr::kIsEntryFunc, Bool(true));
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "AnnotateEntryFunc", {});
}

template <typename FCond>
transform::Pass Filter(FCond fcond) {
  auto fpass = [fcond](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    if (fcond(f)) {
      return f;
    } else {
      return tir::PrimFunc(nullptr);
    }
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "Filter", {});
}

Array<tvm::transform::Pass> CreatePassList(bool disable_loop_partition) {
  transform::PassContext pass_ctx = transform::PassContext::Current();

  bool disable_vectorize = pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
  bool instrument_bound_checkers =
      pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();

  // Get any user-added passes
  Array<Array<ObjectRef>> add_lower_pass =
      pass_ctx->GetConfig<Array<Array<ObjectRef>>>("tir.add_lower_pass", Array<Array<ObjectRef>>())
          .value();

  Array<transform::Pass> user_lower_phase0 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase1 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase2 = Array<transform::Pass>();
  Array<transform::Pass> user_lower_phase3 = Array<transform::Pass>();

  // phase pasees is of the form
  // [[phase_number, pass], [phase_number, pass]... ]
  for (Array<ObjectRef> phase_pass : add_lower_pass) {
    const IntImmNode* phase_num = phase_pass[0].as<IntImmNode>();
    ICHECK(phase_num)
        << "Expected the first entry in the inner Array of tir.add_lower_pass to be an integer";
    int phase_num_val = phase_num->value;

    CHECK_GE(phase_num_val, 0);

    const tvm::transform::PassNode* pass_node = phase_pass[1].as<tvm::transform::PassNode>();
    tvm::transform::Pass pass = GetRef<tvm::transform::Pass>(pass_node);
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
  pass_list.push_back(tir::transform::LowerInitBlock());
  pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
  pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
  pass_list.push_back(tir::transform::CompactBufferAllocation());
  pass_list.push_back(tir::transform::LowerMatchBuffer());
  pass_list.push_back(tir::transform::FlattenBuffer());
  pass_list.push_back(tir::transform::UnifyThreadBinding());
  pass_list.push_back(tir::transform::BF16Legalize());
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
  pass_list.push_back(tir::transform::StorageRewrite());
  pass_list.push_back(tir::transform::UnrollLoop());

  // Add user-defined phase-2 passes
  pass_list.insert(pass_list.end(), user_lower_phase2.begin(), user_lower_phase2.end());

  // PHASE 3
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::RemoveNoOp());
  pass_list.push_back(tir::transform::RewriteUnsafeSelect());
  pass_list.push_back(tir::transform::HoistIfThenElse());

  // Add user-defined phase-3 passes
  pass_list.insert(pass_list.end(), user_lower_phase3.begin(), user_lower_phase3.end());

  if (instrument_bound_checkers) {
    pass_list.push_back(tir::transform::InstrumentBoundCheckers());
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

IRModule ScheduleToModule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                          const std::unordered_map<te::Tensor, tir::Buffer>& binds) {
  // Convert te schedule to IRModule
  Array<ObjectRef> out_arg_list;
  transform::PassContext pass_ctx = transform::PassContext::Current();

  sch = sch.normalize();

  // Before TIR transformation.
  Map<tir::IterVar, Range> bounds = te::InferBound(sch);
  tir::Stmt stmt = te::ScheduleOps(sch, std::move(bounds), false);
  bool compact = te::VerifyCompactBuffer(stmt);

  Map<te::Tensor, tir::Buffer> out_binds;
  GetBinds(args, compact, binds, &out_binds, &out_arg_list);

  // Build the function
  // At this point binds is only te::Tensors
  tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

  // Mark this schedule as being converted from an TE schedule. Makes sure that
  // the correct TE passes are run.
  f = WithAttr(std::move(f), "from_legacy_te_schedule", Bool(true));

  bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();

  if (noalias) {
    f = WithAttr(std::move(f), "tir.noalias", Bool(true));
  }
  return IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(name), f}}));
}

TVM_REGISTER_GLOBAL("driver.schedule_to_module")
    .set_body_typed([](te::Schedule sch, const Array<ObjectRef>& args, const String& name,
                       const Map<te::Tensor, tir::Buffer>& binds) {
      std::unordered_map<te::Tensor, tir::Buffer> c_binds;
      // Check to make sure binds is not null before doing the conversion;
      if (binds.get() != nullptr) {
        for (auto kv : binds) {
          c_binds.insert({kv.first, kv.second});
        }
      }
      IRModule mod = ScheduleToModule(std::move(sch), args, name, c_binds);
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
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds, bool simple_mode) {
  Array<ObjectRef> ref_args;
  for (ObjectRef x : args) {
    ref_args.push_back(x);
  }
  return LowerSchedule(std::move(sch), ref_args, name, binds);
}

IRModule LowerSchedule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                       const std::unordered_map<te::Tensor, tir::Buffer>& binds, bool simple_mode) {
  IRModule mod = ScheduleToModule(std::move(sch), args, name, binds);
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
      return LowerSchedule(std::move(sch), args, name, c_binds, simple_mode);
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

runtime::Module FinalizeModule(const Map<Target, IRModule>& inputs_arg, const Target& host_target) {
  std::vector<runtime::Module> device_modules;
  Map<Target, IRModule> inputs = inputs_arg;
  Target target_host = host_target;

  CheckAndUpdateHostConsistency(&inputs, &target_host);

  if (!target_host.defined()) {
    for (const auto& it : inputs) {
      if (it.first->kind->device_type == kDLCPU || it.first->kind->device_type == kDLMicroDev) {
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

  IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>());

  ICHECK(mhost_all.defined()) << "The host module must be defined";

  for (const auto& it : inputs) {
    if (it.second.defined()) {
      auto pair = SplitMixedModule(it.second, it.first, target_host);
      auto& host_mod = pair.first;
      auto& device_mod = pair.second;

      ICHECK(host_mod.defined()) << "The split host module must be defined";

      ICHECK(mhost_all.defined()) << "The host module must be defined";

      mhost_all->Update(host_mod);

      if (device_mod->functions.size() != 0) {
        device_modules.push_back(codegen::Build(device_mod, it.first));
      }
    }
  }

  runtime::Module complete_mod = codegen::Build(mhost_all, target_host);
  for (const auto& it : device_modules) {
    if (it.operator->()) {
      complete_mod.Import(it);
    }
  }
  return complete_mod;
}

TVM_REGISTER_GLOBAL("driver.finalize_module")
    .set_body_typed([](const Map<Target, IRModule>& inputs_arg, Target host_target) {
      return FinalizeModule(inputs_arg, host_target);
    });

runtime::Module build(const Map<Target, IRModule>& inputs_arg, const Target& target_host_arg) {
  auto pass_ctx = transform::PassContext::Current();

  std::vector<runtime::Module> device_modules;
  Map<Target, IRModule> inputs = inputs_arg;
  Target target_host = target_host_arg;

  // Fetch previous defined target host in targets
  CheckAndUpdateHostConsistency(&inputs, &target_host);

  if (!target_host.defined()) {
    for (const auto& it : inputs) {
      if (it.first->kind->device_type == kDLCPU || it.first->kind->device_type == kDLMicroDev) {
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

  IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>());

  ICHECK(mhost_all.defined()) << "The host module must be defined";

  for (const auto& it : inputs) {
    if (it.second.defined()) {
      auto pair = SplitMixedModule(it.second, it.first, target_host);
      auto& host_mod = pair.first;
      auto& device_mod = pair.second;

      ICHECK(host_mod.defined()) << "The split host module must be defined";

      ICHECK(mhost_all.defined()) << "The host module must be defined";

      mhost_all->Update(host_mod);

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
  return build(updated_inputs, target_host);
}

// Build for homogeneous execution.
runtime::Module build(const IRModule& funcs, const Target& target_arg,
                      const Target& target_host_arg) {
  auto target = target_arg, target_host = target_host_arg;
  CheckAndUpdateHostConsistency(&target, &target_host);
  // More maps of target and target host
  Map<Target, IRModule> inputs = {{target, funcs}};
  return build(inputs, target_host);
}

transform::Sequential MixedModulePassManager(IRModule mixed_mod, Target target) {
  transform::PassContext pass_ctx = transform::PassContext::Current();

  Array<Pass> mixed_pass_list;

  mixed_pass_list.push_back(BindTarget(target));

  mixed_pass_list.push_back(tir::transform::VerifyMemory());
  mixed_pass_list.push_back(tir::transform::MergeDynamicSharedMemoryAllocations());

  if (mixed_mod->functions.size() == 1) {
    // mixed_pass_list.push_back(AnnotateEntryFunc(true));
  }

  bool detect_global_barrier =
      pass_ctx->GetConfig<Bool>("tir.detect_global_barrier", Bool(false)).value();
  if (detect_global_barrier) {
    mixed_pass_list.push_back(tir::transform::ThreadSync("global"));
  }

  mixed_pass_list.push_back(tir::transform::ThreadSync("shared"));
  mixed_pass_list.push_back(tir::transform::ThreadSync("warp"));
  mixed_pass_list.push_back(tir::transform::InferFragment());
  mixed_pass_list.push_back(tir::transform::LowerThreadAllreduce());

  if (target->GetAttr<Bool>("unpacked-api").value_or(Bool(false))) {
    mixed_pass_list.push_back(tir::transform::MakeUnpackedAPI());
  } else {
    mixed_pass_list.push_back(tir::transform::MakePackedAPI(-1));
  }
  mixed_pass_list.push_back(tir::transform::SplitHostDevice());

  return transform::Sequential(mixed_pass_list);
}

TVM_REGISTER_GLOBAL("driver.mixed_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target) {
      return MixedModulePassManager(mixed_mod, target);
    });

transform::Sequential HostModulePassManager(IRModule mixed_mod, Target target_host) {
  Array<tvm::transform::Pass> host_pass_list;
  host_pass_list.push_back(Filter([](const tir::PrimFunc& f) {
    return f->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) !=
           CallingConv::kDeviceKernelLaunch;
  }));

  ICHECK(mixed_mod.defined()) << "This module must be defined";

  host_pass_list.push_back(BindTarget(target_host));

  host_pass_list.push_back(tir::transform::LowerTVMBuiltin());
  host_pass_list.push_back(tir::transform::LowerCustomDatatypes());
  host_pass_list.push_back(tir::transform::LowerIntrin());
  host_pass_list.push_back(tir::transform::LowerDeviceStorageAccessInfo());
  host_pass_list.push_back(tir::transform::CombineContextCall());

  return transform::Sequential(host_pass_list);
}

TVM_REGISTER_GLOBAL("driver.host_mod_passes")
    .set_body_typed([](IRModule mixed_mod, Target target_host) {
      return HostModulePassManager(mixed_mod, target_host);
    });

transform::Sequential DeviceModulePassManager(IRModule mixed_mod, Target target) {
  Array<Pass> device_pass_list;
  device_pass_list.push_back(Filter([](const tir::PrimFunc& f) {
    return f->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) ==
           CallingConv::kDeviceKernelLaunch;
  }));

  device_pass_list.push_back(BindTarget(target));

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
