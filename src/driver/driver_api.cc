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
#include <tvm/te/operation.h>

#include <tvm/tir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/target/codegen.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <mutex>
#include <stack>

namespace tvm {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

bool LLVMEnabled() {
  const runtime::PackedFunc* pf = runtime::Registry::Get("target.build.llvm");
  return pf != nullptr;
}

/*! \return The default host target for a given device target */
Target DefaultTargetHost(Target target) {
  if (target.defined() && target->device_type == kDLCPU) {
    return target;
  } else {
    if (LLVMEnabled()) {
      return target::llvm();
    } else {
      return target::stackvm();
    }
  }
}

tir::Buffer BufferWithOffsetAlignment(Array<PrimExpr> shape,
                                      DataType dtype,
                                      std::string name,
                                      int data_alignment,
                                      int offset_factor,
                                      bool compact) {
  auto data = tir::Var(name, DataType::Handle());
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

  return tir::BufferNode::make(data, dtype, shape, Array<PrimExpr>(), elem_offset, name, "",
    data_alignment, offset_factor, buffer_type);
}

void GetBinds(const Array<te::Tensor>& args,
              bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds,
              Array<ObjectRef>* out_arg_list,
              const BuildConfig& config) {
  *out_binds = binds;

  for (const auto &x : args) {
    if (out_binds->find(x) == out_binds->end()) {
      auto buf = BufferWithOffsetAlignment(x->shape, x->dtype, x->op->name,
        config->data_alignment, config->offset_factor, compact);
      out_binds->Set(x, buf);
      out_arg_list->push_back(buf);
    } else {
      out_arg_list->push_back((*out_binds)[x]);
    }
  }
}

transform::Pass BindTarget(Target target) {
  auto fpass = [target](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    return WithAttr(std::move(f), tvm::attr::kTarget, target);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "BindTarget", {});
}


template<typename FCond>
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


IRModule lower(te::Schedule sch,
               const Array<te::Tensor>& args,
               const std::string& name,
               const std::unordered_map<te::Tensor, tir::Buffer>& binds,
               const BuildConfig& config) {
  Array<ObjectRef> out_arg_list;

  sch = sch.normalize();

  // Before TIR transformation.
  auto bounds = te::InferBound(sch);
  auto stmt = te::ScheduleOps(sch, bounds, false);
  bool compact = tir::VerifyCompactBuffer(stmt);

  Map<te::Tensor, tir::Buffer> out_binds;
  GetBinds(args, compact, binds, &out_binds, &out_arg_list, config);

  // build the function
  tir::PrimFunc f = te::SchedulePostProcToPrimFunc(
      out_arg_list, std::move(stmt), out_binds);
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));
  if (config->restricted_func) {
    f = WithAttr(std::move(f), "tir.noalias", Integer(1));
  }

  auto mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(name), f}}));
  auto pass_list = Array<tvm::transform::Pass>();

  // Phase 0
  pass_list.push_back(tir::transform::InjectPrefetch());
  pass_list.push_back(
      tir::transform::StorageFlatten(64, config->instrument_bound_checkers));
  // Phase 1
  pass_list.push_back(tir::transform::NarrowDataType(32));
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::LoopPartition(config->partition_const_loop));
  pass_list.push_back(tir::transform::VectorizeLoop(!config->disable_vectorize));
  pass_list.push_back(tir::transform::InjectVirtualThread());
  pass_list.push_back(tir::transform::InjectDoubleBuffer(config->double_buffer_split_loop));
  pass_list.push_back(tir::transform::StorageRewrite());
  pass_list.push_back(
      tir::transform::UnrollLoop(config->auto_unroll_max_step,
                                 config->auto_unroll_max_depth,
                                 config->auto_unroll_max_extent,
                                 config->unroll_explicit));
  // Phase 2
  pass_list.push_back(tir::transform::Simplify());
  pass_list.push_back(tir::transform::RemoveNoOp());
  if (!(config->disable_select_rewriting)) {
    pass_list.push_back(tir::transform::RewriteUnsafeSelect());
  }
  if (config->instrument_bound_checkers) {
    pass_list.push_back(tir::transform::InstrumentBoundCheckers());
  }
  // run
  auto optimize = transform::Sequential(pass_list);
  mod = optimize(std::move(mod));
  return mod;
}


std::pair<IRModule, IRModule>
split_dev_host_funcs(IRModule mod_mixed,
                     const Target& target,
                     const Target& target_host,
                     const BuildConfig& config) {
  mod_mixed = BindTarget(target)(std::move(mod_mixed));
  tir::VerifyMemory(mod_mixed);

  Array<tvm::transform::Pass> mixed_pass_list = {BindTarget(target)};
  if (config->detect_global_barrier) {
    mixed_pass_list.push_back(tir::transform::ThreadSync("global"));
  }
  mixed_pass_list.push_back(tir::transform::ThreadSync("shared"));
  mixed_pass_list.push_back(tir::transform::ThreadSync("warp"));
  mixed_pass_list.push_back(tir::transform::InferFragment());
  mixed_pass_list.push_back(tir::transform::LowerThreadAllreduce());
  mixed_pass_list.push_back(tir::transform::MakePackedAPI(0));
  mixed_pass_list.push_back(tir::transform::SplitHostDevice());
  auto opt_mixed = transform::Sequential(mixed_pass_list);
  mod_mixed = opt_mixed(std::move(mod_mixed));

  auto host_pass_list = {
    Filter([](const tir::PrimFunc& f) {
      return f->GetAttr<Integer>(
          tvm::attr::kCallingConv,
          Integer(CallingConv::kDefault)) != CallingConv::kDeviceKernelLaunch;
    }),
    BindTarget(target_host),
    tir::transform::LowerTVMBuiltin(),
    tir::transform::LowerIntrin(),
    tir::transform::LowerDeviceStorageAccessInfo(),
    tir::transform::CombineContextCall(),
  };
  auto opt_host = transform::Sequential(host_pass_list);
  auto mhost = opt_host(mod_mixed);

  // device pipeline
  auto device_pass_list = {
    Filter([](const tir::PrimFunc& f) {
      return f->GetAttr<Integer>(
          tvm::attr::kCallingConv,
          Integer(CallingConv::kDefault)) == CallingConv::kDeviceKernelLaunch;
    }),
    BindTarget(target),
    tir::transform::LowerWarpMemory(),
    tir::transform::Simplify(),
    tir::transform::LowerIntrin(),
    tir::transform::LowerDeviceStorageAccessInfo(),
  };
  auto opt_device = transform::Sequential(device_pass_list);
  auto mdevice = opt_device(mod_mixed);

  // some final misc checks.
  auto keys = target->keys();
  bool target_is_gpu = std::find(keys.begin(), keys.end(), "gpu") != keys.end();
  if (target_is_gpu && mdevice->functions.size() == 0) {
    LOG(WARNING) << "Specified target "
                 << target->str()
                 << " but cannot find device code. Did you forget to bind?";
  }

  if (target->device_type == target::llvm()->device_type &&
      target_host == target) {
    CHECK(mdevice->functions.empty())
        << "No device code should be generated when target "
        << "and host_target are both llvm target."
        << "\n";
  }

  return {mhost, mdevice};
}


// Build for heterogeneous execution.
runtime::Module build(const Map<Target, IRModule>& inputs,
                      const Target& target_host,
                      const BuildConfig& config) {
  std::vector<runtime::Module> device_modules;

  Target target_host_val = target_host;
  if (!target_host.defined()) {
    for (const auto& it : inputs) {
      if (it.first->device_type == kDLCPU) {
        target_host_val = it.first;
        break;
      }
    }
  }

  if (!target_host_val.defined()) {
    target_host_val = DefaultTargetHost(target_host_val);
  }

  IRModule mhost_all = IRModule(Map<GlobalVar, BaseFunc>());

  for (const auto& it : inputs) {
    auto pair =
        split_dev_host_funcs(it.second, it.first, target_host_val, config);
    auto& mhost = pair.first;
    auto& mdevice = pair.second;

    mhost_all->Update(mhost);
    if (mdevice->functions.size() != 0) {
      device_modules.push_back(codegen::Build(mdevice, it.first));
    }
  }

  runtime::Module mhost = codegen::Build(mhost_all, target_host_val);
  // Import all modules
  for (const auto& it : device_modules) {
    if (it.operator->()) {
      mhost.Import(it);
    }
  }
  return mhost;
}

// Build for heterogeneous execution when target is a string.
runtime::Module build(const Map<std::string, IRModule>& inputs,
                      const Target& target_host,
                      const BuildConfig& config) {
  Map<Target, IRModule> updated_input;
  for (const auto& it : inputs) {
    auto target = Target::Create(it.first);
    if (target->device_name == "vta") {
      target = Target::Create("ext_dev");
    }
    updated_input.Set(target, it.second);
  }
  return build(updated_input, target_host, config);
}

// Build for homogeneous execution.
runtime::Module build(const IRModule& funcs,
                      const Target& target,
                      const Target& target_host,
                      const BuildConfig& config) {
  Map<Target, IRModule> inputs = {{target, funcs}};
  return build(inputs, target_host, config);
}

}  // namespace tvm
