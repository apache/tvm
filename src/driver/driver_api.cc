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

/*!
* \brief Build a Stmt given a schedule, args and binds. This function runs the IR passes.
* \param sch The schedule to build.
* \param args The arguments for the schedule.
* \param binds Buffer assignments.
* \param loop_partition True if the LoopPartition pass should be included.
* \param out_arg_list Returns the arguments for the Stmt.
* \param config The build configuration.
* \return The built Stmt.
*/
tir::Stmt BuildStmt(te::Schedule sch,
                    const Array<te::Tensor>& args,
                    const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                    bool loop_partition,
                    Array<ObjectRef> *out_arg_list,
                    const BuildConfig& config) {
  sch = sch.normalize();

  // Phase 0
  auto bounds = te::InferBound(sch);
  auto stmt = te::ScheduleOps(sch, bounds, false);
  stmt = tir::InjectPrefetch(stmt);

  bool compact = tir::VerifyCompactBuffer(stmt);
  Map<te::Tensor, tir::Buffer> out_binds;
  GetBinds(args, compact, binds, &out_binds, out_arg_list, config);

  // Phase 1
  stmt = tir::StorageFlatten(stmt, out_binds, 64,
                            config->instrument_bound_checkers);
  stmt = tir::CanonicalSimplify(stmt);
  if (loop_partition) {
    stmt = tir::LoopPartition(stmt, config->partition_const_loop);
  }
  if (config->disable_vectorize) {
    stmt = tir::SkipVectorize(stmt);
  } else {
    stmt = tir::VectorizeLoop(stmt);
  }
  stmt = tir::InjectVirtualThread(stmt);
  stmt = tir::InjectDoubleBuffer(stmt, config->double_buffer_split_loop);
  stmt = tir::StorageRewrite(stmt);
  stmt = tir::UnrollLoop(stmt, config->auto_unroll_max_step, config->auto_unroll_max_depth,
    config->auto_unroll_max_extent, config->unroll_explicit);

  // Phase 2
  stmt = tir::Simplify(stmt);
  stmt = tir::RemoveNoOp(stmt);

  if (!(config->disable_select_rewriting))
    stmt = tir::RewriteUnsafeSelect(stmt);

  if (config->instrument_bound_checkers)
    stmt = tir::InstrumentBoundCheckers(stmt);

  return stmt;
}

transform::Pass BindTarget(Target target) {
  auto fpass = [target](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    return WithAttr(std::move(f), tvm::attr::kTarget, target);
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "BindTarget", {});
}


template<typename FCond>
transform::Pass FilterBy(FCond fcond) {
  auto fpass = [fcond](tir::PrimFunc f, IRModule m, transform::PassContext ctx) {
    if (fcond(f)) {
      return f;
    } else {
      return tir::PrimFunc(nullptr);
    }
  };
  return tir::transform::CreatePrimFuncPass(fpass, 0, "FilterBy", {});
}


IRModule lower(te::Schedule sch,
               const Array<te::Tensor>& args,
               const std::string& name,
               const std::unordered_map<te::Tensor, tir::Buffer>& binds,
               const BuildConfig& config) {
  Array<ObjectRef> out_arg_list;
  auto stmt = BuildStmt(sch, args, binds, true, &out_arg_list, config);

  Array<tir::Var> params;
  Map<tir::Var, tir::Buffer> buffer_map;

  for (auto var : out_arg_list) {
    if (auto* n = var.as<tir::VarNode>()) {
      params.push_back(GetRef<tir::Var>(n));
    } else {
      tir::Buffer buffer = Downcast<tir::Buffer>(var);
      tir::Var bptr(buffer->name, DataType::Handle());
      params.push_back(bptr);
      buffer_map.Set(bptr, buffer);
    }
  }

  auto f = tir::PrimFunc(params, stmt, VoidType(), buffer_map);
  f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

  if (config->restricted_func) {
    f = WithAttr(std::move(f), "tir.noalias", Integer(1));
  }
  return IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(name), f}}));
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
    FilterBy([](const tir::PrimFunc& f) {
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
    FilterBy([](const tir::PrimFunc& f) {
      return f->GetAttr<Integer>(
          tvm::attr::kCallingConv,
          Integer(CallingConv::kDefault)) == CallingConv::kDeviceKernelLaunch;
    }),
    BindTarget(target),
    tir::transform::LowerWarpMemory(),
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
