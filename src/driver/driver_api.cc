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
#include <tvm/tir/ir_pass.h>
#include <tvm/target/codegen.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <mutex>
#include <stack>

namespace tvm {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;
using tir::LoweredFunc;

bool LLVMEnabled() {
  const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.build_llvm");
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

Array<LoweredFunc> lower(te::Schedule sch,
                         const Array<te::Tensor>& args,
                         const std::string& name,
                         const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                         const BuildConfig& config) {
  Array<ObjectRef> out_arg_list;
  auto stmt = BuildStmt(sch, args, binds, true, &out_arg_list, config);
  return Array<LoweredFunc>({ tir::MakeAPI(stmt, name, out_arg_list, 0, config->restricted_func) });
}

Array<Array<LoweredFunc> > split_dev_host_funcs(const Array<LoweredFunc>& funcs,
                                                const Target& target,
                                                const Target& target_host,
                                                const BuildConfig& config) {
  std::unordered_set<std::string> all_names;
  for (const auto& x : funcs) {
    CHECK(all_names.count(x->name) == 0)
        << "Duplicate function name " << x->name;
    all_names.insert(x->name);
  }

  Array<LoweredFunc> fhost;
  Array<LoweredFunc> fdevice;

  for (const auto& x : funcs) {
    CHECK(tir::VerifyMemory(x, target->device_type))
        << "Direct host side access to device memory is detected in "
        << x->func_name() << ". Did you forget to bind?";

    if (x->func_type == tir::kMixedFunc) {
      auto func = x;
      if (config->detect_global_barrier) {
        func = tir::ThreadSync(func, "global");
      }

      func = tir::ThreadSync(func, "shared");
      func = tir::ThreadSync(func, "warp");
      func = tir::LowerThreadAllreduce(func, target->thread_warp_size);
      auto fsplits = tir::SplitHostDevice(func);
      fhost.push_back(fsplits[0]);
      for (auto f = fsplits.begin() + 1; f != fsplits.end(); ++f) {
        fdevice.push_back(*f);
      }
    } else if (x->func_type == tir::kHostFunc) {
      fhost.push_back(x);
    } else if (x->func_type == tir::kDeviceFunc) {
      fdevice.push_back(x);
    } else {
      LOG(FATAL) << "unknown function type " << x->func_type;
    }
  }

  for (size_t i = 0; i < fdevice.size(); i++) {
    auto warp_size = target->thread_warp_size;
    auto func = fdevice[i];
    func = tir::LowerWarpMemory(fdevice[i], warp_size);
    fdevice.Set(i, func);
  }

  auto keys = target->keys();
  bool target_is_gpu = std::find(keys.begin(), keys.end(), "gpu") != keys.end();
  if (target_is_gpu && fdevice.size() == 0) {
    LOG(WARNING) << "Specified target "
                 << target->str()
                 << " but cannot find device code. Did you forget to bind?";
  }

  for (size_t i = 0; i < fdevice.size(); ++i) {
    auto func = fdevice[i];
    func = tir::LowerIntrin(func, target->target_name);
    fdevice.Set(i, func);
  }

  if (target->device_type == target::llvm()->device_type &&
        target_host == target) {
    CHECK(fdevice.empty()) << "No device code should be generated when target "
                           << "and host_target are both llvm target."
                           << "\n";
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = tir::BindDeviceType(func, target->device_type);
    func = tir::LowerDeviceStorageAccessInfo(func);
    func = tir::LowerTVMBuiltin(func);
    fhost.Set(i, func);
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = tir::LowerIntrin(func, target_host->target_name);
    func = tir::LowerDeviceStorageAccessInfo(func);
    func = tir::CombineContextCall(func);
    fhost.Set(i, func);
  }
  return {fhost, fdevice};
}

// Create a module for a specific device (target). The lowered functions
// associated with the host is returned as well.
runtime::Module DeviceBuild(const Array<LoweredFunc>& fdevice,
                            const Target& target) {
  if (!fdevice.empty()) {
    return codegen::Build(fdevice, target->str());
  } else {
    return runtime::Module(nullptr);
  }
}

// Build for heterogeneous execution.
runtime::Module build(const Map<Target, Array<LoweredFunc>>& inputs,
                      const Target& target_host,
                      const BuildConfig& config) {
  Array<LoweredFunc> fhost_all;
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

  for (const auto& it : inputs) {
    auto host_dev_funcs =
        split_dev_host_funcs(it.second, it.first, target_host_val, config);
    auto& fhost = host_dev_funcs[0];
    auto& fdevice = host_dev_funcs[1];
    // Get the module for a certain target.
    runtime::Module mdev = DeviceBuild(fdevice, it.first);
    for (const auto& it : fhost) {
      fhost_all.push_back(it);
    }
    device_modules.push_back(mdev);
  }

  runtime::Module mhost = codegen::Build(fhost_all, target_host_val->str());
  // Import all modules
  for (const auto& it : device_modules) {
    if (it.operator->()) {
      mhost.Import(it);
    }
  }
  return mhost;
}

// Build for heterogeneous execution when target is a string.
runtime::Module build(const Map<std::string, Array<LoweredFunc>>& inputs,
                      const Target& target_host,
                      const BuildConfig& config) {
  Map<Target, Array<LoweredFunc>> updated_input;
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
runtime::Module build(const Array<LoweredFunc>& funcs,
                      const Target& target,
                      const Target& target_host,
                      const BuildConfig& config) {
  Map<Target, Array<LoweredFunc>> inputs = {{target, funcs}};
  return build(inputs, target_host, config);
}

}  // namespace tvm
