/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/compile_engine.h
 * \brief Internal compilation engine handle function cache.
 *  and interface to low level code generation.
 */
#ifndef TVM_RELAY_FRONTEND_TVM_BUILD_MODULE_H_
#define TVM_RELAY_FRONTEND_TVM_BUILD_MODULE_H_

#include <tvm/build_module.h>
#include <tvm/codegen.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

#include <string>

#include "graph_runtime_codegen.h"
#include "utils.h"

namespace tvm {
/*!
 * \brief Build TVM Runtime
 *
 * \param funcs Lowered functions
 * \param target Target device
 * \param target_host Target host device
 * \param config TVM Build config
 * \return runtime::Module
 */
runtime::Module tvm_build(const Array<LoweredFunc>& funcs, const Target& target,
                          const Target& target_host, const BuildConfig& config) {
  std::unordered_set<std::string> all_names;
  for (const auto& x : funcs) {
    CHECK(all_names.count(x->name) == 0) << "Duplicate function name " << x->name;
    all_names.insert(x->name);
  }

  auto target_host_val = target_host;

  Array<LoweredFunc> fhost;
  Array<LoweredFunc> fdevice;

  for (const auto& x : funcs) {
    CHECK(ir::VerifyMemory(x, target->device_type))
        << "Direct host side access to device memory is detected in " << x->func_name()
        << ". Did you forget to bind?";

    if (x->func_type == kMixedFunc) {
      auto func = x;
      if (config->detect_global_barrier) {
        func = ir::ThreadSync(func, "global");
      }

      func = ir::ThreadSync(func, "shared");
      func = ir::LowerThreadAllreduce(func, target->thread_warp_size);
      auto fsplits = ir::SplitHostDevice(func);
      fhost.push_back(fsplits[0]);
      for (auto f = fsplits.begin() + 1; f != fsplits.end(); ++f) {
        fdevice.push_back(*f);
      }
    } else if (x->func_type == kHostFunc) {
      fhost.push_back(x);
    } else if (x->func_type == kDeviceFunc) {
      fdevice.push_back(x);
    } else {
      LOG(FATAL) << "unknown function type " << x->func_type;
    }
  }

  auto keys = target->keys();
  bool target_is_gpu = std::find(keys.begin(), keys.end(), "gpu") != keys.end();
  if (target_is_gpu && fdevice.size() == 0) {
    LOG(WARNING) << "Specified target " + target->str() +
                        " but cannot find device code. Did you forget to bind?";
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = ir::BindDeviceType(func, target->device_type);
    func = ir::LowerTVMBuiltin(func);
    fhost.Set(i, func);
  }

  for (size_t i = 0; i < fdevice.size(); ++i) {
    auto func = fdevice[i];
    func = ir::LowerIntrin(func, target->target_name);
    fdevice.Set(i, func);
  }

  for (size_t i = 0; i < fhost.size(); ++i) {
    auto func = fhost[i];
    func = ir::LowerIntrin(func, target_host_val->target_name);
    func = ir::CombineContextCall(func);
    fhost.Set(i, func);
  }

  auto mhost = codegen::Build(fhost, target_host_val->str());

  if (fdevice.size() > 0) {
    auto mdev = codegen::Build(fdevice, target->str());
    mhost.Import(mdev);
  }

  return mhost;
}

}  // namespace tvm
#endif  // TVM_RELAY_FRONTEND_TVM_BUILD_MODULE_H_
