/*!
 *  Copyright (c) 2018 by Contributors
 * \file aocl_device_api.cc
 */
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>
#include "aocl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

OpenCLThreadEntry* AOCLWorkspace::GetThreadEntry() {
  return AOCLThreadEntry::ThreadLocal();
}

const std::shared_ptr<OpenCLWorkspace>& AOCLWorkspace::Global() {
  static std::shared_ptr<OpenCLWorkspace> inst = std::make_shared<AOCLWorkspace>();
  return inst;
}

void AOCLWorkspace::Init() {
  OpenCLWorkspace::Init("aocl", "accelerator", "Intel(R) FPGA SDK for OpenCL(TM)");
}

bool AOCLWorkspace::IsOpenCLDevice(TVMContext ctx) {
  return ctx.device_type == static_cast<DLDeviceType>(kDLAOCL);
}

typedef dmlc::ThreadLocalStore<AOCLThreadEntry> AOCLThreadStore;

AOCLThreadEntry* AOCLThreadEntry::ThreadLocal() {
  return AOCLThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.aocl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = AOCLWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
