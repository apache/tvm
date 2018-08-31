/*!
 *  Copyright (c) 2018 by Contributors
 * \file sdaccel_device_api.cc
 */
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>
#include "sdaccel_common.h"

namespace tvm {
namespace runtime {
namespace cl {

OpenCLThreadEntry* SDAccelWorkspace::GetThreadEntry() {
  return SDAccelThreadEntry::ThreadLocal();
}

const std::shared_ptr<OpenCLWorkspace>& SDAccelWorkspace::Global() {
  static std::shared_ptr<OpenCLWorkspace> inst = std::make_shared<SDAccelWorkspace>();
  return inst;
}

void SDAccelWorkspace::Init() {
  OpenCLWorkspace::Init("sdaccel", "accelerator", "Xilinx");
}

bool SDAccelWorkspace::IsOpenCLDevice(TVMContext ctx) {
  return ctx.device_type == static_cast<DLDeviceType>(kDLSDAccel);
}

typedef dmlc::ThreadLocalStore<SDAccelThreadEntry> SDAccelThreadStore;

SDAccelThreadEntry* SDAccelThreadEntry::ThreadLocal() {
  return SDAccelThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.sdaccel")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = SDAccelWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
