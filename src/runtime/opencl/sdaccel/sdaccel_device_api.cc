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
 * \file sdaccel_device_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "sdaccel_common.h"

namespace tvm {
namespace runtime {
namespace cl {

OpenCLThreadEntry* SDAccelWorkspace::GetThreadEntry() { return SDAccelThreadEntry::ThreadLocal(); }

OpenCLWorkspace* SDAccelWorkspace::Global() {
  static OpenCLWorkspace* inst = new SDAccelWorkspace();
  return inst;
}

void SDAccelWorkspace::Init() { OpenCLWorkspace::Init("sdaccel", "accelerator", "Xilinx"); }

bool SDAccelWorkspace::IsOpenCLDevice(Device dev) {
  return dev.device_type == static_cast<DLDeviceType>(kDLSDAccel);
}

typedef dmlc::ThreadLocalStore<SDAccelThreadEntry> SDAccelThreadStore;

SDAccelThreadEntry* SDAccelThreadEntry::ThreadLocal() { return SDAccelThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.sdaccel").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = SDAccelWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
