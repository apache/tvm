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
