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
 * \file sdaccel_common.h
 * \brief SDAccel common header
 */
#ifndef TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_COMMON_H_
#define TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_COMMON_H_

#include <memory>

#include "../opencl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

/*!
 * \brief Process global SDAccel workspace.
 */
class SDAccelWorkspace final : public OpenCLWorkspace {
 public:
  // override OpenCL device API
  void Init() final;
  bool IsOpenCLDevice(Device dev) final;
  OpenCLThreadEntry* GetThreadEntry() final;
  // get the global workspace
  static OpenCLWorkspace* Global();
};

/*! \brief Thread local workspace for SDAccel*/
class SDAccelThreadEntry : public OpenCLThreadEntry {
 public:
  // constructor
  SDAccelThreadEntry()
      : OpenCLThreadEntry(static_cast<DLDeviceType>(kDLSDAccel), SDAccelWorkspace::Global()) {}

  // get the global workspace
  static SDAccelThreadEntry* ThreadLocal();
};
}  // namespace cl
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_COMMON_H_
