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
 * \file Use external miopen utils function
 */
#include "miopen_utils.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <vector>

namespace tvm {
namespace contrib {
namespace miopen {

std::string miopenGetErrorString(int error_code) {
  const std::vector<std::string> mio_err{"StatusSuccess        ", "StatusNotInitialized ",
                                         "StatusInvalidValue   ", "StatusBadParm        ",
                                         "StatusAllocFailed    ", "StatusInternalError  ",
                                         "StatusNotImplemented ", "StatusUnknownError   "};
  return mio_err[error_code];
}

// MiopenThreadEntry
MIOpenThreadEntry::MIOpenThreadEntry() {
  auto stream = runtime::ROCMThreadEntry::ThreadLocal()->stream;
  auto func = runtime::Registry::Get("device_api.rocm");
  void* ret = (*func)();
  rocm_api = static_cast<runtime::DeviceAPI*>(ret);
  MIOPEN_CALL(miopenCreate(&handle));
  MIOPEN_CALL(miopenSetStream(handle, stream));
  conv_entry.rocm_api = rocm_api;
}

MIOpenThreadEntry::~MIOpenThreadEntry() { MIOPEN_CALL(miopenDestroy(handle)); }

typedef dmlc::ThreadLocalStore<MIOpenThreadEntry> MIOpenThreadStore;

MIOpenThreadEntry* MIOpenThreadEntry::ThreadLocal() { return MIOpenThreadStore::Get(); }

// ConvEntry

ConvEntry::ConvEntry() {
  MIOPEN_CALL(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CALL(miopenCreateTensorDescriptor(&filter_desc));
  MIOPEN_CALL(miopenCreateTensorDescriptor(&input_desc));
  MIOPEN_CALL(miopenCreateTensorDescriptor(&output_desc));
}

ConvEntry::~ConvEntry() {
  MIOPEN_CALL(miopenDestroyConvolutionDescriptor(conv_desc));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(filter_desc));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(input_desc));
  MIOPEN_CALL(miopenDestroyTensorDescriptor(output_desc));
  CleanWorkspace();
}

void ConvEntry::UpdateWorkspace(const size_t wsize) {
  if (workspace_size < wsize) {
    if (workspace != nullptr) {
      CleanWorkspace();
    }
    workspace_size = wsize;
    workspace = rocm_api->AllocWorkspace(device, workspace_size);
  }
}

void ConvEntry::CleanWorkspace() {
  if (workspace) rocm_api->FreeWorkspace(device, workspace);
  workspace_size = 0;
}

SoftmaxEntry::SoftmaxEntry() { MIOPEN_CALL(miopenCreateTensorDescriptor(&shape_desc)); }

SoftmaxEntry::~SoftmaxEntry() { MIOPEN_CALL(miopenDestroyTensorDescriptor(shape_desc)); }

}  // namespace miopen
}  // namespace contrib
}  // namespace tvm
