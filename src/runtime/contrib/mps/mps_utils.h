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
 * \file Use external mps utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_MPS_MPS_UTILS_H_
#define TVM_RUNTIME_CONTRIB_MPS_MPS_UTILS_H_

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <vector>

#include "../../metal/metal_common.h"

namespace tvm {
namespace contrib {

/*! breif Convert DLTensor type to MPS type */
struct MPSType {
  static MPSDataType DLTypeToMPSType(const DLDataType& dtype);
};  // struct MPSType

struct MetalThreadEntry {
  MetalThreadEntry();
  ~MetalThreadEntry();
  MPSImage* AllocMPSImage(id<MTLDevice> dev, MPSImageDescriptor* desc);
  MPSTemporaryImage* AllocTempImage(id<MTLCommandBuffer> cb, MPSImageDescriptor* desc);
  runtime::metal::MetalWorkspace* metal_api{nullptr};
  static MetalThreadEntry* ThreadLocal();
  std::vector<MPSImage*> img_table;
};  // MetalThreadEntry

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MPS_MPS_UTILS_H_
