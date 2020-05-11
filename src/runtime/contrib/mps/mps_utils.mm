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

#include "mps_utils.h"

namespace tvm {
namespace contrib {

// MPS Data Type
MPSDataType MPSType::DLTypeToMPSType(const DLDataType& dtype) {
  switch (dtype.code) {
    case kDLInt:
      if (dtype.bits == 8 && dtype.lanes == 1)
        return MPSDataTypeInt8;
      else if (dtype.bits == 16 && dtype.lanes == 1)
        return MPSDataTypeInt16;
      else
        LOG(FATAL) << "Unsupported type";
      break;
    case kDLUInt:
      if (dtype.bits == 8 && dtype.lanes == 1)
        return MPSDataTypeUInt8;
      else if (dtype.bits == 16 && dtype.lanes == 1)
        return MPSDataTypeUInt16;
      else if (dtype.bits == 32 && dtype.lanes == 1)
        return MPSDataTypeUInt32;
      LOG(FATAL) << "Unsupported type";
      break;
    case kDLFloat:
      if (dtype.bits == 16 && dtype.lanes == 1)
        return MPSDataTypeFloat16;
      else if (dtype.bits == 32 && dtype.lanes == 1)
        return MPSDataTypeFloat32;
      else
        LOG(FATAL) << "Unsupported type";
      break;
    default:
      LOG(FATAL) << "Unsupported type";
  }
  return MPSDataTypeFloat32;
}

// MetalThreadEntry

MPSImage* MetalThreadEntry::AllocMPSImage(id<MTLDevice> dev, MPSImageDescriptor* desc) {
  MPSImage* mpsimg = [[MPSImage alloc] initWithDevice:dev imageDescriptor:desc];
  img_table.push_back(mpsimg);
  return mpsimg;
}

MPSTemporaryImage* MetalThreadEntry::AllocTempImage(id<MTLCommandBuffer> cb,
                                                    MPSImageDescriptor* desc) {
  MPSTemporaryImage* mpsimg = [MPSTemporaryImage temporaryImageWithCommandBuffer:cb
                                                                 imageDescriptor:desc];
  return mpsimg;
}

MetalThreadEntry::MetalThreadEntry() {
  auto func = runtime::Registry::Get("device_api.metal");
  void* ret = (*func)();
  metal_api = static_cast<runtime::metal::MetalWorkspace*>(ret);
}

MetalThreadEntry::~MetalThreadEntry() {
  for (int i = 0; i < img_table.size(); ++i) {
    [img_table[i] dealloc];
  }
}

typedef dmlc::ThreadLocalStore<MetalThreadEntry> MetalThreadStore;

MetalThreadEntry* MetalThreadEntry::ThreadLocal() { return MetalThreadStore::Get(); }

}  // namespace contrib
}  // namespace tvm
