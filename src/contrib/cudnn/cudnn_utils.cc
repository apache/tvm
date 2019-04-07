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
 *  Copyright (c) 2017 by Contributors
 * \file Use external cudnn utils function
 */
#include "cudnn_utils.h"
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>


namespace tvm {
namespace contrib {

// CuDNN Data Type
cudnnDataType_t CuDNNDataType::DLTypeToCuDNNType(const DLDataType &dtype) {
  switch (dtype.code) {
      case kDLInt:
        if (dtype.bits == 8 && dtype.lanes == 1) return CUDNN_DATA_INT8;
        else if (dtype.bits == 32 && dtype.lanes == 1) return CUDNN_DATA_INT32;
        else if (dtype.bits == 8 && dtype.lanes == 4) return CUDNN_DATA_INT8x4;
        else
          LOG(FATAL) << "Unsupported type";
        break;
      case kDLUInt:
        LOG(FATAL) << "Unsupported type";
        break;
      case kDLFloat:
        if (dtype.bits == 32 && dtype.lanes == 1) return CUDNN_DATA_FLOAT;
        else if (dtype.bits == 64 && dtype.lanes == 1) return CUDNN_DATA_DOUBLE;
        else if (dtype.bits == 16 && dtype.lanes == 1) return CUDNN_DATA_HALF;
        else
          LOG(FATAL) << "Unsupported type";
        break;
    }
    return CUDNN_DATA_FLOAT;
}

template<>
const void* CuDNNDataType::GetConst<0>(cudnnDataType_t type) {
  static const int int_v = 0;
  static const float float_v = 0;
  static const double double_v = 0;
  if (type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_HALF) {
    return static_cast<const void*>(&float_v);
  }
  if (type == CUDNN_DATA_DOUBLE) {
    return static_cast<const void*>(&double_v);
  }
  if (type == CUDNN_DATA_INT8 || type == CUDNN_DATA_INT32 || type == CUDNN_DATA_INT8x4) {
    return static_cast<const void*>(&int_v);
  }
  return nullptr;
}

template<>
const void* CuDNNDataType::GetConst<1>(cudnnDataType_t type) {
  static const int int_v = 1;
  static const float float_v = 1.f;
  static const double double_v = 1.f;
  if (type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_HALF) {
    return static_cast<const void*>(&float_v);
  }
  if (type == CUDNN_DATA_DOUBLE) {
    return static_cast<const void*>(&double_v);
  }
  if (type == CUDNN_DATA_INT8 || type == CUDNN_DATA_INT32 || type == CUDNN_DATA_INT8x4) {
    return static_cast<const void*>(&int_v);
  }
  return nullptr;
}

// CuDNNThreadEntry

CuDNNThreadEntry::CuDNNThreadEntry() {
  auto stream = runtime::CUDAThreadEntry::ThreadLocal()->stream;
  auto func = runtime::Registry::Get("device_api.gpu");
  void *ret = (*func)();
  cuda_api = static_cast<runtime::DeviceAPI*>(ret);
  CUDNN_CALL(cudnnCreate(&handle));
  CUDNN_CALL(cudnnSetStream(handle, stream));
  conv_entry.cuda_api = cuda_api;
}

CuDNNThreadEntry::~CuDNNThreadEntry() {
  CUDNN_CALL(cudnnDestroy(handle));
}

typedef dmlc::ThreadLocalStore<CuDNNThreadEntry> CuDNNThreadStore;

CuDNNThreadEntry* CuDNNThreadEntry::ThreadLocal() {
  return CuDNNThreadStore::Get();
}

// ConvEntry

ConvEntry::ConvEntry() {
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
}

ConvEntry::~ConvEntry() {
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CleanWorkspace();
}

void ConvEntry::UpdateWorkspace(const size_t wsize) {
  if (workspace_size < wsize) {
    if (workspace != nullptr) {
      CleanWorkspace();
    }
    workspace_size = wsize;
    workspace = cuda_api->AllocWorkspace(ctx, workspace_size);
  }
}

void ConvEntry::CleanWorkspace() {
  if (workspace) cuda_api->FreeWorkspace(ctx, workspace);
  workspace_size = 0;
}

}  // namespace contrib
}  // namespace tvm
