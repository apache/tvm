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
 * \file Use external cudnn utils function
 */

#include "cudnn_utils.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <vector>

namespace tvm {
namespace contrib {

// CuDNN Data Type
cudnnDataType_t CuDNNDataType::DLTypeToCuDNNType(const DLDataType& dtype) {
  switch (dtype.code) {
    case kDLInt:
      if (dtype.bits == 8 && dtype.lanes == 1)
        return CUDNN_DATA_INT8;
      else if (dtype.bits == 32 && dtype.lanes == 1)
        return CUDNN_DATA_INT32;
      else if (dtype.bits == 8 && dtype.lanes == 4)
        return CUDNN_DATA_INT8x4;
      else
        LOG(FATAL) << "Unsupported type";
      break;
    case kDLUInt:
      LOG(FATAL) << "Unsupported type";
      break;
    case kDLFloat:
      if (dtype.bits == 32 && dtype.lanes == 1)
        return CUDNN_DATA_FLOAT;
      else if (dtype.bits == 64 && dtype.lanes == 1)
        return CUDNN_DATA_DOUBLE;
      else if (dtype.bits == 16 && dtype.lanes == 1)
        return CUDNN_DATA_HALF;
      else
        LOG(FATAL) << "Unsupported type";
      break;
  }
  return CUDNN_DATA_FLOAT;
}

template <>
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

template <>
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
  auto func = runtime::Registry::Get("device_api.cuda");
  void* ret = (*func)();
  cuda_api = static_cast<runtime::DeviceAPI*>(ret);

  // If no CuDNN-capable device is present, allow the CuDNNThreadEntry
  // object to be created.  This is needed for
  // CuDNNThreadEntry::exists.
  {
    cudnnStatus_t create_res = cudnnCreate(&handle);
    if (create_res == CUDNN_STATUS_NOT_INITIALIZED) {
      return;
    }
    CUDNN_CALL(create_res);
  }

  CUDNN_CALL(cudnnSetStream(handle, stream));
  conv_entry.cuda_api = cuda_api;
}

CuDNNThreadEntry::~CuDNNThreadEntry() {}

typedef dmlc::ThreadLocalStore<CuDNNThreadEntry> CuDNNThreadStore;

CuDNNThreadEntry* CuDNNThreadEntry::ThreadLocal(bool check_exists) {
  auto* res = CuDNNThreadStore::Get();
  if (check_exists) {
    ICHECK(res->exists()) << "CUDNN_STATUS_NOT_INITIALIZED";
  }

  return res;
}

// ConvEntry

ConvEntry::ConvEntry() {
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc));
}

ConvEntry::~ConvEntry() {
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc));
  CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
  CleanWorkspace();
}

void ConvEntry::UpdateWorkspace(const size_t wsize) {
  if (workspace_size < wsize) {
    if (workspace != nullptr) {
      CleanWorkspace();
    }
    workspace_size = wsize;
    workspace = cuda_api->AllocWorkspace(device, workspace_size);
  }
}

void ConvEntry::CleanWorkspace() {
  if (workspace) cuda_api->FreeWorkspace(device, workspace);
  workspace_size = 0;
}

void SetConvDescriptors(CuDNNThreadEntry* entry_ptr, int format, int dims, int groups,
                        const int pad[], const int stride[], const int dilation[], int64_t x_dim[],
                        int64_t w_dim[], int64_t y_dim[], DLDataType data_dtype,
                        const std::string& conv_dtype) {
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Data Type
  entry_ptr->conv_entry.data_type =
      CuDNNDataType::DLTypeToCuDNNType(runtime::String2DLDataType(conv_dtype));

  cudnnDataType_t cudnn_data_type = CuDNNDataType::DLTypeToCuDNNType(data_dtype);

  // Dims includes N and C
  int full_dims = dims + 2;

  std::vector<int> dim(full_dims);
  std::vector<int> tensor_stride(full_dims);

  // Note: For 2D tenor, using ND setters causes CUDNN_STATUS_NOT_SUPPORTED error
  // in following cudnnGetConvolutionForwardWorkspaceSize() when data type is fp16, int

  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  if (dims == 2) {
    // Set Desc
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        entry_ptr->conv_entry.conv_desc, pad[0], pad[1], stride[0], stride[1], dilation[0],
        dilation[1], entry_ptr->conv_entry.mode, entry_ptr->conv_entry.data_type));
    int ni, ci, hi, wi;
    if (entry_ptr->conv_entry.tensor_format == CUDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }

    // Set Input
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.tensor_format, cudnn_data_type,
        static_cast<int>(x_dim[ni]), static_cast<int>(x_dim[ci]), static_cast<int>(x_dim[hi]),
        static_cast<int>(x_dim[wi])));
    // Set Filter
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        entry_ptr->conv_entry.filter_desc, cudnn_data_type, entry_ptr->conv_entry.tensor_format,
        static_cast<int>(w_dim[ni]), static_cast<int>(w_dim[ci]), static_cast<int>(w_dim[hi]),
        static_cast<int>(w_dim[wi])));
    // Set Output
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->conv_entry.output_desc, entry_ptr->conv_entry.tensor_format, cudnn_data_type,
        static_cast<int>(y_dim[ni]), static_cast<int>(y_dim[ci]), static_cast<int>(y_dim[hi]),
        static_cast<int>(y_dim[wi])));
  } else {
    ICHECK_EQ(format, 0) << "Use of layout CUDNN_TENSOR_NHWC is supported only for 4-D tensors.";

    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc, dims, pad, stride,
                                               dilation, entry_ptr->conv_entry.mode,
                                               entry_ptr->conv_entry.data_type));

    // Set Filter
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(w_dim[i]);
    }
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc, cudnn_data_type,
                                          entry_ptr->conv_entry.tensor_format, full_dims,
                                          dim.data()));
    // Set Input
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(x_dim[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc, cudnn_data_type,
                                          full_dims, dim.data(), tensor_stride.data()));
    // Set Output
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(y_dim[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc, cudnn_data_type,
                                          full_dims, dim.data(), tensor_stride.data()));
  }

  if (cudnnGetVersion() > 7000) {
    CUDNN_CALL(cudnnSetConvolutionMathType(entry_ptr->conv_entry.conv_desc, CUDNN_TENSOR_OP_MATH))
  }
}

// SoftmaxEntry

SoftmaxEntry::SoftmaxEntry() { CUDNN_CALL(cudnnCreateTensorDescriptor(&shape_desc)); }

SoftmaxEntry::~SoftmaxEntry() { CUDNN_CALL(cudnnDestroyTensorDescriptor(shape_desc)); }

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.exists").set_body_typed([]() -> bool {
  return CuDNNThreadEntry::ThreadLocal(false)->exists();
});

}  // namespace contrib
}  // namespace tvm
