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
 * \file utils.h
 * \brief Util functions for pytorch tvm interaction.
 */

#ifndef TVM_CONTRIB_TORCH_UTILS_H_
#define TVM_CONTRIB_TORCH_UTILS_H_

#include <dlpack/dlpack.h>
#include <torch/script.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#ifdef PT_TVMDSOOP_ENABLE_GPU
#include <cuda_runtime.h>
#endif

#include <string>
#include <vector>

namespace tvm {
namespace contrib {
namespace pytorch {

inline bool GetTvmDtype(const caffe2::TypeMeta& dtype, DLDataType* res) noexcept {
  if (dtype == torch::kFloat16) {
    *res = {kDLFloat, 16, 1};
  } else if (dtype == torch::kFloat32) {
    *res = {kDLFloat, 32, 1};
  } else if (dtype == torch::kFloat64) {
    *res = {kDLFloat, 64, 1};
  } else if (dtype == torch::kInt8) {
    *res = {kDLInt, 8, 1};
  } else if (dtype == torch::kInt16) {
    *res = {kDLInt, 16, 1};
  } else if (dtype == torch::kInt32) {
    *res = {kDLInt, 32, 1};
  } else if (dtype == torch::kInt64) {
    *res = {kDLInt, 64, 1};
  } else if (dtype == torch::kUInt8) {
    *res = {kDLUInt, 8, 1};
  } else if (dtype == torch::kBool) {
    *res = {kDLInt, 1, 1};
  } else {
    return false;
  }
  return true;
}

inline bool GetTvmDtype(const caffe2::TypeMeta& dtype, tvm::runtime::DataType* res) noexcept {
  DLDataType dlpack_dtype;

  if (!GetTvmDtype(dtype, &dlpack_dtype)) {
    return false;
  }
  *res = tvm::runtime::DataType(dlpack_dtype);
  return true;
}

inline bool GetTorchDtype(const DLDataType& dtype, c10::ScalarType* res) noexcept {
  if (dtype.lanes != 1) {
    // only scalar type
    return false;
  }
  if (dtype.code == kDLFloat) {
    if (dtype.bits == 16) {
      *res = torch::kFloat16;
    } else if (dtype.bits == 32) {
      *res = torch::kFloat32;
    } else if (dtype.bits == 64) {
      *res = torch::kFloat64;
    } else {
      return false;
    }
  } else if (dtype.code == kDLInt) {
    if (dtype.bits == 16) {
      *res = torch::kInt16;
    } else if (dtype.bits == 32) {
      *res = torch::kInt32;
    } else if (dtype.bits == 64) {
      *res = torch::kInt64;
    } else if (dtype.bits == 1) {
      *res = torch::kBool;
    } else {
      return false;
    }
  } else if (dtype.code == kDLUInt) {
    if (dtype.bits == 8) {
      *res = torch::kUInt8;
    } else if (dtype.bits == 1) {
      *res = torch::kBool;
    } else {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

inline bool GetTorchDtype(const tvm::runtime::DataType& dtype, c10::ScalarType* res) noexcept {
  using tvm::runtime::DataType;
  if (dtype == DataType::Float(16)) {
    *res = torch::kFloat16;
  } else if (dtype == DataType::Float(32)) {
    *res = torch::kFloat32;
  } else if (dtype == DataType::Float(64)) {
    *res = torch::kFloat64;
  } else if (dtype == DataType::Int(32)) {
    *res = torch::kInt32;
  } else if (dtype == DataType::Int(64)) {
    *res = torch::kInt64;
  } else if (dtype == DataType::Int(1)) {
    *res = torch::kBool;
  } else if (dtype == DataType::Int(8)) {
    *res = torch::kInt8;
  } else if (dtype == DataType::Int(16)) {
    *res = torch::kInt16;
  } else if (dtype == DataType::UInt(8)) {
    *res = torch::kUInt8;
  } else if (dtype == DataType::Bool()) {
    *res = torch::kBool;
  } else {
    return false;
  }
  return true;
}

// Buffer information used for actual computation.
// Each buffer is associated with one PyTorch tensor
// whose underlying buffer is record into "origin_buf".
// For input tensor, we copy data from origin_buf to buf
// and for output tensor, copy data from buf to origin_buf
class TensorAsBuf {
 public:
  explicit TensorAsBuf(const at::Tensor& tensor)
      : pt_device_type_(tensor.device().type()),
        device_id_(tensor.device().index()),
        origin_shape_(tensor.sizes().begin(), tensor.sizes().end()) {
    CHECK(pt_device_type_ == torch::kCUDA || pt_device_type_ == torch::kCPU);
    device_type_ = (pt_device_type_ == torch::kCUDA ? kDLCUDA : kDLCPU);

    char* buf = static_cast<char*>(tensor.data_ptr());
    this->origin_buf_ = buf;
    this->size_ = tensor.nbytes();

    // const int alignment = 64;
    const int alignment = tvm::runtime::kAllocAlignment;
    char* aligned = reinterpret_cast<char*>(((uint64_t)buf + alignment - 1) & (~(alignment - 1)));
    if (buf == aligned) {
      this->tensor_ = tensor;
      this->buf_ = buf;
      this->offset_ = 0;
    } else {
      const auto options =
          torch::TensorOptions().dtype(tensor.dtype()).device(pt_device_type_, device_id_);
      this->inline_tensor_ =
          torch::empty({static_cast<int64_t>(tensor.nbytes() + alignment)}, options);
      this->tensor_ = this->inline_tensor_;

      buf = static_cast<char*>(this->tensor_.data_ptr());
      char* buf_aligned = reinterpret_cast<char*>(((uint64_t)buf + alignment) & (~(alignment - 1)));
      this->buf_ = buf;
      this->offset_ = buf_aligned - buf;
    }
  }

  void CopyToOrigin() {
    if (buf_ == origin_buf_) {
      return;
    }
    if (device_type_ == kDLCPU) {
      memcpy(origin_buf_, buf_ + offset_, size_);
#ifdef PT_TVMDSOOP_ENABLE_GPU
    } else if (device_type_ == kDLCUDA) {
      cudaMemcpy(origin_buf_, buf_ + offset_, size_, cudaMemcpyDeviceToDevice);
#endif
    } else {
      LOG(FATAL) << "Only support CPU and CUDA now. Device " << device_type_
                 << " is not implemented currently";
    }
  }

  void CopyFromOrigin() {
    if (buf_ == origin_buf_) {
      return;
    }
    if (device_type_ == kDLCPU) {
      memcpy(buf_ + offset_, origin_buf_, size_);
#ifdef PT_TVMDSOOP_ENABLE_GPU
    } else if (device_type_ == kDLCUDA) {
      cudaMemcpy(buf_ + offset_, origin_buf_, size_, cudaMemcpyDeviceToDevice);
#endif
    } else {
      LOG(FATAL) << "Only support CPU and CUDA now. Device " << device_type_
                 << " is not implemented currently";
    }
  }

  // Create DLPack tensor from PyTorch tensor
  void MakeDLTensor(DLTensor* out) {
    const DLDevice dl_ctx{DLDeviceType(device_type_), device_id_};
    DLDataType dlpack_type;
    const auto& tensor = this->tensor_;
    CHECK(GetTvmDtype(tensor.dtype(), &dlpack_type));

    out->device = dl_ctx;
    out->ndim = origin_shape_.size();
    out->shape = origin_shape_.data();
    out->strides = nullptr;
    out->byte_offset = 0;
    out->dtype = dlpack_type;
    out->data = buf_ + offset_;
  }

  std::string DebugString() {
    std::stringstream ss;
    ss << "dl device: " << device_type_ << "\npt device: " << static_cast<int>(pt_device_type_)
       << "\ndevice_id: " << device_id_ << "\nsize: " << size_ << "\noffset: " << offset_
       << "\nshape:";
    for (auto dim : origin_shape_) {
      ss << ' ' << dim;
    }
    ss << std::endl;
    return ss.str();
  }

 private:
  DLDeviceType device_type_;
  c10::DeviceType pt_device_type_;
  int device_id_;

  at::Tensor inline_tensor_;
  at::Tensor tensor_;
  size_t size_;
  size_t offset_;

  std::vector<int64_t> origin_shape_;

  char* origin_buf_;
  char* buf_;
};
}  // namespace pytorch
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_TORCH_UTILS_H_
