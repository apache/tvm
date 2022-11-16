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
 * \file src/runtime/contrib/arm_compute_lib/acl_utils.cc
 * \brief Utils and common functions for the interface.
 */

#include "acl_utils.h"

#include <arm_compute/runtime/OffsetLifetimeManager.h>
#include <arm_compute/runtime/PoolManager.h>
#include <tvm/runtime/data_type.h>

namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

void CheckACLError(const arm_compute::Status& status) {
  ICHECK(status.error_code() == arm_compute::ErrorCode::OK)
      << "ACL: " << status.error_description();
}

arm_compute::Tensor MakeACLTensor(const JSONGraphNode& tensor_rep, void* data,
                                  const DLTensor* scale, const DLTensor* offset,
                                  bool apply_dim_correction, bool increase_dim_unit,
                                  uint32_t entry_index) {
  arm_compute::Tensor tensor;
  std::vector<int64_t> shape = tensor_rep.GetOpShape()[entry_index];
  DLDataType dtype = tensor_rep.GetOpDataType()[entry_index];
  arm_compute::TensorInfo info =
      MakeACLTensorInfo(shape, dtype, scale, offset, apply_dim_correction, increase_dim_unit);
  info.set_is_resizable(false);
  tensor.allocator()->init(info);
  if (data != nullptr) {
    CheckACLError(tensor.allocator()->import_memory(data));
  }
  return tensor;
}

arm_compute::TensorInfo MakeACLTensorInfo(const std::vector<int64_t>& shape,
                                          const DLDataType& dtype, const DLTensor* scale,
                                          const DLTensor* offset, bool apply_dim_correction,
                                          bool increase_dim_unit) {
  arm_compute::TensorShape acl_shape;
  for (unsigned int i = shape.size(); i > 0; --i) {
    acl_shape.set(shape.size() - i, shape[i - 1], apply_dim_correction, increase_dim_unit);
  }
  arm_compute::DataType acl_dtype = MakeACLDataType(dtype);
  arm_compute::TensorInfo info(acl_shape, 1, acl_dtype, arm_compute::DataLayout::NHWC);

  // If scale and offset provided create quantized ACL tensor.
  if (scale != nullptr && offset != nullptr) {
    std::vector<float> scale_data = GetVectorFromDLTensor<float>(scale);
    std::vector<int> offset_data = GetVectorFromDLTensor<int>(offset);
    ICHECK(scale_data.size() == 1 && offset_data.size() == 1)
        << "Currently only per-layer quantization is supported in the Arm Compute Library runtime.";
    arm_compute::QuantizationInfo qinfo(scale_data[0], offset_data[0]);
    info.set_quantization_info(qinfo);
  }

  return info;
}

std::shared_ptr<arm_compute::MemoryManagerOnDemand> MakeACLMemoryManager() {
  auto lifetime_mgr = std::make_shared<arm_compute::OffsetLifetimeManager>();
  auto pool_mgr = std::make_shared<arm_compute::PoolManager>();
  return std::make_shared<arm_compute::MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);
}

arm_compute::PadStrideInfo MakeACLPadStride(const std::vector<std::string>& pad,
                                            const std::vector<std::string>& stride,
                                            bool ceil_mode) {
  int pad_0 = 0, pad_1 = 0, pad_2 = 0, pad_3 = 0;
  int stride_0 = std::stoi(stride[0]), stride_1 = std::stoi(stride[1]);
  auto dimensions_rounding = arm_compute::DimensionRoundingType::FLOOR;
  size_t size = pad.size();
  if (size == 1) {
    int pad_v = std::stoi(pad[0]);
    pad_0 = pad_v;
    pad_1 = pad_v;
    pad_2 = pad_v;
    pad_3 = pad_v;
  } else if (size == 2) {
    // TVM: height, width -> ACL: left, right, top, bottom
    int pad_h = std::stoi(pad[0]);
    int pad_w = std::stoi(pad[1]);
    pad_0 = pad_w;
    pad_1 = pad_w;
    pad_2 = pad_h;
    pad_3 = pad_h;
  } else if (size == 4) {
    // TVM: top, left, bottom, right -> ACL: left, right, top, bottom
    pad_0 = std::stoi(pad[1]);
    pad_1 = std::stoi(pad[3]);
    pad_2 = std::stoi(pad[0]);
    pad_3 = std::stoi(pad[2]);
  } else {
    LOG(FATAL) << "Unsupported padding dimensions";
  }

  if (ceil_mode) {
    dimensions_rounding = arm_compute::DimensionRoundingType::CEIL;
  }

  return arm_compute::PadStrideInfo(stride_0, stride_1, pad_0, pad_1, pad_2, pad_3,
                                    dimensions_rounding);
}

arm_compute::DataType MakeACLDataType(const DLDataType& data_type) {
  if (data_type.code == DLDataTypeCode::kDLFloat && data_type.bits == 32) {
    return arm_compute::DataType::F32;
  } else if (data_type.code == DLDataTypeCode::kDLUInt && data_type.bits == 8) {
    return arm_compute::DataType::QASYMM8;
  } else if (data_type.code == DLDataTypeCode::kDLInt && data_type.bits == 8) {
    return arm_compute::DataType::QASYMM8_SIGNED;
  } else if (data_type.code == DLDataTypeCode::kDLInt && data_type.bits == 32) {
    return arm_compute::DataType::S32;
  } else {
    LOG(FATAL) << "Datatype " << data_type << " unsupported by ACL runtime";
    return arm_compute::DataType::UNKNOWN;
  }
}

arm_compute::ActivationLayerInfo MakeACLActivationInfo(const std::string& activation_type) {
  auto act_func = arm_compute::ActivationLayerInfo::ActivationFunction::IDENTITY;
  if (activation_type == "relu") {
    act_func = arm_compute::ActivationLayerInfo::ActivationFunction::RELU;
  } else {
    LOG(FATAL) << "Activation " << activation_type << " unsupported by ACL runtime";
  }
  return {act_func};
}

template <typename T>
std::vector<T> GetVectorFromDLTensor(const DLTensor* tensor) {
  ICHECK(tensor) << "Cannot convert a nullptr";
  int len = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    len *= tensor->shape[i];
  }
  T* data = static_cast<T*>(tensor->data);
  return std::vector<T>(data, data + len);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
