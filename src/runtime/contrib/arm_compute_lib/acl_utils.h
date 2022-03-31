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
 * \file src/runtime/contrib/arm_compute_lib/acl_utils.h
 * \brief Utils and common functions for the interface.
 */

#ifndef TVM_RUNTIME_CONTRIB_ARM_COMPUTE_LIB_ACL_UTILS_H_
#define TVM_RUNTIME_CONTRIB_ARM_COMPUTE_LIB_ACL_UTILS_H_

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>
#include <string>
#include <vector>

#include "../json/json_node.h"

namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

/*!
 * \brief Check if there are any errors from acl and forward them to TVM.
 *
 * Status values:
 * - 0 => OK
 * - 1 => RUNTIME_ERROR
 * - 2 => UNSUPPORTED_EXTENSION_USE
 *
 * \param status status of called function.
 */
void CheckACLError(const arm_compute::Status& status);

/*!
 * \brief Make an acl tensor from JSON tensor representation.
 *
 * \param tensor_rep A JSON tensor representation.
 * \param data (optional) Initialize the tensor with memory.
 * \param scale (optional) The quantization scale.
 * \param offset (optional) The quantization offset.
 * \return arm_compute::Tensor.
 */
arm_compute::Tensor MakeACLTensor(const JSONGraphNode& tensor_rep, void* data = nullptr,
                                  const DLTensor* scale = nullptr, const DLTensor* offset = nullptr,
                                  bool apply_dim_correction = true, bool increase_dim_unit = true,
                                  uint32_t entry_index = 0);

/*!
 * \brief Make an acl tensor info object from JSON tensor
 * representation.
 *
 * \param shape The shape of the tensor to create.
 * \param dtype The data type of the tensor to create.
 * \param scale (optional) The quantization scale.
 * \param offset (optional) The quantization offset.
 * \return arm_compute::TensorInfo.
 */
arm_compute::TensorInfo MakeACLTensorInfo(const std::vector<int64_t>& shape,
                                          const DLDataType& dtype, const DLTensor* scale = nullptr,
                                          const DLTensor* offset = nullptr,
                                          bool apply_dim_correction = true,
                                          bool increase_dim_unit = true);

/*!
 * \brief Create a memory manager for use with a layer that
 * requires working memory.
 *
 * \return reference counted memory manager.
 */
std::shared_ptr<arm_compute::MemoryManagerOnDemand> MakeACLMemoryManager();

/*!
 * \brief Convert TVM padding and stride format to acl PadStrideInfo.
 *
 * \param pad The pad vector.
 * \param stride The stride vector.
 * \param ceil_mode Dimensions rounding.
 * \return arm_compute::PadStrideInfo
 */
arm_compute::PadStrideInfo MakeACLPadStride(const std::vector<std::string>& pad,
                                            const std::vector<std::string>& stride,
                                            bool ceil_mode = false);

/*!
 * \brief Convert DLDataType to arm_compute::DataType.
 *
 * \param data_type The data type to convert.
 * \return arm_compute::DataType.
 */
arm_compute::DataType MakeACLDataType(const DLDataType& data_type);

/*!
 * \brief Convert string to arm_compute::ActivationLayerInfo
 *
 * \param activation_type A string representing activation function.
 * Currently supports the following options: "relu".
 * \return arm_compute::ActivationLayerInfo.
 */
arm_compute::ActivationLayerInfo MakeACLActivationInfo(const std::string& activation_type);

/*!
 * \brief Get a vector from DLTensor data.
 * \note Performs a copy of data.
 *
 * \tparam T The type of the vector.
 * \param tensor The tensor to convert.
 * \return Vector of type T.
 */
template <typename T>
std::vector<T> GetVectorFromDLTensor(const DLTensor* tensor);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_ARM_COMPUTE_LIB_ACL_UTILS_H_
