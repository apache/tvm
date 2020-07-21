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
 * \return arm_compute::Tensor.
 */
arm_compute::Tensor MakeTensor(const JSONGraphNode& tensor_rep, void* data = nullptr);

/*!
 * \brief Make an acl tensor from type and shape, without having a JSON representation.
 *
 * \param shape The shape of the tensor to create.
 * \return arm_compute::Tensor.
 */
arm_compute::Tensor MakeOutputTensor(const std::vector<int64_t>& shape);

/*!
 * \brief Make an acl tensor info object from JSON tensor
 * representation.
 *
 * \param shape The shape of the tensor to create.
 * \return arm_compute::TensorInfo.
 */
arm_compute::TensorInfo MakeTensorInfo(const std::vector<int64_t>& shape);

/*!
 * \brief Convert vector object to acl TensorShape.
 * \note This requires reversing the given vector.
 *
 * \param shape The shape of the tensor as a vector.
 * \return arm_compute::TensorShape.
 */
arm_compute::TensorShape MakeTensorShape(const std::vector<int64_t>& shape);

/*!
 * \brief Create a memory manager for use with a layer that
 * requires working memory.
 *
 * \return reference counted memory manager.
 */
std::shared_ptr<arm_compute::MemoryManagerOnDemand> MakeMemoryManager();

/*!
 * \brief Convert TVM padding and stride format to acl PadStrideInfo.
 *
 * \param pad The pad vector.
 * \param stride The stride vector.
 * \return arm_compute::PadStrideInfo
 */
arm_compute::PadStrideInfo ToACLPadStride(const std::vector<std::string>& pad,
                                          const std::vector<std::string>& stride);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_ARM_COMPUTE_LIB_ACL_UTILS_H_
