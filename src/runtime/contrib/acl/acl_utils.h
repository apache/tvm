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
 * \file src/runtime/contrib/acl/acl_utils.h
 * \brief Utils and common functions for the interface.
 */

#ifndef TVM_RUNTIME_CONTRIB_ACL_ACL_UTILS_H_
#define TVM_RUNTIME_CONTRIB_ACL_ACL_UTILS_H_

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/Tensor.h>

#include <vector>

#include "../../../relay/backend/contrib/acl/acl_api.h"

namespace tvm {
namespace runtime {
namespace contrib {
namespace acl {

namespace api = relay::contrib::acl;
namespace acl = arm_compute;

/*!
 * \brief Check if there are any errors from acl and forward them to TVM.
 *
 * \param status status of called function.
 *
 * Status values:
 * - 0 => OK
 * - 1 => RUNTIME_ERROR
 * - 2 => UNSUPPORTED_EXTENSION_USE
 */
void CheckACLError(acl::Status status);

/*!
 * \brief Make an acl tensor from JSON tensor representation.
 *
 * \param tensor_rep A JSON tensor representation.
 * \param data (optional) Initialize the tensor with memory.
 * \return arm_compute::Tensor.
 */
acl::Tensor MakeTensor(const api::JSONTensor& tensor_rep, void* data = nullptr);

/*!
 * \brief Make an acl tensor info object from JSON tensor
 * representation.
 *
 * \param tensor_rep A JSON tensor representation.
 * \return arm_compute::TensorInfo.
 */
acl::TensorInfo MakeTensorInfo(const api::JSONTensor& tensor_rep);

/*!
 * \brief Convert vector object to acl TensorShape.
 * \note This requires reversing the given vector.
 *
 * \param shape The shape of the tensor as a vector.
 * \return acl TensorShape.
 */
acl::TensorShape MakeTensorShape(const std::vector<int>& shape);

}  // namespace acl
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_ACL_ACL_UTILS_H_
