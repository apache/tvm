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
 * \file src/runtime/contrib/acl/acl_utils.cc
 * \brief Utils and common functions for the interface.
 */

#include "acl_utils.h"

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace contrib {
namespace acl {

void CheckACLError(arm_compute::Status status) {
  CHECK(status.error_code() == arm_compute::ErrorCode::OK) << "ACL: " << status.error_description();
}

acl::Tensor MakeTensor(const api::JSONTensor& tensor_rep, void* data) {
  acl::Tensor tensor;
  acl::TensorInfo info = MakeTensorInfo(tensor_rep);
  tensor.allocator()->init(info);
  if (data != nullptr) {
    CheckACLError(tensor.allocator()->import_memory(data));
  }
  return tensor;
}

acl::TensorInfo MakeTensorInfo(const api::JSONTensor& tensor_rep) {
  return acl::TensorInfo(MakeTensorShape(tensor_rep.shape), 1, acl::DataType::F32,
                         acl::DataLayout::NHWC);
}

arm_compute::TensorShape MakeTensorShape(const std::vector<int>& shape) {
  arm_compute::TensorShape acl_shape;
  for (unsigned int i = shape.size(); i > 0; --i) {
    acl_shape.set(shape.size() - i, shape[i - 1]);
  }
  return acl_shape;
}

}  // namespace acl
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
