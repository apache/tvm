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
  CHECK(status.error_code() == arm_compute::ErrorCode::OK) << "ACL: " << status.error_description();
}

arm_compute::Tensor MakeTensor(const JSONGraphNode& tensor_rep, void* data) {
  CHECK(tensor_rep.GetOpType() == "input" || tensor_rep.GetOpType() == "const");
  arm_compute::Tensor tensor;
  arm_compute::TensorInfo info = MakeTensorInfo(tensor_rep.GetOpShape()[0]);
  tensor.allocator()->init(info);
  if (data != nullptr) {
    CheckACLError(tensor.allocator()->import_memory(data));
  }
  return tensor;
}

arm_compute::Tensor MakeOutputTensor(const std::vector<int64_t>& shape) {
  arm_compute::Tensor tensor;
  tensor.allocator()->init(MakeTensorInfo(shape));
  return tensor;
}

arm_compute::TensorInfo MakeTensorInfo(const std::vector<int64_t>& shape) {
  arm_compute::TensorShape acl_shape = MakeTensorShape(shape);
  return arm_compute::TensorInfo(acl_shape, 1, arm_compute::DataType::F32,
                                 arm_compute::DataLayout::NHWC);
}

arm_compute::TensorShape MakeTensorShape(const std::vector<int64_t>& shape) {
  arm_compute::TensorShape acl_shape;
  for (unsigned int i = shape.size(); i > 0; --i) {
    acl_shape.set(shape.size() - i, shape[i - 1]);
  }
  return acl_shape;
}

std::shared_ptr<arm_compute::MemoryManagerOnDemand> MakeMemoryManager() {
  auto lifetime_mgr = std::make_shared<arm_compute::OffsetLifetimeManager>();
  auto pool_mgr = std::make_shared<arm_compute::PoolManager>();
  return std::make_shared<arm_compute::MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);
}

arm_compute::PadStrideInfo ToACLPadStride(const std::vector<std::string>& pad,
                                          const std::vector<std::string>& stride) {
  int pad_0 = 0, pad_1 = 0, pad_2 = 0, pad_3 = 0;
  int stride_0 = std::stoi(stride[0]), stride_1 = std::stoi(stride[1]);
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

  return arm_compute::PadStrideInfo(stride_0, stride_1, pad_0, pad_1, pad_2, pad_3,
                                    arm_compute::DimensionRoundingType::FLOOR);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
