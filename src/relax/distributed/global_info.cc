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

#include <tvm/relax/distributed/global_info.h>

namespace tvm {
namespace relax {
namespace distributed {

DeviceMesh::DeviceMesh(ShapeTuple shape, Array<Integer> device_ids) {
  device_ids = device_ids.Map(
      [](const Integer& id) -> Integer { return IntImm(DataType::Int(64), id->value); });

  int prod = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    prod *= shape[i];
  }
  CHECK_EQ(prod, static_cast<int>(device_ids.size()))
      << "The number of device ids must match the product of the shape";

  ObjectPtr<DeviceMeshNode> n = make_object<DeviceMeshNode>();
  n->shape = std::move(shape);
  n->device_ids = std::move(device_ids);
  data_ = std::move(n);
}

DeviceMesh::DeviceMesh(ShapeTuple shape, Range device_range) {
  int64_t range_start = device_range->min.as<IntImmNode>()->value;
  int64_t range_extent = device_range->extent.as<IntImmNode>()->value;

  device_range = Range::FromMinExtent(IntImm(DataType::Int(64), range_start),
                                      IntImm(DataType::Int(64), range_extent));

  Array<Integer> device_ids;
  for (int64_t i = range_start; i < range_start + range_extent; i++) {
    device_ids.push_back(IntImm(DataType::Int(64), i));
  }
  int prod = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    prod *= shape[i];
  }
  CHECK_EQ(prod, static_cast<int>(device_ids.size()))
      << "The number of device ids must match the product of the shape";

  ObjectPtr<DeviceMeshNode> n = make_object<DeviceMeshNode>();
  n->device_ids = std::move(device_ids);
  n->shape = std::move(shape);
  n->device_range = std::move(device_range);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DeviceMeshNode);
TVM_REGISTER_GLOBAL("relax.distributed.DeviceMesh")
    .set_body_typed([](ShapeTuple shape, Variant<Array<Integer>, Range> device_ids) {
      if (auto arr = device_ids.as<Array<Integer>>()) {
        return DeviceMesh(shape, arr.value());
      } else if (auto range = device_ids.as<Range>()) {
        return DeviceMesh(shape, range.value());
      } else {
        LOG(FATAL) << "Unreachable, "
                   << "variant must contain one of the allowed types";
      }
    });

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
