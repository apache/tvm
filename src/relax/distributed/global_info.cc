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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/distributed/global_info.h>

#include <limits>

namespace tvm {
namespace relax {
namespace distributed {

TVM_FFI_STATIC_INIT_BLOCK() { DeviceMeshNode::RegisterReflection(); }

namespace {

int64_t MeshSize(const ffi::Shape& shape) {
  bool empty = false;
  for (int64_t dim : shape) {
    TVM_FFI_CHECK_GE(dim, 0, ValueError) << "Device mesh dimensions must be non-negative";
    empty |= dim == 0;
  }
  if (empty) return 0;
  int64_t size = 1;
  for (int64_t dim : shape) {
    TVM_FFI_CHECK_LE(size, std::numeric_limits<int64_t>::max() / dim, ValueError)
        << "Device mesh shape product exceeds int64";
    size *= dim;
  }
  return size;
}

}  // namespace

DeviceMesh::DeviceMesh(ffi::Shape shape, ffi::Array<int64_t> device_ids) {
  int64_t size = MeshSize(shape);
  ffi::ObjectPtr<DeviceMeshNode> n = ffi::make_object<DeviceMeshNode>();
  TVM_FFI_ICHECK_EQ(static_cast<uint64_t>(size), device_ids.size())
      << "The number of device ids must match the product of the shape";
  n->shape = std::move(shape);
  n->device_ids = std::move(device_ids);
  data_ = std::move(n);
}

DeviceMesh::DeviceMesh(ffi::Shape shape, Range device_range) {
  ffi::ObjectPtr<DeviceMeshNode> n = ffi::make_object<DeviceMeshNode>();
  ffi::Array<int64_t> device_ids;
  const auto* start = device_range->min.as<IntImmNode>();
  const auto* extent = device_range->extent.as<IntImmNode>();
  TVM_FFI_CHECK(start && extent, ValueError) << "Device mesh range must be constant";
  int64_t range_start = start->value;
  int64_t range_extent = extent->value;
  TVM_FFI_CHECK_GE(range_extent, 0, ValueError) << "Device mesh range extent must be non-negative";
  TVM_FFI_ICHECK_EQ(MeshSize(shape), range_extent)
      << "The number of device ids must match the product of the shape";
  if (range_extent > 0) {
    TVM_FFI_CHECK_LE(range_start, std::numeric_limits<int64_t>::max() - (range_extent - 1),
                     ValueError)
        << "Device mesh range exceeds int64";
  }
  for (int64_t i = 0; i < range_extent; ++i) {
    device_ids.push_back(range_start + i);
  }
  n->device_ids = std::move(device_ids);
  n->shape = std::move(shape);
  n->device_range = std::move(device_range);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.distributed.DeviceMesh",
      [](ffi::Shape shape, ffi::Array<int64_t> device_ids, ffi::Optional<Range> device_range) {
        if (device_range.has_value())
          return DeviceMesh(shape, device_range.value());
        else
          return DeviceMesh(shape, device_ids);
      });
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
