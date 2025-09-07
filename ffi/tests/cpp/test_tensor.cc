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
#include <gtest/gtest.h>
#include <tvm/ffi/container/tensor.h>

namespace {

using namespace tvm::ffi;

struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) { tensor->data = malloc(GetDataSize(*tensor)); }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

inline Tensor Empty(Shape shape, DLDataType dtype, DLDevice device) {
  return Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
}

TEST(Tensor, Basic) {
  Tensor nd = Empty(Shape({1, 2, 3}), DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  Shape shape = nd.shape();
  Shape strides = nd.strides();
  EXPECT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 2);
  EXPECT_EQ(shape[2], 3);
  EXPECT_EQ(strides.size(), 3);
  EXPECT_EQ(strides[0], 6);
  EXPECT_EQ(strides[1], 3);
  EXPECT_EQ(strides[2], 1);
  EXPECT_EQ(nd.dtype(), DLDataType({kDLFloat, 32, 1}));
  for (int64_t i = 0; i < shape.Product(); ++i) {
    reinterpret_cast<float*>(nd->data)[i] = static_cast<float>(i);
  }

  Any any0 = nd;
  Tensor nd2 = any0.as<Tensor>().value();
  EXPECT_EQ(nd2.shape(), shape);
  EXPECT_EQ(nd2.strides(), strides);
  EXPECT_EQ(nd2.dtype(), DLDataType({kDLFloat, 32, 1}));
  for (int64_t i = 0; i < shape.Product(); ++i) {
    EXPECT_EQ(reinterpret_cast<float*>(nd2->data)[i], i);
  }

  EXPECT_EQ(nd.IsContiguous(), true);
  EXPECT_EQ(nd2.use_count(), 3);
}

TEST(Tensor, DLPack) {
  Tensor tensor = Empty({1, 2, 3}, DLDataType({kDLInt, 16, 1}), DLDevice({kDLCPU, 0}));
  DLManagedTensor* dlpack = tensor.ToDLPack();
  EXPECT_EQ(dlpack->dl_tensor.ndim, 3);
  EXPECT_EQ(dlpack->dl_tensor.shape[0], 1);
  EXPECT_EQ(dlpack->dl_tensor.shape[1], 2);
  EXPECT_EQ(dlpack->dl_tensor.shape[2], 3);
  EXPECT_EQ(dlpack->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(dlpack->dl_tensor.dtype.bits, 16);
  EXPECT_EQ(dlpack->dl_tensor.dtype.lanes, 1);
  EXPECT_EQ(dlpack->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlpack->dl_tensor.device.device_id, 0);
  EXPECT_EQ(dlpack->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlpack->dl_tensor.strides[0], 6);
  EXPECT_EQ(dlpack->dl_tensor.strides[1], 3);
  EXPECT_EQ(dlpack->dl_tensor.strides[2], 1);
  EXPECT_EQ(tensor.use_count(), 2);
  {
    Tensor tensor2 = Tensor::FromDLPack(dlpack);
    EXPECT_EQ(tensor2.use_count(), 1);
    EXPECT_EQ(tensor2->data, tensor->data);
    EXPECT_EQ(tensor.use_count(), 2);
    EXPECT_EQ(tensor2.use_count(), 1);
  }
  EXPECT_EQ(tensor.use_count(), 1);
}

TEST(Tensor, DLPackVersioned) {
  DLDataType dtype = DLDataType({kDLFloat4_e2m1fn, 4, 1});
  EXPECT_EQ(GetDataSize(2, dtype), 2 * 4 / 8);
  Tensor tensor = Empty({2}, dtype, DLDevice({kDLCPU, 0}));
  DLManagedTensorVersioned* dlpack = tensor.ToDLPackVersioned();
  EXPECT_EQ(dlpack->version.major, DLPACK_MAJOR_VERSION);
  EXPECT_EQ(dlpack->version.minor, DLPACK_MINOR_VERSION);
  EXPECT_EQ(dlpack->dl_tensor.ndim, 1);
  EXPECT_EQ(dlpack->dl_tensor.shape[0], 2);
  EXPECT_EQ(dlpack->dl_tensor.dtype.code, kDLFloat4_e2m1fn);
  EXPECT_EQ(dlpack->dl_tensor.dtype.bits, 4);
  EXPECT_EQ(dlpack->dl_tensor.dtype.lanes, 1);
  EXPECT_EQ(dlpack->dl_tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dlpack->dl_tensor.device.device_id, 0);
  EXPECT_EQ(dlpack->dl_tensor.byte_offset, 0);
  EXPECT_EQ(dlpack->dl_tensor.strides[0], 1);

  EXPECT_EQ(tensor.use_count(), 2);
  {
    Tensor tensor2 = Tensor::FromDLPackVersioned(dlpack);
    EXPECT_EQ(tensor2.use_count(), 1);
    EXPECT_EQ(tensor2->data, tensor->data);
    EXPECT_EQ(tensor.use_count(), 2);
    EXPECT_EQ(tensor2.use_count(), 1);
  }
  EXPECT_EQ(tensor.use_count(), 1);
}
}  // namespace
