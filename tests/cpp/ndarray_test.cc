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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/ndarray.h>

using namespace tvm;

TEST(NDArrayTest, IsContiguous_ContiguousStride) {
  auto array = runtime::NDArray::Empty({5, 10}, DataType::Float(32), {kDLCPU});
  DLManagedTensor* managed_tensor = array.ToDLPack();

  int64_t strides[] = {10, 1};
  managed_tensor->dl_tensor.strides = strides;

  ICHECK(runtime::IsContiguous(managed_tensor->dl_tensor));

  managed_tensor->deleter(managed_tensor);
}

TEST(NDArrayTest, IsContiguous_NullStride) {
  auto array = runtime::NDArray::Empty({5, 10}, DataType::Float(32), {kDLCPU});
  DLManagedTensor* managed_tensor = array.ToDLPack();

  managed_tensor->dl_tensor.strides = nullptr;

  ICHECK(runtime::IsContiguous(managed_tensor->dl_tensor));

  managed_tensor->deleter(managed_tensor);
}

TEST(NDArrayTest, IsContiguous_AnyStrideForSingular) {
  auto array = runtime::NDArray::Empty({5, 1, 10}, DataType::Float(32), {kDLCPU});
  DLManagedTensor* managed_tensor = array.ToDLPack();

  int64_t strides[] = {10, 1, 1};  // strides[1] is normalized to 1 because shape[1] == 1.
  managed_tensor->dl_tensor.strides = strides;

  ICHECK(runtime::IsContiguous(managed_tensor->dl_tensor));

  managed_tensor->dl_tensor.strides = nullptr;
  managed_tensor->deleter(managed_tensor);
}

TEST(NDArrayTest, IsContiguous_UncontiguousStride) {
  auto array = runtime::NDArray::Empty({5, 1, 10}, DataType::Float(32), {kDLCPU});
  DLManagedTensor* managed_tensor = array.ToDLPack();

  int64_t strides[] = {1, 1, 1};
  managed_tensor->dl_tensor.strides = strides;

  ICHECK(!runtime::IsContiguous(managed_tensor->dl_tensor));

  managed_tensor->dl_tensor.strides = nullptr;
  managed_tensor->deleter(managed_tensor);
}
