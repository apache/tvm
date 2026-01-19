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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/module.h>

#include <cmath>
#include <random>

#include "../src/runtime/vulkan/vulkan_device_api.h"

using tvm::runtime::memory::AllocatorType;
using tvm::runtime::memory::MemoryManager;
using tvm::runtime::memory::Storage;

class VulkanTextureCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bool enabled = tvm::runtime::RuntimeEnabled("vulkan");
    if (!enabled) {
      GTEST_SKIP() << "Skip texture copy test because Vulkan runtime is disabled.\n";
    }
  }
};

TEST_F(VulkanTextureCopyTest, ViewBufferAsBuffer) {
  using namespace tvm;
  std::vector<int64_t> shape{1, 16, 16, 8};
  std::vector<int64_t> same_shape{1, 8, 16, 16};
  auto cpu_arr = runtime::Tensor::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr_ret = runtime::Tensor::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});

  ffi::String mem_scope = "global";

  DLDevice cl_dev = {kDLVulkan, 0};
  auto allocator = MemoryManager::GetOrCreateAllocator(cl_dev, AllocatorType::kPooled);
  auto buffer = allocator->Alloc(cl_dev, ffi::Shape(shape), {kDLFloat, 32, 1});
  auto stor = Storage(buffer, allocator);

  auto vulkan_memobj = stor->AllocTensorScoped(0, ffi::Shape(shape), {kDLFloat, 32, 1}, mem_scope);
  auto vulkan_memview =
      stor->AllocTensorScoped(0, ffi::Shape(same_shape), {kDLFloat, 32, 1}, mem_scope);

  std::random_device dev;
  std::mt19937 mt(dev());
  std::uniform_real_distribution<> random(-10.0, 10.0);

  size_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    size *= static_cast<size_t>(shape[i]);
  }

  /* Check original object round trip */
  // Random initialize host pool storage
  for (size_t i = 0; i < size; i++) {
    static_cast<float*>(cpu_arr->data)[i] = random(mt);
  }
  // Copy to VulkanBuffer
  cpu_arr.CopyTo(vulkan_memobj);
  // Copy from VulkanBuffer
  vulkan_memobj.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }

  /* Check view object round trip */
  // Random initialize host pool storage
  for (size_t i = 0; i < size; i++) {
    static_cast<float*>(cpu_arr->data)[i] = random(mt);
  }
  // Copy to VulkanBuffer
  cpu_arr.CopyTo(vulkan_memview);
  // Copy from VulkanBuffer
  vulkan_memview.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }
}

TEST_F(VulkanTextureCopyTest, ViewBufferAsImage) {
  using namespace tvm;
  // Shape that doesn't cause padding for image row
  std::vector<int64_t> shape{1, 16, 16, 8, 4};
  std::vector<int64_t> same_shape{1, 8, 16, 16, 4};
  auto cpu_arr = runtime::Tensor::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr_ret = runtime::Tensor::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});

  DLDevice cl_dev = {kDLVulkan, 0};
  auto allocator = MemoryManager::GetOrCreateAllocator(cl_dev, AllocatorType::kPooled);
  auto buffer = allocator->Alloc(cl_dev, ffi::Shape(shape), {kDLFloat, 32, 1});
  auto stor = Storage(buffer, allocator);

  auto vulkan_buf_obj = stor->AllocTensorScoped(0, ffi::Shape(shape), {kDLFloat, 32, 1}, "global");
  auto vulkan_img_obj =
      stor->AllocTensorScoped(0, ffi::Shape(same_shape), {kDLFloat, 32, 1}, "global.texture");

  std::random_device dev;
  std::mt19937 mt(dev());
  std::uniform_real_distribution<> random(-10.0, 10.0);

  size_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    size *= static_cast<size_t>(shape[i]);
  }

  /* Check original object round trip */
  // Random initialize host pool storage
  for (size_t i = 0; i < size; i++) {
    static_cast<float*>(cpu_arr->data)[i] = random(mt);
  }
  // Copy to VulkanBuffer
  cpu_arr.CopyTo(vulkan_buf_obj);
  // Copy from VulkanBuffer
  vulkan_buf_obj.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }

  /* Check view object round trip */
  // Random initialize host pool storage
  for (size_t i = 0; i < size; i++) {
    static_cast<float*>(cpu_arr->data)[i] = random(mt);
  }
  // Copy to VulkanBuffer
  cpu_arr.CopyTo(vulkan_img_obj);
  // Copy from VulkanBuffer
  vulkan_img_obj.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }
}
