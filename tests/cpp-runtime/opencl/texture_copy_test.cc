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
#include <tvm/ffi/function.h>
#include <tvm/runtime/logging.h>

#include <cmath>
#include <random>

#include "../src/runtime/opencl/opencl_common.h"

using tvm::runtime::kAllocAlignment;
using tvm::runtime::memory::AllocatorType;
using tvm::runtime::memory::Buffer;
using tvm::runtime::memory::MemoryManager;
using tvm::runtime::memory::Storage;

class TextureCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bool enabled = tvm::runtime::RuntimeEnabled("opencl");
    if (!enabled) {
      GTEST_SKIP() << "Skip texture copy test because opencl runtime is disabled.\n";
    }
    // Check hardware support
    tvm::runtime::cl::OpenCLWorkspace* workspace = tvm::runtime::cl::OpenCLWorkspace::Global();
    tvm::runtime::cl::OpenCLThreadEntry* thr = workspace->GetThreadEntry();
    if (!workspace->IsBufferToImageSupported(thr->device.device_id)) {
      GTEST_SKIP() << "Skip test case as BufferToImage is not supported \n";
    }
    (void)tvm::runtime::memory::MemoryManager::GetOrCreateAllocator(
        thr->device, tvm::runtime::memory::AllocatorType::kPooled);
  }
};

TEST(TextureCopy, HostDeviceRT) {
  using namespace tvm;
  bool enabled = tvm::runtime::RuntimeEnabled("opencl");
  if (!enabled) {
    GTEST_SKIP() << "Skip texture copy test because opencl runtime is disabled.\n";
  }
  tvm::runtime::cl::OpenCLWorkspace* workspace = tvm::runtime::cl::OpenCLWorkspace::Global();
  tvm::runtime::cl::OpenCLThreadEntry* thr = workspace->GetThreadEntry();
  (void)tvm::runtime::memory::MemoryManager::GetOrCreateAllocator(
      thr->device, tvm::runtime::memory::AllocatorType::kPooled);
  std::vector<int64_t> shape{16, 16, 4};
  auto cpu_arr0 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr1 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  String mem_scope = "global.texture";
  auto opencl_txarr0 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLOpenCL, 0}, mem_scope);

  size_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    size *= static_cast<size_t>(shape[i]);
  }

  std::random_device dev;
  std::mt19937 mt(dev());
  std::uniform_real_distribution<> random(-10.0, 10.0);

  // Random initialize host ndarray
  for (size_t i = 0; i < size; i++) {
    static_cast<float*>(cpu_arr0->data)[i] = random(mt);
  }

  // Do a roundtrip from host storage to opencl texture storage and back
  cpu_arr0.CopyTo(opencl_txarr0);
  opencl_txarr0.CopyTo(cpu_arr1);
  for (size_t i = 0; i < size; ++i) {
    ICHECK_LT(
        std::fabs(static_cast<float*>(cpu_arr1->data)[i] - static_cast<float*>(cpu_arr0->data)[i]),
        1e-5);
  }
}

TEST_F(TextureCopyTest, ViewBufferAsBuffer) {
  using namespace tvm;
  std::vector<int64_t> shape{1, 16, 16, 8};
  std::vector<int64_t> same_shape{1, 8, 16, 16};
  auto cpu_arr = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr_ret = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});

  String mem_scope = "global";

  DLDevice cl_dev = {kDLOpenCL, 0};
  auto allocator = MemoryManager::GetOrCreateAllocator(cl_dev, AllocatorType::kPooled);
  auto buffer = allocator->Alloc(cl_dev, ffi::Shape(shape), {kDLFloat, 32, 1});
  auto stor = Storage(buffer, allocator);

  auto opencl_memobj = stor->AllocNDArrayScoped(0, ffi::Shape(shape), {kDLFloat, 32, 1}, mem_scope);
  auto opencl_memview =
      stor->AllocNDArrayScoped(0, ffi::Shape(same_shape), {kDLFloat, 32, 1}, mem_scope);

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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_memobj);
  // Copy from OpenCLBuffer
  opencl_memobj.CopyTo(cpu_arr_ret);
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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_memview);
  // Copy from OpenCLBuffer
  opencl_memview.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }
}

TEST_F(TextureCopyTest, ViewBufferAsImage) {
  using namespace tvm;
  // Shape that doesn't cause padding for image row
  std::vector<int64_t> shape{1, 16, 16, 8, 4};
  std::vector<int64_t> same_shape{1, 8, 16, 16, 4};
  auto cpu_arr = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr_ret = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});

  DLDevice cl_dev = {kDLOpenCL, 0};
  auto allocator = MemoryManager::GetOrCreateAllocator(cl_dev, AllocatorType::kPooled);
  auto buffer = allocator->Alloc(cl_dev, ffi::Shape(shape), {kDLFloat, 32, 1});
  auto stor = Storage(buffer, allocator);

  auto opencl_buf_obj = stor->AllocNDArrayScoped(0, ffi::Shape(shape), {kDLFloat, 32, 1}, "global");
  auto opencl_img_obj =
      stor->AllocNDArrayScoped(0, ffi::Shape(same_shape), {kDLFloat, 32, 1}, "global.texture");

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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_buf_obj);
  // Copy from OpenCLBuffer
  opencl_buf_obj.CopyTo(cpu_arr_ret);
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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_img_obj);
  // Copy from OpenCLBuffer
  opencl_img_obj.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }
}

TEST_F(TextureCopyTest, ViewImageAsBuffer) {
  using namespace tvm;
  // Shape that doesn't cause padding for image row
  std::vector<int64_t> shape{1, 16, 16, 8, 4};
  std::vector<int64_t> same_shape{1, 8, 16, 16, 4};
  auto cpu_arr = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr_ret = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});

  DLDevice cl_dev = {kDLOpenCL, 0};
  auto allocator = MemoryManager::GetOrCreateAllocator(cl_dev, AllocatorType::kPooled);
  auto buffer = allocator->Alloc(cl_dev, ffi::Shape(shape), {kDLFloat, 32, 1});
  auto stor = Storage(buffer, allocator);

  auto opencl_img_obj =
      stor->AllocNDArrayScoped(0, ffi::Shape(shape), {kDLFloat, 32, 1}, "global.texture");
  auto opencl_buf_obj =
      stor->AllocNDArrayScoped(0, ffi::Shape(same_shape), {kDLFloat, 32, 1}, "global");

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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_buf_obj);
  // Copy from OpenCLBuffer
  opencl_buf_obj.CopyTo(cpu_arr_ret);
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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_img_obj);
  // Copy from OpenCLBuffer
  opencl_img_obj.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }
}

TEST_F(TextureCopyTest, ViewImageAsImage) {
  using namespace tvm;
  // Shape that doesn't cause padding for image row
  std::vector<int64_t> shape{1, 16, 16, 8, 4};
  std::vector<int64_t> same_shape{1, 8, 16, 16, 4};
  auto cpu_arr = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr_ret = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});

  DLDevice cl_dev = {kDLOpenCL, 0};
  auto allocator = MemoryManager::GetOrCreateAllocator(cl_dev, AllocatorType::kPooled);
  auto buffer = allocator->Alloc(cl_dev, ffi::Shape(shape), {kDLFloat, 32, 1});
  auto stor = Storage(buffer, allocator);

  auto opencl_img_obj_1 =
      stor->AllocNDArrayScoped(0, ffi::Shape(shape), {kDLFloat, 32, 1}, "global.texture");
  auto opencl_img_obj_2 =
      stor->AllocNDArrayScoped(0, ffi::Shape(same_shape), {kDLFloat, 32, 1}, "global.texture");

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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_img_obj_1);
  // Copy from OpenCLBuffer
  opencl_img_obj_1.CopyTo(cpu_arr_ret);
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
  // Copy to OpenCLBuffer
  cpu_arr.CopyTo(opencl_img_obj_2);
  // Copy from OpenCLBuffer
  opencl_img_obj_2.CopyTo(cpu_arr_ret);
  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] -
                        static_cast<float*>(cpu_arr_ret->data)[i]),
              1e-5);
  }
}
