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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <exception>

#include "../../../../src/runtime/memory/pooled_allocator.h"

namespace tvm {
namespace runtime {
namespace memory {

// MemoryManangerWrapper is necessary because in class MemoryManager we don't have access to its
// protected members. In this class we add a new method which allow us to clear internal state of
// the global memory manager.
class MemoryManagerWrapper : public MemoryManager {
 public:
  static MemoryManagerWrapper* Global() {
    return reinterpret_cast<MemoryManagerWrapper*>(MemoryManager::Global());
  }
  void clear() { allocators_.clear(); }
};

class TvmVMMemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear allocators from previous tests
    MemoryManagerWrapper::Global()->clear();
  }
};

TEST_F(TvmVMMemoryManagerTest, NaiveAllocBasic) {
  Device dev = {kDLCPU, 0};
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kNaive);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto buff = allocator->Alloc(64, 32, DataType::Float(32));
  EXPECT_EQ(allocator->UsedMemory(), 64);
  allocator->Free(buff);
  EXPECT_EQ(allocator->UsedMemory(), 0);
}

TEST_F(TvmVMMemoryManagerTest, PooledAllocBasic) {
  Device dev = {kDLCPU, 0};
  size_t nbytes = 64;
  size_t page_size = PooledAllocator::kDefaultPageSize;
  size_t size = ((nbytes + page_size - 1) / page_size) * page_size;
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kPooled);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto buff = allocator->Alloc(nbytes, 32, DataType::Float(32));
  EXPECT_EQ(allocator->UsedMemory(), size);
  allocator->Free(buff);
  EXPECT_EQ(allocator->UsedMemory(), size);
}

TEST_F(TvmVMMemoryManagerTest, NaiveEmptyBasic) {
  Device dev = {kDLCPU, 0};
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kNaive);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  ShapeTuple shape = {1, 3, 6, 6};
  {
    auto ndarray = allocator->Empty(shape, dt, dev);
    EXPECT_EQ(allocator->UsedMemory(), nbytes);
  }
  EXPECT_EQ(allocator->UsedMemory(), 0);
}

TEST_F(TvmVMMemoryManagerTest, BothAllocatorsCoexists) {
  Device dev = {kDLCPU, 0};
  // Initialize and use Naive allocator
  Allocator* nallocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kNaive);
  EXPECT_EQ(nallocator->UsedMemory(), 0);
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  ShapeTuple shape = {1, 3, 6, 6};
  {
    auto ndarray = nallocator->Empty(shape, dt, dev);
    EXPECT_EQ(nallocator->UsedMemory(), nbytes);
  }
  EXPECT_EQ(nallocator->UsedMemory(), 0);
  auto naive_buff = nallocator->Alloc(shape, dt);
  EXPECT_EQ(nallocator->UsedMemory(), nbytes);

  // Initialize and use Pooled allocator
  Allocator* pallocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kPooled);
  EXPECT_EQ(pallocator->UsedMemory(), 0);
  auto pooled_buff = pallocator->Alloc(shape, dt);
  EXPECT_NE(pallocator->UsedMemory(), 0);

  // Operate on Naive allocator
  EXPECT_EQ(nallocator->UsedMemory(), nbytes);
  nallocator->Free(naive_buff);
  EXPECT_EQ(nallocator->UsedMemory(), 0);

  // Operate on Pooled allocator
  pallocator->Free(pooled_buff);
  EXPECT_NE(pallocator->UsedMemory(), 0);
}

TEST_F(TvmVMMemoryManagerTest, PooledEmptyBasic) {
  Device dev = {kDLCPU, 0};
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kPooled);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  size_t page_size = PooledAllocator::kDefaultPageSize;
  size_t size = ((nbytes + page_size - 1) / page_size) * page_size;
  ShapeTuple shape = {1, 3, 6, 6};
  {
    auto ndarray = allocator->Empty(shape, dt, dev);
    EXPECT_EQ(allocator->UsedMemory(), size);
  }
  EXPECT_EQ(allocator->UsedMemory(), size);
}

TEST_F(TvmVMMemoryManagerTest, NaiveAllocWithShape) {
  Device dev = {kDLCPU, 0};
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kNaive);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  ShapeTuple shape = {1, 3, 6, 6};
  auto buff = allocator->Alloc(shape, dt);
  EXPECT_EQ(allocator->UsedMemory(), nbytes);
  allocator->Free(buff);
  EXPECT_EQ(allocator->UsedMemory(), 0);

  try {
    auto texture = allocator->Alloc(shape, dt, "global.texture");
    (void)texture;
    FAIL();
  } catch (std::exception& e) {
    std::string pattern =
        "Device does not support allocate data space with specified memory scope: global.texture";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
}

TEST_F(TvmVMMemoryManagerTest, PooledAllocWithShape) {
  Device dev = {kDLCPU, 0};
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kPooled);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  size_t page_size = PooledAllocator::kDefaultPageSize;
  size_t size = ((nbytes + page_size - 1) / page_size) * page_size;
  ShapeTuple shape = {1, 3, 6, 6};
  auto buff = allocator->Alloc(shape, dt);
  EXPECT_EQ(allocator->UsedMemory(), size);
  allocator->Free(buff);
  EXPECT_EQ(allocator->UsedMemory(), size);

  try {
    auto texture = allocator->Alloc(shape, dt, "global.texture");
    (void)texture;
    FAIL();
  } catch (std::exception& e) {
    std::string pattern =
        "Device does not support allocate data space with specified memory scope: global.texture";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
}

TEST_F(TvmVMMemoryManagerTest, NaiveAllocOpenCLTexture) {
  bool enabled = tvm::runtime::RuntimeEnabled("opencl");
  if (!enabled) {
    LOG(INFO) << "Skip OpenCL Texture alloc test because opencl runtime is disabled.\n";
    return;
  }
  Device dev = {kDLOpenCL, 0};
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kNaive);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  ShapeTuple shape = {1, 3, 6, 6};
  auto buff = allocator->Alloc(shape, dt);
  EXPECT_EQ(allocator->UsedMemory(), nbytes);
  allocator->Free(buff);
  EXPECT_EQ(allocator->UsedMemory(), 0);

  auto texture = allocator->Alloc(shape, dt, "global.texture");
  EXPECT_EQ(allocator->UsedMemory(), nbytes);
  allocator->Free(texture);
  EXPECT_EQ(allocator->UsedMemory(), 0);
}

TEST_F(TvmVMMemoryManagerTest, PooledAllocOpenCLTexture) {
  bool enabled = tvm::runtime::RuntimeEnabled("opencl");
  if (!enabled) {
    LOG(INFO) << "Skip OpenCL Texture alloc test because opencl runtime is disabled.\n";
    return;
  }
  Device dev = {kDLOpenCL, 0};
  Allocator* allocator = MemoryManagerWrapper::GetOrCreateAllocator(dev, kPooled);
  EXPECT_EQ(allocator->UsedMemory(), 0);
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  size_t page_size = PooledAllocator::kDefaultPageSize;
  size_t size = ((nbytes + page_size - 1) / page_size) * page_size;
  ShapeTuple shape = {1, 3, 6, 6};
  auto buff = allocator->Alloc(shape, dt);
  EXPECT_EQ(allocator->UsedMemory(), size);
  allocator->Free(buff);
  EXPECT_EQ(allocator->UsedMemory(), size);

  auto texture = allocator->Alloc(shape, dt, "global.texture");
  allocator->Free(texture);
}

TEST_F(TvmVMMemoryManagerTest, DeviceConflictsNaive) {
  bool enabled = tvm::runtime::RuntimeEnabled("opencl");
  if (!enabled) {
    LOG(INFO) << "Skip OpenCL Texture alloc test because opencl runtime is disabled.\n";
    return;
  }
  // Initializations
  Device cpu_dev = {kDLCPU, 0};
  Device cl_dev = {kDLOpenCL, 0};
  auto dt = DataType::Float(32);
  size_t nbytes = 1 * 3 * 6 * 6 * dt.bytes();
  ShapeTuple shape = {1, 3, 6, 6};
  // CL Allocator
  Allocator* cl_allocator = MemoryManagerWrapper::GetOrCreateAllocator(cl_dev, kNaive);
  EXPECT_EQ(cl_allocator->UsedMemory(), 0);
  {
    auto ndarray = cl_allocator->Empty(shape, dt, cl_dev);
    EXPECT_EQ(cl_allocator->UsedMemory(), nbytes);
  }
  EXPECT_EQ(cl_allocator->UsedMemory(), 0);
  // CPU Allocator
  Allocator* cpu_allocator = MemoryManagerWrapper::GetOrCreateAllocator(cpu_dev, kNaive);
  {
    auto ndarray = cpu_allocator->Empty(shape, dt, cpu_dev);
    EXPECT_EQ(cpu_allocator->UsedMemory(), nbytes);
  }
  EXPECT_EQ(cpu_allocator->UsedMemory(), 0);

  // Try allocating CL Tensor on CPU Allocator
  try {
    auto cpu_buff = cpu_allocator->Empty(shape, dt, cl_dev);
    (void)cpu_buff;
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
  auto cpu_buff = cpu_allocator->Alloc(shape, dt);
  EXPECT_EQ(cpu_allocator->UsedMemory(), nbytes);

  // Try allocating CPU Tensor on CL Allocator
  try {
    auto cl_buff = cl_allocator->Empty(shape, dt, cpu_dev);
    (void)cl_buff;
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
  auto cl_buff = cl_allocator->Alloc(shape, dt);
  EXPECT_EQ(cl_allocator->UsedMemory(), nbytes);

  // Attempt conflicting Free calls
  try {
    cl_allocator->Free(cpu_buff);
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
  try {
    cpu_allocator->Free(cl_buff);
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }

  // Regular Free calls
  cl_allocator->Free(cl_buff);
  EXPECT_EQ(cl_allocator->UsedMemory(), 0);

  cpu_allocator->Free(cpu_buff);
  EXPECT_EQ(cpu_allocator->UsedMemory(), 0);
}

TEST_F(TvmVMMemoryManagerTest, DeviceConflictsPooled) {
  bool enabled = tvm::runtime::RuntimeEnabled("opencl");
  if (!enabled) {
    LOG(INFO) << "Skip OpenCL Texture alloc test because opencl runtime is disabled.\n";
    return;
  }
  // Initializations
  Device cpu_dev = {kDLCPU, 0};
  Device cl_dev = {kDLOpenCL, 0};
  auto dt = DataType::Float(32);
  ShapeTuple shape = {1, 3, 6, 6};
  // CL Allocator
  Allocator* cl_allocator = MemoryManagerWrapper::GetOrCreateAllocator(cl_dev, kPooled);
  EXPECT_EQ(cl_allocator->UsedMemory(), 0);
  {
    auto ndarray = cl_allocator->Empty(shape, dt, cl_dev);
    EXPECT_NE(cl_allocator->UsedMemory(), 0);
  }
  EXPECT_NE(cl_allocator->UsedMemory(), 0);
  // CPU Allocator
  Allocator* cpu_allocator = MemoryManagerWrapper::GetOrCreateAllocator(cpu_dev, kPooled);
  {
    auto ndarray = cpu_allocator->Empty(shape, dt, cpu_dev);
    EXPECT_NE(cpu_allocator->UsedMemory(), 0);
  }
  EXPECT_NE(cpu_allocator->UsedMemory(), 0);

  // Try allocating CL Tensor on CPU Allocator
  try {
    auto cpu_buff = cpu_allocator->Empty(shape, dt, cl_dev);
    (void)cpu_buff;
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
  auto cpu_buff = cpu_allocator->Alloc(shape, dt);
  EXPECT_NE(cpu_allocator->UsedMemory(), 0);

  // Try allocating CPU Tensor on CL Allocator
  try {
    auto cl_buff = cl_allocator->Empty(shape, dt, cpu_dev);
    (void)cl_buff;
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
  auto cl_buff = cl_allocator->Alloc(shape, dt);
  EXPECT_NE(cl_allocator->UsedMemory(), 0);

  // Attempt conflicting Free calls
  try {
    cl_allocator->Free(cpu_buff);
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
  try {
    cpu_allocator->Free(cl_buff);
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Device mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }

  // Regular Free calls
  cl_allocator->Free(cl_buff);
  EXPECT_NE(cl_allocator->UsedMemory(), 0);

  cpu_allocator->Free(cpu_buff);
  EXPECT_NE(cpu_allocator->UsedMemory(), 0);
}

TEST_F(TvmVMMemoryManagerTest, AllocatorConflicts) {
  // Initializations
  Device cpu_dev = {kDLCPU, 0};
  auto dt = DataType::Float(32);
  ShapeTuple shape = {1, 3, 6, 6};
  Allocator* nallocator = MemoryManagerWrapper::GetOrCreateAllocator(cpu_dev, kNaive);
  Allocator* pallocator = MemoryManagerWrapper::GetOrCreateAllocator(cpu_dev, kPooled);
  auto n_buff = nallocator->Alloc(shape, dt);
  auto p_buff = pallocator->Alloc(shape, dt);

  // Attempt conflicting Free calls
  try {
    nallocator->Free(p_buff);
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Allocator type mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }
  try {
    pallocator->Free(n_buff);
    FAIL();
  } catch (std::exception& e) {
    std::string pattern = "Allocator type mismatch, expected";
    std::string what = e.what();
    EXPECT_NE(what.find(pattern), std::string::npos) << what;
  }

  // Regular Free calls
  nallocator->Free(n_buff);
  pallocator->Free(p_buff);
}

}  // namespace memory
}  // namespace runtime
}  // namespace tvm
