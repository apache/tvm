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
#include <tvm/runtime/container/optional.h>

#include "../src/runtime/opencl/opencl_common.h"
#include "../src/runtime/texture.h"

using namespace tvm::runtime;
using namespace tvm::runtime::cl;

// PoolWrapper is necessary because in class Pool2D we don't have an access to
// its protected members. In this class we add new methods which allow us to
// get and check internal state of class Pool
class PoolWrapper : public Pool2D {
 public:
  inline size_t FreeListSize() const { return free_list_.size(); }
  inline size_t AllocatedListSize() const { return allocated_.size(); }
  inline std::pair<size_t, size_t> FreeListItemSize(size_t idx) const {
    return std::make_pair(free_list_[idx].x, free_list_[idx].y);
  }
  inline std::pair<size_t, size_t> AllocatedListItemSize(size_t idx) const {
    return std::make_pair(allocated_[idx].x, allocated_[idx].y);
  }
};

TEST(OpenCLTexturePool, textures_reallocation_optimal_size) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();
  OpenCLThreadEntry* t = workspace->GetThreadEntry();
  PoolWrapper pool;
  EXPECT_EQ(pool.AllocatedListSize(), 0);
  EXPECT_EQ(pool.FreeListSize(), 0);

  DLDataType type{kDLFloat, 16, 1};
  void* data1 = pool.Alloc(t->device, workspace, 1024, 768, type);
  EXPECT_EQ(pool.AllocatedListSize(), 1);
  EXPECT_EQ(pool.FreeListSize(), 0);
  auto item = pool.AllocatedListItemSize(0);
  EXPECT_EQ(item.first, 1024);
  EXPECT_EQ(item.second, 768);

  pool.Alloc(t->device, workspace, 64, 12455, type);
  EXPECT_EQ(pool.AllocatedListSize(), 2);
  EXPECT_EQ(pool.FreeListSize(), 0);
  item = pool.AllocatedListItemSize(1);
  EXPECT_EQ(item.first, 64);
  EXPECT_EQ(item.second, 12455);

  pool.Free(data1);
  EXPECT_EQ(pool.AllocatedListSize(), 1);
  EXPECT_EQ(pool.FreeListSize(), 1);
  item = pool.AllocatedListItemSize(0);
  EXPECT_EQ(item.first, 64);
  EXPECT_EQ(item.second, 12455);
  item = pool.FreeListItemSize(0);
  EXPECT_EQ(item.first, 1024);
  EXPECT_EQ(item.second, 768);

  pool.Alloc(t->device, workspace, 768, 1024, type);
  EXPECT_EQ(pool.AllocatedListSize(), 2);
  EXPECT_EQ(pool.FreeListSize(), 0);
  item = pool.AllocatedListItemSize(0);
  EXPECT_EQ(item.first, 64);
  EXPECT_EQ(item.second, 12455);
  item = pool.AllocatedListItemSize(1);
  EXPECT_EQ(item.first, 1024);
  EXPECT_EQ(item.second, 1024);
}

TEST(OpenCLTexturePool, avoid_reusing_too_big_textures) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();
  OpenCLThreadEntry* t = workspace->GetThreadEntry();
  PoolWrapper pool;
  EXPECT_EQ(pool.AllocatedListSize(), 0);
  EXPECT_EQ(pool.FreeListSize(), 0);

  DLDataType type{kDLFloat, 16, 1};
  void* data1 = pool.Alloc(t->device, workspace, 12455, 64, type);
  EXPECT_EQ(pool.AllocatedListSize(), 1);
  EXPECT_EQ(pool.FreeListSize(), 0);
  auto item = pool.AllocatedListItemSize(0);
  EXPECT_EQ(item.first, 12455);
  EXPECT_EQ(item.second, 64);

  pool.Free(data1);
  EXPECT_EQ(pool.AllocatedListSize(), 0);
  EXPECT_EQ(pool.FreeListSize(), 1);
  item = pool.FreeListItemSize(0);
  EXPECT_EQ(item.first, 12455);
  EXPECT_EQ(item.second, 64);

  pool.Alloc(t->device, workspace, 1024, 768, type);
  EXPECT_EQ(pool.AllocatedListSize(), 1);
  EXPECT_EQ(pool.FreeListSize(), 1);
  item = pool.FreeListItemSize(0);
  EXPECT_EQ(item.first, 12455);
  EXPECT_EQ(item.second, 64);
  item = pool.AllocatedListItemSize(0);
  EXPECT_EQ(item.first, 1024);
  EXPECT_EQ(item.second, 768);
}

TEST(OpenCLTexturePool, avoid_reusing_too_small_textures) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();
  OpenCLThreadEntry* t = workspace->GetThreadEntry();
  PoolWrapper pool;
  EXPECT_EQ(pool.AllocatedListSize(), 0);
  EXPECT_EQ(pool.FreeListSize(), 0);

  DLDataType type{kDLFloat, 16, 1};
  void* data1 = pool.Alloc(t->device, workspace, 1024, 64, type);
  EXPECT_EQ(pool.AllocatedListSize(), 1);
  EXPECT_EQ(pool.FreeListSize(), 0);
  auto item = pool.AllocatedListItemSize(0);
  EXPECT_EQ(item.first, 1024);
  EXPECT_EQ(item.second, 64);

  pool.Free(data1);
  EXPECT_EQ(pool.AllocatedListSize(), 0);
  EXPECT_EQ(pool.FreeListSize(), 1);
  item = pool.FreeListItemSize(0);
  EXPECT_EQ(item.first, 1024);
  EXPECT_EQ(item.second, 64);

  pool.Alloc(t->device, workspace, 12544, 64, type);
  EXPECT_EQ(pool.AllocatedListSize(), 1);
  EXPECT_EQ(pool.FreeListSize(), 1);
  item = pool.FreeListItemSize(0);
  EXPECT_EQ(item.first, 1024);
  EXPECT_EQ(item.second, 64);
  item = pool.AllocatedListItemSize(0);
  EXPECT_EQ(item.first, 12544);
  EXPECT_EQ(item.second, 64);
}
