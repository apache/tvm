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

#include "../src/relay/backend/token_allocator.h"

namespace tvm {
namespace relay {

// TokenAllocator2d is necessary because in class TokenAllocator2D we don't
// have an access to its protected members. In this class we add new methods
// which allow us to get and check internal state of class TokenAllocator2D
class TokenAllocator2DWrapper : public TokenAllocator2D {
 public:
  inline size_t FreeListSize() const { return free_list_.size(); }
  inline size_t BlockMapSize() const { return blocks_.size(); }
};

TEST(Token2DAlloc, OneToken) {
  TokenAllocator2DWrapper alloc;
  int storage_ids = 0;
  EXPECT_EQ(alloc.BlockMapSize(), 0);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  TensorType tt1({1, 22, 20, 20, 4}, DataType(kDLFloat, 32, 1));
  VirtualDevice vd1(kDLOpenCL, 0, {}, MemoryScope("global.texture-nhwc"));
  StorageToken tok1 = {
      1,    // ref_counter
      0,    // max bytes
      tt1,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto size2d = alloc.GetSize2D(&tok1);
  EXPECT_EQ(size2d.channel, 4);
  EXPECT_EQ(size2d.height, 22);
  EXPECT_EQ(size2d.width, 400);
  EXPECT_EQ(alloc.Request(&tok1), nullptr);

  alloc.Alloc(&tok1, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  tok1.ref_counter -= 1;
  alloc.CheckForRelease(&tok1);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);
}

TEST(Token2DAlloc, EqualSizeTokenReuse) {
  TokenAllocator2DWrapper alloc;
  int storage_ids = 0;
  EXPECT_EQ(alloc.BlockMapSize(), 0);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  TensorType tt1({1, 22, 20, 20, 4}, DataType(kDLFloat, 32, 1));
  VirtualDevice vd1(kDLOpenCL, 0, {}, MemoryScope("global.texture-nhwc"));
  StorageToken tok1 = {
      1,    // ref_counter
      0,    // max bytes
      tt1,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto size2d = alloc.GetSize2D(&tok1);
  EXPECT_EQ(size2d.channel, 4);
  EXPECT_EQ(size2d.height, 22);
  EXPECT_EQ(size2d.width, 400);
  EXPECT_EQ(alloc.Request(&tok1), nullptr);

  alloc.Alloc(&tok1, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  tok1.ref_counter -= 1;
  alloc.CheckForRelease(&tok1);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  StorageToken tok2 = {
      1,    // ref_counter
      0,    // max bytes
      tt1,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto req = alloc.Request(&tok2);
  EXPECT_NE(req, nullptr);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);
  EXPECT_EQ(req->storage_id, storage_ids - 1);
  EXPECT_EQ(req->ref_counter, 1);
  auto sizeReq = alloc.GetSize2D(req);
  EXPECT_EQ(sizeReq.channel, 4);
  EXPECT_EQ(sizeReq.height, 22);
  EXPECT_EQ(sizeReq.width, 400);
}

TEST(Token2DAlloc, EqualSizeDiffTypes) {
  TokenAllocator2DWrapper alloc;
  int storage_ids = 0;
  EXPECT_EQ(alloc.BlockMapSize(), 0);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  TensorType tt1({1, 22, 20, 20, 4}, DataType(kDLFloat, 32, 1));
  VirtualDevice vd1(kDLOpenCL, 0, {}, MemoryScope("global.texture-nhwc"));
  StorageToken tok1 = {
      1,    // ref_counter
      0,    // max bytes
      tt1,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto size2d = alloc.GetSize2D(&tok1);
  EXPECT_EQ(size2d.channel, 4);
  EXPECT_EQ(size2d.height, 22);
  EXPECT_EQ(size2d.width, 400);
  EXPECT_EQ(alloc.Request(&tok1), nullptr);

  alloc.Alloc(&tok1, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  tok1.ref_counter -= 1;
  alloc.CheckForRelease(&tok1);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  TensorType tt2({1, 22, 20, 20, 4}, DataType(kDLFloat, 16, 1));
  StorageToken tok2 = {
      1,    // ref_counter
      0,    // max bytes
      tt2,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  EXPECT_EQ(alloc.Request(&tok2), nullptr);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  alloc.Alloc(&tok2, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 2);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  tok2.ref_counter -= 1;
  alloc.CheckForRelease(&tok2);
  EXPECT_EQ(alloc.BlockMapSize(), 2);
  EXPECT_EQ(alloc.FreeListSize(), 2);
}

TEST(Token2DAlloc, DifferentSizesTokenReuse) {
  TokenAllocator2DWrapper alloc;
  int storage_ids = 0;
  EXPECT_EQ(alloc.BlockMapSize(), 0);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  TensorType tt1({1, 22, 20, 20, 4}, DataType(kDLFloat, 32, 1));
  VirtualDevice vd1(kDLOpenCL, 0, {}, MemoryScope("global.texture-nhwc"));
  StorageToken tok1 = {
      1,    // ref_counter
      0,    // max bytes
      tt1,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto size2d = alloc.GetSize2D(&tok1);
  EXPECT_EQ(size2d.channel, 4);
  EXPECT_EQ(size2d.height, 22);
  EXPECT_EQ(size2d.width, 400);
  EXPECT_EQ(alloc.Request(&tok1), nullptr);

  alloc.Alloc(&tok1, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  tok1.ref_counter -= 1;
  alloc.CheckForRelease(&tok1);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  TensorType tt2({1, 40, 30, 30, 4}, DataType(kDLFloat, 32, 1));
  StorageToken tok2 = {
      1,    // ref_counter
      0,    // max bytes
      tt2,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto req = alloc.Request(&tok2);
  EXPECT_NE(req, nullptr);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);
  EXPECT_EQ(req->storage_id, storage_ids - 1);
  EXPECT_EQ(req->ref_counter, 2);
  auto sizeReq = alloc.GetSize2D(req);
  EXPECT_EQ(sizeReq.channel, 4);
  EXPECT_EQ(sizeReq.height, 40);
  EXPECT_EQ(sizeReq.width, 900);

  tok2.ref_counter -= 1;
  req->ref_counter -= 1;
  alloc.CheckForRelease(&tok1);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  TensorType tt3({1, 25, 30, 30, 4}, DataType(kDLFloat, 32, 1));
  StorageToken tok3 = {
      1,    // ref_counter
      0,    // max bytes
      tt3,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto req2 = alloc.Request(&tok3);
  EXPECT_NE(req2, nullptr);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);
  EXPECT_EQ(req2->storage_id, storage_ids - 1);
  EXPECT_EQ(req2->ref_counter, 1);
  auto sizeReq2 = alloc.GetSize2D(req2);
  EXPECT_EQ(sizeReq2.channel, 4);
  EXPECT_EQ(sizeReq2.height, 40);
  EXPECT_EQ(sizeReq2.width, 900);
}

TEST(Token2DAlloc, DifferentSizesTokenReuse2) {
  TokenAllocator2DWrapper alloc;
  int storage_ids = 0;
  EXPECT_EQ(alloc.BlockMapSize(), 0);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  TensorType tt1({1, 22, 20, 20, 4}, DataType(kDLFloat, 32, 1));
  VirtualDevice vd1(kDLOpenCL, 0, {}, MemoryScope("global.texture-nhwc"));
  StorageToken tok1 = {
      1,    // ref_counter
      0,    // max bytes
      tt1,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto size2d = alloc.GetSize2D(&tok1);
  EXPECT_EQ(size2d.channel, 4);
  EXPECT_EQ(size2d.height, 22);
  EXPECT_EQ(size2d.width, 400);
  EXPECT_EQ(alloc.Request(&tok1), nullptr);

  alloc.Alloc(&tok1, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  tok1.ref_counter -= 1;
  alloc.CheckForRelease(&tok1);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  TensorType tt2({1, 5, 30, 20, 4}, DataType(kDLFloat, 32, 1));
  StorageToken tok2 = {
      1,    // ref_counter
      0,    // max bytes
      tt2,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto req = alloc.Request(&tok2);
  EXPECT_NE(req, nullptr);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);
  EXPECT_EQ(req->storage_id, storage_ids - 1);
  EXPECT_EQ(req->ref_counter, 2);
  auto sizeReq = alloc.GetSize2D(req);
  EXPECT_EQ(sizeReq.channel, 4);
  EXPECT_EQ(sizeReq.height, 5);
  EXPECT_EQ(sizeReq.width, 600);
}

TEST(Token2DAlloc, SameSizesButDiffMemoryScopes) {
  TokenAllocator2DWrapper alloc;
  int storage_ids = 0;
  EXPECT_EQ(alloc.BlockMapSize(), 0);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  TensorType tt1({28, 676, 1, 1, 4}, DataType(kDLFloat, 32, 1));
  VirtualDevice vd1(kDLOpenCL, 0, {}, MemoryScope("global.texture-weight"));
  StorageToken tok1 = {
      1,    // ref_counter
      0,    // max bytes
      tt1,  // tensor type
      vd1,  // virtual device
      -1    // storage_id
  };
  auto size2d = alloc.GetSize2D(&tok1);
  EXPECT_EQ(size2d.channel, 4);
  EXPECT_EQ(size2d.height, 28);
  EXPECT_EQ(size2d.width, 676);
  EXPECT_EQ(alloc.Request(&tok1), nullptr);

  alloc.Alloc(&tok1, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 0);

  tok1.ref_counter -= 1;
  alloc.CheckForRelease(&tok1);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  TensorType tt2({1, 28, 26, 26, 4}, DataType(kDLFloat, 32, 1));
  VirtualDevice vd2(kDLOpenCL, 0, {}, MemoryScope("global.texture-nhwc"));
  StorageToken tok2 = {
      1,    // ref_counter
      0,    // max bytes
      tt2,  // tensor type
      vd2,  // virtual device
      -1    // storage_id
  };
  auto tok2Size = alloc.GetSize2D(&tok2);
  EXPECT_EQ(tok2Size.channel, 4);
  EXPECT_EQ(tok2Size.height, 28);
  EXPECT_EQ(tok2Size.width, 676);

  EXPECT_EQ(alloc.Request(&tok2), nullptr);
  EXPECT_EQ(alloc.BlockMapSize(), 1);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  alloc.Alloc(&tok2, storage_ids++);
  EXPECT_EQ(alloc.BlockMapSize(), 2);
  EXPECT_EQ(alloc.FreeListSize(), 1);

  tok2.ref_counter -= 1;
  alloc.CheckForRelease(&tok2);
  EXPECT_EQ(alloc.BlockMapSize(), 2);
  EXPECT_EQ(alloc.FreeListSize(), 2);
}
}  // namespace relay
}  // namespace tvm
