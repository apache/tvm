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
#include <hexagon/hexagon/hexagon_buffer.h>
#include <tvm/runtime/container/optional.h>

using namespace tvm::runtime;
using namespace tvm::runtime::hexagon;

TEST(HexagonBuffer, default_scope) {
  Optional<String> scope;
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);
}

TEST(HexagonBuffer, ddr_scope) {
  Optional<String> scope("global");
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);
}

TEST(HexagonBuffer, vtcm_scope) {
  Optional<String> scope("global.vtcm");
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_EQ(hb.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);
}

TEST(HexagonBuffer, invalid_scope) {
  Optional<String> scope("invalid");
  EXPECT_THROW(HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, scope), InternalError);
}

TEST(HexagonBuffer, copy_from) {
  Optional<String> scope("global");
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, scope);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data.data(), data.size());

  uint8_t* ptr = static_cast<uint8_t*>(hb.GetPointer()[0]);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST(HexagonBuffer, copy_from_invalid_size) {
  Optional<String> scope("global");
  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};

  // HexagonBuffer too small
  HexagonBuffer toosmall(4 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_THROW(toosmall.CopyFrom(data.data(), data.size()), InternalError);

  // HexagonBuffer too big
  HexagonBuffer toobig(16 /* nbytes */, 16 /* alignment */, scope);
  EXPECT_THROW(toobig.CopyFrom(data.data(), data.size()), InternalError);
}

TEST(HexagonBuffer, nd) {
  Optional<String> def;
  HexagonBuffer hb_default(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, def);
  EXPECT_EQ(hb_default.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  Optional<String> global("global");
  HexagonBuffer hb_global(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, global);
  EXPECT_EQ(hb_global.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  Optional<String> vtcm("global.vtcm");
  HexagonBuffer hb_vtcm(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, vtcm);
  EXPECT_EQ(hb_vtcm.GetStorageScope(), HexagonBuffer::StorageScope::kVTCM);

  Optional<String> invalid("invalid");
  EXPECT_THROW(HexagonBuffer hb_invalid(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, invalid),
               InternalError);
}

TEST(HexagonBuffer, nd_copy_from) {
  Optional<String> scope("global");
  HexagonBuffer hb(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, scope);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data.data(), data.size());

  uint8_t* ptr = static_cast<uint8_t*>(hb.GetPointer()[0]);
  EXPECT_EQ(ptr[0], data[0]);
  EXPECT_EQ(ptr[1], data[1]);
  EXPECT_EQ(ptr[2], data[2]);
  EXPECT_EQ(ptr[3], data[3]);

  ptr = static_cast<uint8_t*>(hb.GetPointer()[1]);
  EXPECT_EQ(ptr[0], data[4]);
  EXPECT_EQ(ptr[1], data[5]);
  EXPECT_EQ(ptr[2], data[6]);
  EXPECT_EQ(ptr[3], data[7]);
}

TEST(HexagonBuffer, 1d_copy_from_1d) {
  Optional<String> global("global");
  HexagonBuffer from(8 /* nbytes */, 8 /* alignment */, global);

  Optional<String> vtcm("global.vtcm");
  HexagonBuffer to(8 /* nbytes */, 8 /* alignment */, vtcm);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  from.CopyFrom(data.data(), data.size());
  to.CopyFrom(from);

  uint8_t* ptr = static_cast<uint8_t*>(to.GetPointer()[0]);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST(HexagonBuffer, 2d_copy_from_1d) {
  Optional<String> vtcm("global.vtcm");
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, vtcm);

  Optional<String> global("global");
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, global);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb1d.CopyFrom(data.data(), data.size());
  hb2d.CopyFrom(hb1d);

  uint8_t* ptr = static_cast<uint8_t*>(hb2d.GetPointer()[0]);
  EXPECT_EQ(ptr[0], data[0]);
  EXPECT_EQ(ptr[1], data[1]);
  EXPECT_EQ(ptr[2], data[2]);
  EXPECT_EQ(ptr[3], data[3]);

  ptr = static_cast<uint8_t*>(hb2d.GetPointer()[1]);
  EXPECT_EQ(ptr[0], data[4]);
  EXPECT_EQ(ptr[1], data[5]);
  EXPECT_EQ(ptr[2], data[6]);
  EXPECT_EQ(ptr[3], data[7]);
}

TEST(HexagonBuffer, 1d_copy_from_2d) {
  Optional<String> vtcm("global.vtcm");
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, vtcm);

  Optional<String> global("global.vtcm");
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, global);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb2d.CopyFrom(data.data(), data.size());
  hb1d.CopyFrom(hb2d);

  uint8_t* ptr = static_cast<uint8_t*>(hb1d.GetPointer()[0]);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST(HexagonBuffer, nd_copy_from_nd_invalid_size) {
  Optional<String> scope("global");
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, scope);
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, scope);

  HexagonBuffer toosmall1d(4 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_THROW(hb1d.CopyFrom(toosmall1d), InternalError);
  EXPECT_THROW(hb2d.CopyFrom(toosmall1d), InternalError);

  HexagonBuffer toosbig1d(16 /* nbytes */, 16 /* alignment */, scope);
  EXPECT_THROW(hb1d.CopyFrom(toosbig1d), InternalError);
  EXPECT_THROW(hb2d.CopyFrom(toosbig1d), InternalError);

  HexagonBuffer toosmall2d(2 /* ndim */, 2 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_THROW(hb1d.CopyFrom(toosmall2d), InternalError);
  EXPECT_THROW(hb2d.CopyFrom(toosmall2d), InternalError);

  HexagonBuffer toobig2d(2 /* ndim */, 16 /* nbytes */, 16 /* alignment */, scope);
  EXPECT_THROW(hb1d.CopyFrom(toobig2d), InternalError);
  EXPECT_THROW(hb2d.CopyFrom(toobig2d), InternalError);
}

TEST(HexagonBuffer, md_copy_from_nd) {
  Optional<String> scope("global");
  HexagonBuffer hb3d(3 /* ndim */, 4 /* nbytes */, 8 /* alignment */, scope);
  HexagonBuffer hb4d(4 /* ndim */, 3 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_THROW(hb3d.CopyFrom(hb4d), InternalError);
  EXPECT_THROW(hb4d.CopyFrom(hb3d), InternalError);
}

TEST(HexagonBuffer, copy_to) {
  Optional<String> scope("global");
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, scope);

  std::vector<uint8_t> data_in{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data_in.data(), data_in.size());

  std::vector<uint8_t> data_out{7, 6, 5, 4, 3, 2, 1, 0};
  hb.CopyTo(data_out.data(), data_out.size());

  for (size_t i = 0; i < data_in.size(); ++i) {
    EXPECT_EQ(data_in[i], data_out[i]);
  }
}

TEST(HexagonBuffer, nd_copy_to) {
  Optional<String> scope("global");
  HexagonBuffer hb(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, scope);

  std::vector<uint8_t> data_in{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data_in.data(), data_in.size());

  std::vector<uint8_t> data_out{7, 6, 5, 4, 3, 2, 1, 0};
  hb.CopyTo(data_out.data(), data_out.size());

  for (size_t i = 0; i < data_in.size(); ++i) {
    EXPECT_EQ(data_in[i], data_out[i]);
  }
}

TEST(HexagonBuffer, external) {
  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};

  Optional<String> def;
  HexagonBuffer hb_default(data.data(), data.size(), def);
  EXPECT_EQ(hb_default.GetPointer()[0], data.data());
  EXPECT_EQ(hb_default.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  Optional<String> global("global");
  HexagonBuffer hb_global(data.data(), data.size(), global);
  EXPECT_EQ(hb_global.GetPointer()[0], data.data());
  EXPECT_EQ(hb_global.GetStorageScope(), HexagonBuffer::StorageScope::kDDR);

  Optional<String> vtcm("global.vtcm");
  EXPECT_THROW(HexagonBuffer hb_vtcm(data.data(), data.size(), vtcm), InternalError);

  Optional<String> invalid("invalid");
  EXPECT_THROW(HexagonBuffer hb_vtcm(data.data(), data.size(), invalid), InternalError);
}
