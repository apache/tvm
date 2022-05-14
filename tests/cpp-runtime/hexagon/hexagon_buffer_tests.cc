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

#include "../src/runtime/hexagon/hexagon_buffer.h"

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

TEST(HexagonBuffer, micro_copies_corresponding_regions) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(16)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 16);

  std::vector<void*> dest_ptr{ptr(64), ptr(80)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 16);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(micro_copies.size(), 2);
  for (size_t i = 0; i < micro_copies.size(); i++) {
    EXPECT_EQ(micro_copies[i].src, ptr(16 * i));
    EXPECT_EQ(micro_copies[i].dest, ptr(64 + 16 * i));
    EXPECT_EQ(micro_copies[i].num_bytes, 16);
  }
}

TEST(HexagonBuffer, micro_copies_src_bigger) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(16)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 16);

  std::vector<void*> dest_ptr{ptr(64), ptr(72), ptr(80), ptr(88)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 8);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(micro_copies.size(), 4);
  for (size_t i = 0; i < micro_copies.size(); i++) {
    EXPECT_EQ(micro_copies[i].src, ptr(8 * i));
    EXPECT_EQ(micro_copies[i].dest, ptr(64 + 8 * i));
    EXPECT_EQ(micro_copies[i].num_bytes, 8);
  }
}

TEST(HexagonBuffer, micro_copies_dest_bigger) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(8), ptr(16), ptr(24)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 8);

  std::vector<void*> dest_ptr{ptr(64), ptr(80)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 16);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 32);
  EXPECT_EQ(micro_copies.size(), 4);
  for (size_t i = 0; i < micro_copies.size(); i++) {
    EXPECT_EQ(micro_copies[i].src, ptr(8 * i));
    EXPECT_EQ(micro_copies[i].dest, ptr(64 + 8 * i));
    EXPECT_EQ(micro_copies[i].num_bytes, 8);
  }
}

TEST(HexagonBuffer, micro_copies_src_overlaps_dest_region) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(16)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 16);

  std::vector<void*> dest_ptr{ptr(64), ptr(76)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 12);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 24);
  EXPECT_EQ(micro_copies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(micro_copies[0].src, ptr(0));
  EXPECT_EQ(micro_copies[0].dest, ptr(64));
  EXPECT_EQ(micro_copies[0].num_bytes, 12);

  // First region of source, second region of dest
  EXPECT_EQ(micro_copies[1].src, ptr(12));
  EXPECT_EQ(micro_copies[1].dest, ptr(76));
  EXPECT_EQ(micro_copies[1].num_bytes, 4);

  // Second region of source, second region of dest
  EXPECT_EQ(micro_copies[2].src, ptr(16));
  EXPECT_EQ(micro_copies[2].dest, ptr(80));
  EXPECT_EQ(micro_copies[2].num_bytes, 8);
}

TEST(HexagonBuffer, micro_copies_dest_overlaps_src_region) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(12)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 12);

  std::vector<void*> dest_ptr{ptr(64), ptr(80)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 16);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 24);
  EXPECT_EQ(micro_copies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(micro_copies[0].src, ptr(0));
  EXPECT_EQ(micro_copies[0].dest, ptr(64));
  EXPECT_EQ(micro_copies[0].num_bytes, 12);

  // Second region of source, first region of dest
  EXPECT_EQ(micro_copies[1].src, ptr(12));
  EXPECT_EQ(micro_copies[1].dest, ptr(76));
  EXPECT_EQ(micro_copies[1].num_bytes, 4);

  // Second region of source, second region of dest
  EXPECT_EQ(micro_copies[2].src, ptr(16));
  EXPECT_EQ(micro_copies[2].dest, ptr(80));
  EXPECT_EQ(micro_copies[2].num_bytes, 8);
}

TEST(HexagonBuffer, micro_copies_discontiguous_regions) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  // Stride of 16, but only first 11 bytes in each region belong to
  // this buffer.
  std::vector<void*> src_ptr{ptr(0), ptr(16)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 11);

  std::vector<void*> dest_ptr{ptr(64), ptr(80)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 13);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 16);
  EXPECT_EQ(micro_copies.size(), 3);

  // First region of source, first region of dest
  EXPECT_EQ(micro_copies[0].src, ptr(0));
  EXPECT_EQ(micro_copies[0].dest, ptr(64));
  EXPECT_EQ(micro_copies[0].num_bytes, 11);

  // Second region of source, first region of dest
  EXPECT_EQ(micro_copies[1].src, ptr(16));
  EXPECT_EQ(micro_copies[1].dest, ptr(75));
  EXPECT_EQ(micro_copies[1].num_bytes, 2);

  // Second region of source, second region of dest
  EXPECT_EQ(micro_copies[2].src, ptr(18));
  EXPECT_EQ(micro_copies[2].dest, ptr(80));
  EXPECT_EQ(micro_copies[2].num_bytes, 3);
}

TEST(HexagonBuffer, micro_copies_invalid_size) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(16)};
  std::vector<void*> dest_ptr{ptr(64), ptr(80)};

  {
    BufferSet src(src_ptr.data(), 1, 16);
    BufferSet dest(dest_ptr.data(), 2, 16);
    EXPECT_THROW(BufferSet::MemoryCopies(dest, src, 24), InternalError);
  }

  {
    BufferSet src(src_ptr.data(), 2, 16);
    BufferSet dest(dest_ptr.data(), 1, 16);
    EXPECT_THROW(BufferSet::MemoryCopies(dest, src, 24), InternalError);
  }
}

TEST(HexagonBuffer, macro_copies_adjacent_corresponding_regions_merged) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(16)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 16);

  std::vector<void*> dest_ptr{ptr(64), ptr(80)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 16);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 32);
  auto macro_copies = MemoryCopy::MergeAdjacent(std::move(micro_copies));

  ASSERT_EQ(macro_copies.size(), 1);
  EXPECT_EQ(macro_copies[0].src, ptr(0));
  EXPECT_EQ(macro_copies[0].dest, ptr(64));
  EXPECT_EQ(macro_copies[0].num_bytes, 32);
}

TEST(HexagonBuffer, macro_copies_discontiguous_regions_not_merged) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(16)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 12);

  std::vector<void*> dest_ptr{ptr(64), ptr(80)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 12);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 24);
  auto macro_copies = MemoryCopy::MergeAdjacent(std::move(micro_copies));

  ASSERT_EQ(macro_copies.size(), 2);

  EXPECT_EQ(macro_copies[0].src, ptr(0));
  EXPECT_EQ(macro_copies[0].dest, ptr(64));
  EXPECT_EQ(macro_copies[0].num_bytes, 12);

  EXPECT_EQ(macro_copies[1].src, ptr(16));
  EXPECT_EQ(macro_copies[1].dest, ptr(80));
  EXPECT_EQ(macro_copies[1].num_bytes, 12);
}

TEST(HexagonBuffer, macro_copies_overlapping_regions_merged) {
  auto ptr = [](auto val) { return reinterpret_cast<void*>(val); };

  std::vector<void*> src_ptr{ptr(0), ptr(12)};
  BufferSet src(src_ptr.data(), src_ptr.size(), 12);

  std::vector<void*> dest_ptr{ptr(64), ptr(80)};
  BufferSet dest(dest_ptr.data(), dest_ptr.size(), 16);

  auto micro_copies = BufferSet::MemoryCopies(dest, src, 24);
  auto macro_copies = MemoryCopy::MergeAdjacent(std::move(micro_copies));

  ASSERT_EQ(macro_copies.size(), 1);
  EXPECT_EQ(macro_copies[0].src, ptr(0));
  EXPECT_EQ(macro_copies[0].dest, ptr(64));
  EXPECT_EQ(macro_copies[0].num_bytes, 24);
}

TEST(HexagonBuffer, copy_from) {
  Optional<String> scope("global");
  HexagonBuffer hb(8 /* nbytes */, 8 /* alignment */, scope);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb.CopyFrom(data.data(), data.size());

  uint8_t* ptr = static_cast<uint8_t*>(hb.GetPointer());
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
}

TEST(HexagonBuffer, copy_from_smaller_size) {
  Optional<String> scope("global");
  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};

  // HexagonBuffer is big
  HexagonBuffer big(16 /* nbytes */, 16 /* alignment */, scope);
  EXPECT_NO_THROW(big.CopyFrom(data.data(), data.size()));
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

  uint8_t** ptr = static_cast<uint8_t**>(hb.GetPointer());
  EXPECT_EQ(ptr[0][0], data[0]);
  EXPECT_EQ(ptr[0][1], data[1]);
  EXPECT_EQ(ptr[0][2], data[2]);
  EXPECT_EQ(ptr[0][3], data[3]);
  EXPECT_EQ(ptr[1][0], data[4]);
  EXPECT_EQ(ptr[1][1], data[5]);
  EXPECT_EQ(ptr[1][2], data[6]);
  EXPECT_EQ(ptr[1][3], data[7]);
}

TEST(HexagonBuffer, 1d_copy_from_1d) {
  Optional<String> global("global");
  HexagonBuffer from(8 /* nbytes */, 8 /* alignment */, global);

  Optional<String> vtcm("global.vtcm");
  HexagonBuffer to(8 /* nbytes */, 8 /* alignment */, vtcm);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  from.CopyFrom(data.data(), data.size());
  to.CopyFrom(from, 8);

  uint8_t* ptr = static_cast<uint8_t*>(to.GetPointer());
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
  hb2d.CopyFrom(hb1d, 8);

  uint8_t** ptr = static_cast<uint8_t**>(hb2d.GetPointer());
  EXPECT_EQ(ptr[0][0], data[0]);
  EXPECT_EQ(ptr[0][1], data[1]);
  EXPECT_EQ(ptr[0][2], data[2]);
  EXPECT_EQ(ptr[0][3], data[3]);
  EXPECT_EQ(ptr[1][0], data[4]);
  EXPECT_EQ(ptr[1][1], data[5]);
  EXPECT_EQ(ptr[1][2], data[6]);
  EXPECT_EQ(ptr[1][3], data[7]);
}

TEST(HexagonBuffer, 1d_copy_from_2d) {
  Optional<String> vtcm("global.vtcm");
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, vtcm);

  Optional<String> global("global.vtcm");
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, global);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7};
  hb2d.CopyFrom(data.data(), data.size());
  hb1d.CopyFrom(hb2d, 8);

  uint8_t* ptr = static_cast<uint8_t*>(hb1d.GetPointer());
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(ptr[i], data[i]);
  }
}

TEST(HexagonBuffer, nd_copy_from_nd_invalid_size) {
  Optional<String> scope("global");
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, scope);
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, scope);

  HexagonBuffer toosbig1d(16 /* nbytes */, 16 /* alignment */, scope);
  EXPECT_THROW(hb1d.CopyFrom(toosbig1d, 16), InternalError);
  EXPECT_THROW(hb2d.CopyFrom(toosbig1d, 16), InternalError);

  HexagonBuffer toobig2d(2 /* ndim */, 16 /* nbytes */, 16 /* alignment */, scope);
  EXPECT_THROW(hb1d.CopyFrom(toobig2d, 32), InternalError);
  EXPECT_THROW(hb2d.CopyFrom(toobig2d, 32), InternalError);
}

TEST(HexagonBuffer, nd_copy_from_nd_smaller_size) {
  Optional<String> scope("global");
  HexagonBuffer hb1d(8 /* nbytes */, 8 /* alignment */, scope);
  HexagonBuffer hb2d(2 /* ndim */, 4 /* nbytes */, 8 /* alignment */, scope);

  HexagonBuffer small1d(4 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_NO_THROW(hb1d.CopyFrom(small1d, 4));
  EXPECT_NO_THROW(hb2d.CopyFrom(small1d, 4));

  HexagonBuffer small2d(2 /* ndim */, 2 /* nbytes */, 8 /* alignment */, scope);
  EXPECT_NO_THROW(hb1d.CopyFrom(small2d, 4));
  EXPECT_NO_THROW(hb2d.CopyFrom(small2d, 4));
}

TEST(HexagonBuffer, md_copy_from_nd) {
  Optional<String> scope("global");
  HexagonBuffer hb3d(3 /* ndim */, 4 /* nbytes */, 8 /* alignment */, scope);
  HexagonBuffer hb4d(4 /* ndim */, 3 /* nbytes */, 8 /* alignment */, scope);

  std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  hb3d.CopyFrom(data.data(), data.size());
  hb4d.CopyFrom(hb3d, data.size());

  uint8_t** hb3d_ptr = static_cast<uint8_t**>(hb3d.GetPointer());
  uint8_t** hb4d_ptr = static_cast<uint8_t**>(hb4d.GetPointer());
  for (size_t i = 0; i < 12; i++) {
    EXPECT_EQ(hb3d_ptr[i / 4][i % 4], hb4d_ptr[i / 3][i % 3]);
  }
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
