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

#include <dlpack/dlpack.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <ctime>
#include <functional>
#include <string>
#include <tuple>

#include "conv2d.h"
#include "hexagon_conv_utils_test.h"

using namespace tvm::runtime::hexagon;

// Parameterized test fixture with 4 params representing n, h, w, c
class HexagonUtilsActivationsBlockizeTest
    : public HexagonUtilsTest<uint16_t>,
      public ::testing::WithParamInterface<std::tuple<
          std::tuple<int64_t, int64_t, int64_t, int64_t>, std::tuple<int, int, int, int>>> {};

// TODO (quic-sanirudh): See if we can test with random generated indices
INSTANTIATE_TEST_SUITE_P(
    BlockizeDeblockizeTestFixtures, HexagonUtilsActivationsBlockizeTest,
    ::testing::Combine(::testing::Values(std::make_tuple(1, 14, 7, 60)),
                       ::testing::Values(std::make_tuple(0, 0, 0, 0),   // first element
                                         std::make_tuple(0, 7, 3, 31),  // last element
                                         // Remaining are random element tests
                                         std::make_tuple(0, 13, 6, 59),
                                         std::make_tuple(0, 0, 0, 32), std::make_tuple(0, 0, 4, 32),
                                         std::make_tuple(0, 2, 3, 4), std::make_tuple(0, 5, 6, 7),
                                         std::make_tuple(0, 10, 4, 12))),
    [](const ::testing::TestParamInfo<HexagonUtilsActivationsBlockizeTest::ParamType>& info) {
      // Can use info.param here to generate the test suffix
      auto indices = std::get<1>(info.param);
      int h = std::get<1>(indices);
      int w = std::get<2>(indices);
      int c = std::get<3>(indices);
      // Generate test name as "hwc0x0x0" if the indices of hwc are 0,0,0
      std::string name =
          "hwc" + std::to_string(h) + "x" + std::to_string(w) + "x" + std::to_string(c);
      return name;
    });

TEST_F(HexagonUtilsActivationsBlockizeTest, prepare_nhwc) {
  auto shape = std::make_tuple(1, 14, 7, 60);
  auto [n, h, w, c] = shape;
  setupTensor(shape, float16);

  // // copy_data is set to false here as there's a separate test for blockize when copy_data
  // becomes true
  auto blocked_tensor =
      prepare_nhwc<uint16_t, 8, 4, 32>(device_api, &flat_tensor, /*copy_data=*/false);

  EXPECT_EQ(blocked_tensor.shape[0], n);
  EXPECT_EQ(blocked_tensor.shape[1], round_up(h, 8) / 8);
  EXPECT_EQ(blocked_tensor.shape[2], round_up(w, 4) / 4);
  EXPECT_EQ(blocked_tensor.shape[3], round_up(c, 32) / 32);

  TearDownTensor();
  release(device_api, blocked_tensor);
}

TEST_P(HexagonUtilsActivationsBlockizeTest, blockize_hwc_16b) {
  auto shape_tuple = std::get<0>(GetParam());
  setupTensor(shape_tuple, float16);
  auto [n, h, w, c] = shape_tuple;
  int64_t shape[] = {n, h, w, c};

  int h_rounded = round_up(h, 8);
  int w_rounded = round_up(w, 4);
  int c_rounded = round_up(c, 32);
  int64_t shape_2d[2] = {(n * h_rounded * w_rounded * c_rounded) / (8 * 4 * 32), 8 * 4 * 32};

  void* blocked_mem = device_api->AllocDataSpace(hexagon_device, 2, shape_2d, float16, vtcm_scope);
  int64_t blocked_shape[] = {n, h_rounded / 8, w_rounded / 4, c_rounded / 32};
  blockize_hwc<uint16_t, 8, 4, 32>(blocked_mem, flat_mem, h, w, c);

  std::function<int(int, int, int, int, int64_t*)> flatten =
      HexagonUtilsActivationsBlockizeTest::flattened_idx;

  auto getBlockedElem = [&blocked_shape, blocked_mem, flatten](int nn, int hh, int ww, int cc) {
    auto* blocks = static_cast<uintptr_t*>(blocked_mem);
    int blockIdx = flatten(nn, hh / 8, ww / 4, cc / 32, blocked_shape);
    uint16_t* block = reinterpret_cast<uint16_t*>(blocks[blockIdx]);
    return block[yxc_to_sm_16b(hh % 8, ww % 4, cc % 32)];
  };

  auto [nn, hh, ww, cc] = std::get<1>(GetParam());

  EXPECT_EQ(flat_mem_data[flattened_idx(nn, hh, ww, cc, shape)], getBlockedElem(nn, hh, ww, cc));

  TearDownTensor();
  device_api->FreeDataSpace(hexagon_device, blocked_mem);
}

TEST_P(HexagonUtilsActivationsBlockizeTest, deblockize_hwc_16b) {
  auto shape_tuple = std::get<0>(GetParam());
  setupTensor(shape_tuple, float16);
  auto [n, h, w, c] = shape_tuple;
  int64_t shape[] = {n, h, w, c};
  int64_t shape_1d[1] = {n * h * w * c};

  int h_rounded = round_up(h, 8);
  int w_rounded = round_up(w, 4);
  int c_rounded = round_up(c, 32);
  int64_t shape_2d[2] = {(n * h_rounded * w_rounded * c_rounded) / (8 * 4 * 32), 8 * 4 * 32};

  void* blocked_mem = device_api->AllocDataSpace(hexagon_device, 2, shape_2d, float16, vtcm_scope);
  blockize_hwc<uint16_t, 8, 4, 32>(blocked_mem, flat_mem, h, w, c);

  void* deblocked_flat_mem =
      device_api->AllocDataSpace(hexagon_device, 1, shape_1d, float16, vtcm_scope);
  deblockize_hwc<uint16_t, 8, 4, 32>(deblocked_flat_mem, blocked_mem, h, w, c);
  auto* deblocked_flat_mem_data = static_cast<uint16_t*>(deblocked_flat_mem);

  auto [nn, hh, ww, cc] = std::get<1>(GetParam());

  auto idx = flattened_idx(nn, hh, ww, cc, shape);
  EXPECT_EQ(flat_mem_data[idx], deblocked_flat_mem_data[idx]);

  TearDownTensor();
  device_api->FreeDataSpace(hexagon_device, blocked_mem);
  device_api->FreeDataSpace(hexagon_device, deblocked_flat_mem);
}

class HexagonUtilsWeightsChunkifyTest
    : public HexagonUtilsTest<uint16_t>,
      public ::testing::WithParamInterface<std::tuple<
          std::tuple<int64_t, int64_t, int64_t, int64_t>, std::tuple<int, int, int, int>>> {};

INSTANTIATE_TEST_SUITE_P(
    ChunkifyDechunkifyTests, HexagonUtilsWeightsChunkifyTest,
    ::testing::Combine(::testing::Values(std::make_tuple(3, 3, 40, 40)),
                       ::testing::Values(std::make_tuple(0, 0, 0, 0),    // first element
                                         std::make_tuple(2, 2, 39, 39),  // Last element
                                         // Remaining are random element tests
                                         std::make_tuple(1, 1, 28, 33),
                                         std::make_tuple(1, 2, 8, 38),
                                         std::make_tuple(1, 0, 12, 15),
                                         std::make_tuple(2, 1, 9, 22), std::make_tuple(0, 2, 6, 7),
                                         std::make_tuple(1, 2, 3, 4))),
    [](const ::testing::TestParamInfo<HexagonUtilsWeightsChunkifyTest::ParamType>& info) {
      // Can use info.param here to generate the test suffix
      auto indices = std::get<1>(info.param);
      int h = std::get<0>(indices);
      int w = std::get<1>(indices);
      int i = std::get<2>(indices);
      int o = std::get<3>(indices);
      // Generate test name as "hwc0x0x0" if the indices of hwc are 0,0,0
      std::string name = "hwio" + std::to_string(h) + std::to_string(w) + "x" + std::to_string(i) +
                         "x" + std::to_string(o);
      return name;
    });

TEST_F(HexagonUtilsWeightsChunkifyTest, calculate_num_weight_chunks) {
  int64_t shape[] = {3, 3, 40, 40};
  int num_wgt_chunks =
      calculate_num_weight_chunks(shape, /* chunk_height */ 8, /* chunk_width */ 4,
                                  /* chunk_in_channel */ 32, /* chunk_out_channel */ 32);
  EXPECT_EQ(num_wgt_chunks, 4);
}

TEST_F(HexagonUtilsWeightsChunkifyTest, prepare_hwio) {
  int64_t shape[] = {3, 3, 40, 40};
  auto [h, w, i, o] = shape;
  auto shape_tuple = std::make_tuple(h, w, i, o);
  setupTensor(shape_tuple, float16);

  // copy_data is set to false here as there's a separate test for blockize when copy_data becomes
  // true
  auto num_wgt_chunks = calculate_num_weight_chunks(shape, 8, 4, 32, 32);
  auto wgt_ptr_table =
      reinterpret_cast<void**>(__builtin_alloca(num_wgt_chunks * sizeof(uintptr_t)));
  auto chunked_tensor = prepare_hwio(device_api, &flat_tensor, num_wgt_chunks, wgt_ptr_table);

  EXPECT_EQ(chunked_tensor.shape[0], round_up(h, 8) / 8);
  EXPECT_EQ(chunked_tensor.shape[1], round_up(w, 4) / 4);
  EXPECT_EQ(chunked_tensor.shape[2], round_up(i, 32) / 32);
  EXPECT_EQ(chunked_tensor.shape[3], round_up(o, 32) / 32);

  release(device_api, chunked_tensor);
  TearDownTensor();
}

TEST_P(HexagonUtilsWeightsChunkifyTest, chunkify_hwio_16b) {
  auto [shape_tuple, indices] = GetParam();
  auto [h, w, i, o] = shape_tuple;
  setupTensor(shape_tuple, float16);
  int64_t shape[] = {h, w, i, o};

  auto num_wgt_chunks = calculate_num_weight_chunks(shape, 8, 4, 32, 32);
  auto wgt_ptr_table =
      reinterpret_cast<void**>(__builtin_alloca(num_wgt_chunks * sizeof(uintptr_t)));
  auto chunked_tensor = prepare_hwio(device_api, &flat_tensor, num_wgt_chunks, wgt_ptr_table);

  int rd = w - (w % 4);  // round down by 4 for width
  int thin_w = w - rd;

  auto getChunkedElem = [thin_w, chunked_tensor](int hh, int ww, int ii, int oo) {
    int fcw = 0;
    if (ww >= thin_w) {
      fcw = (ww - thin_w) / 4 + 1;
      ww = (ww - thin_w) % 4;
    }
    auto chunk = hwio_at(chunked_tensor, hh / 8, fcw, ii / 32, oo / 32);
    auto chunk_uint16 = reinterpret_cast<uint16_t*>(chunk);
    return chunk_uint16[hwio_to_sm_16b(thin_w, hh % 8, ww, ii % 32, oo % 32)];
  };

  auto [hh, ww, ii, oo] = indices;

  EXPECT_EQ(flat_mem_data[flattened_idx(hh, ww, ii, oo, shape)], getChunkedElem(hh, ww, ii, oo));
  release(device_api, chunked_tensor);
}
