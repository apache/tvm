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

#include <HAP_farf.h>
#include <dlpack/dlpack.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <ctime>
#include <functional>
#include <string>
#include <tuple>

#include "tvm/runtime/hexagon/ops/conv2d.h"

using namespace tvm::runtime::hexagon;

// Parameterized test fixture with 4 params representing n, h, w, c
class HexagonConvUtilsTest : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {
 public:
  void SetUp() override {
    vtcm_scope = "global.vtcm";
    device_api = tvm::runtime::DeviceAPI::Get(hexagon_device, false);
    float16.code = kDLFloat;
    float16.bits = 16;
    float16.lanes = 1;

    // TODO (quic-sanirudh): Figure out if it's possible to test with randomized shapes
    shape[0] = 1;
    shape[1] = 14;
    shape[2] = 7;
    shape[3] = 60;

    auto [n, h, w, c] = shape;
    int64_t shape_1d[1] = {n * h * w * c};

    flat_mem = device_api->AllocDataSpace(hexagon_device, 1, shape_1d, float16, vtcm_scope);
    flat_mem_data = static_cast<uint16_t*>(flat_mem);
    fill_vals(flat_mem_data, shape_1d[0]);
  }

  tvm::runtime::DeviceAPI* device_api;
  tvm::runtime::String vtcm_scope;
  tvm::runtime::String global_scope;
  DLDataType float16;
  void* flat_mem;
  uint16_t* flat_mem_data;
  int shape[4];

  static int flattened_idx(int nn, int hh, int ww, int cc, int* shape) {
    int h = shape[1];
    int w = shape[2];
    int c = shape[3];
    return cc + c * (ww + w * (hh + h * (nn)));
  }

  void TearDown() override { device_api->FreeDataSpace(hexagon_device, flat_mem); }

  static void fill_vals(uint16_t* arr, int size) {
    // Testing with uint16 instead of float16 as generating random float16 is not easy within c++
    uint16_t max = UINT16_MAX;
    srand(42);
    for (int i = 0; i < size; ++i) {
      arr[i] = static_cast<uint16_t>(std::rand() % max);
    }
  }
};

// TODO (quic-sanirudh): See if we can test with random generated indices
INSTANTIATE_TEST_SUITE_P(
    BlockizeDeblockizeTestFixtures, HexagonConvUtilsTest,
    ::testing::Values(std::make_tuple(0, 0, 0, 0),    // first element
                      std::make_tuple(0, 7, 3, 31),   // last element of first block
                      std::make_tuple(0, 13, 6, 59),  // Last element of entire data
                      std::make_tuple(0, 0, 0, 32),   // First element of second block
                      std::make_tuple(0, 0, 4, 32),   // First element of fourth block
                      std::make_tuple(0, 2, 3, 4),    // Random element 1
                      std::make_tuple(0, 5, 6, 7),    // Random element 2
                      std::make_tuple(0, 10, 4, 12)   // Random element 3
                      ),
    [](const ::testing::TestParamInfo<HexagonConvUtilsTest::ParamType>& info) {
      // Can use info.param here to generate the test suffix
      int h = std::get<1>(info.param);
      int w = std::get<2>(info.param);
      int c = std::get<3>(info.param);
      // Generate test name as "hwc0x0x0" if the indices of hwc are 0,0,0
      std::string name =
          "hwc" + std::to_string(h) + "x" + std::to_string(w) + "x" + std::to_string(c);
      return name;
    });

TEST_P(HexagonConvUtilsTest, blockize_hwc_16b) {
  auto [n, h, w, c] = shape;

  int h_rounded = round_up(h, 8);
  int w_rounded = round_up(w, 4);
  int c_rounded = round_up(c, 32);
  int64_t shape_2d[2] = {(n * h_rounded * w_rounded * c_rounded) / (8 * 4 * 32), 8 * 4 * 32};

  void* blocked_mem = device_api->AllocDataSpace(hexagon_device, 2, shape_2d, float16, vtcm_scope);
  int blocked_shape[] = {n, h_rounded / 8, w_rounded / 4, c_rounded / 32};
  blockize_hwc_16b(blocked_mem, flat_mem, h, w, c);

  std::function<int(int, int, int, int, int*)> flatten = HexagonConvUtilsTest::flattened_idx;

  auto getBlockedElem = [&blocked_shape, blocked_mem, flatten](int nn, int hh, int ww, int cc) {
    auto* blocks = static_cast<uintptr_t*>(blocked_mem);
    int blockIdx = flatten(nn, hh / 8, ww / 4, cc / 32, blocked_shape);
    uint16_t* block = reinterpret_cast<uint16_t*>(blocks[blockIdx]);
    return block[xyc_to_sm_16b(hh % 8, ww % 4, cc % 32)];
  };

  auto [nn, hh, ww, cc] = GetParam();

  EXPECT_EQ(flat_mem_data[flattened_idx(nn, hh, ww, cc, shape)], getBlockedElem(nn, hh, ww, cc));

  device_api->FreeDataSpace(hexagon_device, blocked_mem);
}

TEST_P(HexagonConvUtilsTest, deblockize_hwc_16b) {
  auto [n, h, w, c] = shape;
  int64_t shape_1d[1] = {n * h * w * c};

  int h_rounded = round_up(h, 8);
  int w_rounded = round_up(w, 4);
  int c_rounded = round_up(c, 32);
  int64_t shape_2d[2] = {(n * h_rounded * w_rounded * c_rounded) / (8 * 4 * 32), 8 * 4 * 32};

  void* blocked_mem = device_api->AllocDataSpace(hexagon_device, 2, shape_2d, float16, vtcm_scope);
  blockize_hwc_16b(blocked_mem, flat_mem, h, w, c);

  void* deblocked_flat_mem =
      device_api->AllocDataSpace(hexagon_device, 1, shape_1d, float16, vtcm_scope);
  deblockize_hwc_16b(deblocked_flat_mem, blocked_mem, h, w, c);
  auto* deblocked_flat_mem_data = static_cast<uint16_t*>(deblocked_flat_mem);

  auto [nn, hh, ww, cc] = GetParam();

  auto idx = flattened_idx(nn, hh, ww, cc, shape);
  EXPECT_EQ(flat_mem_data[idx], deblocked_flat_mem_data[idx]);

  device_api->FreeDataSpace(hexagon_device, blocked_mem);
  device_api->FreeDataSpace(hexagon_device, deblocked_flat_mem);
}
