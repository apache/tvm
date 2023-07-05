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

#ifndef TVM_TESTS_CPPRUNTIME_HEXAGON_HEXAGON_CONV_UTILS_H
#define TVM_TESTS_CPPRUNTIME_HEXAGON_HEXAGON_CONV_UTILS_H

#include <dlpack/dlpack.h>
#include <gtest/gtest.h>

#include <limits>

#include "conv2d.h"

using namespace tvm::runtime::hexagon::conv_utils;

template <typename T>
class HexagonUtilsTest : public ::testing::Test {
 public:
  void SetUp() override {
    vtcm_scope = "global.vtcm";
    device_api = tvm::runtime::DeviceAPI::Get(hexagon_device, false);
    float16.code = kDLFloat;
    float16.bits = 16;
    float16.lanes = 1;

    uint8.code = kDLUInt;
    uint8.bits = 8;
    uint8.lanes = 1;

    int8.code = kDLInt;
    int8.bits = 8;
    int8.lanes = 1;
  }

  void setupTensor(std::tuple<int64_t, int64_t, int64_t, int64_t> shape, DLDataType dtype) {
    auto [s1, s2, s3, s4] = shape;
    tensor_shape[0] = s1;
    tensor_shape[1] = s2;
    tensor_shape[2] = s3;
    tensor_shape[3] = s4;
    int64_t shape_1d[1] = {s1 * s2 * s3 * s4};

    flat_mem = device_api->AllocDataSpace(hexagon_device, 1, shape_1d, dtype, vtcm_scope);
    flat_mem_data = static_cast<T*>(flat_mem);
    fill_vals(flat_mem_data, shape_1d[0]);

    flat_tensor.data = flat_mem;
    flat_tensor.device = hexagon_device;
    flat_tensor.ndim = 4;
    flat_tensor.dtype = dtype;
    flat_tensor.shape = tensor_shape;
    flat_tensor.strides = nullptr;
    flat_tensor.byte_offset = 0;
  }

  void TearDownTensor() {
    if (flat_tensor.data) device_api->FreeDataSpace(hexagon_device, flat_mem);
  }

  static void fill_vals(T* arr, int size) {
    // Testing with uint16 instead of float16 as generating random float16 is not easy within c++
    auto max = std::numeric_limits<T>::max();
    srand(std::time(0));
    for (int i = 0; i < size; ++i) {
      arr[i] = static_cast<T>(std::rand() % max);
    }
  }

  static int flattened_idx(int nn, int hh, int ww, int cc, int64_t* shape) {
    int h = shape[1];
    int w = shape[2];
    int c = shape[3];
    return cc + c * (ww + w * (hh + h * (nn)));
  }

  DLTensor flat_tensor;
  void* flat_mem;
  T* flat_mem_data;
  tvm::runtime::DeviceAPI* device_api;
  tvm::runtime::String vtcm_scope;
  DLDataType float16;
  DLDataType int8, uint8;
  int64_t tensor_shape[4];
};

#endif
