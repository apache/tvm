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

#ifdef TVM_USE_CMSISNN

#include "../../../../../../src/relay/backend/contrib/cmsisnn/buffer_size.h"

#include <gtest/gtest.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <cmath>
#include <list>
#include <random>
#include <string>

#include "../../../../../../src/relay/backend/contrib/cmsisnn/compiler_attrs.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_int_distribution<> fake_parameters(2, 100);

class CMSISNNCalculatedBufferSize : public testing::TestWithParam<std::array<int32_t, 3>> {};

TEST(CMSISNNConv2dBufferSizeInt8, Conv1x1) {
  int32_t any = fake_parameters(gen);
  auto conv2d_1x1 = [=](bool has_mve, bool has_dsp, Dims& input_dims) {
    return Convolve1x1S8FastGetBufferSize(has_mve, has_dsp, &input_dims);
  };

  std::list<int> input_c = {4, 8, 12, 16, 32};
  for (const auto& c : input_c) {
    Dims input_dims = {any, any, any, c};
    ASSERT_EQ(conv2d_1x1(false, false, input_dims), 0);
  }
  for (const auto& c : input_c) {
    Dims input_dims = {any, any, any, c};
    ASSERT_EQ(conv2d_1x1(false, true, input_dims), 0);
  }
  for (const auto& c : input_c) {
    Dims input_dims = {any, any, any, c};
    ASSERT_EQ(conv2d_1x1(true, true, input_dims), 0);
  }
}

TEST(CMSISNNConv2dBufferSizeInt8, Conv1xN) {
  int32_t any = fake_parameters(gen);
  int32_t input_c = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = 1;
  int32_t calculated_buffer = (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);

  auto conv2d_1xn = [=](bool has_mve, bool has_dsp, Dims& input_dims, Dims& filter_dims) {
    return Convolve1XNS8GetBufferSize(has_mve, has_dsp, &input_dims, &filter_dims);
  };

  Dims input_dims = {any, any, any, input_c};
  Dims filter_dims = {any, filter_h, filter_w, any};
  ASSERT_EQ(conv2d_1xn(false, false, input_dims, filter_dims), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(false, true, input_dims, filter_dims), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(true, true, input_dims, filter_dims), 0);
}

TEST(CMSISNNConv2dBufferSizeInt8, Default) {
  int32_t any = fake_parameters(gen);
  int32_t input_c = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);
  int32_t calculated_buffer = (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
  int32_t col_length = input_c * filter_w * filter_h;
  col_length = (col_length + 7) / 8;
  int32_t calculated_buffer_mve = 4 * col_length * 8 * (int32_t)sizeof(int8_t);

  auto conv2d = [=](bool has_mve, bool has_dsp, Dims& input_dims, Dims& filter_dims) {
    return ConvolveS8GetBufferSize(has_mve, has_dsp, &input_dims, &filter_dims);
  };

  Dims input_dims = {any, any, any, input_c};
  Dims filter_dims = {any, filter_h, filter_w, any};
  ASSERT_EQ(conv2d(false, false, input_dims, filter_dims), calculated_buffer);
  ASSERT_EQ(conv2d(false, true, input_dims, filter_dims), calculated_buffer);
  ASSERT_EQ(conv2d(true, true, input_dims, filter_dims), calculated_buffer_mve);
}

TEST(CMSISNNConv2dBufferSizeInt16, Default) {
  int32_t any = fake_parameters(gen);
  ConvParams conv_params = {any, any, {any, any}, {any, any}, {1, 1}, {any, any}};
  Dims output_dims = {any, any, any, any};

  auto conv2d_int16 = [=](bool has_mve, bool has_dsp, int32_t input_c, int32_t filter_w,
                          int32_t filter_h) {
    Dims input_dims = {any, any, any, input_c};
    Dims filter_dims = {any, filter_h, filter_w, any};
    return ConvolveWrapperS16GetBufferSize(has_mve, has_dsp, &conv_params, &input_dims,
                                           &filter_dims, &output_dims);
  };

  auto calculated_buffer = [=](int32_t input_c, int32_t filter_w, int32_t filter_h) {
    return (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
  };

  ASSERT_EQ(conv2d_int16(false, false, 3, 5, 5), 0);
  ASSERT_EQ(conv2d_int16(false, false, 32, 3, 3), 0);

  ASSERT_EQ(conv2d_int16(false, true, 3, 3, 3), calculated_buffer(3, 3, 3));
  ASSERT_EQ(conv2d_int16(false, true, 12, 5, 5), calculated_buffer(12, 5, 5));
  ASSERT_EQ(conv2d_int16(false, true, 24, 5, 5), 0);

  ASSERT_EQ(conv2d_int16(true, true, 3, 3, 3), 0);
  ASSERT_EQ(conv2d_int16(true, true, 12, 5, 5), 0);
  ASSERT_EQ(conv2d_int16(true, true, 24, 5, 5), 0);
}

TEST(CMSISNNDepthwiseConv2dBufferSizeInt8, UnEvenChannels) {
  int32_t any = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);
  int32_t input_n = 1;
  DepthwiseConvParams conv_params = {any, any, any, {any, any}, {any, any}, {1, 1}, {any, any}};
  Dims filter_dims = {any, filter_h, filter_w, any};

  auto depthwise = [=](bool has_mve, bool has_dsp, int32_t input_c, int32_t output_c) {
    Dims input_dims = {input_n, any, any, input_c};
    Dims output_dims = {input_n, any, any, any};
    return DepthwiseConvWrapperS8GetBufferSize(has_mve, has_dsp, &conv_params, &input_dims,
                                               &filter_dims, &output_dims);
  };

  ASSERT_EQ(depthwise(false, false, 4, 6), 0);
  ASSERT_EQ(depthwise(false, false, 8, 7), 0);
  ASSERT_EQ(depthwise(false, true, 4, 6), 0);
  ASSERT_EQ(depthwise(false, true, 8, 7), 0);
  ASSERT_EQ(depthwise(true, true, 4, 6), 0);
  ASSERT_EQ(depthwise(true, true, 8, 7), 0);
}

TEST(CMSISNNDepthwiseConv2dBufferSizeInt8, MultipleBatches) {
  int32_t any = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);
  DepthwiseConvParams conv_params = {any, any, any, {any, any}, {any, any}, {1, 1}, {any, any}};
  Dims filter_dims = {any, filter_h, filter_w, any};

  auto depthwise = [=](bool has_mve, bool has_dsp, int32_t input_n) {
    Dims input_dims = {input_n, any, any, any};
    Dims output_dims = {input_n, any, any, any};
    return DepthwiseConvWrapperS8GetBufferSize(has_mve, has_dsp, &conv_params, &input_dims,
                                               &filter_dims, &output_dims);
  };

  ASSERT_EQ(depthwise(false, false, 4), 0);
  ASSERT_EQ(depthwise(false, false, 8), 0);
  ASSERT_EQ(depthwise(false, true, 4), 0);
  ASSERT_EQ(depthwise(false, true, 8), 0);
  ASSERT_EQ(depthwise(true, true, 4), 0);
  ASSERT_EQ(depthwise(true, true, 8), 0);
}

TEST(CMSISNNDepthwiseConv2dBufferSizeInt8, Default) {
#define CH_IN_BLOCK_MVE (124)
  int32_t any = fake_parameters(gen);
  int32_t input_output_c = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);
  int32_t input_n = 1;
  DepthwiseConvParams conv_params = {any, any, any, {any, any}, {any, any}, {1, 1}, {any, any}};
  Dims filter_dims = {any, filter_h, filter_w, any};
  Dims input_dims = {input_n, any, any, input_output_c};
  Dims output_dims = {input_n, any, any, input_output_c};

  int32_t mve_calculated_buffer =
      (4 * CH_IN_BLOCK_MVE * filter_w * filter_h) * (int32_t)sizeof(int8_t);
  int32_t dsp_calculated_buffer = (input_output_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);

  auto depthwise = [=](bool has_mve, bool has_dsp) {
    return DepthwiseConvWrapperS8GetBufferSize(has_mve, has_dsp, &conv_params, &input_dims,
                                               &filter_dims, &output_dims);
  };

  ASSERT_EQ(depthwise(false, false), 0);
  ASSERT_EQ(depthwise(false, false), 0);
  ASSERT_EQ(depthwise(false, true), dsp_calculated_buffer);
  ASSERT_EQ(depthwise(false, true), dsp_calculated_buffer);
  ASSERT_EQ(depthwise(true, true), mve_calculated_buffer);
  ASSERT_EQ(depthwise(true, true), mve_calculated_buffer);
}

TEST(CMSISNNDepthwiseConv2dBufferSizeInt16, Default) {
  int32_t any = fake_parameters(gen);
  int32_t input_n = 1;
  DepthwiseConvParams conv_params = {any, any, 1, {any, any}, {any, any}, {1, 1}, {any, any}};

  auto depthwise = [=](bool has_mve, bool has_dsp, int32_t input_c, int32_t filter_w,
                       int32_t filter_h) {
    Dims filter_dims = {any, filter_h, filter_w, any};
    Dims input_dims = {input_n, any, any, input_c};
    Dims output_dims = {input_n, any, any, input_c};
    return DepthwiseConvWrapperS16GetBufferSize(has_mve, has_dsp, &conv_params, &input_dims,
                                                &filter_dims, &output_dims);
  };

  auto dsp_only_buffer = [=](int32_t input_c, int32_t filter_w, int32_t filter_h) {
    return (input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
  };

  auto dsp_mve_buffer = [=](int32_t input_c, int32_t filter_w, int32_t filter_h) {
    return (4 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t) + 8;
  };

  ASSERT_EQ(depthwise(false, false, 3, 5, 5), 0);
  ASSERT_EQ(depthwise(false, false, 32, 3, 3), 0);

  ASSERT_EQ(depthwise(false, true, 3, 3, 3), dsp_only_buffer(3, 3, 3));
  ASSERT_EQ(depthwise(false, true, 12, 5, 5), dsp_only_buffer(12, 5, 5));
  ASSERT_EQ(depthwise(false, true, 24, 5, 5), 0);

  ASSERT_EQ(depthwise(true, true, 3, 3, 3), dsp_mve_buffer(3, 3, 3));
  ASSERT_EQ(depthwise(true, true, 12, 5, 5), dsp_mve_buffer(12, 5, 5));
  ASSERT_EQ(depthwise(true, true, 24, 5, 5), 0);
}

TEST(CMSISNNAvgPoolBufferSize, Default) {
  int32_t input_c = fake_parameters(gen);
  int32_t output_w = fake_parameters(gen);

  int32_t calculated_buffer = (input_c * sizeof(int32_t));

  auto avg_pool = [=](bool has_mve, bool has_dsp) {
    return AvgpoolS8GetBufferSize(has_mve, has_dsp, output_w, input_c);
  };

  ASSERT_EQ(avg_pool(false, false), 0);
  ASSERT_EQ(avg_pool(false, true), calculated_buffer);
  ASSERT_EQ(avg_pool(true, true), 0);
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif
