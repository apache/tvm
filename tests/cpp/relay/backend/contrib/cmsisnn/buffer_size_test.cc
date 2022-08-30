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

static const Target kHasMVE("cmsis-nn -mcpu=cortex-m55");
static const Target kHasDSP("cmsis-nn -mcpu=cortex-m55 -mattr=+nomve");
static const Target kNoExt("cmsis-nn -mcpu=cortex-m55 -mattr=+nodsp,+nomve");

class CMSISNNCalculatedBufferSize : public testing::TestWithParam<std::array<int32_t, 3>> {};

TEST(CMSISNNConv2dBufferSize, Conv1x1) {
  int32_t any = fake_parameters(gen);
  auto conv2d_1x1 = [=](Target target, int32_t input_c) {
    return Conv2dBufferSize(target, 0, 0, any, any, input_c, any, any, 1, 1, 1, 1, 1, 1);
  };

  ASSERT_EQ(conv2d_1x1(kNoExt, 4), 0);
  ASSERT_EQ(conv2d_1x1(kNoExt, 8), 0);
  ASSERT_EQ(conv2d_1x1(kNoExt, 12), 0);
  ASSERT_EQ(conv2d_1x1(kNoExt, 16), 0);
  ASSERT_EQ(conv2d_1x1(kNoExt, 32), 0);

  ASSERT_EQ(conv2d_1x1(kHasDSP, 4), 0);
  ASSERT_EQ(conv2d_1x1(kHasDSP, 8), 0);
  ASSERT_EQ(conv2d_1x1(kHasDSP, 12), 0);
  ASSERT_EQ(conv2d_1x1(kHasDSP, 16), 0);
  ASSERT_EQ(conv2d_1x1(kHasDSP, 32), 0);

  ASSERT_EQ(conv2d_1x1(kHasMVE, 4), 0);
  ASSERT_EQ(conv2d_1x1(kHasMVE, 8), 0);
  ASSERT_EQ(conv2d_1x1(kHasMVE, 12), 0);
  ASSERT_EQ(conv2d_1x1(kHasMVE, 16), 0);
  ASSERT_EQ(conv2d_1x1(kHasMVE, 32), 0);
}

TEST(CMSISNNConv2dBufferSize, Conv1xN) {
  int32_t any = fake_parameters(gen);
  int32_t input_c = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = 1;
  int32_t calculated_buffer = (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);

  auto conv2d_1xn = [=](Target target, int32_t output_w) {
    return Conv2dBufferSize(target, any, any, 1, 1, input_c, 1, output_w, any, any, 1, 1, filter_w,
                            filter_h);
  };

  ASSERT_EQ(conv2d_1xn(kNoExt, 4), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kNoExt, 8), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kNoExt, 12), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kNoExt, 16), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kNoExt, 32), calculated_buffer);

  ASSERT_EQ(conv2d_1xn(kHasDSP, 4), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kHasDSP, 8), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kHasDSP, 12), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kHasDSP, 16), calculated_buffer);
  ASSERT_EQ(conv2d_1xn(kHasDSP, 32), calculated_buffer);

  ASSERT_EQ(conv2d_1xn(kHasMVE, 4), 0);
  ASSERT_EQ(conv2d_1xn(kHasMVE, 8), 0);
  ASSERT_EQ(conv2d_1xn(kHasMVE, 12), 0);
  ASSERT_EQ(conv2d_1xn(kHasMVE, 16), 0);
  ASSERT_EQ(conv2d_1xn(kHasMVE, 32), 0);
}

TEST(CMSISNNConv2dBufferSize, Default) {
  int32_t any = fake_parameters(gen);

  int32_t input_c = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);
  int32_t calculated_buffer = (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
  int32_t col_length = input_c * filter_w * filter_h;
  col_length = (col_length + 7) / 8;
  int32_t calculated_buffer_mve = 4 * col_length * 8 * (int32_t)sizeof(int8_t);

  auto conv2d = [=](Target target, int32_t output_w) {
    return Conv2dBufferSize(target, any, any, 1, 1, input_c, 1, output_w, any, any, any, any,
                            filter_w, filter_h);
  };

  ASSERT_EQ(conv2d(kNoExt, 4), calculated_buffer);
  ASSERT_EQ(conv2d(kNoExt, 8), calculated_buffer);
  ASSERT_EQ(conv2d(kNoExt, 12), calculated_buffer);
  ASSERT_EQ(conv2d(kNoExt, 16), calculated_buffer);
  ASSERT_EQ(conv2d(kNoExt, 32), calculated_buffer);

  ASSERT_EQ(conv2d(kHasDSP, 4), calculated_buffer);
  ASSERT_EQ(conv2d(kHasDSP, 8), calculated_buffer);
  ASSERT_EQ(conv2d(kHasDSP, 12), calculated_buffer);
  ASSERT_EQ(conv2d(kHasDSP, 16), calculated_buffer);
  ASSERT_EQ(conv2d(kHasDSP, 32), calculated_buffer);

  ASSERT_EQ(conv2d(kHasMVE, 4), calculated_buffer_mve);
  ASSERT_EQ(conv2d(kHasMVE, 8), calculated_buffer_mve);
  ASSERT_EQ(conv2d(kHasMVE, 12), calculated_buffer_mve);
  ASSERT_EQ(conv2d(kHasMVE, 16), calculated_buffer_mve);
  ASSERT_EQ(conv2d(kHasMVE, 32), calculated_buffer_mve);
}

TEST(CMSISNNDepthwiseConv2dBufferSize, UnEvenChannels) {
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);
  int32_t input_n = 1;

  auto depthwise_conv2d_with_channels = [=](Target target, int32_t input_c, int32_t output_c) {
    return DepthwiseConv2dBufferSize(target, input_n, input_c, output_c, filter_w, filter_h);
  };

  ASSERT_EQ(depthwise_conv2d_with_channels(kNoExt, 4, 6), 0);
  ASSERT_EQ(depthwise_conv2d_with_channels(kNoExt, 8, 7), 0);
  ASSERT_EQ(depthwise_conv2d_with_channels(kHasDSP, 4, 6), 0);
  ASSERT_EQ(depthwise_conv2d_with_channels(kHasDSP, 8, 7), 0);
  ASSERT_EQ(depthwise_conv2d_with_channels(kHasMVE, 4, 6), 0);
  ASSERT_EQ(depthwise_conv2d_with_channels(kHasMVE, 8, 7), 0);
}

TEST(CMSISNNDepthwiseConv2dBufferSize, MultipleBatches) {
  int32_t input_output_c = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);

  auto depthwise_conv2d_with_batch = [=](Target target, int32_t input_n) {
    return DepthwiseConv2dBufferSize(target, input_n, input_output_c, input_output_c, filter_w,
                                     filter_h);
  };

  ASSERT_EQ(depthwise_conv2d_with_batch(kNoExt, 4), 0);
  ASSERT_EQ(depthwise_conv2d_with_batch(kNoExt, 7), 0);
  ASSERT_EQ(depthwise_conv2d_with_batch(kHasDSP, 4), 0);
  ASSERT_EQ(depthwise_conv2d_with_batch(kHasDSP, 7), 0);
  ASSERT_EQ(depthwise_conv2d_with_batch(kHasMVE, 4), 0);
  ASSERT_EQ(depthwise_conv2d_with_batch(kHasMVE, 7), 0);
}

TEST(CMSISNNDepthwiseConv2dBufferSize, Default) {
  int32_t input_output_c = fake_parameters(gen);
  int32_t filter_w = fake_parameters(gen);
  int32_t filter_h = fake_parameters(gen);
  int32_t input_n = 1;

  int32_t mve_calculated_buffer =
      (2 * input_output_c * filter_w * filter_h) * (int32_t)sizeof(int16_t) + 4;
  int32_t dsp_calculated_buffer = (input_output_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);

  auto depthwise_conv2d = [=](Target target) {
    return DepthwiseConv2dBufferSize(target, input_n, input_output_c, input_output_c, filter_w,
                                     filter_h);
  };

  ASSERT_EQ(depthwise_conv2d(kNoExt), 0);
  ASSERT_EQ(depthwise_conv2d(kNoExt), 0);
  ASSERT_EQ(depthwise_conv2d(kHasDSP), dsp_calculated_buffer);
  ASSERT_EQ(depthwise_conv2d(kHasDSP), dsp_calculated_buffer);
  ASSERT_EQ(depthwise_conv2d(kHasMVE), mve_calculated_buffer);
  ASSERT_EQ(depthwise_conv2d(kHasMVE), mve_calculated_buffer);
}

TEST(CMSISNNAvgPoolBufferSize, Default) {
  int32_t input_c = fake_parameters(gen);
  int32_t calculated_buffer = (input_c * sizeof(int32_t));

  auto avg_pool = [=](Target target) { return AvgPoolBufferSize(target, input_c); };

  ASSERT_EQ(avg_pool(kNoExt), 0);
  ASSERT_EQ(avg_pool(kHasDSP), calculated_buffer);
  ASSERT_EQ(avg_pool(kHasMVE), 0);
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif
