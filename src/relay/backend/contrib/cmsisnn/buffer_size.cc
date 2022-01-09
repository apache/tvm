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

#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>

#include "compiler_attrs.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

int Conv2dBufferSize(CMSISNNFlags flags, int32_t padding_w, int32_t padding_h, int32_t input_n,
                     int32_t input_h, int32_t input_c, int32_t output_h, int32_t output_w,
                     int32_t stride_w, int32_t stride_h, int32_t filter_w, int32_t filter_h) {
  bool is1x1 = (padding_w == 0) && (padding_h == 0) && (input_c % 4 == 0) && (stride_w == 1) &&
               (stride_h == 1) && (filter_w == 1) && (filter_h == 1);
  bool is1xN =
      (output_h == 1) && (input_h == 1) && (filter_h == 1) && (output_w % 4 == 0) && (input_n == 1);

  if (is1x1) {
    return 0;
  }

  if (is1xN) {
    if (flags.dsp && !flags.mve) {
      return (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
    }
    return 0;
  }

  if (flags.dsp) {
    return (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
  }
  return 0;
}

int DepthwiseConv2dBufferSize(CMSISNNFlags flags, int32_t input_n, int32_t input_c,
                              int32_t output_c, int32_t filter_w, int32_t filter_h) {
  if (input_c == output_c && input_n == 1) {
    if (flags.mve) {
      return (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t) + 4;
    }
    if (flags.dsp) {
      return (input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
    }
  }
  return 0;
}

int AvgPoolBufferSize(CMSISNNFlags flags, int32_t input_c) {
  if (flags.dsp && !flags.mve) {
    return (input_c * sizeof(int32_t));
  }
  return 0;
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
