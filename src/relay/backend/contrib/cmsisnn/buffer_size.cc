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

#include "buffer_size.h"

#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>

#include "compiler_attrs.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

int Conv2dBufferSize(Target target, int32_t padding_w, int32_t padding_h, int32_t input_n,
                     int32_t input_h, int32_t input_c, int32_t output_h, int32_t output_w,
                     int32_t stride_w, int32_t stride_h, int32_t dilation_w, int32_t dilation_h,
                     int32_t filter_w, int32_t filter_h) {
  bool is1x1 = (padding_w == 0) && (padding_h == 0) && (input_c % 4 == 0) && (stride_w == 1) &&
               (stride_h == 1) && (filter_w == 1) && (filter_h == 1) && (dilation_w == 1) &&
               (dilation_h == 1);
  bool is1xN = (output_h == 1) && (input_h == 1) && (filter_h == 1) && (output_w % 4 == 0) &&
               (input_n == 1) && (dilation_w == 1) && (dilation_h == 1);

  bool has_mve = target->GetFeature<Bool>("has_mve").value_or(Bool(false));

  if (is1x1) {
    return 0;
  }

  if (is1xN) {
    if (has_mve) {
      return 0;
    }
    return (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
  }

  if (has_mve || is1xN) {
    int32_t col_length = input_c * filter_w * filter_h;
    col_length = (col_length + 7) / 8;
    return 4 * col_length * 8 * (int32_t)sizeof(int8_t);
  } else {
    return (2 * input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
  }
  return 0;
}

int DepthwiseConv2dBufferSize(Target target, int32_t input_n, int32_t input_c, int32_t output_c,
                              int32_t filter_w, int32_t filter_h, int32_t dilation_w,
                              int32_t dilation_h) {
  bool has_mve = target->GetFeature<Bool>("has_mve").value_or(Bool(false));
  bool has_dsp = target->GetFeature<Bool>("has_dsp").value_or(Bool(false));

  if (input_c == output_c && input_n == 1 && dilation_w == 1 && dilation_h == 1) {
    if (has_mve) {
      return (4 * CH_IN_BLOCK_MVE * filter_w * filter_h) * (int32_t)sizeof(int8_t);
    } else if (has_dsp) {
      return (input_c * filter_w * filter_h) * (int32_t)sizeof(int16_t);
    }
  }
  return 0;
}

int AvgPoolBufferSize(Target target, int32_t input_c) {
  bool has_mve = target->GetFeature<Bool>("has_mve").value_or(Bool(false));
  bool has_dsp = target->GetFeature<Bool>("has_dsp").value_or(Bool(false));

  if (has_dsp && !has_mve) {
    return (input_c * sizeof(int32_t));
  }
  return 0;
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
