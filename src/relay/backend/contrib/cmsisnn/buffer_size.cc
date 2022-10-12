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
/*!
 * \file src/relay/backend/contrib/cmsisnn/buffer_size.cc
 * \brief This file contains CMSIS-NN buffer size functions similar to present
 *  here:
 * https://github.com/ARM-software/CMSIS_5/tree/51263182d16c92649a48144ba56c0945f9fce60e/CMSIS/NN
 */

#include "buffer_size.h"

#include <stdint.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

#define USE_FAST_DW_CONV_FUNCTION(dw_conv_params, filter_dims, input_dims) \
  dw_conv_params->ch_mult == 1 && dw_conv_params->dilation.w == 1 &&       \
      dw_conv_params->dilation.h == 1 && filter_dims->w * filter_dims->h * input_dims->c < 512

#define CH_IN_BLOCK_MVE (124)

int32_t AvgpoolS16GetBufferSize(bool has_mve, bool has_dsp, const int output_x, const int ch_src) {
  (void)output_x;
  if (has_dsp && !has_mve) {
    return (ch_src * (int32_t)sizeof(int32_t));
  } else {
    (void)ch_src;
  }
  return 0;
}

int32_t AvgpoolS8GetBufferSize(bool has_mve, bool has_dsp, const int output_x, const int ch_src) {
  (void)output_x;

  if (has_dsp && !has_mve) {
    return (ch_src * sizeof(int32_t));
  } else {
    (void)ch_src;
    return 0;
  }
}

int32_t ConvolveFastS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                     const Dims* filter_dims) {
  if (has_dsp && !has_mve) {
    return (2 * input_dims->c * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int16_t);
  } else {
    (void)input_dims;
    (void)filter_dims;
    return 0;
  }
}

int32_t ConvolveS8GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                const Dims* filter_dims) {
  if (has_mve) {
    int32_t col_length = input_dims->c * filter_dims->w * filter_dims->h;
    // Get number of complete int16 lanes(multiple of 8) for given col_length. This is dependent on
    // implementation of  arm_nn_mat_mult_s8
    col_length = (col_length + 7) / 8;
    // 4 -> number of im2col buffers, 8 -> 8 elements per Q register
    return 4 * col_length * 8 * (int32_t)sizeof(int8_t);
  } else {
    return (2 * input_dims->c * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int16_t);
  }
}

int32_t DepthwiseConvWrapperS8GetBufferSize(bool has_mve, bool has_dsp,
                                            const DepthwiseConvParams* dw_conv_params,
                                            const Dims* input_dims, const Dims* filter_dims,
                                            const Dims* output_dims) {
  (void)dw_conv_params;
  int32_t size = 0;

  if (input_dims->c == output_dims->c && input_dims->n == 1 && dw_conv_params->dilation.w == 1 &&
      dw_conv_params->dilation.h == 1) {
    size = DepthwiseConvS8OptGetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
  }

  return size;
}

int32_t DepthwiseConvWrapperS16GetBufferSize(bool has_mve, bool has_dsp,
                                             const DepthwiseConvParams* dw_conv_params,
                                             const Dims* input_dims, const Dims* filter_dims,
                                             const Dims* output_dims) {
  (void)dw_conv_params;
  (void)input_dims;
  (void)filter_dims;
  (void)output_dims;
  int32_t size = 0;

  if (USE_FAST_DW_CONV_FUNCTION(dw_conv_params, filter_dims, input_dims)) {
    size = DepthwiseConvFastS16GetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
  }

  return size;
}

int32_t DepthwiseConvS8OptGetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                        const Dims* filter_dims) {
  if (has_mve) {
    (void)input_dims;
    return (4 * CH_IN_BLOCK_MVE * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int8_t);
  } else if (has_dsp) {
    return (input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
  } else {
    (void)input_dims;
    (void)filter_dims;
    return 0;
  }
}

int32_t ConvolveWrapperS8GetBufferSize(bool has_mve, bool has_dsp, const ConvParams* conv_params,
                                       const Dims* input_dims, const Dims* filter_dims,
                                       const Dims* output_dims) {
  if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) &&
      (conv_params->stride.w == 1) && (conv_params->stride.h == 1) && (filter_dims->w == 1) &&
      (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1)) {
    return Convolve1x1S8FastGetBufferSize(has_mve, has_dsp, input_dims);
  } else if ((input_dims->h == 1) && (output_dims->w % 4 == 0) && (conv_params->dilation.w == 1) &&
             (filter_dims->h == 1)) {
    return Convolve1XNS8GetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
  } else {
    return ConvolveS8GetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
  }
}

int32_t ConvolveS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                 const Dims* filter_dims) {
  (void)input_dims;
  (void)filter_dims;
  return 0;
}

int32_t DepthwiseConvFastS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                          const Dims* filter_dims) {
  if (has_dsp) {
    if (has_mve) {
      return 4 * input_dims->c * filter_dims->w * filter_dims->h * sizeof(int16_t) + 8;
    } else {
      return input_dims->c * filter_dims->w * filter_dims->h * sizeof(int16_t);
    }
  } else {
    (void)input_dims;
    (void)filter_dims;
    return 0;
  }
}

int32_t Convolve1XNS8GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                   const Dims* filter_dims) {
  if (!has_mve) {
    return ConvolveS8GetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
  } else {
    (void)input_dims;
    (void)filter_dims;
    return 0;
  }
}

int32_t ConvolveWrapperS16GetBufferSize(bool has_mve, bool has_dsp, const ConvParams* conv_params,
                                        const Dims* input_dims, const Dims* filter_dims,
                                        const Dims* output_dims) {
  (void)conv_params;
  (void)output_dims;

  if (has_dsp && !has_mve) {
    if (filter_dims->w * filter_dims->h * input_dims->c < 512 &&
        (conv_params->dilation.w == 1 && conv_params->dilation.h == 1)) {
      return ConvolveFastS16GetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
    }

    return ConvolveS16GetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
  } else {
    return ConvolveS16GetBufferSize(has_mve, has_dsp, input_dims, filter_dims);
  }
}

int32_t Convolve1x1S8FastGetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims) {
  (void)input_dims;
  return 0;
}

int32_t FullyConnectedS8GetBufferSize(bool has_mve, bool has_dsp, const Dims* filter_dims) {
  (void)filter_dims;
  return 0;
}

int32_t FullyConnectedS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* filter_dims) {
  (void)filter_dims;
  return 0;
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
