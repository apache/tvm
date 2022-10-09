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
 * \file src/relay/backend/contrib/cmsisnn/buffer_size.h
 * \brief CMSIS-NN Buffer Size calculation functions
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CMSISNN_BUFFER_SIZE_H_
#define TVM_RELAY_BACKEND_CONTRIB_CMSISNN_BUFFER_SIZE_H_

#include <tvm/ir/transform.h>

#include "compiler_attrs.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

#define CH_IN_BLOCK_MVE (124)

/*!
 * \brief Calculates the appropriate buffer size for CMSIS-NN Convolutions
 * See:
 * https://github.com/ARM-software/CMSIS_5/blob/8c60448c0e1e50e426180b26db9bc31ddf774361/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c#L108-L127
 *
 * \param is_int16 - type of conv2d
 * \param target - CMSIS-NN Target
 * \param padding_w - Width padding
 * \param padding_h - Height padding
 * \param input_n - Input batch size
 * \param input_h - Input height
 * \param input_c - Input channels
 * \param output_h - Output height
 * \param output_w - Output width
 * \param stride_w - Stride width
 * \param stride_h - Stride height
 * \param filter_w - Filter width
 * \param filter_h - Filter height
 *
 * \return Size of buffer to allocate for convolution
 */
int Conv2dBufferSize(bool is_int16, Target target, int32_t padding_w, int32_t padding_h,
                     int32_t input_n, int32_t input_h, int32_t input_c, int32_t output_h,
                     int32_t output_w, int32_t stride_w, int32_t stride_h, int32_t dilation_w,
                     int32_t dilation_h, int32_t filter_w, int32_t filter_h);

int Conv2dBufferSizeInt8(Target target, int32_t padding_w, int32_t padding_h, int32_t input_n,
                         int32_t input_h, int32_t input_c, int32_t output_h, int32_t output_w,
                         int32_t stride_w, int32_t stride_h, int32_t dilation_w, int32_t dilation_h,
                         int32_t filter_w, int32_t filter_h);

int Conv2dBufferSizeInt16(Target target, int32_t padding_w, int32_t padding_h, int32_t input_n,
                          int32_t input_h, int32_t input_c, int32_t output_h, int32_t output_w,
                          int32_t stride_w, int32_t stride_h, int32_t dilation_w,
                          int32_t dilation_h, int32_t filter_w, int32_t filter_h);

/*!
 * \brief Calculates the appropriate buffer size for CMSIS-NN Depthwise Convolutions
 * See:
 * https://github.com/ARM-software/CMSIS_5/blob/325443e52637b6c7eedbd160d238a6c462e89c9f/CMSIS/NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c#L115-L129
 *
 * \param is_int16 - type of conv2d
 * \param target - CMSIS-NN Target
 * \param input_n - Input batch size
 * \param input_c - Input channels
 * \param output_c - Output channels
 * \param filter_w - Filter width
 * \param filter_h - Filter height
 * \param dilation_w - Dilation width
 * \param dilation_h - Dilation height
 * \param depth_multiplier - Depth Multiplier for Depthwise Convolution
 *
 * \return Size of buffer to allocate for depthwise convolution
 */
int DepthwiseConv2dBufferSize(bool is_int16, Target target, int32_t input_n, int32_t input_c,
                              int32_t output_c, int32_t filter_w, int32_t filter_h,
                              int32_t dilation_w, int32_t dilation_h, int32_t depth_multiplier);

int DepthwiseConv2dBufferSizeInt8(Target target, int32_t input_n, int32_t input_c, int32_t output_c,
                                  int32_t filter_w, int32_t filter_h, int32_t dilation_w,
                                  int32_t dilation_h, int32_t depth_multiplier);

int DepthwiseConv2dBufferSizeInt16(Target target, int32_t input_n, int32_t input_c,
                                   int32_t output_c, int32_t filter_w, int32_t filter_h,
                                   int32_t dilation_w, int32_t dilation_h,
                                   int32_t depth_multiplier);

/*!
 * \brief Calculates the appropriate buffer size for CMSIS-NN Average Pooling
 * See:
 * https://github.com/ARM-software/CMSIS_5/blob/bff28575f0c96a4ee9008947fea2b018a69b4900/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c#L388-L398
 *
 * \param target - CMSIS-NN Target
 * \param input_c - Input channels
 *
 * \return Size of buffer to allocate for average pooling
 */
int AvgPoolBufferSize(Target target, int32_t input_c);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CMSISNN_BUFFER_SIZE_H_
