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
 * \brief This file contains CMSIS-NN buffer size functions similar to present
 *  here:
 * https://github.com/ARM-software/CMSIS_5/tree/51263182d16c92649a48144ba56c0945f9fce60e/CMSIS/NN
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CMSISNN_BUFFER_SIZE_H_
#define TVM_RELAY_BACKEND_CONTRIB_CMSISNN_BUFFER_SIZE_H_

#include "types.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

int32_t ConvolveWrapperS8GetBufferSize(bool has_mve, bool has_dsp, const ConvParams* conv_params,
                                       const Dims* input_dims, const Dims* filter_dims,
                                       const Dims* output_dims);

int32_t ConvolveWrapperS16GetBufferSize(bool has_mve, bool has_dsp, const ConvParams* conv_params,
                                        const Dims* input_dims, const Dims* filter_dims,
                                        const Dims* output_dims);

int32_t ConvolveS8GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                const Dims* filter_dims);

int32_t ConvolveS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                 const Dims* filter_dims);

int32_t ConvolveFastS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                     const Dims* filter_dims);

int32_t Convolve1x1S8FastGetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims);

int32_t Convolve1XNS8GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                   const Dims* filter_dims);

int32_t DepthwiseConvWrapperS8GetBufferSize(bool has_mve, bool has_dsp,
                                            const DepthwiseConvParams* dw_conv_params,
                                            const Dims* input_dims, const Dims* filter_dims,
                                            const Dims* output_dims);

int32_t DepthwiseConvWrapperS16GetBufferSize(bool has_mve, bool has_dsp,
                                             const DepthwiseConvParams* dw_conv_params,
                                             const Dims* input_dims, const Dims* filter_dims,
                                             const Dims* output_dims);

int32_t DepthwiseConvFastS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                          const Dims* filter_dims);

int32_t DepthwiseConvS8OptGetBufferSize(bool has_mve, bool has_dsp, const Dims* input_dims,
                                        const Dims* filter_dims);

int32_t FullyConnectedS8GetBufferSize(bool has_mve, bool has_dsp, const Dims* filter_dims);

int32_t FullyConnectedS16GetBufferSize(bool has_mve, bool has_dsp, const Dims* filter_dims);

int32_t AvgpoolS8GetBufferSize(bool has_mve, bool has_dsp, const int dim_dst_width,
                               const int ch_src);

int32_t AvgpoolS16GetBufferSize(bool has_mve, bool has_dsp, const int dim_dst_width,
                                const int ch_src);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CMSISNN_BUFFER_SIZE_H_
