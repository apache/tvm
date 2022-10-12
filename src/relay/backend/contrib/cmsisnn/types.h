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
 * \file src/relay/backend/contrib/cmsisnn/types.h
 * \brief Includes structs and data types needed by CMSIS-NN APIs.
 * https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/Include/arm_nn_types.h
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CMSISNN_TYPES_H_
#define TVM_RELAY_BACKEND_CONTRIB_CMSISNN_TYPES_H_

#include <stdint.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*!
 * \brief Object that contains the width and height of a tile
 */
typedef struct {
  /*! \brief Tile width */
  int32_t w;
  /*! \brief Tile height */
  int32_t h;
} Tile;

/*!
 * \brief Dimensions of tensors
 */
typedef struct {
  /*! \brief Either contains batch size or number of channels. */
  int32_t n;
  int32_t h;
  int32_t w;
  int32_t c;
} Dims;

/*!
 * \brief Quantized Relu activation
 */
typedef struct {
  int32_t min;
  int32_t max;
} Activation;

/*!
 * \brief Convolution layer parameters
 */
typedef struct {
  /*! \brief Zero point for the input tensor. */
  int32_t input_offset;
  /*! \brief Zero point for the output tensor. */
  int32_t output_offset;
  Tile stride;
  Tile padding;
  Tile dilation;
  Activation activation;
} ConvParams;

/*!
 * \brief Depthwise convolution layer parameters
 */
typedef struct {
  /*! \brief Zero point for the input tensor. */
  int32_t input_offset;
  /*! \brief Zero point for the output tensor. */
  int32_t output_offset;
  /*! \brief Channel Multiplier. ch_mult * in_ch = out_ch */
  int32_t ch_mult;
  Tile stride;
  Tile padding;
  Tile dilation;
  Activation activation;
} DepthwiseConvParams;

/*!
 * \brief Pooling layer parameters
 */
typedef struct {
  Tile stride;
  Tile padding;
  Activation activation;
} PoolParams;

/*!
 * \brief Fully Connected layer parameters
 */
typedef struct {
  /*! \brief Zero point for the input tensor. */
  int32_t input_offset;
  /*! \brief Zero point for the filter tensor. */
  int32_t filter_offset;
  /*! \brief Zero point for the input tensor. */
  int32_t output_offset;
  Activation activation;
} FCParams;

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CMSISNN_TYPES_H_
