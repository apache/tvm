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
 * \file src/relay/op/contrib/ethosu/common.h
 * \brief Functions for all Arm(R) Ethos(TM)-U NPU operators to use.
 */

#ifndef TVM_RELAY_OP_CONTRIB_ETHOSU_COMMON_H_
#define TVM_RELAY_OP_CONTRIB_ETHOSU_COMMON_H_

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

/*! \brief Infer the output tensor shape for convolution and pooling operators.
 * \param ifm_shape The shape of Input Feature Map.
 * \param ifm_layout The layout of the IFM (NHWC or NHCWB16).
 * \param ofm_layout The layout of the OFM (NHWC or NHCWB16).
 * \param kernel_shape Kernel shape in format (height, width).
 * \param ofm_channels The number of Output Feature Map channels.
 * \param dilation The 2-dimensional dilation as (dilation_height, dilation_width).
 * \param strides The 2 dimensional strides as (stride_height, stride_width).
 * \param padding The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).
 * \return The shape of the output tensor.
 */
Array<IndexExpr> EthosuInferKernelOutput(Array<IndexExpr> ifm_shape, String ifm_layout,
                                         String ofm_layout, Array<IndexExpr> kernel_shape,
                                         IndexExpr ofm_channels, Array<IndexExpr> dilation,
                                         Array<IndexExpr> strides, Array<IndexExpr> padding);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_CONTRIB_ETHOSU_COMMON_H_
