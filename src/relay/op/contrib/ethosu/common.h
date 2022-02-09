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

/*! \brief Infer the output tensor shape for binary elementwise operators.
 * \param ifm_shape The shape of Input Feature Map.
 * \param ifm_layout The layout of the IFM (NHWC or NHCWB16).
 * \param ofm_layout The layout of the OFM (NHWC or NHCWB16).
 * \param ofm_channels The number of Output Feature Map channels.
 * \return The shape of the output tensor.
 */
Array<IndexExpr> EthosuInferElementwiseOutputShape(Array<IndexExpr> ifm_shape, String ifm_layout,
                                                   String ofm_layout, IndexExpr ofm_channels);

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

/*! \brief Infer the Output Feature Map shape for operations that use upscaling.
 * \param ifm_shape The shape of the Input Feature Map.
 * \param ifm_layout The layout of the Input Feature Map.
 */
Array<IndexExpr> EthosuInferUpscaledInput(Array<IndexExpr> ifm_shape, String ifm_layout);

/*! \brief Get data type from string representation.
 * \param dtype Data type in lower case format followed by number of bits e.g. "int8".
 */
DataType DataTypeFromString(const String& dtype);

/*! \brief Check the data type for a given input matches one given in allowed_data_types. Raise a
 * type inference error if not.
 * \param reporter The infer type reporter.
 * \param data_type The data type to check.
 * \param allowed_data_types An initializer list of allowed data types.
 * \param operator_name The name of the operator to report.
 * \param tensor_name The name of the tensor to report e.g. "ifm", "ofm".
 * \param operator_type The type of the operator to report e.g. "ADD" for binary_elementwise.
 */
void CheckDataType(const TypeReporter& reporter, const DataType& data_type,
                   const std::initializer_list<DataType>& allowed_data_types,
                   const String& operator_name, const String& tensor_name,
                   const String& operator_type = "");

/*! \brief Check the upscale method matches one given in allowed_upscale_methods. Raise a type
 * inference error if not.
 * \param reporter The infer type reporter.
 * \param upscale_method The upscale method string to check.
 * \param allowed_upscale_methods An initializer list of allowed upscale methods.
 * \param operator_name The name of the operator to report.
 * \param operator_type The type of the operator to report e.g. "ADD" for binary_elementwise.
 */
void CheckUpscaleMethod(const TypeReporter& reporter, const String& upscale_method,
                        const std::initializer_list<String>& allowed_upscale_methods,
                        const String& operator_name, const String& operator_type = "");

/*! \brief Check the data type matches that of the second data type provided. Raise a type inference
 * error if not.
 * \param reporter The infer type reporter.
 * \param data_type The data type to check.
 * \param data_type2 The second data type to check.
 * \param operator_name The name of the operator to report.
 * \param tensor_name The name of the tensor to report e.g. "ifm", "ofm".
 * \param tensor_name2 The name of the second tensor to report e.g. "ifm2".
 * \param operator_type The type of the operator to report e.g. "ADD" for binary_elementwise.
 */
void CheckDataTypeMatch(const TypeReporter& reporter, const DataType& data_type,
                        const DataType& data_type2, const String& operator_name,
                        const String& tensor_name, const String& tensor_name2,
                        const String& operator_type = "");

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_CONTRIB_ETHOSU_COMMON_H_
