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
 * \file src/relay/op/contrib/ethosu/common.cc
 * \brief A set of utilities and common functionality for Arm(R) Ethos(TM)-U NPU QNN ops.
 */

#include "common.h"

#include <sstream>

#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

Array<IndexExpr> EthosuInferElementwiseOutputShape(Array<IndexExpr> ifm_shape, String ifm_layout,
                                                   String ofm_layout, IndexExpr ofm_channels) {
  // In the case of NHCWB16, convert the ifm shape to NHW (C not required for this function)
  if (ifm_layout == "NHCWB16") {
    ifm_shape = {ifm_shape[0], ifm_shape[1], ifm_shape[3]};
  }
  Array<IndexExpr> oshape({ifm_shape[0], ifm_shape[1], ifm_shape[2], ofm_channels});

  // If the ofm is NHCWB16, convert the layout
  if (ofm_layout == "NHCWB16") {
    int channel_bricks = 1 + (oshape[3].as<IntImmNode>()->value - 1) / 16;
    oshape = {oshape[0], oshape[1], channel_bricks, oshape[2], 16};
  }

  return oshape;
}

Array<IndexExpr> EthosuInferKernelOutput(Array<IndexExpr> ifm_shape, String ifm_layout,
                                         String ofm_layout, Array<IndexExpr> kernel_shape,
                                         IndexExpr ofm_channels, Array<IndexExpr> dilation,
                                         Array<IndexExpr> strides, Array<IndexExpr> padding) {
  // In the case of NHCWB16, convert the ifm shape to NHW (C not required for this function)
  if (ifm_layout == "NHCWB16") {
    ifm_shape = {ifm_shape[0], ifm_shape[1], ifm_shape[3]};
  }
  Array<IndexExpr> output_shape({ifm_shape[0], 0, 0, ofm_channels});

  IndexExpr dilated_ksize_y = 1 + (kernel_shape[0] - 1) * dilation[0];
  IndexExpr dilated_ksize_x = 1 + (kernel_shape[1] - 1) * dilation[1];
  IndexExpr pad_h, pad_w;
  GetPaddingHeightWidth(padding, &pad_h, &pad_w);
  output_shape.Set(1, indexdiv(ifm_shape[1] + pad_h - dilated_ksize_y, strides[0]) + 1);
  output_shape.Set(2, indexdiv(ifm_shape[2] + pad_w - dilated_ksize_x, strides[1]) + 1);

  // If the ofm is NHCWB16, convert the layout
  if (ofm_layout == "NHCWB16") {
    int channel_bricks = 1 + (output_shape[3].as<IntImmNode>()->value - 1) / 16;
    output_shape = {output_shape[0], output_shape[1], channel_bricks, output_shape[2], 16};
  }

  return output_shape;
}

Array<IndexExpr> EthosuInferUpscaledInput(Array<IndexExpr> ifm_shape, String ifm_layout) {
  if (ifm_layout == "NHCWB16") {
    ifm_shape = {ifm_shape[0], ifm_shape[1], ifm_shape[3], ifm_shape[2] * 16};
  }

  const int scale_factor = 2;
  Array<IndexExpr> new_ifm_shape = {ifm_shape[0], ifm_shape[1] * scale_factor,
                                    ifm_shape[2] * scale_factor, ifm_shape[3]};

  if (ifm_layout == "NHCWB16") {
    int channel_bricks = 1 + (new_ifm_shape[3].as<IntImmNode>()->value - 1) / 16;
    new_ifm_shape = {new_ifm_shape[0], new_ifm_shape[1], channel_bricks, new_ifm_shape[2], 16};
  }

  return new_ifm_shape;
}

DataType DataTypeFromString(const String& dtype) {
  DLDataType dl_dtype = tvm::runtime::String2DLDataType(dtype);
  return DataType(dl_dtype);
}

void CheckDataType(const TypeReporter& reporter, const DataType& data_type,
                   const std::initializer_list<DataType>& allowed_data_types,
                   const String& operator_name, const String& tensor_name,
                   const String& operator_type) {
  for (const auto& i : allowed_data_types) {
    if (data_type == i) {
      return;
    }
  }

  std::ostringstream message;
  message << "Invalid operator: expected " << operator_name << " ";
  if (operator_type != "") {
    message << operator_type << " ";
  }
  message << "to have type in {";
  for (auto it = allowed_data_types.begin(); it != allowed_data_types.end(); ++it) {
    message << *it;
    if (std::next(it) != allowed_data_types.end()) {
      message << ", ";
    }
  }
  message << "}";
  message << " for " << tensor_name << " but was " << data_type << ".";

  reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan()) << message.str());
}

void CheckUpscaleMethod(const TypeReporter& reporter, const String& upscale_method,
                        const std::initializer_list<String>& allowed_upscale_methods,
                        const String& operator_name, const String& operator_type) {
  for (const auto& i : allowed_upscale_methods) {
    if (upscale_method == i) {
      return;
    }
  }

  std::ostringstream message;
  message << "Invalid operator: expected " << operator_name << " ";
  if (operator_type != "") {
    message << operator_type << " ";
  }
  message << "to have upscale method in {";
  for (auto it = allowed_upscale_methods.begin(); it != allowed_upscale_methods.end(); ++it) {
    message << *it;
    if (std::next(it) != allowed_upscale_methods.end()) {
      message << ", ";
    }
  }
  message << "}";
  message << " but was " << upscale_method << ".";

  reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan()) << message.str());
}

void CheckDataTypeMatch(const TypeReporter& reporter, const DataType& data_type,
                        const DataType& data_type2, const String& operator_name,
                        const String& tensor_name, const String& tensor_name2,
                        const String& operator_type) {
  if (data_type == data_type2) {
    return;
  }

  std::ostringstream message;
  message << "Invalid operator: expected " << operator_name << " ";
  if (operator_type != "") {
    message << operator_type << " ";
  }
  message << "data types for " << tensor_name << " and " << tensor_name2 << " to match, but was "
          << data_type << " and " << data_type2;

  reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan()) << message.str());
}

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
