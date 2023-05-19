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
 * \file src/relay/op/contrib/ethosu/pooling.cc
 * \brief Pooling operators definitions for the Arm(R) Ethos(TM)-U NPU.
 */
#include <tvm/relay/op.h>

#include "common.h"
#include "op_attrs.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

bool EthosuPoolingRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  int ifm_index = 0;
  int result_index = 2;
  ICHECK_EQ(types.size(), result_index + 1);

  const auto* ifm = types[ifm_index].as<TensorTypeNode>();
  if (ifm == nullptr) return false;

  const auto* param = attrs.as<EthosuPoolingAttrs>();
  ICHECK(param != nullptr) << "EthosuPoolingAttrs cannot be nullptr.";

  const String operator_name = "ethosu_pooling";

  if (param->pooling_type != "AVG" && param->pooling_type != "MAX" &&
      param->pooling_type != "SUM") {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "Invalid operator: expected " << operator_name
                                     << " type 'AVG', 'MAX', or 'SUM' but was "
                                     << param->pooling_type);
    return false;
  }

  std::initializer_list<DataType> max_avg_pooling_ifm_dtypes = {DataType::UInt(8), DataType::Int(8),
                                                                DataType::Int(16)};
  std::initializer_list<DataType> sum_pooling_ifm_dtypes = {DataType::UInt(8), DataType::Int(8),
                                                            DataType::Int(16), DataType::Int(32)};

  std::initializer_list<DataType>& allowed_ifm_dtypes = max_avg_pooling_ifm_dtypes;
  if (param->pooling_type == "SUM") {
    allowed_ifm_dtypes = sum_pooling_ifm_dtypes;
  }

  CheckDataType(reporter, ifm->dtype, allowed_ifm_dtypes, operator_name, "ifm",
                param->pooling_type);

  DataType ofm_dtype = DataTypeFromString(param->ofm_dtype);

  std::initializer_list<DataType> max_avg_pooling_ofm_dtypes = {DataType::Int(8), DataType::UInt(8),
                                                                DataType::Int(16)};
  if (param->pooling_type == "AVG" || param->pooling_type == "MAX") {
    CheckDataType(reporter, ofm_dtype, max_avg_pooling_ofm_dtypes, operator_name, "ofm",
                  param->pooling_type);
    CheckDataTypeMatch(reporter, ofm_dtype, ifm->dtype, operator_name, "ifm", "ofm",
                       param->pooling_type);
  } else {
    CheckDataType(reporter, ofm_dtype, {DataType::Int(32)}, operator_name, "ofm",
                  param->pooling_type);
  }

  CheckUpscaleMethod(reporter, param->upscale, {"NONE", "ZEROS", "NEAREST"}, operator_name);

  Array<IndexExpr> ifm_shape = ifm->shape;
  if (param->upscale != "NONE") {
    ifm_shape = EthosuInferUpscaledInput(ifm_shape, param->ifm_layout);
  }

  // Assign ofm shape
  auto ofm_shape = EthosuInferKernelOutput(
      ifm_shape, param->ifm_layout, param->ofm_layout, param->pool_shape, param->ofm_channels,
      Array<IndexExpr>({1, 1}), param->strides, param->padding);

  reporter->Assign(types[result_index], TensorType(ofm_shape, ofm_dtype));
  return true;
}

Expr MakeEthosuPooling(Expr ifm, Expr lut, String pooling_type, double ifm_scale,
                       int ifm_zero_point, double ofm_scale, int ofm_zero_point,
                       Array<IndexExpr> pool_shape, IndexExpr ofm_channels, String ofm_dtype,
                       Array<IndexExpr> strides, Array<IndexExpr> padding, String activation,
                       int clip_min, int clip_max, String rounding_mode, String upscale,
                       String ifm_layout, String ofm_layout) {
  auto attrs = make_object<EthosuPoolingAttrs>();
  attrs->pooling_type = std::move(pooling_type);
  attrs->ifm_scale = ifm_scale;
  attrs->ifm_zero_point = ifm_zero_point;
  attrs->ofm_scale = ofm_scale;
  attrs->ofm_zero_point = ofm_zero_point;
  attrs->pool_shape = std::move(pool_shape);
  attrs->ofm_channels = std::move(ofm_channels);
  attrs->ofm_dtype = std::move(ofm_dtype);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->activation = std::move(activation);
  attrs->clip_min = clip_min;
  attrs->clip_max = clip_max;
  attrs->rounding_mode = std::move(rounding_mode);
  attrs->upscale = std::move(upscale);
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  static const Op& op = Op::Get("contrib.ethosu.pooling");
  return Call(op, {ifm, lut}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ethosu_pooling").set_body_typed(MakeEthosuPooling);

RELAY_REGISTER_OP("contrib.ethosu.pooling")
    .describe(R"code(Arm(R) Ethos(TM)-U NPU 2D quantized pooling operator.

This Relay operator corresponds to the hardware-implemented quantized
pooling operation found on Ethos(TM)-U NPU. It accepts either NHWC
or NHCWB16 format for the input data (input feature map, or IFM).

Reference: https://developer.arm.com/documentation/102420/0200/

- **ifm**: NHWC - (1, ifm_height, ifm_width, ifm_channels)
           NHCWB16 - (1, ifm_height, ifm_channels // 16, ifm_width, 16)
- **ofm**: (1, ofm_height, ofm_width, ofm_channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EthosuPoolingAttrs>()
    .set_num_inputs(2)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .add_argument("lut", "Tensor", "The look-up table of values to use if activation = 'LUT'")
    .set_support_level(11)
    .add_type_rel("EthosuPooling", EthosuPoolingRel);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
