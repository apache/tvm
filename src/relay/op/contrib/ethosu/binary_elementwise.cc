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
 * \file src/relay/op/contrib/ethosu/binary_elementwise.cc
 * \brief Binary elementwise operators definitions for the Arm(R) Ethos(TM)-U NPU.
 */
#include <tvm/relay/op.h>

#include "common.h"
#include "op_attrs.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

bool EthosuBinaryElementwiseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                const TypeReporter& reporter) {
  const int ifm_index = 0;
  const int ifm2_index = 1;
  const int result_index = 3;
  ICHECK_EQ(types.size(), result_index + 1);

  const auto* ifm = types[ifm_index].as<TensorTypeNode>();
  const auto* ifm2 = types[ifm2_index].as<TensorTypeNode>();
  if (ifm == nullptr) return false;
  if (ifm2 == nullptr) return false;

  const auto* param = attrs.as<EthosuBinaryElementwiseAttrs>();
  ICHECK(param != nullptr) << "EthosuBinaryElementwiseAttrs cannot be nullptr.";

  const String operator_name = "ethosu_binary_elementwise";
  const String operator_type = param->operator_type;
  const DataType ifm_dtype = ifm->dtype;
  const DataType ifm2_dtype = ifm2->dtype;
  const DataType ofm_dtype = DataTypeFromString(param->ofm_dtype);

  CheckDataTypeMatch(reporter, ifm_dtype, ifm2_dtype, operator_name, "ifm", "ifm2", operator_type);

  if (operator_type == "ADD" || operator_type == "SUB" || operator_type == "MUL") {
    auto allowed_types = {DataType::Int(8), DataType::UInt(8), DataType::Int(16),
                          DataType::Int(32)};
    CheckDataType(reporter, ifm_dtype, allowed_types, operator_name, "ifm", operator_type);
    CheckDataType(reporter, ofm_dtype, allowed_types, operator_name, "ofm", operator_type);
  } else if (operator_type == "MIN" || operator_type == "MAX") {
    auto allowed_types = {DataType::Int(8), DataType::UInt(8)};
    CheckDataType(reporter, ifm_dtype, allowed_types, operator_name, "ifm", operator_type);
    CheckDataTypeMatch(reporter, ifm_dtype, ofm_dtype, operator_name, "ifm", "ofm", operator_type);
  } else if (operator_type == "SHR") {
    CheckDataType(reporter, ifm_dtype, {DataType::Int(32)}, operator_name, "ifm", operator_type);
    CheckDataType(reporter, ofm_dtype, {DataType::UInt(8), DataType::Int(8), DataType::Int(32)},
                  operator_name, "ofm", operator_type);
  } else if (operator_type == "SHL") {
    CheckDataType(reporter, ifm_dtype, {DataType::Int(32)}, operator_name, "ifm", operator_type);
    CheckDataType(reporter, ofm_dtype, {DataType::Int(32)}, operator_name, "ofm", operator_type);
  } else {
    reporter->GetDiagCtx().EmitFatal(
        Diagnostic::Error(reporter->GetSpan())
        << "Invalid operator: expected " << operator_name << " 'ADD' or 'SUB' or 'MUL' or "
        << "'MIN' or 'MAX' or 'SHR' or 'SHL' for operator_type but was " << param->operator_type);
    return false;
  }

  // Assign ofm type
  auto ofm_shape = EthosuInferElementwiseOutputShape(ifm->shape, param->ifm_layout,
                                                     param->ofm_layout, param->ifm_channels);
  reporter->Assign(types[result_index], TensorType(ofm_shape, ofm_dtype));
  return true;
}

Expr MakeEthosuBinaryElementwise(Expr ifm, Expr ifm2, Expr lut, String operator_type,
                                 double ifm_scale, int ifm_zero_point, double ifm2_scale,
                                 int ifm2_zero_point, double ofm_scale, int ofm_zero_point,
                                 IndexExpr ifm_channels, IndexExpr ifm2_channels,
                                 bool reversed_operands, String activation, int clip_min,
                                 int clip_max, String rounding_mode, String ifm_layout,
                                 String ifm2_layout, String ofm_layout, String ofm_dtype,
                                 bool use_rescale, int rescale_scale, int rescale_shift) {
  auto attrs = make_object<EthosuBinaryElementwiseAttrs>();

  attrs->operator_type = std::move(operator_type);
  attrs->ifm_scale = ifm_scale;
  attrs->ifm_zero_point = ifm_zero_point;
  attrs->ifm2_scale = ifm2_scale;
  attrs->ifm2_zero_point = ifm2_zero_point;
  attrs->ofm_scale = ofm_scale;
  attrs->ofm_zero_point = ofm_zero_point;
  attrs->ifm_channels = std::move(ifm_channels);
  attrs->ifm2_channels = std::move(ifm2_channels);
  attrs->reversed_operands = reversed_operands;
  attrs->activation = std::move(activation);
  attrs->clip_min = clip_min;
  attrs->clip_max = clip_max;
  attrs->rounding_mode = std::move(rounding_mode);
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ifm2_layout = std::move(ifm2_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  attrs->ofm_dtype = std::move(ofm_dtype);
  attrs->use_rescale = use_rescale;
  attrs->rescale_scale = rescale_scale;
  attrs->rescale_shift = rescale_shift;

  static const Op& op = Op::Get("contrib.ethosu.binary_elementwise");
  return Call(op, {ifm, ifm2, lut}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ethosu_binary_elementwise")
    .set_body_typed(MakeEthosuBinaryElementwise);

RELAY_REGISTER_OP("contrib.ethosu.binary_elementwise")
    .describe(R"code(Arm(R) Ethos(TM)-U NPU quantized binary elementwise operator.

This Relay operator corresponds to the hardware-implemented quantized
binary elementwise operation found on Ethos(TM)-U NPU. It accepts either NHWC
or NHCWB16 format for the inputs data (input feature maps, or IFMs).

Reference: https://developer.arm.com/documentation/102420/0200/

- **ifm**: NHWC - (1, ifm_height, ifm_width, ifm_channels)
           NHCWB16 - (1, ifm_height, ifm_channels // 16, ifm_width, 16)
- **ifm2**: NHWC - (1, ifm_height, ifm_width, ifm_channels)
           NHCWB16 - (1, ifm_height, ifm_channels // 16, ifm_width, 16)
- **ofm**: (1, ofm_height, ofm_width, ifm_channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EthosuBinaryElementwiseAttrs>()
    .set_num_inputs(3)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .add_argument("ifm2", "Tensor", "The Input Feature Map tensor 2 (IFM2).")
    .add_argument("lut", "Tensor", "The look-up table of values to use if activation = 'LUT'")
    .set_support_level(11)
    .add_type_rel("EthosuBinaryElementwise", EthosuBinaryElementwiseRel);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
