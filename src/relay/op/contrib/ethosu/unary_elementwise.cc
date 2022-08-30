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
 * \file src/relay/op/contrib/ethosu/unary_elementwise.cc
 * \brief Property def of the Arm(R) Ethos(TM)-U unary elementwise ops.
 */
#include <tvm/relay/op.h>

#include "common.h"
#include "op_attrs.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

bool EthosuUnaryElementwiseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                               const TypeReporter& reporter) {
  const int ifm_index = 0;
  const int result_index = 2;
  ICHECK_EQ(types.size(), result_index + 1);

  const auto* ifm = types[ifm_index].as<TensorTypeNode>();
  if (ifm == nullptr) return false;

  const auto* param = attrs.as<EthosuUnaryElementwiseAttrs>();
  CHECK(param != nullptr) << "EthosuUnaryElementwiseAttrs cannot be nullptr.";

  const String operator_name = "ethosu_unary_elementwise";
  const String operator_type = param->operator_type;
  if (operator_type != "ABS" && operator_type != "CLZ") {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "Invalid operator: expected << " << operator_name
                                     << "  'ABS' or 'CLZ' for operator_type but was"
                                     << operator_type);
    return false;
  }

  const DataType ifm_dtype = ifm->dtype;
  if (operator_type == "CLZ") {
    CheckDataType(reporter, ifm_dtype, {DataType::Int(32)}, operator_name, "ifm", operator_type);
  } else {
    CheckDataType(reporter, ifm_dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "ifm",
                  operator_type);
  }

  // Assign ofm type
  auto ofm_shape = EthosuInferElementwiseOutputShape(ifm->shape, param->ifm_layout,
                                                     param->ofm_layout, param->ofm_channels);
  reporter->Assign(types[result_index], TensorType(ofm_shape, ifm_dtype));
  return true;
}

Expr MakeEthosuUnaryElementwise(Expr ifm, Expr lut, String operator_type, double ifm_scale,
                                int ifm_zero_point, double ofm_scale, int ofm_zero_point,
                                IndexExpr ofm_channels, String activation, int clip_min,
                                int clip_max, String rounding_mode, String ifm_layout,
                                String ofm_layout) {
  auto attrs = make_object<EthosuUnaryElementwiseAttrs>();

  attrs->operator_type = std::move(operator_type);
  attrs->ifm_scale = ifm_scale;
  attrs->ifm_zero_point = ifm_zero_point;
  attrs->ofm_scale = ofm_scale;
  attrs->ofm_zero_point = ofm_zero_point;
  attrs->ofm_channels = std::move(ofm_channels);
  attrs->activation = std::move(activation);
  attrs->clip_min = clip_min;
  attrs->clip_max = clip_max;
  attrs->rounding_mode = std::move(rounding_mode);
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);

  static const Op& op = Op::Get("contrib.ethosu.unary_elementwise");
  return Call(op, {ifm, lut}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ethosu_unary_elementwise")
    .set_body_typed(MakeEthosuUnaryElementwise);

RELAY_REGISTER_OP("contrib.ethosu.unary_elementwise")
    .describe(R"code(Quantized unary elementwise operator for Arm(R) Ethos(TM)-U NPUs.

This Relay operator corresponds to the hardware-implemented quantized
unary elementwise operation found on NPUs. It accepts either NHWC
or NHCWB16 format for the inputs data (input feature maps, or IFMs).

Reference: https://developer.arm.com/documentation/102420/0200/

- **ifm**: NHWC - (1, ifm_height, ifm_width, ifm_channels)
           NHCWB16 - (1, ifm_height, ifm_channels // 16, ifm_width, 16)
- **ofm**: (1, ofm_height, ofm_width, ofm_channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EthosuUnaryElementwiseAttrs>()
    .set_num_inputs(2)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .add_argument("lut", "Tensor", "The look-up table values to use if activation = 'LUT'")
    .set_support_level(11)
    .add_type_rel("EthosuUnaryElementwise", EthosuUnaryElementwiseRel);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
