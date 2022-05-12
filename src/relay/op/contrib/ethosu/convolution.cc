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
 * \file src/relay/op/contrib/ethosu/convolution.cc
 * \brief Operator definitions for the Arm(R) Ethos(TM)-U NPU convolution ops.
 */
#include "../../nn/convolution.h"

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_layout.h>

#include "../../../qnn/utils.h"
#include "common.h"
#include "op_attrs.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

bool EthosuConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto* ifm = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  const auto* scale_bias = types[2].as<TensorTypeNode>();
  if (ifm == nullptr || weight == nullptr) return false;
  const auto* param = attrs.as<EthosuConv2DAttrs>();
  CHECK(param != nullptr) << "EthosuConv2DAttrs cannot be nullptr.";
  const String operator_name = "ethosu_conv2d";

  CheckDataType(reporter, ifm->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "ifm");
  CheckDataType(reporter, weight->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name,
                "weight");
  CheckDataType(reporter, scale_bias->dtype, {DataType::UInt(8)}, operator_name, "scale bias");

  CheckUpscaleMethod(reporter, param->upscale, {"NONE", "ZEROS", "NEAREST"}, operator_name);

  // The scale_bias should be provided as a tensor of size {ofm_channels, 10}
  reporter->Assign(types[2], TensorType({weight->shape[0], 10}, DataType::UInt(8)));

  // Assign weight type {ofm_channels, kernel_height, kernel_width, ifm_channels}
  reporter->Assign(types[1], TensorType({param->ofm_channels, param->kernel_shape[0],
                                         param->kernel_shape[1], weight->shape[3]},
                                        weight->dtype));

  Array<IndexExpr> ifm_shape = ifm->shape;
  if (param->upscale != "NONE") {
    ifm_shape = EthosuInferUpscaledInput(ifm_shape, param->ifm_layout);
  }

  // Assign ofm type
  auto ofm_shape =
      EthosuInferKernelOutput(ifm_shape, param->ifm_layout, param->ofm_layout, param->kernel_shape,
                              param->ofm_channels, param->dilation, param->strides, param->padding);

  reporter->Assign(types[4], TensorType(ofm_shape, ifm->dtype));
  return true;
}

Expr MakeEthosuConv2D(Expr ifm, Expr weight, Expr scale_bias, Expr lut, double ifm_scale,
                      int ifm_zero_point, int weight_zero_point, double ofm_scale,
                      int ofm_zero_point, Array<IndexExpr> kernel_shape, IndexExpr ofm_channels,
                      Array<IndexExpr> strides, Array<IndexExpr> padding, Array<IndexExpr> dilation,
                      String activation, int clip_min, int clip_max, String rounding_mode,
                      String upscale, String ifm_layout, String ofm_layout) {
  auto attrs = make_object<EthosuConv2DAttrs>();
  attrs->ifm_scale = ifm_scale;
  attrs->ifm_zero_point = ifm_zero_point;
  attrs->weight_zero_point = weight_zero_point;
  attrs->ofm_scale = ofm_scale;
  attrs->ofm_zero_point = ofm_zero_point;
  attrs->kernel_shape = std::move(kernel_shape);
  attrs->ofm_channels = std::move(ofm_channels);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->activation = std::move(activation);
  attrs->clip_min = clip_min;
  attrs->clip_max = clip_max;
  attrs->rounding_mode = std::move(rounding_mode);
  attrs->upscale = std::move(upscale);
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  static const Op& op = Op::Get("contrib.ethosu.conv2d");
  return Call(op, {ifm, weight, scale_bias, lut}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ethosu_conv2d").set_body_typed(MakeEthosuConv2D);

RELAY_REGISTER_OP("contrib.ethosu.conv2d")
    .describe(R"code(Arm(R) Ethos(TM)-U NPU 2D quantized convolution operator.

This Relay operator corresponds to the hardware-implemented quantized
convolution operation found on Ethos(TM)-U NPU. It accepts either NHWC
or NHCWB16 format for the input data (Input Feature Map, or IFM) and
OHWI format for the kernel weights.

Reference: https://developer.arm.com/documentation/102420/0200/

Note that the per-channel weight scale and bias tensor must be packed together into
a combined tensor of uint80s. This is represented in TVM by a (channels, 10) tensor
of type uint8. For more detail, refer to the Technical Reference Manual linked above.

- **ifm**: NHWC - (1, ifm_height, ifm_width, ifm_channels)
           NHCWB16 - (1, ifm_height, ifm_channels // 16, ifm_width, 16)
- **weight**: (ofm_channels, kernel_shape[0], kernel_shape[1], ifm_channels)
- **scale_bias**: (ofm_channels, 10)
- **ofm**: (1, ofm_height, ofm_width, ofm_channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EthosuConv2DAttrs>()
    .set_num_inputs(4)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .add_argument("scale_bias", "Tensor", "The packed per-channel weight scale and bias tensor.")
    .add_argument("lut", "Tensor", "The look-up table of values to use if activation = 'LUT'.")
    .set_support_level(11)
    .add_type_rel("EthosuConv2D", EthosuConv2DRel);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
