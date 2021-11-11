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

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

/*! \brief Attributes used by the Ethos(TM)-U NPU pooling operator */
struct EthosuPoolingAttrs : public tvm::AttrsNode<EthosuPoolingAttrs> {
  String pooling_type;
  double ifm_scale;
  int ifm_zero_point;
  double ofm_scale;
  int ofm_zero_point;
  Array<IndexExpr> pool_shape;
  IndexExpr ofm_channels;
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  String activation;
  int clip_min;
  int clip_max;
  String upscale;
  String ifm_layout;
  String ofm_layout;

  TVM_DECLARE_ATTRS(EthosuPoolingAttrs, "relay.attrs.EthosuPoolingAttrs") {
    TVM_ATTR_FIELD(pooling_type)
        .describe("The type of the pooling. 'AVG' - average pool, 'MAX' - max pool.");
    TVM_ATTR_FIELD(ifm_scale).describe("The quantization scale for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ifm_zero_point)
        .describe("The quantization zero point for the Input Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_scale).describe("The quantization scale for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(ofm_zero_point)
        .describe("The quantization zero point for the Output Feature Map tensor.");
    TVM_ATTR_FIELD(pool_shape)
        .describe("The 2 dimensional pool shape as (pool_shape_height, pool_shape_width).")
        .set_default(NullValue<Array<IndexExpr> >());
    TVM_ATTR_FIELD(ofm_channels)
        .describe(" The number of the Output Feature Map channels.")
        .set_default(NullValue<IndexExpr>());
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The 2 dimensional strides as (stride_height, stride_width).");
    TVM_ATTR_FIELD(padding)
        .describe("The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).")
        .set_default(Array<IndexExpr>({0, 0, 0, 0}));
    TVM_ATTR_FIELD(activation)
        .describe(
            "The activation function to use. "
            "'NONE' - no activation function. "
            "'CLIP' - clip the output between clip_min and clip_max. "
            "'TANH' - tanh activation function. "
            "'SIGMOID' - sigmoid activation function. "
            "'LUT' - use a look-up table to perform the activation function.")
        .set_default("NONE");
    TVM_ATTR_FIELD(clip_min)
        .describe("The minimum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(clip_max)
        .describe("The maximum clipping value if activation = 'CLIP'.")
        .set_default(0);
    TVM_ATTR_FIELD(upscale)
        .describe(
            "The 2x2 upscaling mode to apply to the Input Feature Map tensor. "
            "'NONE' - no upscaling. "
            "'NEAREST' - upscale using nearest neighbour. "
            "'ZEROS' - upscale using zeros.")
        .set_default("NONE");
    TVM_ATTR_FIELD(ifm_layout)
        .describe("The layout of the Input Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
    TVM_ATTR_FIELD(ofm_layout)
        .describe("The layout of the Output Feature Map tensor. Can be 'NHWC' or 'NHCWB16'.")
        .set_default("NHWC");
  }
};

TVM_REGISTER_NODE_TYPE(EthosuPoolingAttrs);

bool EthosuPoolingRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  int ifm_index = 0;
  int result_index = 2;
  ICHECK_EQ(types.size(), result_index + 1);

  const auto* ifm = types[ifm_index].as<TensorTypeNode>();
  if (ifm == nullptr) return false;

  const auto* param = attrs.as<EthosuPoolingAttrs>();
  ICHECK(param != nullptr) << "EthosuPoolingAttrs cannot be nullptr.";

  if (param->pooling_type != "AVG" && param->pooling_type != "MAX") {
    reporter->GetDiagCtx().EmitFatal(
        Diagnostic::Error(reporter->GetSpan())
        << "Invalid operator: expected pooling_type 'AVG' or 'MAX' but was "
        << param->pooling_type);
    return false;
  }

  if (ifm->dtype != DataType::UInt(8) && ifm->dtype != DataType::Int(8)) {
    reporter->GetDiagCtx().EmitFatal(
        Diagnostic::Error(reporter->GetSpan())
        << "Invalid operator: Expected pool type(uint8) or type(int8) for ifm but was "
        << ifm->dtype);
    return false;
  }

  // Assign ofm type
  auto ofm_shape = EthosuInferKernelOutput(
      ifm->shape, param->ifm_layout, param->ofm_layout, param->pool_shape, param->ofm_channels,
      Array<IndexExpr>({1, 1}), param->strides, param->padding);
  reporter->Assign(types[result_index], TensorType(ofm_shape, ifm->dtype));
  return true;
}

Expr MakeEthosuPooling(Expr ifm, Expr lut, String pooling_type, double ifm_scale,
                       int ifm_zero_point, double ofm_scale, int ofm_zero_point,
                       Array<IndexExpr> pool_shape, IndexExpr ofm_channels,
                       Array<IndexExpr> strides, Array<IndexExpr> padding, String activation,
                       int clip_min, int clip_max, String upscale, String ifm_layout,
                       String ofm_layout) {
  auto attrs = make_object<EthosuPoolingAttrs>();
  attrs->pooling_type = std::move(pooling_type);
  attrs->ifm_scale = ifm_scale;
  attrs->ifm_zero_point = ifm_zero_point;
  attrs->ofm_scale = ofm_scale;
  attrs->ofm_zero_point = ofm_zero_point;
  attrs->pool_shape = std::move(pool_shape);
  attrs->ofm_channels = std::move(ofm_channels);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->activation = std::move(activation);
  attrs->clip_min = clip_min;
  attrs->clip_max = clip_max;
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
