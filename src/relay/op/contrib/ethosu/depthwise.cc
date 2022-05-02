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
 * \file src/relay/op/contrib/ethosu/depthwise.cc
 * \brief Depthwise convolution 2D operator definition for the Arm(R) Ethos(TM)-U NPU
 */
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_layout.h>

#include "../../../qnn/utils.h"
#include "../../nn/convolution.h"
#include "common.h"
#include "op_attrs.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace ethosu {

bool EthosuDepthwiseConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                              const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 5);
  const auto* ifm = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  const auto* scale_bias = types[2].as<TensorTypeNode>();
  if (ifm == nullptr || weight == nullptr) return false;

  const auto* param = attrs.as<EthosuDepthwiseConv2DAttrs>();
  ICHECK(param != nullptr) << "EthosuDepthwiseConv2DAttrs cannot be nullptr.";

  const String operator_name = "ethosu_depthwise_conv2d";

  CheckDataType(reporter, ifm->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "ifm");
  CheckDataType(reporter, weight->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name,
                "weight");
  CheckDataType(reporter, scale_bias->dtype, {DataType::UInt(8)}, operator_name, "scale bias");

  DataType ofm_dtype = DataTypeFromString(param->ofm_dtype);
  auto ofm_dtypes = {DataType::UInt(8), DataType::Int(8), DataType::Int(16), DataType::Int(32)};
  CheckDataType(reporter, ofm_dtype, ofm_dtypes, operator_name, "ofm");

  // Collect the ifm, weight and ofm tensors for using in the inference function
  Array<Type> tensor_types = {types[0], types[1], types[4]};

  // Assign weight type {ofm_channels, kernel_height, kernel_width, 1}
  reporter->Assign(types[1], TensorType({param->ofm_channels, param->kernel_shape[0],
                                         param->kernel_shape[1], weight->shape[3]},
                                        weight->dtype));

  // Assign ofm type
  auto ofm_shape =
      EthosuInferKernelOutput(ifm->shape, param->ifm_layout, param->ofm_layout, param->kernel_shape,
                              param->ofm_channels, param->dilation, param->strides, param->padding);

  reporter->Assign(types[4], TensorType(ofm_shape, ofm_dtype));

  return true;
}

Expr MakeEthosuDepthwiseConv2D(Expr ifm, Expr weight, Expr scale_bias, Expr lut, double ifm_scale,
                               int ifm_zero_point, int weight_zero_point, double ofm_scale,
                               int ofm_zero_point, Array<IndexExpr> kernel_shape,
                               IndexExpr ofm_channels, Array<IndexExpr> strides,
                               Array<IndexExpr> padding, Array<IndexExpr> dilation,
                               String activation, int clip_min, int clip_max, String rounding_mode,
                               String upscale, String ifm_layout, String ofm_layout,
                               String ofm_dtype) {
  auto attrs = make_object<EthosuDepthwiseConv2DAttrs>();
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
  attrs->ofm_dtype = std::move(ofm_dtype);
  static const Op& op = Op::Get("contrib.ethosu.depthwise_conv2d");
  return Call(op, {ifm, weight, scale_bias, lut}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.ethosu_depthwise_conv2d")
    .set_body_typed(MakeEthosuDepthwiseConv2D);

RELAY_REGISTER_OP("contrib.ethosu.depthwise_conv2d")
    .describe(R"code(Arm(R) Ethos(TM)-U NPU 2D quantized depthwise operator.

This Relay operator corresponds to the hardware-implemented quantized
depthwise operation found on Ethos(TM)-U NPU. It accepts either NHWC or NHCWB16 format
for the input data (input feature map, or IFM) and OHWI format for the kernel weights.

- **ifm**: NHWC - (1, ifm_height, ifm_width, ifm_channels)
           NHCWB16 - (1, ifm_height, ifm_channels // 16, ifm_width, 16)
- **weight**: (ofm_channels, kernel_shape[0], kernel_shape[1], 1 (depth multiplier))
- **scale_bias**: (ofm_channels, 10)
- **ofm**: (1, ofm_height, ofm_width, ofm_channels)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<EthosuDepthwiseConv2DAttrs>()
    .set_num_inputs(4)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .add_argument("scale_bias", "Tensor", "The packed per-channel weight scale and bias tensor.")
    .add_argument("lut", "Tensor", "The look-up table of values to use if activation = 'LUT'")
    .set_support_level(11)
    .add_type_rel("EthosuDepthwiseConv2D", EthosuDepthwiseConv2DRel);

}  // namespace ethosu
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
