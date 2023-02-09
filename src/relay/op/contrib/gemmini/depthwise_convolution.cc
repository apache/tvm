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
 * \file src/relay/op/contrib/gemmini/depthwise_convolution.cc
 * \brief 2D depthwise convolution operator definition for Gemmini.
 * \author Federico Peccia <https://fPecc.github.io/>
 */
#include <tvm/relay/op.h>

#include "../../../qnn/utils.h"
#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace gemmini {

/*! \brief Attributes used by the Gemmini 2D depthwise convolution operator */
struct GemminiDepthwiseConv2dAttrs : public tvm::AttrsNode<GemminiDepthwiseConv2dAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  double ifm_scale;
  Expr ifm_offset;
  double weights_scale;
  double weights_offset;
  Expr bias_scale;
  Expr bias_offset;
  Expr ofm_scale;
  Expr ofm_offset;
  bool activation;

  TVM_DECLARE_ATTRS(GemminiDepthwiseConv2dAttrs, "relay.attrs.GemminiDepthwiseConv2dAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("The 2 dimensional strides as (stride_height, stride_width).");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0, 0, 0}))
        .describe("The 4 dimensional padding.");
    TVM_ATTR_FIELD(ifm_scale).set_default(1.0).describe("Input quantization scale");
    TVM_ATTR_FIELD(ifm_offset).describe("Input quantization offset");
    TVM_ATTR_FIELD(weights_scale).set_default(1.0).describe("Weights quantization scale");
    TVM_ATTR_FIELD(weights_offset).set_default(0.0).describe("Weights quantization offset");
    TVM_ATTR_FIELD(bias_scale).describe("Bias quantization scale");
    TVM_ATTR_FIELD(bias_offset).describe("Bias quantization offset");
    TVM_ATTR_FIELD(ofm_scale).describe("Output quantization scale");
    TVM_ATTR_FIELD(ofm_offset).describe("Output quantization offset");
    TVM_ATTR_FIELD(activation)
        .set_default(false)
        .describe("If it has a ReLu activation (True) or not (False)");
  }
};

TVM_REGISTER_NODE_TYPE(GemminiDepthwiseConv2dAttrs);

bool GemminiDepthwiseConv2dRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                               const TypeReporter& reporter) {
  const int data_index = 0;
  const int weights_index = 1;
  const int bias_index = 2;
  const int result_index = 3;

  const auto* data = types[data_index].as<TensorTypeNode>();
  const auto* weights = types[weights_index].as<TensorTypeNode>();
  const auto* bias = types[bias_index].as<TensorTypeNode>();
  if (data == nullptr) return false;
  if (weights == nullptr) return false;
  if (bias == nullptr) return false;

  const auto* params = attrs.as<GemminiDepthwiseConv2dAttrs>();
  ICHECK(params != nullptr) << "GemminiDepthwiseConv2dAttrs cannot be nullptr.";

  DataType ofm_dtype = DataType::Int(8);

  // Assign ofm type
  Array<IndexExpr> ofm_shape(
      {data->shape[0],
       ((data->shape[1] + (params->padding[0] + params->padding[2]) - weights->shape[1]) /
        params->strides[0]) +
           1,
       ((data->shape[2] + (params->padding[1] + params->padding[3]) - weights->shape[2]) /
        params->strides[1]) +
           1,
       weights->shape[0]});
  reporter->Assign(types[result_index], TensorType(ofm_shape, ofm_dtype));
  return true;
}

Expr MakeGemminiDepthwiseConv2d(Expr data, Expr weights, Expr bias, Array<IndexExpr> strides,
                                Array<IndexExpr> padding, double ifm_scale, Expr ifm_offset,
                                double weights_scale, double weights_offset, Expr bias_scale,
                                Expr bias_offset, Expr ofm_scale, Expr ofm_offset,
                                bool activation) {
  auto attrs = make_object<GemminiDepthwiseConv2dAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->activation = std::move(activation);
  attrs->ifm_scale = std::move(ifm_scale);
  attrs->ifm_offset = std::move(ifm_offset);
  attrs->weights_scale = std::move(weights_scale);
  attrs->weights_offset = std::move(weights_offset);
  attrs->bias_scale = std::move(bias_scale);
  attrs->bias_offset = std::move(bias_offset);
  attrs->ofm_scale = std::move(ofm_scale);
  attrs->ofm_offset = std::move(ofm_offset);

  static const Op& op = Op::Get("contrib.gemmini.depthwiseconv2d");

  // Bias change
  // Term 3
  auto reduced_t3 = Sum(Cast(weights, DataType::Int(32)), {1, 2}, false, false);
  auto term3 = Multiply(attrs->ifm_offset, reduced_t3);

  auto new_bias = Subtract(bias, term3);
  auto scale = Divide(attrs->bias_scale, attrs->ofm_scale);
  auto bias_fix = Divide(Cast(attrs->ofm_offset, DataType::Float(32)), scale);
  new_bias = Add(new_bias, Cast(bias_fix, DataType::Int(32)));

  auto conv2d_output = Call(op, {data, weights, new_bias}, Attrs(attrs), {});
  return conv2d_output;
}

TVM_REGISTER_GLOBAL("relay.op._make.gemmini_depthwise_conv2d")
    .set_body_typed(MakeGemminiDepthwiseConv2d);

RELAY_REGISTER_OP("contrib.gemmini.depthwiseconv2d")
    .describe("Gemmini 2D depthwise convolution operator.")
    .set_attrs_type<GemminiDepthwiseConv2dAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The Input Feature Map tensor.")
    .add_argument("weights", "Tensor", "The Weights tensor.")
    .add_argument("bias", "Tensor", "The bias tensor.")
    .set_support_level(11)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .add_type_rel("GemminiDepthwiseConv2d", GemminiDepthwiseConv2dRel);

}  // namespace gemmini
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
