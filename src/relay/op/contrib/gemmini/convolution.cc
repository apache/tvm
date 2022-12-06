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
 * \file src/relay/op/contrib/gemmini/convolution.cc
 * \brief 2D convolution operator definition for Gemmini.
 * \author Federico Peccia <https://fPecc.github.io/>
 */
#include <tvm/relay/op.h>

#include "../../../qnn/utils.h"
#include "../../op_common.h"
//#include "common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace gemmini {

/*! \brief Attributes used by the Gemmini 2D convolution operator */
struct GemminiConv2dAttrs : public tvm::AttrsNode<GemminiConv2dAttrs> {
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
  bool has_pool;
  Array<IndexExpr> pool_size;
  Array<IndexExpr> pool_strides;
  Array<IndexExpr> pool_dilation;
  Array<IndexExpr> pool_padding;
  Expr input_req_offset_out;
  Expr activation_scale_in;
  Expr activation_offset_in;
  Expr activation_scale_out;
  Expr activation_offset_out;
  bool has_activation;

  TVM_DECLARE_ATTRS(GemminiConv2dAttrs, "relay.attrs.GemminiConv2dAttrs") {
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
    TVM_ATTR_FIELD(has_pool).set_default(false).describe(
        "If it has a pool layer (True) or not (False)");
    TVM_ATTR_FIELD(pool_size).describe("Pooling window size");
    TVM_ATTR_FIELD(pool_strides).describe("Pooling window strides");
    TVM_ATTR_FIELD(pool_dilation).describe("Pooling window dilation");
    TVM_ATTR_FIELD(pool_padding).describe("Pooling padding");
    TVM_ATTR_FIELD(input_req_offset_out).describe("Requantization output offset");
    TVM_ATTR_FIELD(activation_scale_in).describe("Activation input scaling factor");
    TVM_ATTR_FIELD(activation_offset_in).describe("Activation input offset");
    TVM_ATTR_FIELD(activation_scale_out).describe("Activation output scaling factor");
    TVM_ATTR_FIELD(activation_offset_out).describe("Activation output offset");
    TVM_ATTR_FIELD(has_activation).describe("Has activation?");
  }
};

TVM_REGISTER_NODE_TYPE(GemminiConv2dAttrs);

bool GemminiConv2dRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
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

  const auto* params = attrs.as<GemminiConv2dAttrs>();
  ICHECK(params != nullptr) << "GemminiConv2dAttrs cannot be nullptr.";

  DataType ofm_dtype = DataType::Int(8);

  // Assign ofm type
  PrimExpr conv2d_output_h =
      ((data->shape[1] + (params->padding[0] + params->padding[2]) - weights->shape[0]) /
       params->strides[0]) +
      1;
  PrimExpr conv2d_output_w =
      ((data->shape[2] + (params->padding[1] + params->padding[3]) - weights->shape[1]) /
       params->strides[1]) +
      1;
  PrimExpr max_pool2d_h = conv2d_output_h;
  PrimExpr max_pool2d_w = conv2d_output_w;
  if (params->has_pool) {
    max_pool2d_h = ((conv2d_output_h + (params->pool_padding[0] + params->pool_padding[2]) -
                     params->pool_size[0]) /
                    params->pool_strides[0]) +
                   1;
    max_pool2d_w = ((conv2d_output_w + (params->pool_padding[1] + params->pool_padding[3]) -
                     params->pool_size[1]) /
                    params->pool_strides[1]) +
                   1;
  }
  Array<IndexExpr> ofm_shape({data->shape[0], max_pool2d_h, max_pool2d_w, weights->shape[3]});
  reporter->Assign(types[result_index], TensorType(ofm_shape, ofm_dtype));
  return true;
}

Expr MakeGemminiConv2d(Expr data, Expr weights, Expr bias, Array<IndexExpr> strides,
                       Array<IndexExpr> padding, double ifm_scale, Expr ifm_offset,
                       double weights_scale, double weights_offset, Expr bias_scale,
                       Expr bias_offset, Expr ofm_scale, Expr ofm_offset, bool activation,
                       bool has_pool, Array<IndexExpr> pool_size, Array<IndexExpr> pool_strides,
                       Array<IndexExpr> pool_dilation, Array<IndexExpr> pool_padding,
                       Expr input_req_offset_out, bool has_activation, Expr activation_scale_in,
                       Expr activation_offset_in, Expr activation_scale_out,
                       Expr activation_offset_out) {
  auto attrs = make_object<GemminiConv2dAttrs>();
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
  attrs->has_pool = std::move(has_pool);
  attrs->pool_size = std::move(pool_size);
  attrs->pool_strides = std::move(pool_strides);
  attrs->pool_dilation = std::move(pool_dilation);
  attrs->pool_padding = std::move(pool_padding);
  attrs->input_req_offset_out = std::move(input_req_offset_out);
  attrs->activation_scale_in = std::move(activation_scale_in);
  attrs->activation_offset_in = std::move(activation_offset_in);
  attrs->activation_scale_out = std::move(activation_scale_out);
  attrs->activation_offset_out = std::move(activation_offset_out);
  attrs->has_activation = std::move(has_activation);

  static const Op& op = Op::Get("contrib.gemmini.conv2d");

  auto zero_const = MakeConstantScalar(DataType::Int(32), 0);
  auto one_const = MakeConstantScalar(DataType::Int(32), 0);

  auto new_bias = bias;
  // Bias change
  // Term 3
  auto reduced_t3 = Sum(Cast(weights, DataType::Int(32)), {0, 1, 2}, false, false);
  auto term3 = Multiply(attrs->ifm_offset, reduced_t3);
  auto input_req_bias_term = Multiply(attrs->input_req_offset_out, reduced_t3);

  new_bias = Add(Subtract(bias, term3), input_req_bias_term);
  auto scale_1 = Divide(attrs->bias_scale, attrs->ofm_scale);
  auto bias_fix = Divide(Cast(attrs->ofm_offset, DataType::Float(32)), scale_1);
  new_bias = Add(new_bias, Cast(bias_fix, DataType::Int(32)));

  if (attrs->has_activation) {
    auto scale_2 = Divide(attrs->activation_scale_in, attrs->activation_scale_out);
    auto term_1 = Cast(attrs->activation_offset_in, DataType::Float(32));
    auto term_2 = Divide(Cast(attrs->activation_offset_out, DataType::Float(32)), scale_2);
    auto bias_fix = Divide(Subtract(term_2, term_1), scale_1);
    new_bias = Add(new_bias, Cast(bias_fix, DataType::Int(32)));
  }

  auto conv2d_output = Call(op, {data, weights, new_bias}, Attrs(attrs), {});
  return conv2d_output;
}

TVM_REGISTER_GLOBAL("relay.op._make.gemmini_conv2d").set_body_typed(MakeGemminiConv2d);

RELAY_REGISTER_OP("contrib.gemmini.conv2d")
    .describe("Gemmini 2D convolution operator")
    .set_attrs_type<GemminiConv2dAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "The Input Feature Map tensor.")
    .add_argument("weights", "Tensor", "The Weights tensor.")
    .add_argument("bias", "Tensor", "The bias tensor.")
    .set_support_level(11)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .add_type_rel("GemminiConv2d", GemminiConv2dRel);

}  // namespace gemmini
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
