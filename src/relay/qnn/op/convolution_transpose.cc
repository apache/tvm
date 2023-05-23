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
 * \file src/relay/qnn/op/convolution_transpose.cc
 * \brief Property def of qnn transpose convolution operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_layout.h>

#include "../../op/nn/convolution.h"
#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.conv2d_transpose

inline Expr MakeQnnConv2DTranspose(Expr data, Expr weight, Expr input_zero_point,
                                   Expr kernel_zero_point, Expr input_scale, Expr kernel_scale,
                                   Array<IndexExpr> strides, Array<IndexExpr> padding,
                                   Array<IndexExpr> dilation, int groups, IndexExpr channels,
                                   Array<IndexExpr> kernel_size, std::string data_layout,
                                   std::string kernel_layout, std::string out_layout,
                                   Array<IndexExpr> output_padding, DataType out_dtype) {
  auto attrs = make_object<Conv2DTransposeAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->output_padding = std::move(output_padding);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get("qnn.conv2d_transpose");
  return Call(op, {data, weight, input_zero_point, kernel_zero_point, input_scale, kernel_scale},
              Attrs(attrs), {});
}

InferCorrectLayoutOutput QnnConvTransposeInferCorrectLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  // Use Relay Conv2D transpose Infer correct layout.
  auto conv_transpose_new_layouts = ConvInferCorrectLayout<Conv2DTransposeAttrs>(
      attrs, new_in_layouts, old_in_layouts, old_in_types);

  // Fill the layouts of remaining input tensors - scales and zero points. The layouts of these
  // tensors can be treated as channel layout.
  Layout channel_layout = Layout("C");
  Array<Layout> input_layouts = {conv_transpose_new_layouts->input_layouts[0],
                                 conv_transpose_new_layouts->input_layouts[1],
                                 channel_layout,
                                 channel_layout,
                                 channel_layout,
                                 channel_layout};
  Array<Layout> output_layouts = conv_transpose_new_layouts->output_layouts;
  return InferCorrectLayoutOutput(input_layouts, output_layouts, attrs);
}

bool QnnConv2DTransposeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                           const TypeReporter& reporter) {
  // Expected Types: data, weight, input_zero_point, weight_zero_point, input_scale, weight_scale,
  // out_type
  ICHECK_EQ(types.size(), 7);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  if (data == nullptr || weight == nullptr) return false;
  const auto* param = attrs.as<Conv2DTransposeAttrs>();
  ICHECK(param != nullptr) << "Conv2DTransposeAttrs cannot be nullptr.";
  ICHECK(data->dtype == DataType::Int(8) || data->dtype == DataType::UInt(8) ||
         data->dtype == DataType::Int(16) || data->dtype == DataType::UInt(16))
      << "Expected qnn conv2d type(int8, uint8, int16) for input but was " << data->dtype;
  ICHECK(weight->dtype == DataType::Int(8) || weight->dtype == DataType::UInt(8))
      << "Expected qnn conv2d type(int8, uint8) for weight but was " << weight->dtype;
  ICHECK(param->out_dtype == DataType::Int(16) || param->out_dtype == DataType::Int(32) ||
         data->dtype == DataType::Int(64))
      << "Expected qnn conv2d type(int16, int32, int64) for output but was " << param->out_dtype;
  ICHECK(param->out_dtype.bits() > 0) << "Output dtype bits should be greater than 0.";

  // Check the types of scale and zero points.
  for (size_t i = 2; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }

  const auto* weight_zp_type = types[3].as<TensorTypeNode>();
  ICHECK(weight_zp_type->dtype == DataType::Int(32));  // weight_zero_point

  bool input_zp_is_scalar = (types[2].as<TensorTypeNode>())->shape.size() == 0 ||
                            get_const_int((types[2].as<TensorTypeNode>())->Size()) == 1;
  bool input_scale_is_scalar = (types[4].as<TensorTypeNode>())->shape.size() == 0 ||
                               get_const_int((types[4].as<TensorTypeNode>())->Size()) == 1;

  ICHECK(input_scale_is_scalar && input_zp_is_scalar)
      << "Zero point or scale should be scalar or a vector with one element.";

  // Assign types for input scale and zero point.
  AssignType(types[2], DataType::Int(32), Integer(1), reporter);    // input_zero_point
  AssignType(types[4], DataType::Float(32), Integer(1), reporter);  // input_scale

  // Kernel scale can be a vector of length output_channels or a scalar.
  if (param->groups == 1) {
    size_t axis = param->kernel_layout.find('O');
    ICHECK(axis != std::string::npos) << "Kernel layout attribute is not defined";
    AssignType(types[5], DataType::Float(32), weight->shape[axis], reporter);  // weight_scale
  } else {
    // Here, total number of output channels depend on depth multiplier.
    size_t o_axis = param->kernel_layout.find('O');
    size_t i_axis = param->kernel_layout.find('I');
    ICHECK(o_axis != std::string::npos || i_axis != std::string::npos)
        << "Kernel layout attribute is not defined";
    AssignType(types[5], DataType::Float(32), weight->shape[i_axis] * weight->shape[o_axis],
               reporter);  // kernel scale
  }

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // Conv2D infer type function.
  Array<Type> tensor_types = {types[0], types[1], types[6]};
  return Conv2DTransposeRel(tensor_types, 3, attrs, reporter);
}

RELAY_REGISTER_OP("qnn.conv2d_transpose")
    .describe(R"code(Quantized transposed 2D convolution layer (sometimes called Deconvolution).
This operator deconvolves quantized weight with quantized data. The scale of the
output quantized tensor is the product of the weight_scale and input_scale of
the input quantized tensors. The zero point of the output quantized tensor is
0. By default, the dtype of output is int32. Please also refer to Requantize
operator to understand how to scale back the int32 output to (u)int8.
- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<Conv2DTransposeAttrs>()
    .set_num_inputs(6)
    .add_argument("data", "Tensor", "The quantized input data tensor.")
    .add_argument("weight", "Tensor", "The quantized weight tensor.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("weight_scale", "Tensor", "The quantization scale of the weight tensor.")
    .add_argument("weight_zero_point", "Tensor",
                  "The quantization zero_point of the weight tensor.")
    .set_support_level(11)
    .add_type_rel("QnnConv2DTranspose", QnnConv2DTransposeRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", QnnConvTransposeInferCorrectLayout);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.conv2d_transpose").set_body_typed(MakeQnnConv2DTranspose);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
