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
 * \file src/relay/qnn/op/avg_pool2d.cc
 * \brief Quantized avg_pool2d operator
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/expr.h>

#include "../../op/nn/nn.h"
#include "../../op/nn/pooling.h"
#include "../../op/nn/pooling_common.h"
#include "../../op/tensor/transform.h"
#include "../../transforms/infer_layout_utils.h"
#include "../../transforms/pattern_utils.h"
#include "../utils.h"
#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.avg_pool2d
bool QnnAvgPool2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  // Expected Types: data, input_zero_point, input_scale, output_zero_point, output_scale
  // out_type

  ICHECK_EQ(types.size(), 6);

  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  ICHECK(data->dtype == DataType::Int(8) || data->dtype == DataType::UInt(8))
      << "Expected quantized avg_pool2d type(int8, uint8) for input but was " << data->dtype;

  const auto* param = attrs.as<AvgPool2DAttrs>();
  ICHECK(param != nullptr) << "AvgPool2DAttrs cannot be nullptr.";

  // Check the types of scale and zero points.
  for (size_t i = 1; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }

  ICHECK(IsScalarType(types[1], DataType::Float(32)));  // input_scale
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // input_zero_point
  ICHECK(IsScalarType(types[3], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[4], DataType::Int(32)));    // output_zero_point

  // Find the output shape and data type
  const auto dshape = data->shape;
  ICHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";

  // Check input and output layout
  Layout layout(param->layout);
  // The Layout is always NHWC
  ICHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
         !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid input layout " << layout
      << ". qnn_avg_pool2d inut layout must have H and W, which cannot be split";

  // Find the output shape and data type
  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));

  IndexExpr pad_h, pad_w;
  if (param->padding.size() == 1) {
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[0] * 2;
  } else if (param->padding.size() == 2) {
    // (top, left)
    pad_h = param->padding[0] * 2;
    pad_w = param->padding[1] * 2;
  } else if (param->padding.size() == 4) {
    // (top, left, bottom, right)
    pad_h = param->padding[0] + param->padding[2];
    pad_w = param->padding[1] + param->padding[3];
  } else {
    return false;
  }

  std::vector<IndexExpr> oshape(dshape.begin(), dshape.end());
  if (dshape[hidx].as<tir::AnyNode>()) {
    oshape[hidx] = dshape[hidx];
  } else {
    oshape[hidx] =
        calculate_pool_dimension(dshape[hidx], pad_h, param->pool_size[0], param->dilation[0],
                                 param->strides[0], param->ceil_mode);
  }
  if (dshape[widx].as<tir::AnyNode>()) {
    oshape[widx] = dshape[widx];
  } else {
    oshape[widx] =
        calculate_pool_dimension(dshape[widx], pad_w, param->pool_size[1], param->dilation[1],
                                 param->strides[1], param->ceil_mode);
  }

  // assign output type
  reporter->Assign(types[5], TensorType(oshape, data->dtype));
  return true;
}

InferCorrectLayoutOutput QnnAvgPoolInferCorrectLayout(const Attrs& attrs,
                                                      const Array<Layout>& new_in_layouts,
                                                      const Array<Layout>& old_in_layouts,
                                                      const Array<tvm::relay::Type>& old_in_types) {
  // Use Relay AvgPool2D Infer correct layout.
  auto avgpool_new_layouts =
      PoolInferCorrectLayout<AvgPool2DAttrs>(attrs, new_in_layouts, old_in_layouts, old_in_types);

  // Scales and zero points are scalars, use the "undef" layout for them.
  Array<Layout> input_layouts = {avgpool_new_layouts->input_layouts[0], Layout::Undef(),
                                 Layout::Undef(), Layout::Undef(), Layout::Undef()};
  Array<Layout> output_layouts = avgpool_new_layouts->output_layouts;
  return InferCorrectLayoutOutput(input_layouts, output_layouts, attrs);
}

/*
 * \brief Forward rewrite the qnn avg_pool2d op.
 * \param attrs The QNN avg_pool2d attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for qnn avg_pool2d op.
 * \note Lowering of the qnn.avg_pool2d operator

 *  Quantized avg_pool2d will take one quantized input tensor and returns another
 *  quantized tensor. Since the input qnn params can be different from the output
 *  qnn params, first, we requantize the input tensors with output qnn params and
 *  cast the results into Int32. Then we call relay.nn.avg_pool2d on that requantized
 *  inputs. Finally, the results are cast into the quantized output data type.

 * Note: The RequantizeOrUpcast function only perform requantization if the input
 * and output qnn params are different, otherwise it only does casting to Int32.
 */

Expr QnnAvgPoolCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                            const Array<tvm::relay::Type>& arg_types) {
  ICHECK_EQ(new_args.size(), 5);
  Expr input_data = new_args[0];
  Expr input_scale = new_args[1];
  Expr input_zero_point = new_args[2];
  Expr output_scale = new_args[3];
  Expr output_zero_point = new_args[4];
  const auto in_shape = get_shape(arg_types[0]);
  const auto* avgpool_attrs = attrs.as<AvgPool2DAttrs>();
  auto requantized_input = RequantizeOrUpcast(input_data, input_scale, input_zero_point,
                                              output_scale, output_zero_point, in_shape);
  Expr nn_avg = AvgPool2D(requantized_input, avgpool_attrs->pool_size, avgpool_attrs->strides,
                          avgpool_attrs->dilation, avgpool_attrs->padding, avgpool_attrs->layout,
                          avgpool_attrs->out_layout, avgpool_attrs->ceil_mode,
                          avgpool_attrs->count_include_pad);

  const auto* data = arg_types[5].as<TensorTypeNode>();
  const int32_t min_val = GetQmin(data->dtype);
  const int32_t max_val = GetQmax(data->dtype);
  return Cast(Clip(nn_avg, min_val, max_val), data->dtype);
}

// Positional relay function to create quantized avg_pool2d operator used by frontend FFI.
Expr MakeQuantizedAvgPool2D(Expr data, Expr input_scale, Expr input_zero_point, Expr output_scale,
                            Expr output_zero_point, Array<IndexExpr> pool_size,
                            Array<IndexExpr> strides, Array<IndexExpr> padding,
                            Array<IndexExpr> dilation, bool ceil_mode, bool count_include_pad,
                            String layout, String output_layout) {
  auto attrs = make_object<AvgPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(output_layout);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  static const Op& op = Op::Get("qnn.avg_pool2d");
  return Call(op, {data, input_scale, input_zero_point, output_scale, output_zero_point},
              Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.avg_pool2d")
    .describe("Customized? qnn_avg_pool2d for quantized tensors.")
    .set_attrs_type<AvgPool2DAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Quantized Tensor", "The input data.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("QnnAvgPool2D", QnnAvgPool2DRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", QnnAvgPoolInferCorrectLayout)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnAvgPoolCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.avg_pool2d").set_body_typed(MakeQuantizedAvgPool2D);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
