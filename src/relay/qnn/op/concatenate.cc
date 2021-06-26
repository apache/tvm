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
 * \file src/relay/qnn/op/concatenate.cc
 * \brief QNN concatenate operator. It concatenates quantized input tensors along a given axis.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/expr.h>

#include "../../op/tensor/transform.h"
#include "../../transforms/infer_layout_utils.h"
#include "../../transforms/pattern_utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

bool QnnConcatenateRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  // Expected Types: data, input_scales, input_zero_points, output_scale, output_zero_point,
  // out_type
  ICHECK_EQ(types.size(), 6);

  if (types[0].as<IncompleteTypeNode>()) {
    return false;
  }
  // Check the scale and zero point types
  const auto* input_scales_tuple = types[1].as<TupleTypeNode>();
  if (input_scales_tuple == nullptr) {
    if (types[1].as<IncompleteTypeNode>()) {
      return false;
    } else {
      throw CompileError(
          ErrorBuilder()
          << "qnn concatenate requires a tuple of scales as the second argument, found "
          << PrettyPrint(types[1]));
    }
  }
  for (const auto& input_scale : input_scales_tuple->fields) {
    if (input_scale.as<IncompleteTypeNode>()) {
      return false;
    }
    ICHECK(IsScalarType(input_scale, DataType::Float(32)));  // input_scales[idx]
  }

  const auto* input_zero_points_tuple = types[2].as<TupleTypeNode>();
  if (input_zero_points_tuple == nullptr) {
    if (types[2].as<IncompleteTypeNode>()) {
      return false;
    } else {
      throw CompileError(
          ErrorBuilder()
          << "qnn concatenate requires a tuple of zero_points as the third argument, found "
          << PrettyPrint(types[2]));
    }
  }
  for (const auto& input_zero_point : input_zero_points_tuple->fields) {
    if (input_zero_point.as<IncompleteTypeNode>()) {
      return false;
    }
    ICHECK(IsScalarType(input_zero_point, DataType::Int(32)));  // input_zero_points[idx]
  }

  for (size_t i = 3; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  ICHECK(IsScalarType(types[3], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[4], DataType::Int(32)));    // output_zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // Concatenate infer type function.
  Array<Type> tensor_types = {types[0], types[5]};
  return ConcatenateRel<ConcatenateAttrs>(tensor_types, 2, attrs, reporter);
}

InferCorrectLayoutOutput QnnConcatenateLayout(const Attrs& attrs,
                                              const Array<Layout>& new_in_layouts,
                                              const Array<Layout>& old_in_layouts,
                                              const Array<tvm::relay::Type>& old_in_types) {
  // Collect the layouts and types to reuse Relay Concatenate Infer Correct Layout.
  ICHECK_EQ(old_in_types.size(), 5);
  auto input_tuple_type = old_in_types[0].as<TupleTypeNode>();
  ICHECK(input_tuple_type);
  auto num_input_tensors = input_tuple_type->fields.size();

  Array<Layout> relay_new_in_layouts(nullptr);
  if (new_in_layouts.defined()) {
    relay_new_in_layouts =
        Array<Layout>(new_in_layouts.begin(), new_in_layouts.begin() + num_input_tensors);
  }
  Array<Layout> relay_old_in_layouts(nullptr);
  if (old_in_layouts.defined()) {
    relay_old_in_layouts =
        Array<Layout>(old_in_layouts.begin(), old_in_layouts.begin() + num_input_tensors);
  }

  // Use Relay Concatenate Infer Correct layout to infer the layouts for data tensors.
  auto concat_new_layout =
      ConcatenateLayout(attrs, relay_new_in_layouts, relay_old_in_layouts, {old_in_types[0]});

  // Fill the layouts of remaining input tensors - scales and zero points. The layouts of these
  // tensors can be treated as channel layout. Total number of these tensors are 2 * num of data
  // tensors (scale and zero point for each input data tensor) + 2 for the output data tensor.
  Layout channel_layout = Layout("C");
  Array<Layout> input_layouts = concat_new_layout->input_layouts;

  for (size_t i = 0; i < 2 * num_input_tensors + 2; i++) {
    input_layouts.push_back(channel_layout);
  }
  Array<Layout> output_layouts = concat_new_layout->output_layouts;
  return InferCorrectLayoutOutput(input_layouts, output_layouts, concat_new_layout->new_attrs);
}

Expr MakeQnnConcatenate(Expr data, Expr input_scales, Expr input_zero_points, Expr output_scale,
                        Expr output_zero_point, int axis) {
  auto attrs = make_object<ConcatenateAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("qnn.concatenate");
  return Call(op, {data, input_scales, input_zero_points, output_scale, output_zero_point},
              Attrs(attrs), {});
}

/*
 * \brief Canonicalizes the QNN concatenate op.
 * \param attrs The QNN concatenate attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for concatenate op.
 */
Expr ConcatenateQnnCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                                const Array<tvm::relay::Type>& arg_types) {
  // Get the attrs.
  ICHECK_EQ(new_args.size(), 5);
  auto& data = new_args[0];
  auto& input_scales = new_args[1];
  auto& input_zero_points = new_args[2];
  auto& output_scale = new_args[3];
  auto& output_zero_point = new_args[4];
  const auto* concatenate_attrs = attrs.as<ConcatenateAttrs>();
  ICHECK(concatenate_attrs != nullptr);

  // Get the input dtype and shape.
  ICHECK_GE(arg_types.size(), 1);
  auto tuple_type = arg_types[0].as<TupleTypeNode>();
  ICHECK(tuple_type != nullptr);

  // FIXME (anijain2305) - The lowering can be further optimized. Instead of inserting requantize in
  // the start, we can insert requantize at the end if and only if all the input tensors have same
  // qnn params. This can be done in future.

  // If the output qnn params do not match the input qnn params, we can call requantize on the input
  // expr first, followed by a concatenate on the requantized input exprs.

  Array<Expr> tuple_exprs;
  if (data->IsInstance<TupleNode>()) {
    tuple_exprs = data.as<TupleNode>()->fields;
  } else if (data->IsInstance<CallNode>()) {  // if the data is a CallNode, use TupleGetItems
    auto call = Downcast<Call>(data);
    for (size_t i = 0; i < tuple_type->fields.size(); i++) {
      tuple_exprs.push_back(TupleGetItem(call, i));
    }
  }
  ICHECK(!tuple_exprs.empty());

  auto tuple_input_scales = input_scales.as<TupleNode>();
  ICHECK(tuple_input_scales != nullptr);

  auto tuple_input_zero_points = input_zero_points.as<TupleNode>();
  ICHECK(tuple_input_zero_points != nullptr);

  int idx = 0;
  Array<Expr> requantized_exprs;
  for (auto quantized_expr : tuple_exprs) {
    // Get the input scale for the idx quantized input tensor.
    auto input_scale = tuple_input_scales->fields[idx];

    // Get the zero point for the idx quantized input tensor.
    auto input_zero_point = tuple_input_zero_points->fields[idx];

    // Check if output and input qnn params are same. If not, requantize.
    if (!IsEqualScalar(input_scale, output_scale) ||
        !IsEqualScalar(input_zero_point, output_zero_point)) {
      // Get the input shape and dtype.
      auto tensor_type = tuple_type->fields[idx].as<TensorTypeNode>();
      auto input_dtype = tensor_type->dtype;
      auto input_shape = tensor_type->shape;

      // Requantize the input.
      auto requantized_expr = Requantize(quantized_expr, input_shape, input_scale, input_zero_point,
                                         output_scale, output_zero_point, input_dtype);
      requantized_exprs.push_back(requantized_expr);
    } else {
      requantized_exprs.push_back(quantized_expr);
    }
    idx++;
  }
  return MakeConcatenate(Tuple(requantized_exprs), concatenate_attrs->axis);
}

RELAY_REGISTER_OP("qnn.concatenate")
    .describe(R"code(Concatenate the quantized input tensors along the given axis.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<ConcatenateAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "The tensor to concatenate.")
    .add_argument("input_scales", "Tensor", "The quantization scales of the input tensors.")
    .add_argument("input_zero_points", "Tensor",
                  "The quantization zero_points of the input tensors.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("QnnConcatenate", QnnConcatenateRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", ConcatenateQnnCanonicalize)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", QnnConcatenateLayout);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.concatenate").set_body_typed(MakeQnnConcatenate);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
