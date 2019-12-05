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

#include <tvm/ir.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../../op/tensor/transform.h"
#include "../../pass/pattern_util.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(QnnConcatenateAttrs);

Expr MakeQnnConcatenate(Expr data, Array<tvm::Expr> input_scales,
                        Array<tvm::Expr> input_zero_points, double output_scale,
                        int32_t output_zero_point, int axis) {
  auto attrs = make_node<QnnConcatenateAttrs>();
  attrs->input_scales = std::move(input_scales);
  attrs->input_zero_points = std::move(input_zero_points);
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->axis = axis;
  static const Op& op = Op::Get("qnn.concatenate");
  return CallNode::make(op, {data}, Attrs(attrs), {});
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
  CHECK_EQ(new_args.size(), 1);
  auto& data = new_args[0];
  const auto* concatenate_attrs = attrs.as<QnnConcatenateAttrs>();
  CHECK(concatenate_attrs != nullptr);
  auto input_scales = concatenate_attrs->input_scales;
  auto input_zero_points = concatenate_attrs->input_zero_points;
  auto output_scale = concatenate_attrs->output_scale;
  auto output_zero_point = concatenate_attrs->output_zero_point;

  // Get the input dtype and shape.
  CHECK_GE(arg_types.size(), 1);
  auto tuple_type = arg_types[0].as<TupleTypeNode>();
  CHECK(tuple_type != nullptr);

  // FIXME (anijain2305) - The lowering can be further optimized. Instead of inserting requantize in
  // the start, we can insert requantize at the end if and only if all the input tensors have same
  // qnn params. This can be done in future.

  // If the output qnn params do not match the input qnn params, we can call requantize on the input
  // expr first, followed by a concatenate on the requantized input exprs.

  auto tuple_data = data.as<TupleNode>();
  CHECK(tuple_data != nullptr);

  int idx = 0;
  Array<Expr> requantized_exprs;
  for (auto quantized_expr : tuple_data->fields) {
    // Get the input scale for the idx quantized input tensor.
    auto input_scale_expr = input_scales[idx].as<tvm::ir::FloatImm>();
    CHECK(input_scale_expr != nullptr);
    auto input_scale = input_scale_expr->value;

    // Get the zero point for the idx quantized input tensor.
    auto input_zero_point_expr = input_zero_points[idx].as<tvm::ir::IntImm>();
    CHECK(input_zero_point_expr != nullptr);
    auto input_zero_point = input_zero_point_expr->value;

    // Check if output and input qnn params are same. If not, requantize.
    if (input_scale != output_scale || input_zero_point != output_zero_point) {
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
  return MakeConcatenate(TupleNode::make(requantized_exprs), concatenate_attrs->axis);
}

RELAY_REGISTER_OP("qnn.concatenate")
.describe(R"code(Concatenate the quantized input tensors along the given axis.
)code" TVM_ADD_FILELINE)
.set_attrs_type<QnnConcatenateAttrs>()
.set_num_inputs(1)
.add_argument("data", "Tensor", "The tensor to concatenate.")
.set_support_level(11)
.add_type_rel("QnnConcatenate", ConcatenateRel<QnnConcatenateAttrs>)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", ConcatenateQnnCanonicalize);

TVM_REGISTER_API("relay.qnn.op._make.concatenate")
.set_body_typed(MakeQnnConcatenate);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
