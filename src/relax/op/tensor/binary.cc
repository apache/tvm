/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

#include <algorithm>

namespace tvm {
namespace relax {

template <typename FType>
StructInfo InferStructInfoBroadcast(const Call& call, const BlockBuilder& ctx,
                                    FType f_compute_out_dtype) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo x1_sinfo = input_sinfo[0];
  TensorStructInfo x2_sinfo = input_sinfo[1];

  // DateType
  DataType output_dtype = f_compute_out_dtype(call, ctx, x1_sinfo, x2_sinfo);

  // VDevice
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, x1_sinfo, x2_sinfo);

  // ndims
  int output_ndim;
  if (x1_sinfo->IsUnknownNdim() || x2_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(x1_sinfo->ndim, x2_sinfo->ndim);
  }

  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  // Shapes and ndims
  if (x1_shape && x2_shape) {
    // If all inputs have shapes, directly infer shapes
    Optional<Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
    if (!output_shape.defined()) {
      return TensorStructInfo(output_dtype, /*ndim=*/output_ndim, vdevice);

    } else {
      ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype, vdevice);
    }
  } else if (x1_sinfo->shape.defined() && x1_sinfo->shape.same_as(x2_sinfo->shape)) {
    return TensorStructInfo(x1_sinfo->shape.value(), output_dtype, vdevice);
  } else {
    return TensorStructInfo(output_dtype, /*ndim=*/output_ndim, vdevice);
  }
}

StructInfo InferStructInfoBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

StructInfo InferStructInfoBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorStructInfo& x1_sinfo,
         const TensorStructInfo& x2_sinfo) { return DataType::Bool(); });
}

InferLayoutOutput InferLayoutBinaryEwise(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));
  LayoutDecision layout1 = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision layout2 = GetLayoutDecision(var_layout_map, call->args[1]);

  auto* x1_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  auto* x2_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);

  ICHECK(!x1_sinfo->IsUnknownNdim() && !x2_sinfo->IsUnknownNdim())
      << "Unknown dim tensors should not be handled by this function";

  if (x1_sinfo->ndim <= x2_sinfo->ndim) {
    if (x1_sinfo->ndim == 0) {
      LayoutDecision out_layout = layout2;
      return InferLayoutOutput({LayoutDecision(""), layout2}, {out_layout}, Attrs(call->attrs));
    }
    LayoutDecision out_layout = FollowDecision(layout1, x2_sinfo->ndim);
    return InferLayoutOutput({layout1, out_layout}, {out_layout}, Attrs(call->attrs));
  } else {
    if (x2_sinfo->ndim == 0) {
      LayoutDecision out_layout = layout1;
      return InferLayoutOutput({layout1, LayoutDecision("")}, {out_layout}, Attrs(call->attrs));
    }
    LayoutDecision out_layout = FollowDecision(layout2, x1_sinfo->ndim);

    return InferLayoutOutput({out_layout, layout2}, {out_layout}, Attrs(call->attrs));
  }
}

/***************** Arithmetic operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(add);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(divide);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(floor_divide);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(multiply);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(power);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(subtract);

/***************** Comparison operators *****************/

RELAX_REGISTER_CMP_OP_AND_IMPL(equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(greater);
RELAX_REGISTER_CMP_OP_AND_IMPL(greater_equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(less);
RELAX_REGISTER_CMP_OP_AND_IMPL(less_equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(not_equal);

/***************** Min/Max operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(minimum);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(maximum);

/***************** Logical operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(logical_and);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(logical_or);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(logical_xor);

/***************** Bitwise operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(bitwise_and);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(bitwise_or);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(bitwise_xor);

}  // namespace relax
}  // namespace tvm
