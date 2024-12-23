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
  Op op = Downcast<Op>(call->op);
  size_t n_input = op->arguments.size();
  if (call->args.size() != n_input) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << call->op << " op should have " << n_input << " arguments");
  }

  auto lhs_sinfo = GetStructInfo(call->args[0]);
  auto rhs_sinfo = GetStructInfo(call->args[1]);

  CHECK(lhs_sinfo.as<PrimStructInfoNode>() || lhs_sinfo.as<TensorStructInfoNode>())
      << "TypeError: "
      << "Arguments to binary operators must be either R.Tensor or R.Prim types, "
      << "but expression " << call << " has LHS " << call->args[0] << ", which has StructInfo "
      << lhs_sinfo;
  CHECK(rhs_sinfo.as<PrimStructInfoNode>() || rhs_sinfo.as<TensorStructInfoNode>())
      << "TypeError: "
      << "Arguments to binary operators must be either R.Tensor or R.Prim types, "
      << "but expression " << call << " has RHS " << call->args[1] << ", which has StructInfo "
      << rhs_sinfo;

  // DateType
  DataType output_dtype = f_compute_out_dtype(call, ctx, lhs_sinfo, rhs_sinfo);

  if (lhs_sinfo.as<PrimStructInfoNode>() && rhs_sinfo.as<PrimStructInfoNode>()) {
    return PrimStructInfo(output_dtype);
  }

  // VDevice
  Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, lhs_sinfo, rhs_sinfo);

  auto get_ndim = [&](const StructInfo& sinfo) -> int {
    if (sinfo.as<PrimStructInfoNode>()) {
      return 1;
    } else if (const auto* tensor = sinfo.as<TensorStructInfoNode>()) {
      return tensor->ndim;
    } else {
      return kUnknownNDim;
    }
  };

  // ndims
  int output_ndim = [&]() {
    int lhs_ndim = get_ndim(lhs_sinfo);
    int rhs_ndim = get_ndim(rhs_sinfo);
    if (lhs_ndim == kUnknownNDim || rhs_ndim == kUnknownNDim) {
      return kUnknownNDim;
    } else {
      return std::max(lhs_ndim, rhs_ndim);
    }
  }();

  // Shapes

  auto get_shape = [](const StructInfo& sinfo) -> Optional<Array<PrimExpr>> {
    if (sinfo.as<PrimStructInfoNode>()) {
      return Array<PrimExpr>{IntImm(DataType::Int(64), 1)};
    } else if (const auto* tensor = sinfo.as<TensorStructInfoNode>()) {
      return tensor->GetShape();
    } else {
      return NullOpt;
    }
  };

  // If both inputs have a known shape, directly infer the shape of
  // the output.
  auto lhs_shape = get_shape(lhs_sinfo);
  auto rhs_shape = get_shape(rhs_sinfo);
  if (lhs_shape && rhs_shape) {
    Optional<Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, lhs_shape.value(), rhs_shape.value());
    if (output_shape.defined()) {
      ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      return TensorStructInfo(ShapeExpr(output_shape.value()), output_dtype, vdevice);
    }
  }

  auto get_shape_expr = [](const StructInfo& sinfo) -> Optional<Expr> {
    if (const auto* tensor = sinfo.as<TensorStructInfoNode>()) {
      return tensor->shape;
    } else {
      return NullOpt;
    }
  };

  // If the input shape is unknown, but both inputs have the same
  // `ShapeStructInfo`variable for their shape, then propagate that
  // variable to the output.
  auto lhs_shape_expr = get_shape_expr(lhs_sinfo);
  auto rhs_shape_expr = get_shape_expr(rhs_sinfo);
  if (lhs_shape_expr.defined() && lhs_shape_expr.same_as(rhs_shape_expr)) {
    return TensorStructInfo(lhs_shape_expr.value(), output_dtype, vdevice);
  }

  // If neither of those cases holds, then fall back to an unknown
  // shape with `output_ndim` dimensionality.
  return TensorStructInfo(output_dtype, output_ndim, vdevice);
}

StructInfo InferStructInfoBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

StructInfo InferStructInfoBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const StructInfo& lhs_sinfo,
         const StructInfo& rhs_sinfo) { return DataType::Bool(); });
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

  Optional<ShapeExpr> shape1 = GetRef<ShapeExpr>(x1_sinfo->shape.as<ShapeExprNode>());
  Optional<ShapeExpr> shape2 = GetRef<ShapeExpr>(x2_sinfo->shape.as<ShapeExprNode>());
  // Lets handle sub indexing as long as primal dims are matching
  if (layout1->layout.ndim_primal() == layout2->layout.ndim_primal()) {
    if ((layout1->layout.ndim() >= layout2->layout.ndim()) && shape2.defined()) {
      if (CanProveLayoutTransform(layout2->layout, layout1->layout, shape2.value()->values)) {
        return InferLayoutOutput({layout1, layout1}, {layout1}, Attrs(call->attrs));
      }
    } else if (shape1.defined()) {
      if (CanProveLayoutTransform(layout1->layout, layout2->layout, shape1.value()->values)) {
        return InferLayoutOutput({layout2, layout2}, {layout2}, Attrs(call->attrs));
      }
    }
  }

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
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(mod);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(floor_mod);

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
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(left_shift);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(right_shift);

}  // namespace relax
}  // namespace tvm
