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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>

#include <algorithm>

namespace tvm {
namespace relax {
using namespace tvm::prim;

template <typename FType>
Type InferTypeBroadcast(const Call& call, const BlockBuilder& ctx, FType f_compute_out_dtype) {
  Op op = call->op.as_or_throw<Op>();
  size_t n_input = op->args_info.size();
  if (call->args.size() != n_input) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << call->op << " op should have " << n_input << " arguments";
  }

  auto lhs_ty = GetType(call->args[0]);
  auto rhs_ty = GetType(call->args[1]);

  TVM_FFI_CHECK(lhs_ty.as<PrimTypeNode>() || lhs_ty.as<TensorTypeNode>(), TypeError)
      << "Arguments to binary operators must be either R.Tensor or R.Prim types, "
      << "but expression " << call << " has LHS " << call->args[0] << ", which has Type " << lhs_ty;
  TVM_FFI_CHECK(rhs_ty.as<PrimTypeNode>() || rhs_ty.as<TensorTypeNode>(), TypeError)
      << "Arguments to binary operators must be either R.Tensor or R.Prim types, "
      << "but expression " << call << " has RHS " << call->args[1] << ", which has Type " << rhs_ty;

  // Dtype
  ffi::Optional<PrimType> output_dtype = f_compute_out_dtype(call, ctx, lhs_ty, rhs_ty);

  if (lhs_ty.as<PrimTypeNode>() && rhs_ty.as<PrimTypeNode>()) {
    TVM_FFI_ICHECK(output_dtype.has_value());
    return output_dtype.value();
  }

  // VDevice
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, lhs_ty, rhs_ty);

  auto get_ndim = [&](const Type& ty) -> int {
    if (ty.as<PrimTypeNode>()) {
      return 1;
    } else if (const auto* tensor = ty.as<TensorTypeNode>()) {
      return tensor->ndim;
    } else {
      return kUnknownNDim;
    }
  };

  // ndims
  int output_ndim = [&]() {
    int lhs_ndim = get_ndim(lhs_ty);
    int rhs_ndim = get_ndim(rhs_ty);
    if (lhs_ndim == kUnknownNDim || rhs_ndim == kUnknownNDim) {
      return kUnknownNDim;
    } else {
      return std::max(lhs_ndim, rhs_ndim);
    }
  }();

  // Shapes

  auto get_shape = [](const Type& ty) -> ffi::Optional<ffi::Array<PrimExpr>> {
    if (ty.as<PrimTypeNode>()) {
      return ffi::Array<PrimExpr>{IntImm::Int64(1)};
    } else if (const auto* tensor = ty.as<TensorTypeNode>()) {
      return tensor->GetShape();
    } else {
      return std::nullopt;
    }
  };

  // If both inputs have a known shape, directly infer the shape of
  // the output.
  auto lhs_shape = get_shape(lhs_ty);
  auto rhs_shape = get_shape(rhs_ty);
  if (lhs_shape && rhs_shape) {
    ffi::Optional<ffi::Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, lhs_shape.value(), rhs_shape.value());
    if (output_shape.has_value()) {
      TVM_FFI_ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      return TensorType(ShapeExpr(output_shape.value()), output_dtype, vdevice);
    }
  }

  auto get_shape_expr = [](const Type& ty) -> ffi::Optional<Expr> {
    if (const auto* tensor = ty.as<TensorTypeNode>()) {
      return tensor->shape;
    } else {
      return std::nullopt;
    }
  };

  // If the input shape is unknown, but both inputs have the same
  // `ShapeType`variable for their shape, then propagate that
  // variable to the output.
  auto lhs_shape_expr = get_shape_expr(lhs_ty);
  auto rhs_shape_expr = get_shape_expr(rhs_ty);
  if (lhs_shape_expr.has_value() && lhs_shape_expr.same_as(rhs_shape_expr)) {
    return TensorType(lhs_shape_expr.value(), output_dtype, vdevice);
  }

  // If neither of those cases holds, then fall back to an unknown
  // shape with `output_ndim` dimensionality.
  return TensorType(output_dtype, output_ndim, vdevice);
}

Type InferTypeBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferTypeBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

Type InferTypeBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferTypeBroadcast(call, ctx,
                            [](const Call& call, const BlockBuilder& ctx, const Type& lhs_ty,
                               const Type& rhs_ty) { return PrimType::Bool(); });
}

InferLayoutOutput InferLayoutBinaryEwise(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  LayoutDecision layout1 = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision layout2 = GetLayoutDecision(var_layout_map, call->args[1]);

  auto* x1_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  auto* x2_ty = GetTypeAs<TensorTypeNode>(call->args[1]);

  TVM_FFI_ICHECK(!x1_ty->IsUnknownNdim() && !x2_ty->IsUnknownNdim())
      << "Unknown dim tensors should not be handled by this function";

  ffi::Optional<ShapeExpr> shape1 = ffi::GetRef<ShapeExpr>(x1_ty->shape.as<ShapeExprNode>());
  ffi::Optional<ShapeExpr> shape2 = ffi::GetRef<ShapeExpr>(x2_ty->shape.as<ShapeExprNode>());
  // Lets handle sub indexing as long as primal dims are matching
  if ((layout1->layout.ndim() != layout1->layout.ndim_primal()) ||
      (layout2->layout.ndim() != layout2->layout.ndim_primal())) {
    if (layout1->layout.ndim_primal() == layout2->layout.ndim_primal()) {
      if ((layout1->layout.ndim() >= layout2->layout.ndim()) && shape2.has_value()) {
        if (CanProveLayoutTransform(InitialLayout(shape2.value()->values.size()), layout1->layout,
                                    shape2.value()->values)) {
          return InferLayoutOutput({layout1, layout1}, {layout1}, Attrs(call->attrs));
        }
      } else if (shape1.has_value()) {
        if (CanProveLayoutTransform(InitialLayout(shape1.value()->values.size()), layout2->layout,
                                    shape1.value()->values)) {
          return InferLayoutOutput({layout2, layout2}, {layout2}, Attrs(call->attrs));
        }
      }
    }
  }

  if (x1_ty->ndim <= x2_ty->ndim) {
    if (x1_ty->ndim == 0) {
      LayoutDecision out_layout = layout2;
      return InferLayoutOutput({LayoutDecision(""), layout2}, {out_layout}, Attrs(call->attrs));
    }
    LayoutDecision out_layout = FollowDecision(layout1, x2_ty->ndim);
    return InferLayoutOutput({layout1, out_layout}, {out_layout}, Attrs(call->attrs));
  } else {
    if (x2_ty->ndim == 0) {
      LayoutDecision out_layout = layout1;
      return InferLayoutOutput({layout1, LayoutDecision("")}, {out_layout}, Attrs(call->attrs));
    }
    LayoutDecision out_layout = FollowDecision(layout2, x1_ty->ndim);

    return InferLayoutOutput({out_layout, layout2}, {out_layout}, Attrs(call->attrs));
  }
}

/***************** Arithmetic operators *****************/

Expr add(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.add");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr divide(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.divide");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr floor_divide(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.floor_divide");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr log_add_exp(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.log_add_exp");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr multiply(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.multiply");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr power(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.power");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr atan2(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.atan2");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr subtract(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.subtract");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr mod(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.mod");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr floor_mod(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.floor_mod");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr equal(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.equal");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr greater(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.greater");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr greater_equal(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.greater_equal");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr less(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.less");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr less_equal(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.less_equal");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr not_equal(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.not_equal");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr minimum(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.minimum");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr maximum(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.maximum");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr logical_and(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.logical_and");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr logical_or(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.logical_or");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr logical_xor(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.logical_xor");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr bitwise_and(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.bitwise_and");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr bitwise_or(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.bitwise_or");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr bitwise_xor(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.bitwise_xor");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr left_shift(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.left_shift");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

Expr right_shift(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.right_shift");
  return Call(Type::Missing(), op, {x1, x2}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("relax.op.add", add);

  OpDef("relax.add")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.divide", divide);

  OpDef("relax.divide")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.floor_divide", floor_divide);

  OpDef("relax.floor_divide")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.log_add_exp", log_add_exp);

  OpDef("relax.log_add_exp")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.multiply", multiply);

  OpDef("relax.multiply")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.power", power);

  OpDef("relax.power")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.atan2", atan2);

  OpDef("relax.atan2")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.subtract", subtract);

  OpDef("relax.subtract")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.mod", mod);

  OpDef("relax.mod")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.floor_mod", floor_mod);

  OpDef("relax.floor_mod")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  /***************** Comparison operators *****************/

  tvm::ffi::reflection::GlobalDef().def("relax.op.equal", equal);

  OpDef("relax.equal")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastCMP);

  tvm::ffi::reflection::GlobalDef().def("relax.op.greater", greater);

  OpDef("relax.greater")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastCMP);

  tvm::ffi::reflection::GlobalDef().def("relax.op.greater_equal", greater_equal);

  OpDef("relax.greater_equal")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastCMP);

  tvm::ffi::reflection::GlobalDef().def("relax.op.less", less);

  OpDef("relax.less")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastCMP);

  tvm::ffi::reflection::GlobalDef().def("relax.op.less_equal", less_equal);

  OpDef("relax.less_equal")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastCMP);

  tvm::ffi::reflection::GlobalDef().def("relax.op.not_equal", not_equal);

  OpDef("relax.not_equal")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastCMP);

  /***************** Min/Max operators *****************/

  tvm::ffi::reflection::GlobalDef().def("relax.op.minimum", minimum);

  OpDef("relax.minimum")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.maximum", maximum);

  OpDef("relax.maximum")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  /***************** Logical operators *****************/

  tvm::ffi::reflection::GlobalDef().def("relax.op.logical_and", logical_and);

  OpDef("relax.logical_and")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.logical_or", logical_or);

  OpDef("relax.logical_or")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.logical_xor", logical_xor);

  OpDef("relax.logical_xor")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  /***************** Bitwise operators *****************/

  tvm::ffi::reflection::GlobalDef().def("relax.op.bitwise_and", bitwise_and);

  OpDef("relax.bitwise_and")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.bitwise_or", bitwise_or);

  OpDef("relax.bitwise_or")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.bitwise_xor", bitwise_xor);

  OpDef("relax.bitwise_xor")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.left_shift", left_shift);

  OpDef("relax.left_shift")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);

  tvm::ffi::reflection::GlobalDef().def("relax.op.right_shift", right_shift);

  OpDef("relax.right_shift")
      .set_num_inputs(2)
      .arg<Expr>("x1", "The first input tensor.")
      .arg<Expr>("x2", "The second input tensor.")
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
      .set_attr<bool>("FPurity", true)
      .set_attr<FInferType>("FInferType", InferTypeBroadcastArith);
}

}  // namespace relax
}  // namespace tvm
