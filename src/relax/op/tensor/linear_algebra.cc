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
 * \file linear_algebra.cc
 * \brief Linear algebra operators.
 */

#include "linear_algebra.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/topi/einsum.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  MatmulAttrs::RegisterReflection();
  EinsumAttrs::RegisterReflection();
}

/* relax.matmul */

Expr matmul(Expr x1, Expr x2, ffi::Optional<DataType> out_dtype) {
  ffi::ObjectPtr<MatmulAttrs> attrs = ffi::make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype.value_or(DataType::Void());

  static const Op& op = Op::Get("relax.matmul");
  return Call(op, {std::move(x1), std::move(x2)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.matmul", matmul);
}

Type InferTypeMatmul(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  Expr lhs = call->args[0];
  Expr rhs = call->args[1];
  TensorType x1_ty = input_ty[0];
  TensorType x2_ty = input_ty[1];

  VDevice vdev = VDevice();
  if (x1_ty->vdevice.defined() && x2_ty->vdevice.defined()) {
    if (x1_ty->vdevice.value() == x2_ty->vdevice.value()) {
      vdev = x1_ty->vdevice.value();
    }
  } else if (x1_ty->vdevice.defined()) {
    vdev = x1_ty->vdevice.value();
  } else if (x2_ty->vdevice.defined()) {
    vdev = x2_ty->vdevice.value();
  }

  const auto* attrs = call->attrs.as<MatmulAttrs>();
  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, x1_ty, x2_ty)
                           : attrs->out_dtype;

  if (x1_ty->IsUnknownNdim() || x2_ty->IsUnknownNdim()) {
    if (vdev.defined()) {
      return TensorType(out_dtype, kUnknownNDim, vdev);
    }
    return TensorType(out_dtype, kUnknownNDim);
  }
  int x1_ndim = x1_ty->ndim;
  int x2_ndim = x2_ty->ndim;
  if (x1_ndim == 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul operands must not be scalar.  "
        << "However, the expression " << call << " has a LHS of " << lhs << " with type " << x1_ty
        << ", which is scalar (zero-dimensional) tensor.";
  }
  if (x2_ndim == 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul operands must not be scalar.  "
        << "However, the expression " << call << " has a RHS of " << rhs << " with type " << x2_ty
        << ", which is scalar (zero-dimensional) tensor.";
  }

  int x1_prepended = 0;
  int x2_appended = 0;
  if (x1_ndim == 1) {
    x1_ndim = 2;
    x1_prepended = 1;
  }
  if (x2_ndim == 1) {
    x2_ndim = 2;
    x2_appended = 1;
  }
  int output_ndim = std::max(x1_ndim, x2_ndim) - x1_prepended - x2_appended;

  const auto* x1_shape = x1_ty->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_ty->shape.as<ShapeExprNode>();
  if (x1_shape == nullptr || x2_shape == nullptr) {
    if (vdev.defined()) {
      return TensorType(out_dtype, output_ndim, vdev);
    }
    return TensorType(out_dtype, output_ndim);
  }

  ffi::Array<PrimExpr> x1_shape_prefix{x1_shape->values.begin(),
                                       x1_shape->values.end() - 2 + x1_prepended};
  ffi::Array<PrimExpr> x2_shape_prefix{x2_shape->values.begin(),
                                       x2_shape->values.end() - 2 + x2_appended};
  ffi::Optional<ffi::Array<PrimExpr>> output_shape_prefix =
      InferBinaryBroadcastShape(call, ctx, x1_shape_prefix, x2_shape_prefix);
  if (!output_shape_prefix.defined()) {
    if (vdev.defined()) {
      return TensorType(out_dtype, output_ndim, vdev);
    }
    return TensorType(out_dtype, output_ndim);
  }

  arith::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr x1_reduction_length = x1_shape->values[x1_ty->ndim - 1];
  PrimExpr x2_reduction_length = x2_shape->values[x2_ndim - 2];
  if (analyzer->CanProve(x1_reduction_length != x2_reduction_length)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul requires the reduction length of the operands to be equal.  "
        << "However, the LHS " << lhs << " has shape " << x1_ty->shape << ", while the RHS " << rhs
        << " has shape " << x2_ty->shape << ".  The reduction dimensions of " << x1_reduction_length
        << " and " << x2_reduction_length << " are not equal.";
  }

  ffi::Array<PrimExpr> output_shape = output_shape_prefix.value();
  if (!x1_prepended) {
    output_shape.push_back(x1_shape->values[x1_ndim - 2]);
  }
  if (!x2_appended) {
    output_shape.push_back(x2_shape->values[x2_ndim - 1]);
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(output_shape.size()), output_ndim);
  if (vdev.defined()) {
    return TensorType(ShapeExpr(output_shape), out_dtype, vdev);
  }
  return TensorType(ShapeExpr(output_shape), out_dtype);
}

Call InferMixedPrecisionMatmul(const Call& call, const DataType& out_dtype) {
  return matmul(call->args[0], call->args[1], out_dtype).as_or_throw<Call>();
}

TVM_REGISTER_OP("relax.matmul")
    .set_num_inputs(2)
    .add_argument("x1", "Tensor", "The first input tensor.")
    .add_argument("x2", "Tensor", "The second input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeMatmul)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionMatmul)
    .set_attr<bool>("FPurity", true);

/* relax.einsum */

Expr einsum(Expr operands, ffi::String subscripts) {
  ffi::ObjectPtr<EinsumAttrs> attrs = ffi::make_object<EinsumAttrs>();
  attrs->subscripts = std::move(subscripts);

  static const Op& op = Op::Get("relax.einsum");
  return Call(op, {std::move(operands)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.einsum", einsum);
}

Type InferTypeEinsum(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Einsum op should take 1 argument";
  }
  ffi::Array<TensorType> operands_tensor_ty = GetTensorTypeFromTuple(call, ctx, call->args[0]);
  if (operands_tensor_ty.empty()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Einsum op expects at least one tensor in the input Tuple. However, the "
           "given input Tuple is empty.";
  }

  const auto* attrs = call->attrs.as<EinsumAttrs>();

  bool vdevice_unknown = false;
  VDevice vdev = VDevice();
  for (TensorType ty : operands_tensor_ty) {
    if (!vdevice_unknown) {
      if (ty->vdevice.defined()) {
        if (!vdev.defined()) {
          vdev = ty->vdevice.value();
        } else if (ty->vdevice.value()->target.defined()) {
          // mismatch
          if (ty->vdevice.value() != vdev) {
            vdevice_unknown = true;
          }
        }
      }
    }
  }

  ffi::String subscripts = attrs->subscripts;

  DataType operand_dtype = operands_tensor_ty[0]->dtype;
  std::vector<ffi::Array<PrimExpr>> input_shapes;
  input_shapes.reserve(operands_tensor_ty.size());

  for (TensorType tensor_ty : operands_tensor_ty) {
    // Check the input tuple consists of tensors with same dtype
    if (tensor_ty->dtype != operand_dtype) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "Einsum expects all input tensors to have the same dtype. However, the "
             "input contains tensors with dtype "
          << operand_dtype << " and " << tensor_ty->dtype;
    }

    // Get input shapes
    const auto* shape_expr = tensor_ty->shape.as<ShapeExprNode>();
    if (shape_expr != nullptr) {
      input_shapes.push_back(shape_expr->values);
    } else {
      if (!vdevice_unknown) {
        return TensorType(operand_dtype, tensor_ty->ndim, vdev);
      }
      return TensorType(operand_dtype, tensor_ty->ndim);
    }
  }
  // Calculate output shape using InferEinsumShape in topi
  ffi::Array<PrimExpr> oshape = topi::InferEinsumShape(subscripts, input_shapes);

  if (!vdevice_unknown) {
    return TensorType(ShapeExpr(oshape), operand_dtype, vdev);
  }
  return TensorType(ShapeExpr(oshape), operand_dtype);
}

TVM_REGISTER_OP("relax.einsum")
    .set_attrs_type<EinsumAttrs>()
    .set_num_inputs(1)
    .add_argument("operands", "Tensor", "The input tensors.")
    .set_attr<FInferType>("FInferType", InferTypeEinsum)
    .set_attr<bool>("FPurity", true);

/* relax.outer */

Expr outer(Expr x1, Expr x2) {
  static const Op& op = Op::Get("relax.outer");
  return Call(op, {std::move(x1), std::move(x2)}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.outer", outer);
}

Type InferTypeOuter(const Call& call, const BlockBuilder& ctx) {
  auto input_ty = GetInputTensorType(call, ctx);
  auto x1_ty = input_ty[0];
  auto x2_ty = input_ty[1];

  // Ensure both inputs are 1D tensors
  if (x1_ty->ndim != 1 || x2_ty->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "torch.outer requires both inputs to be 1D tensors.";
  }

  // Determine output shape
  auto x1_shape = x1_ty->shape.as<ShapeExprNode>();
  auto x2_shape = x2_ty->shape.as<ShapeExprNode>();
  if (!x1_shape || !x2_shape) {
    return TensorType(x1_ty->dtype, 2);
  }
  ffi::Array<PrimExpr> output_shape = {x1_shape->values[0], x2_shape->values[0]};
  return TensorType(ShapeExpr(output_shape), x1_ty->dtype);
}

TVM_REGISTER_OP("relax.outer")
    .set_num_inputs(2)
    .add_argument("x1", "Tensor", "The first input tensor.")
    .add_argument("x2", "Tensor", "The second input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeOuter)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<bool>("FPurity", true);

}  // namespace relax
}  // namespace tvm
