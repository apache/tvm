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

#include <tvm/topi/einsum.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.matmul */
TVM_REGISTER_NODE_TYPE(MatmulAttrs);

Expr matmul(Expr x1, Expr x2, DataType out_dtype) {
  ObjectPtr<MatmulAttrs> attrs = make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.matmul");
  return Call(op, {std::move(x1), std::move(x2)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.matmul").set_body_typed(matmul);

StructInfo InferStructInfoMatmul(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  Expr lhs = call->args[0];
  Expr rhs = call->args[1];
  TensorStructInfo x1_sinfo = input_sinfo[0];
  TensorStructInfo x2_sinfo = input_sinfo[1];

  VDevice vdev = VDevice();
  if (x1_sinfo->vdevice.defined() && x2_sinfo->vdevice.defined()) {
    if (x1_sinfo->vdevice.value() == x2_sinfo->vdevice.value()) {
      vdev = x1_sinfo->vdevice.value();
    }
  } else if (x1_sinfo->vdevice.defined()) {
    vdev = x1_sinfo->vdevice.value();
  } else if (x2_sinfo->vdevice.defined()) {
    vdev = x2_sinfo->vdevice.value();
  }

  const auto* attrs = call->attrs.as<MatmulAttrs>();
  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, x1_sinfo, x2_sinfo)
                           : attrs->out_dtype;

  if (x1_sinfo->IsUnknownNdim() || x2_sinfo->IsUnknownNdim()) {
    if (vdev.defined()) {
      return TensorStructInfo(out_dtype, kUnknownNDim, vdev);
    }
    return TensorStructInfo(out_dtype, kUnknownNDim);
  }
  int x1_ndim = x1_sinfo->ndim;
  int x2_ndim = x2_sinfo->ndim;
  if (x1_ndim == 0) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Matmul operands must not be scalar.  "
                     << "However, the expression " << call << " has a LHS of " << lhs
                     << " with struct info " << x1_sinfo
                     << ", which is scalar (zero-dimensional) tensor.");
  }
  if (x2_ndim == 0) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Matmul operands must not be scalar.  "
                     << "However, the expression " << call << " has a RHS of " << rhs
                     << " with struct info " << x2_sinfo
                     << ", which is scalar (zero-dimensional) tensor.");
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

  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  if (x1_shape == nullptr || x2_shape == nullptr) {
    if (vdev.defined()) {
      return TensorStructInfo(out_dtype, output_ndim, vdev);
    }
    return TensorStructInfo(out_dtype, output_ndim);
  }

  Array<PrimExpr> x1_shape_prefix{x1_shape->values.begin(),
                                  x1_shape->values.end() - 2 + x1_prepended};
  Array<PrimExpr> x2_shape_prefix{x2_shape->values.begin(),
                                  x2_shape->values.end() - 2 + x2_appended};
  Optional<Array<PrimExpr>> output_shape_prefix =
      InferBinaryBroadcastShape(call, ctx, x1_shape_prefix, x2_shape_prefix);
  if (!output_shape_prefix.defined()) {
    if (vdev.defined()) {
      return TensorStructInfo(out_dtype, output_ndim, vdev);
    }
    return TensorStructInfo(out_dtype, output_ndim);
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr x1_reduction_length = x1_shape->values[x1_sinfo->ndim - 1];
  PrimExpr x2_reduction_length = x2_shape->values[x2_ndim - 2];
  if (analyzer->CanProve(x1_reduction_length != x2_reduction_length)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Matmul requires the reduction length of the operands to be equal.  "
                     << "However, the LHS " << lhs << " has shape " << x1_sinfo->shape
                     << ", while the RHS " << rhs << " has shape " << x2_sinfo->shape
                     << ".  The reduction dimensions of " << x1_reduction_length << " and "
                     << x2_reduction_length << " are not equal.");
  }

  Array<PrimExpr> output_shape = output_shape_prefix.value();
  if (!x1_prepended) {
    output_shape.push_back(x1_shape->values[x1_ndim - 2]);
  }
  if (!x2_appended) {
    output_shape.push_back(x2_shape->values[x2_ndim - 1]);
  }
  ICHECK_EQ(static_cast<int>(output_shape.size()), output_ndim);
  if (vdev.defined()) {
    return TensorStructInfo(ShapeExpr(output_shape), out_dtype, vdev);
  }
  return TensorStructInfo(ShapeExpr(output_shape), out_dtype);
}

Call InferMixedPrecisionMatmul(const Call& call, const DataType& out_dtype) {
  return Downcast<Call>(matmul(call->args[0], call->args[1], out_dtype));
}

TVM_REGISTER_OP("relax.matmul")
    .set_num_inputs(2)
    .add_argument("x1", "Tensor", "The first input tensor.")
    .add_argument("x2", "Tensor", "The second input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMatmul)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionMatmul)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.einsum */
TVM_REGISTER_NODE_TYPE(EinsumAttrs);

Expr einsum(Expr operands, String subscripts) {
  ObjectPtr<EinsumAttrs> attrs = make_object<EinsumAttrs>();
  attrs->subscripts = std::move(subscripts);

  static const Op& op = Op::Get("relax.einsum");
  return Call(op, {std::move(operands)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.einsum").set_body_typed(einsum);

StructInfo InferStructInfoEinsum(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Einsum op should take 1 argument");
  }
  Array<TensorStructInfo> operands_tensor_sinfo =
      GetTensorStructInfoFromTuple(call, ctx, call->args[0]);
  if (operands_tensor_sinfo.empty()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Einsum op expects at least one tensor in the input Tuple. However, the "
                        "given input Tuple is empty.");
  }

  const auto* attrs = call->attrs.as<EinsumAttrs>();

  bool vdevice_unknown = false;
  VDevice vdev = VDevice();
  for (TensorStructInfo sinfo : operands_tensor_sinfo) {
    if (!vdevice_unknown) {
      if (sinfo->vdevice.defined()) {
        if (!vdev.defined()) {
          vdev = sinfo->vdevice.value();
        } else if (sinfo->vdevice.value()->target.defined()) {
          // mismatch
          if (sinfo->vdevice.value() != vdev) {
            vdevice_unknown = true;
          }
        }
      }
    }
  }

  String subscripts = attrs->subscripts;

  DataType operand_dtype = operands_tensor_sinfo[0]->dtype;
  std::vector<Array<PrimExpr>> input_shapes;
  input_shapes.reserve(operands_tensor_sinfo.size());

  for (TensorStructInfo tensor_sinfo : operands_tensor_sinfo) {
    // Check the input tuple consists of tensors with same dtype
    if (tensor_sinfo->dtype != operand_dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Einsum expects all input tensors to have the same dtype. However, the "
                          "input contains tensors with dtype "
                       << operand_dtype << " and " << tensor_sinfo->dtype);
    }

    // Get input shapes
    const auto* shape_expr = tensor_sinfo->shape.as<ShapeExprNode>();
    if (shape_expr != nullptr) {
      input_shapes.push_back(shape_expr->values);
    } else {
      if (!vdevice_unknown) {
        return TensorStructInfo(operand_dtype, tensor_sinfo->ndim, vdev);
      }
      return TensorStructInfo(operand_dtype, tensor_sinfo->ndim);
    }
  }
  // Calculate output shape using InferEinsumShape in topi
  Array<PrimExpr> oshape = topi::InferEinsumShape(subscripts, input_shapes);

  if (!vdevice_unknown) {
    return TensorStructInfo(ShapeExpr(oshape), operand_dtype, vdev);
  }
  return TensorStructInfo(ShapeExpr(oshape), operand_dtype);
}

TVM_REGISTER_OP("relax.einsum")
    .set_attrs_type<EinsumAttrs>()
    .set_num_inputs(1)
    .add_argument("operands", "Tensor", "The input tensors.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoEinsum)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
