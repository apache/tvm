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
 * \file search.cc
 * \brief Searching operators.
 */

#include "search.h"

#include <algorithm>
#include <utility>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK({
  ArgmaxArgminAttrs::RegisterReflection();
  BucketizeAttrs::RegisterReflection();
});

/* relax.bucketize */
TVM_REGISTER_NODE_TYPE(BucketizeAttrs);

Expr bucketize(Expr input_tensor, Expr boundaries, bool out_int32, bool right) {
  auto attrs = make_object<BucketizeAttrs>();
  attrs->out_int32 = std::move(out_int32);
  attrs->right = std::move(right);
  static const Op& op = Op::Get("relax.bucketize");
  return Call(op, {std::move(input_tensor), std::move(boundaries)}, Attrs(attrs), {});
}

TVM_FFI_REGISTER_GLOBAL("relax.op.bucketize").set_body_typed(bucketize);

StructInfo InferStructInfoBucketize(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo input_tensor_info = input_sinfo[0];
  TensorStructInfo boundaries_info = input_sinfo[1];

  if (!boundaries_info->IsUnknownNdim() && boundaries_info->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Bucketize requires boundary to be 1-D array but got "
                     << boundaries_info->ndim);
  }

  auto attrs = call->attrs.as<BucketizeAttrs>();
  DataType out_dtype;
  out_dtype = DataType::Int(64);
  if (attrs->out_int32) {
    out_dtype = DataType::Int(32);
  }

  const auto* data_shape = input_tensor_info->shape.as<ShapeExprNode>();
  if (data_shape) {
    return TensorStructInfo(ShapeExpr(data_shape->values), out_dtype, input_tensor_info->vdevice);
  }
  return TensorStructInfo(out_dtype, input_tensor_info->ndim, input_tensor_info->vdevice);
}

TVM_REGISTER_OP("relax.bucketize")
    .set_num_inputs(2)
    .add_argument("input_tensor", "Tensor",
                  " N-D tensor or a Scalar containing the search value(s).")
    .add_argument("boundaries", "Tensor",
                  "1-D tensor, must contain a strictly increasing sequence, or the return value is "
                  "undefined.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBucketize)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.where */
Expr where(Expr condition, Expr x1, Expr x2) {
  static const Op& op = Op::Get("relax.where");
  return Call(op, {std::move(condition), std::move(x1), std::move(x2)}, Attrs(), {});
}

TVM_FFI_REGISTER_GLOBAL("relax.op.where").set_body_typed(where);

StructInfo InferStructInfoWhere(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo cond_sinfo = input_sinfo[0];
  TensorStructInfo x1_sinfo = input_sinfo[1];
  TensorStructInfo x2_sinfo = input_sinfo[2];

  VDevice vdev = VDevice();
  for (int i = 0; i < 3; ++i) {
    if (input_sinfo[i]->vdevice.defined()) {
      if (!vdev.defined()) {
        vdev = input_sinfo[i]->vdevice.value();
      } else if (input_sinfo[i]->vdevice.value()->target.defined()) {
        // mismatch
        if (input_sinfo[i]->vdevice.value() != vdev) {
          vdev = VDevice();
          break;
        }
      }
    }
  }

  if (!cond_sinfo->dtype.is_bool()) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Where requires the input condition tensor to have boolean dtype. However, "
                        "the given condition dtype is "
                     << cond_sinfo->dtype);
  }
  DataType output_dtype = InferBinaryArithOpOutDtype(call, ctx, x1_sinfo, x2_sinfo);

  int output_ndim;
  if (cond_sinfo->IsUnknownNdim() || x1_sinfo->IsUnknownNdim() || x2_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(cond_sinfo->ndim, std::max(x1_sinfo->ndim, x2_sinfo->ndim));
  }

  const auto* cond_shape = cond_sinfo->shape.as<ShapeExprNode>();
  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  if (cond_shape && x1_shape && x2_shape) {
    // Step 1. Compute the broadcasted shape of x1's and x2's
    Optional<Array<PrimExpr>> broadcasted_shape =
        InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
    if (!broadcasted_shape.defined()) {
      if (vdev.defined()) {
        return TensorStructInfo(output_dtype, output_ndim, vdev);
      }
      return TensorStructInfo(output_dtype, output_ndim);
    }
    // Step 2. Compute the broadcasted shape of cond's and the previous broadcasted shape.
    broadcasted_shape =
        InferBinaryBroadcastShape(call, ctx, cond_shape->values, broadcasted_shape.value());
    if (!broadcasted_shape.defined()) {
      if (vdev.defined()) {
        return TensorStructInfo(output_dtype, output_ndim, vdev);
      }
      return TensorStructInfo(output_dtype, output_ndim);
    }
    ICHECK_EQ(static_cast<int>(broadcasted_shape.value().size()), output_ndim);
    if (vdev.defined()) {
      return TensorStructInfo(ShapeExpr(broadcasted_shape.value()), output_dtype, vdev);
    }
    return TensorStructInfo(ShapeExpr(broadcasted_shape.value()), output_dtype);
  } else if (cond_sinfo->shape.defined() &&                 //
             x1_sinfo->shape.defined() &&                   //
             x2_sinfo->shape.defined() &&                   //
             cond_sinfo->shape.same_as(x1_sinfo->shape) &&  //
             cond_sinfo->shape.same_as(x2_sinfo->shape)) {
    if (vdev.defined()) {
      return TensorStructInfo(cond_sinfo->shape.value(), output_dtype, vdev);
    }
    return TensorStructInfo(cond_sinfo->shape.value(), output_dtype);
  } else {
    if (vdev.defined()) {
      return TensorStructInfo(output_dtype, output_ndim, vdev);
    }
    return TensorStructInfo(output_dtype, output_ndim);
  }
}

TVM_REGISTER_OP("relax.where")
    .set_num_inputs(3)
    .add_argument("condition", "Tensor", "When True, yield `x1`; otherwise, yield `x2`.")
    .add_argument("x1", "Tensor", "The first input tensor.")
    .add_argument("x2", "Tensor", "The second input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoWhere)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.argmax & relax.argmin */
TVM_REGISTER_NODE_TYPE(ArgmaxArgminAttrs);

StructInfo InferStructInfoArgmaxArgmin(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<ArgmaxArgminAttrs>();

  int axis = -1;
  if (!data_sinfo->IsUnknownNdim() && attrs->axis.has_value()) {
    axis = NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_sinfo->ndim;
  } else if (!attrs->axis.has_value()) {
    out_ndim = 0;
  } else if (data_sinfo->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_sinfo->ndim - 1;
    ICHECK_GE(out_ndim, 0);
  }

  DataType out_dtype = DataType::Int(64);
  // The inference rule for reduction operator output shapes:
  // - axes is None, keepdims is false -> return the zero-rank shape;
  // - axes is None, keepdims is true -> return the shape whose ndim is the same as input and every
  // value is 1.
  // - axes is not None, keepdims is false -> the returned shape does not contain the input axes.
  // - axes is not None, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axes
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    if (!attrs->axis.has_value() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorStructInfo(ShapeExpr(Array<PrimExpr>(out_ndim, IntImm(out_dtype, /*value=*/1))),
                              out_dtype, data_sinfo->vdevice);
    } else {
      return out_ndim == 0
                 ? TensorStructInfo(ShapeExpr(Array<PrimExpr>()), out_dtype, data_sinfo->vdevice)
                 : TensorStructInfo(out_dtype, out_ndim, data_sinfo->vdevice);
    }
  }

  if (data_sinfo->ndim > 0) {
    out_dtype = data_shape->values[0]->dtype;
  }

  Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    if (attrs->axis.has_value() && i != axis) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm(out_dtype, /*value=*/1));
    }
  }
  ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype, data_sinfo->vdevice);
}

#define RELAX_REGISTER_ARGMAX_ARGMIN_OP(OpName)                                    \
  Expr OpName(Expr x, Optional<int64_t> axis, bool keepdims) {                     \
    ObjectPtr<ArgmaxArgminAttrs> attrs = make_object<ArgmaxArgminAttrs>();         \
    attrs->axis = std::move(axis);                                                 \
    attrs->keepdims = std::move(keepdims);                                         \
    static const Op& op = Op::Get("relax." #OpName);                               \
    return Call(op, {std::move(x)}, Attrs(attrs));                                 \
  }                                                                                \
  TVM_FFI_REGISTER_GLOBAL("relax.op." #OpName).set_body_typed(OpName);             \
  TVM_REGISTER_OP("relax." #OpName)                                                \
      .set_num_inputs(1)                                                           \
      .add_argument("x", "Tensor", "The input data tensor")                        \
      .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoArgmaxArgmin) \
      .set_attr<Bool>("FPurity", Bool(true));

RELAX_REGISTER_ARGMAX_ARGMIN_OP(argmax);
RELAX_REGISTER_ARGMAX_ARGMIN_OP(argmin);

}  // namespace relax
}  // namespace tvm
