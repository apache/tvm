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
 * \file create.cc
 * \brief Creation operators.
 */

#include "create.h"

#include <tvm/arith/analyzer.h>

#include <string>
#include <utility>

namespace tvm {
namespace relax {

/* Initialization operators */
TVM_REGISTER_NODE_TYPE(InitAttrs);

/* relax.full */
Expr full(ObjectRef shape, Expr fill_value, DataType dtype) {
  Expr shape_in_expr{nullptr};
  if (const auto* expr = shape.as<ExprNode>()) {
    shape_in_expr = GetRef<Expr>(expr);
  } else if (const auto* _array = shape.as<ArrayNode>()) {
    shape_in_expr = ShapeExpr(GetRef<Array<PrimExpr>>(_array));
  } else {
    LOG(FATAL) << "Full only expects the input shape to be either an Expr or an Array of PrimExpr. "
                  "However, the given one is "
               << shape->GetTypeKey();
  }

  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.full");
  return Call(op, {std::move(shape_in_expr), std::move(fill_value)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.full").set_body_typed(full);

StructInfo InferStructInfoFull(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Full op should have 2 arguments");
  }
  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[0]);
  const auto* fill_value_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  if (shape_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Full requires the input shape to be a Shape. However, the given one is "
                     << call->args[0]->struct_info_->GetTypeKey());
  }
  if (fill_value_sinfo == nullptr || fill_value_sinfo->ndim != 0) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Full requires the input fill value to be zero rank Tensor. However, the given one is "
        << call->args[1]->struct_info_);
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  DataType out_dtype = attrs->dtype.is_void() ? fill_value_sinfo->dtype : attrs->dtype;
  return TensorStructInfo(/*shape=*/call->args[0], out_dtype, fill_value_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.full")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .add_argument("fill_value", "Tensor", "The scalar tensor, denoting the value to fill.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFull)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.full_like */
Expr full_like(Expr x, Expr fill_value, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.full_like");
  return Call(op, {std::move(x), std::move(fill_value)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.full_like").set_body_typed(full_like);

StructInfo InferStructInfoFullLike(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo fill_value_sinfo = input_sinfo[1];
  if (fill_value_sinfo->ndim != 0) {
    ctx->ReportFatal(Diagnostic::Error(call) << "FullLike requires the input fill value to be zero "
                                                "rank Tensor. However, the given one has ndim"
                                             << fill_value_sinfo->ndim);
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  if (attrs->dtype.is_void()) {
    return data_sinfo;
  } else {
    auto output_sinfo = make_object<TensorStructInfoNode>(*data_sinfo.get());
    output_sinfo->dtype = attrs->dtype;
    return TensorStructInfo(output_sinfo);
  }
}

TVM_REGISTER_OP("relax.full_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("fill_value", "Tensor", "The scalar value to fill.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFullLike)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

// Structure info inference for ones and zeros
StructInfo InferStructInfoOnesZeros(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Ones/Zeros should have 1 argument");
  }

  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[0]);
  if (shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Ones/Zeros requires the input shape to be a Shape. However, the given one is "
        << call->args[0]->struct_info_->GetTypeKey());
  }
  const auto* attrs = call->attrs.as<InitAttrs>();
  return TensorStructInfo(/*shape=*/call->args[0], attrs->dtype);
}

// Structure info inference for ones_like and zeros_like
StructInfo InferStructInfoOnesLikeZerosLike(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<InitAttrs>();
  if (attrs->dtype.is_void()) {
    return data_sinfo;
  } else {
    auto output_sinfo = make_object<TensorStructInfoNode>(*data_sinfo.get());
    output_sinfo->dtype = attrs->dtype;
    return TensorStructInfo(output_sinfo);
  }
}

/* relax.ones & relax.ones_like */
Expr ones(Expr shape, DataType dtype) {
  CHECK(!dtype.is_void()) << "Ones op expects the input dtype not to be void";
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.ones");
  return Call(op, {std::move(shape)}, Attrs(attrs), {});
}

Expr ones_like(Expr x, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.ones_like");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.ones").set_body_typed(ones);
TVM_REGISTER_GLOBAL("relax.op.ones_like").set_body_typed(ones_like);

TVM_REGISTER_OP("relax.ones")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoOnesZeros)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.ones_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoOnesLikeZerosLike)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.zeros & relax.zeros_like */
Expr zeros(Expr shape, DataType dtype) {
  CHECK(!dtype.is_void()) << "Zeros op expects the input dtype not to be void";
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.zeros");
  return Call(op, {std::move(shape)}, Attrs(attrs), {});
}

Expr zeros_like(Expr x, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.zeros_like");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.zeros").set_body_typed(zeros);
TVM_REGISTER_GLOBAL("relax.op.zeros_like").set_body_typed(zeros_like);

TVM_REGISTER_OP("relax.zeros")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoOnesZeros)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.zeros_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoOnesLikeZerosLike)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.arange */
Expr arange(PrimValue start, PrimValue stop, PrimValue step, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.arange");
  return Call(op, {std::move(start), std::move(stop), std::move(step)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.arange").set_body_typed(arange);

StructInfo InferStructInfoArange(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Arange should have 3 arguments, which are `start`, `end` and `step`, but got "
        << call->args.size() << " arguments");
  }
  // TODO(Siyuan): Support indirect prim_values
  auto get_prim_value = [&ctx](const Expr& expr, std::string key) {
    if (!expr->IsInstance<PrimValueNode>()) {
      ctx->ReportFatal(Diagnostic::Error(expr)
                       << "Arange expects the `" << key << "` to be a PrimValue, but got "
                       << expr->GetTypeKey());
    }
    return expr.as<PrimValueNode>()->value;
  };
  PrimExpr start = get_prim_value(call->args[0], "start");
  PrimExpr end = get_prim_value(call->args[1], "end");
  PrimExpr step = get_prim_value(call->args[2], "step");
  DataType dtype = call->attrs.as<InitAttrs>()->dtype;
  PrimExpr num_elem;
  if (start.dtype().is_int() && end.dtype().is_int() && step.dtype().is_int()) {
    num_elem = tvm::floordiv((end - start + step - 1), step);
  } else {
    num_elem = tvm::cast(tvm::DataType::Int(64),
                         tvm::ceil(tvm::cast(tvm::DataType::Float(32), end - start) / step));
  }
  arith::Analyzer analyzer;
  num_elem = analyzer.Simplify(num_elem);
  return TensorStructInfo(ShapeExpr({num_elem}), dtype);
}

TVM_REGISTER_OP("relax.arange")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(3)
    .add_argument("start", "PrimValue", "The starting value for the set of points.")
    .add_argument("end", "PrimValue", "The ending value for the set of points.")
    .add_argument("step", "PrimValue", "The gap between each pair of adjacent points.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoArange)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.tril & relax.triu */
TVM_REGISTER_NODE_TYPE(TriluAttrs);

Expr tril(Expr x, Expr k) {
  static const Op& op = Op::Get("relax.tril");
  return Call(op, {x, k});
}

Expr tril(Expr x, int k) { return tril(x, relax::PrimValue::Int64(k)); }

Expr triu(Expr x, Expr k) {
  static const Op& op = Op::Get("relax.triu");
  return Call(op, {x, k});
}

Expr triu(Expr x, int k) { return triu(x, relax::PrimValue::Int64(k)); }

TVM_REGISTER_GLOBAL("relax.op.tril").set_body_typed(static_cast<Expr (*)(Expr, Expr)>(tril));
TVM_REGISTER_GLOBAL("relax.op.triu").set_body_typed(static_cast<Expr (*)(Expr, Expr)>(triu));

StructInfo InferStructInfoTrilTriu(const Call& call, const BlockBuilder& ctx) {
  auto [data_sinfo, offset] = GetArgStructInfo<TensorStructInfo, PrimStructInfo>(call, ctx);

  if (!data_sinfo->IsUnknownNdim() && data_sinfo->ndim < 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << call->op
                                             << " requires the input tensor to have at least two "
                                                "dimensions. However, the given input has "
                                             << data_sinfo->ndim << " dimension(s).");
  }
  return data_sinfo;
}

TVM_REGISTER_OP("relax.tril")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("k", "PrimValue", "The offset of the diagonal.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTrilTriu)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.triu")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("k", "PrimValue", "The offset of the diagonal.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTrilTriu)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
