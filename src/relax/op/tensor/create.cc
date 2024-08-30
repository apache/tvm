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
StructInfo InferStructInfoFull(const Call& call) {
  if (call->args.size() != 2) {
    LOG(FATAL) << "Full op should have 2 arguments";
  }
  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[0]);
  const auto* fill_value_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  if (shape_sinfo == nullptr) {
    LOG(FATAL) << "Full requires the input shape to be a Shape. However, the given one is "
               << call->args[0]->struct_info_->GetTypeKey();
  }
  if (fill_value_sinfo == nullptr || fill_value_sinfo->ndim != 0) {
    LOG(FATAL)
        << "Full requires the input fill value to be zero rank Tensor. However, the given one is "
        << call->args[1]->struct_info_;
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  DataType out_dtype = attrs->dtype.is_void() ? fill_value_sinfo->dtype : attrs->dtype;
  return TensorStructInfo(/*shape=*/call->args[0], out_dtype, fill_value_sinfo->vdevice);
}

Expr full(Variant<Expr, Array<PrimExpr>> shape, Expr fill_value, DataType dtype) {
  Expr shape_in_expr = [&]() -> Expr {
    if (const auto* expr = shape.as<ExprNode>()) {
      return GetRef<Expr>(expr);
    } else if (const auto* _array = shape.as<ArrayNode>()) {
      return ShapeExpr(GetRef<Array<PrimExpr>>(_array));
    } else {
      LOG(FATAL)
          << "Full only expects the input shape to be either an Expr or an Array of PrimExpr. "
             "However, the given one is "
          << shape->GetTypeKey();
    }
  }();

  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.full");
  auto call = Call(op, {shape_in_expr, fill_value}, Attrs(attrs), {});

  if (fill_value->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoFull(call));
  }

  return call;
}

TVM_REGISTER_GLOBAL("relax.op.full").set_body_typed(full);

TVM_REGISTER_OP("relax.full")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .add_argument("fill_value", "Tensor", "The scalar tensor, denoting the value to fill.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoFull(call);
                                })
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.full_like */
StructInfo InferStructInfoFullLike(const Call& call) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo fill_value_sinfo = input_sinfo[1];
  if (fill_value_sinfo->ndim != 0) {
    LOG(FATAL) << "FullLike requires the input fill value to be zero "
                  "rank Tensor. However, the given one has ndim"
               << fill_value_sinfo->ndim;
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

Expr full_like(Expr x, Expr fill_value, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.full_like");
  auto call = Call(op, {x, fill_value}, Attrs(attrs), {});

  if (x->struct_info_.defined() && fill_value->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoFullLike(call));
  }

  return call;
}

TVM_REGISTER_GLOBAL("relax.op.full_like").set_body_typed(full_like);

TVM_REGISTER_OP("relax.full_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("fill_value", "Tensor", "The scalar value to fill.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoFullLike(call);
                                })
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

// Structure info inference for ones and zeros
StructInfo InferStructInfoOnesZeros(const Call& call) {
  CheckNumArguments(call);

  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[0]);
  if (shape_sinfo == nullptr) {
    LOG(FATAL) << "Operator " << call->op << " requires the input shape to be a Shape.  "
               << "However, the argument " << call->args[0] << " is of type "
               << call->args[0]->struct_info_;
  }
  const auto* attrs = call->attrs.as<InitAttrs>();
  return TensorStructInfo(/*shape=*/call->args[0], attrs->dtype);
}

// Structure info inference for ones_like and zeros_like
StructInfo InferStructInfoOnesLikeZerosLike(const Call& call) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call);
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
  Call call(op, {shape}, Attrs(attrs), {});

  if (shape->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoOnesZeros(call));
  }

  return call;
}

Expr ones_like(Expr x, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.ones_like");
  Call call(op, {x}, Attrs(attrs), {});

  if (x->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoOnesLikeZerosLike(call));
  }

  return call;
}

TVM_REGISTER_GLOBAL("relax.op.ones").set_body_typed(ones);
TVM_REGISTER_GLOBAL("relax.op.ones_like").set_body_typed(ones_like);

TVM_REGISTER_OP("relax.ones")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoOnesZeros(call);
                                })
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.ones_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoOnesLikeZerosLike(call);
                                })
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.zeros & relax.zeros_like */
Expr zeros(Expr shape, DataType dtype) {
  CHECK(!dtype.is_void()) << "Zeros op expects the input dtype not to be void";
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.zeros");
  Call call(op, {shape}, Attrs(attrs), {});

  if (shape->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoOnesZeros(call));
  }

  return call;
}

Expr zeros_like(Expr x, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.zeros_like");
  Call call(op, {x}, Attrs(attrs), {});

  if (x->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoOnesLikeZerosLike(call));
  }

  return call;
}

TVM_REGISTER_GLOBAL("relax.op.zeros").set_body_typed(zeros);
TVM_REGISTER_GLOBAL("relax.op.zeros_like").set_body_typed(zeros_like);

TVM_REGISTER_OP("relax.zeros")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoOnesZeros(call);
                                })
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.zeros_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoOnesLikeZerosLike(call);
                                })
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.arange */
StructInfo InferStructInfoArange(const Call& call) {
  if (call->args.size() != 3) {
    LOG(FATAL) << "Operator " << call->op
               << " expects 3 arguments, which are `start`, `end` and `step`, "
               << "but received " << call->args.size() << " arguments";
  }
  // TODO(Siyuan): Support indirect prim_values
  auto get_prim_value = [&](const Expr& expr, std::string key) {
    if (!expr->IsInstance<PrimValueNode>()) {
      LOG(FATAL) << "Operator" << call->op << " expects the `" << key
                 << "` parameter to be a PrimValue, "
                 << "but argument " << expr << " was of type " << expr->GetTypeKey();
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

Expr arange(PrimValue start, PrimValue stop, PrimValue step, DataType dtype) {
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.arange");
  Call call(op, {std::move(start), std::move(stop), std::move(step)}, Attrs(attrs), {});

  UpdateStructInfo(call, InferStructInfoArange(call));

  return call;
}

TVM_REGISTER_GLOBAL("relax.op.arange").set_body_typed(arange);

TVM_REGISTER_OP("relax.arange")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(3)
    .add_argument("start", "PrimValue", "The starting value for the set of points.")
    .add_argument("end", "PrimValue", "The ending value for the set of points.")
    .add_argument("step", "PrimValue", "The gap between each pair of adjacent points.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoArange(call);
                                })
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.tril & relax.triu */
TVM_REGISTER_NODE_TYPE(TriluAttrs);

StructInfo InferStructInfoTrilTriu(const Call& call) {
  auto [data_sinfo, offset] = GetArgStructInfo<TensorStructInfo, PrimStructInfo>(call);

  if (!data_sinfo->IsUnknownNdim() && data_sinfo->ndim < 2) {
    LOG(FATAL) << "Operator " << call->op
               << " expects an input tensor with at least two dimensions.  "
               << "However, the argument " << call->args[0] << " has type " << data_sinfo
               << " with " << data_sinfo->ndim << " dimension(s).";
  }
  return data_sinfo;
}

Expr tril(Expr x, Expr k) {
  static const Op& op = Op::Get("relax.tril");
  Call call(op, {x, k});

  if (x->struct_info_.defined() && k->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoTrilTriu(call));
  }

  return call;
}

Expr tril(Expr x, int k) { return tril(x, relax::PrimValue::Int64(k)); }

Expr triu(Expr x, Expr k) {
  static const Op& op = Op::Get("relax.triu");
  Call call(op, {x, k});

  if (x->struct_info_.defined() && k->struct_info_.defined()) {
    UpdateStructInfo(call, InferStructInfoTrilTriu(call));
  }

  return call;
}

Expr triu(Expr x, int k) { return triu(x, relax::PrimValue::Int64(k)); }

TVM_REGISTER_GLOBAL("relax.op.tril").set_body_typed(static_cast<Expr (*)(Expr, Expr)>(tril));
TVM_REGISTER_GLOBAL("relax.op.triu").set_body_typed(static_cast<Expr (*)(Expr, Expr)>(triu));

TVM_REGISTER_OP("relax.tril")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("k", "PrimValue", "The offset of the diagonal.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoTrilTriu(call);
                                })
    .set_attr<Bool>("FPurity", Bool(true));

TVM_REGISTER_OP("relax.triu")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("k", "PrimValue", "The offset of the diagonal.")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder&) -> StructInfo {
                                  return InferStructInfoTrilTriu(call);
                                })
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
