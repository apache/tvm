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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <utility>

#include "tvm/relax/expr.h"

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  InitAttrs::RegisterReflection();
  TriluAttrs::RegisterReflection();
}

/* Initialization operators */

/* relax.full */
Expr full(ffi::Variant<Expr, ffi::Array<PrimExpr>> shape, Expr fill_value,
          ffi::Optional<DLDataType> dtype) {
  Expr shape_in_expr{nullptr};
  if (const auto* expr = shape.as<ExprNode>()) {
    shape_in_expr = ffi::GetRef<Expr>(expr);
  } else if (const auto* _array = shape.as<ffi::ArrayObj>()) {
    shape_in_expr =
        ShapeExpr(ffi::GetRef<ffi::ObjectRef>(_array).as_or_throw<ffi::Array<PrimExpr>>());
  } else {
    TVM_FFI_THROW(InternalError)
        << "Full only expects the input shape to be either an Expr or an Array of PrimExpr. ";
  }

  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.full");
  return Call(op, {std::move(shape_in_expr), std::move(fill_value)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.full", full);
}

Type InferTypeFull(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Full op should have 2 arguments";
  }
  const auto* shape_ty = GetTypeAs<ShapeTypeNode>(call->args[0]);
  const auto* fill_value_ty = GetTypeAs<TensorTypeNode>(call->args[1]);
  if (shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Full requires the input shape to be a Shape. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (fill_value_ty == nullptr || fill_value_ty->ndim != 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Full requires the input fill value to be zero rank Tensor. However, the given one is "
        << call->args[1]->ty;
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  ffi::Optional<PrimType> out_dtype = attrs->dtype.has_value()
                                          ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                                          : fill_value_ty->dtype;
  return TensorType(/*shape=*/call->args[0], out_dtype, fill_value_ty->vdevice);
}

TVM_REGISTER_OP("relax.full")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .add_argument("fill_value", "Tensor", "The scalar tensor, denoting the value to fill.")
    .set_attr<FInferType>("FInferType", InferTypeFull)
    .set_attr<bool>("RequiresArgumentShapes", false)
    .set_attr<bool>("FDataDependent", true)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

/* relax.full_like */
Expr full_like(Expr x, Expr fill_value, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.full_like");
  return Call(op, {std::move(x), std::move(fill_value)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.full_like", full_like);
}

Type InferTypeFullLike(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType data_ty = input_ty[0];
  TensorType fill_value_ty = input_ty[1];
  if (fill_value_ty->ndim != 0) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "FullLike requires the input fill value to be zero "
                                             "rank Tensor. However, the given one has ndim"
                                          << fill_value_ty->ndim;
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  if (!attrs->dtype.has_value()) {
    return data_ty;
  } else {
    auto output_ty = ffi::make_object<TensorTypeNode>(*data_ty.get());
    output_ty->dtype = PrimType(attrs->dtype.value());
    return TensorType(output_ty);
  }
}

TVM_REGISTER_OP("relax.full_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("fill_value", "Tensor", "The scalar value to fill.")
    .set_attr<FInferType>("FInferType", InferTypeFullLike)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

// Structure info inference for ones and zeros
Type InferTypeOnesZeros(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Ones/Zeros should have 1 argument";
  }

  const auto* shape_ty = GetTypeAs<ShapeTypeNode>(call->args[0]);
  if (shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Ones/Zeros requires the input shape to be a Shape. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  return TensorType(/*shape=*/call->args[0], PrimType(attrs->dtype.value()));
}

// Structure info inference for ones_like and zeros_like
Type InferTypeOnesLikeZerosLike(const Call& call, const BlockBuilder& ctx) {
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  const auto* attrs = call->attrs.as<InitAttrs>();
  if (!attrs->dtype.has_value()) {
    return data_ty;
  } else {
    auto output_ty = ffi::make_object<TensorTypeNode>(*data_ty.get());
    output_ty->dtype = PrimType(attrs->dtype.value());
    return TensorType(output_ty);
  }
}

/* relax.ones & relax.ones_like */
Expr ones(Expr shape, DLDataType dtype) {
  TVM_FFI_ICHECK((dtype != DLDataType{kDLOpaqueHandle, 0, 0}))
      << "Ones op expects the input dtype not to be void";
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.ones");
  return Call(op, {std::move(shape)}, Attrs(attrs), {});
}

Expr ones_like(Expr x, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.ones_like");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.ones", ones).def("relax.op.ones_like", ones_like);
}

TVM_REGISTER_OP("relax.ones")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .set_attr<FInferType>("FInferType", InferTypeOnesZeros)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

TVM_REGISTER_OP("relax.ones_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeOnesLikeZerosLike)
    .set_attr<bool>("FPurity", true);

/* relax.zeros & relax.zeros_like */
Expr zeros(Expr shape, DLDataType dtype) {
  TVM_FFI_ICHECK((dtype != DLDataType{kDLOpaqueHandle, 0, 0}))
      << "Zeros op expects the input dtype not to be void";
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.zeros");
  return Call(op, {std::move(shape)}, Attrs(attrs), {});
}

Expr zeros_like(Expr x, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.zeros_like");
  return Call(op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.zeros", zeros).def("relax.op.zeros_like", zeros_like);
}

TVM_REGISTER_OP("relax.zeros")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .set_attr<FInferType>("FInferType", InferTypeOnesZeros)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

TVM_REGISTER_OP("relax.zeros_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", InferTypeOnesLikeZerosLike)
    .set_attr<bool>("FPurity", true);

/* relax.eye & relax.eye_like */
Expr eye(PrimExpr n, PrimExpr m, PrimExpr k, DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.eye");
  return Call(op, {std::move(n), std::move(m), std::move(k)}, Attrs(attrs), {});
}

Expr eye_like(Expr x, PrimExpr k, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.eye_like");
  return Call(op, {std::move(x), std::move(k)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.eye", eye).def("relax.op.eye_like", eye_like);
}

Type InferTypeEye(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Eye op should have 3 arguments: n, m, and k, but got "
                                          << call->args.size() << " arguments";
  }

  auto get_prim_value = [&ctx](const Expr& expr, std::string key) {
    if (!expr->IsInstance<PrimExprNode>()) {
      TVM_FFI_VISIT_THROW(TypeError, expr)
          << "Eye expects the `" << key << "` to be a PrimExpr, but got " << expr->GetTypeKey();
    }
    return expr.as_or_throw<PrimExpr>();
  };

  PrimExpr n = get_prim_value(call->args[0], "n");
  PrimExpr m = get_prim_value(call->args[1], "m");

  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  DLDataType dtype = attrs->dtype.value();
  return TensorType(ShapeExpr({n, m}), PrimType(dtype));
}

Type InferTypeEyeLike(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Eye_like op should have 2 arguments: x and k, but got " << call->args.size()
        << " arguments";
  }

  const auto* x_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  if (x_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Eye_like expects the input `x` to be a Tensor, but got "
        << call->args[0]->ty->GetTypeKey();
  }
  if (x_ty->ndim != 2 && x_ty->ndim != kUnknownNDim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Eye_like expects the input tensor to be 2-dimensional, but got " << x_ty->ndim
        << " dimensions";
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  ffi::Optional<PrimType> out_dtype = attrs->dtype.has_value()
                                          ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                                          : x_ty->dtype;

  return TensorType(x_ty->shape.value(), out_dtype, x_ty->vdevice);
}

TVM_REGISTER_OP("relax.eye")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(3)
    .add_argument("n", "PrimExpr", "Number of rows in the output.")
    .add_argument("m", "PrimExpr", "Number of columns in the output.")
    .add_argument("k", "PrimExpr", "Index of the diagonal.")
    .set_attr<FInferType>("FInferType", InferTypeEye)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

TVM_REGISTER_OP("relax.eye_like")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("k", "PrimExpr", "Index of the diagonal.")
    .set_attr<FInferType>("FInferType", InferTypeEyeLike)
    .set_attr<bool>("FPurity", true);

/* relax.arange */
Expr arange(PrimExpr start, PrimExpr stop, PrimExpr step, DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.arange");
  return Call(op, {std::move(start), std::move(stop), std::move(step)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.arange", arange);
}

Type InferTypeArange(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Arange should have 3 arguments, which are `start`, `end` and `step`, but got "
        << call->args.size() << " arguments";
  }
  // TODO(Siyuan): Support indirect prim_values
  auto get_prim_value = [&ctx](const Expr& expr, std::string key) {
    if (!expr->IsInstance<PrimExprNode>()) {
      TVM_FFI_VISIT_THROW(TypeError, expr)
          << "Arange expects the `" << key << "` to be a PrimExpr, but got " << expr->GetTypeKey();
    }
    return expr.as_or_throw<PrimExpr>();
  };
  PrimExpr start = get_prim_value(call->args[0], "start");
  PrimExpr end = get_prim_value(call->args[1], "end");
  PrimExpr step = get_prim_value(call->args[2], "step");
  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  DLDataType dtype = attrs->dtype.value();
  PrimExpr num_elem;
  if (start.ty().code() == DLDataTypeCode::kDLInt && end.ty().code() == DLDataTypeCode::kDLInt &&
      step.ty().code() == DLDataTypeCode::kDLInt) {
    num_elem = tvm::floordiv((end - start + step - 1), step);
  } else {
    num_elem = tvm::cast(tvm::PrimType::Int(64),
                         tvm::ceil(tvm::cast(tvm::PrimType::Float(32), end - start) / step));
  }
  arith::Analyzer analyzer;
  num_elem = analyzer->Simplify(num_elem);
  return TensorType(ShapeExpr({num_elem}), PrimType(dtype));
}

TVM_REGISTER_OP("relax.arange")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(3)
    .add_argument("start", "PrimExpr", "The starting value for the set of points.")
    .add_argument("end", "PrimExpr", "The ending value for the set of points.")
    .add_argument("step", "PrimExpr", "The gap between each pair of adjacent points.")
    .set_attr<FInferType>("FInferType", InferTypeArange)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

/* relax.hamming_window */
Expr hamming_window(PrimExpr window_size, PrimExpr periodic, PrimExpr alpha, PrimExpr beta,
                    DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("relax.hamming_window");
  return Call(op, {std::move(window_size), std::move(periodic), std::move(alpha), std::move(beta)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.hamming_window", hamming_window);
}

Type InferTypeHammingWindow(const Call& call, const BlockBuilder& ctx) {
  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  DLDataType dtype = attrs->dtype.value();
  if (dtype.code == DLDataTypeCode::kDLInt || dtype.code == DLDataTypeCode::kDLUInt) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Hamming Window expects the datatype to be float but got " << dtype;
  }
  auto get_prim_value = [&ctx](const Expr& expr, std::string key) {
    if (!expr->IsInstance<PrimExprNode>()) {
      TVM_FFI_VISIT_THROW(TypeError, expr) << "Hamming_window expects the `" << key
                                           << "` to be a PrimExpr, but got " << expr->GetTypeKey();
    }
    return expr.as_or_throw<PrimExpr>();
  };
  PrimExpr window_size = get_prim_value(call->args[0], "window_size");

  arith::Analyzer analyzer;
  if (analyzer->CanProveLess(window_size, 1)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Hamming_window expects the window_size must be greater than zero but got "
        << window_size;
  }
  window_size = analyzer->Simplify(window_size);
  return TensorType(ShapeExpr({window_size}), PrimType(dtype));
}

TVM_REGISTER_OP("relax.hamming_window")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(4)
    .add_argument("window_size", "PrimExpr", "The size of the window")
    .add_argument("periodic", "PrimExpr",
                  "If True, returns a window to be used as periodic function. If False, return a "
                  "symmetric window")
    .add_argument("alpha", "PrimExpr", "The coefficient alpha")
    .add_argument("beta", "PrimExpr", "The coefficient beta")
    .set_attr<FInferType>("FInferType", InferTypeHammingWindow)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<bool>("FPurity", true);

/* relax.tril & relax.triu */

Expr tril(Expr x, Expr k) {
  static const Op& op = Op::Get("relax.tril");
  return Call(op, {x, k});
}

Expr tril(Expr x, int k) { return tril(x, IntImm::Int64(k)); }

Expr triu(Expr x, Expr k) {
  static const Op& op = Op::Get("relax.triu");
  return Call(op, {x, k});
}

Expr triu(Expr x, int k) { return triu(x, IntImm::Int64(k)); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.op.tril", static_cast<Expr (*)(Expr, Expr)>(tril))
      .def("relax.op.triu", static_cast<Expr (*)(Expr, Expr)>(triu));
}

Type InferTypeTrilTriu(const Call& call, const BlockBuilder& ctx) {
  auto [data_ty, offset] = GetArgType<TensorType, PrimType>(call, ctx);

  if (!data_ty->IsUnknownNdim() && data_ty->ndim < 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << call->op
                                          << " requires the input tensor to have at least two "
                                             "dimensions. However, the given input has "
                                          << data_ty->ndim << " dimension(s).";
  }
  return data_ty;
}

TVM_REGISTER_OP("relax.tril")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("k", "PrimExpr", "The offset of the diagonal.")
    .set_attr<FInferType>("FInferType", InferTypeTrilTriu)
    .set_attr<bool>("FPurity", true);

TVM_REGISTER_OP("relax.triu")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input tensor.")
    .add_argument("k", "PrimExpr", "The offset of the diagonal.")
    .set_attr<FInferType>("FInferType", InferTypeTrilTriu)
    .set_attr<bool>("FPurity", true);

}  // namespace relax
}  // namespace tvm
