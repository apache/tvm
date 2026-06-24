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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/utils.h>

#include "../transform/utils.h"
#include "op_common.h"

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  CallTIRWithGradAttrs::RegisterReflection();
  CallTIRInplaceAttrs::RegisterReflection();
  CallInplacePackedAttrs::RegisterReflection();
  ToVDeviceAttrs::RegisterReflection();
  HintOnDeviceAttrs::RegisterReflection();
}

bool EqualConstInt(const PrimExpr& lhs, int64_t value) {
  if (const int64_t* pvalue = tirx::as_const_int(lhs)) {
    return pvalue[0] == value;
  }
  return false;
}

bool EqualCheck(const PrimExpr& lhs, const PrimExpr& rhs) {
  PrimExpr diff = lhs - rhs;
  if (const int64_t* pdiff = tirx::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  tvm::arith::Analyzer ana;
  diff = ana->Simplify(diff);
  if (const int64_t* pdiff = tirx::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  return false;
}

Type ReturnVoidType(const Call& call, const BlockBuilder& ctx) {
  return TupleType(ffi::Array<Type>());
}

Type ReturnObjectType(const Call& call, const BlockBuilder& ctx) { return ObjectType(); }

Type InferTypeShapeOf(const Call& call, const BlockBuilder& ctx) {
  // use the Type of the argument
  auto arg_ty = GetType(call->args[0]);
  auto* tensor_ty = GetType(call->args[0]).as<TensorTypeNode>();
  TVM_FFI_ICHECK(tensor_ty) << "shape_of expects a tensor input, but received " << arg_ty
                            << "; use MatchCast if necessary";
  if (tensor_ty->ndim == kUnknownNDim) {
    return ShapeType(kUnknownNDim);
  }
  // if the tensor shape is a Relax var or omitted, do not try to construct a shape expr from it
  if (!tensor_ty->shape.defined() || tensor_ty->shape.as<VarNode>()) {
    return ShapeType(tensor_ty->ndim);
  }
  // otherwise, copy over the values from the tensor shape
  auto* tensor_shape = tensor_ty->shape.as<ShapeExprNode>();
  TVM_FFI_ICHECK(tensor_shape);
  return ShapeType(tensor_shape->values);
}

// call_pure_packed

Type InferTypeCallPurePacked(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() < 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "call_pure_packed must be called with at least one argument";
  }

  // the callee must be an opaque function
  auto callee = call->args[0];
  TVM_FFI_ICHECK(!callee.as<OpNode>()) << "call_pure_packed cannot be used with an op node";
  auto opt = MatchType<FuncType>(callee);
  TVM_FFI_ICHECK(opt) << "Callee must have a function type";
  FuncType finfo = opt.value();
  TVM_FFI_ICHECK(finfo->IsOpaque())
      << "call_pure_packed must be called with an opaque function, but " << callee
      << " is not opaque";

  // same logic as from DeriveCallRetType for ordinary calls
  if (finfo->derive_func.defined()) {
    // derive using custom derivation function.
    return finfo->derive_func.value()(call, ctx);
  } else {
    // directly return the normal value.
    return finfo->ret;
  }
}

TVM_REGISTER_OP("relax.call_pure_packed")
    .set_num_inputs(-1)
    .add_argument("args", "ffi::Array<Expr>",
                  "The first argument is the function being called. The rest are the "
                  "arguments to that function.")
    .set_attr<FInferType>("FInferType", InferTypeCallPurePacked)
    .set_attr<bool>("FPurity", true);

Expr MakeCallPurePacked(const Expr& callee, ffi::Array<Expr> args, const Attrs& attrs,
                        ffi::Array<Type> ty_args) {
  static const Op& op = Op::Get("relax.call_pure_packed");
  ffi::Array<Expr> call_args = {callee};
  for (auto arg : args) {
    call_args.push_back(arg);
  }
  return Call(op, call_args, attrs, ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_pure_packed", MakeCallPurePacked);
}

// call_inplace_packed

Type InferTypeCallInplacePacked(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() <= 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "call_inplace_packed must be called with at least two arguments"
        << " (the packed call and at least one argument to the packed call"
        << "if the packed call does not need arguments, use call_pure_packed instead)";
  }

  // the callee must be an opaque function
  auto callee = call->args[0];
  TVM_FFI_ICHECK(!callee.as<OpNode>()) << "call_pure_packed cannot be used with an op node";
  auto opt = MatchType<FuncType>(callee);
  TVM_FFI_ICHECK(opt) << "Callee must have a function type";
  FuncType finfo = opt.value();
  TVM_FFI_ICHECK(finfo->IsOpaque())
      << "call_pure_packed must be called with an opaque function, but " << callee
      << " is not opaque";

  // check the range for inplace indices, make sure at least one is not -1, ensure they're unique
  const auto* attrs = call->attrs.as<CallInplacePackedAttrs>();
  size_t num_args = call->args.size() - 1;
  std::unordered_set<int> encountered;
  for (size_t i = 0; i < attrs->inplace_indices.size(); i++) {
    int index = attrs->inplace_indices[i];
    if (index < -1 || index >= static_cast<int>(num_args)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "In-place index " << i << " is out of range (must be between -1 and " << (num_args - 1)
          << ", inclusive, but is " << index << ")";
    }
    if (index != -1) {
      if (encountered.count(index)) {
        TVM_FFI_VISIT_THROW(ValueError, call) << "All in-place indices must be unique, but index "
                                              << index << " appears more than once.";
      }
      encountered.insert(index);
    }
  }
  if (encountered.empty()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "At least one index must have a value other than "
                                             "-1 (or else simply use call_pure_packed)";
  }

  // same logic as from DeriveCallRetType for ordinary calls
  Type ret;
  if (finfo->derive_func.defined()) {
    // derive using custom derivation function.
    ret = finfo->derive_func.value()(call, ctx);
  } else {
    // directly return the normal value.
    ret = finfo->ret;
  }

  // make sure that the derived return type matches that of the in-place args
  // (note: arg 0 is the packed func, so we add 1 to the arg index)
  if (attrs->inplace_indices.size() == 1) {
    auto arg_idx = attrs->inplace_indices[0] + 1;
    auto arg_ty = GetType(call->args[arg_idx]);
    if (!IsBaseOf(ret, arg_ty, ctx->GetAnalyzer())) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The derived return Type does not match that for "
          << "the in-place argument at index " << (arg_idx - 1) << ": " << ret << " vs " << arg_ty;
    }
  } else {
    auto* tup_info = ret.as<TupleTypeNode>();
    if (!tup_info) {
      TVM_FFI_VISIT_THROW(ValueError, call) << "Multiple outputs given via the inplace indices "
                                               "but the derived Type is not a tuple";
    }
    for (size_t i = 0; i < attrs->inplace_indices.size(); i++) {
      if (attrs->inplace_indices[i] == -1) {
        continue;
      }
      auto arg_idx = attrs->inplace_indices[i] + 1;
      auto arg_ty = GetType(call->args[arg_idx]);
      auto ret_ty = tup_info->fields[i];
      if (!IsBaseOf(ret_ty, arg_ty, ctx->GetAnalyzer())) {
        TVM_FFI_VISIT_THROW(ValueError, call) << "The derived return Type does not match that for "
                                              << "the in-place argument at index " << (arg_idx - 1)
                                              << ": " << ret_ty << " vs " << arg_ty;
      }
    }
  }

  return ret;
}

TVM_REGISTER_OP("relax.call_inplace_packed")
    .set_num_inputs(-1)
    .set_attrs_type<CallInplacePackedAttrs>()
    .add_argument("args", "ffi::Array<Expr>",
                  "The first argument is the function being called. The rest are the "
                  "arguments to that function.")
    .set_attr<FInferType>("FInferType", InferTypeCallInplacePacked)
    // Warning: considered pure, but it has the potential to create visible effects!
    // This should only be used if it has been *checked* that it is safe (no aliases, in-place
    // arguments will no longer be live) and the user believes the packed func to have no
    // side effects other than modifying the arguments specified as "inplace"
    .set_attr<bool>("FPurity", true);

Expr MakeCallInplacePacked(Expr func, ffi::Array<Expr> args, ffi::Array<int64_t> inplace_indices,
                           ffi::Array<Type> ty_args) {
  ffi::ObjectPtr<CallInplacePackedAttrs> attrs = ffi::make_object<CallInplacePackedAttrs>();
  attrs->inplace_indices = ffi::Array<int64_t>(inplace_indices.begin(), inplace_indices.end());

  static const Op& op = Op::Get("relax.call_inplace_packed");
  ffi::Array<Expr> call_args = {func};
  call_args.insert(call_args.end(), args.begin(), args.end());
  return Call(op, call_args, Attrs(attrs), ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_inplace_packed", MakeCallInplacePacked);
}

// call_tir

/* If possible, infer a legal value of `arg_ty`
 *
 * The `R.call_tir` operator and its variants accept an `arg_ty`
 * parameter, which specifies the shape of the tensor or tensors
 * returned by a PrimFunc.  This output shape must be compatible with
 * the shape defined by the PrimFunc's signature.
 *
 * For dynamic shapes, it is not always possible to infer the output
 * of a TIR PrimFunc from its inputs.  For example, a PrimFunc that
 * accepts input buffer `T.Buffer([16], "float32")` and output buffer
 * `T.Buffer([M, N], "float32")` infers the values of `M` and `N` from
 * the shape of the provided output buffer.
 *
 * If the arguments provided are not compatible with the PrimFunc's
 * signature, an error will be raised.  If the arguments are
 * compatible with the PrimFunc's signature, but are not sufficient to
 * determine the output's Type, then `std::nullopt` will be returned.
 *
 * \param func_ty The Type of the TIR callee.
 * \param arg_ty The Type of the argument tuple.
 * \param packed_ints_ty The Type of the ffi::Shape argument,
 *     if present.
 * \param opt_inplace_indices For `R.call_tir_inplace`, an array of
 *     indices indicating which outputs are constructed from in-place
 *     mutation of the inputs.  See
 *     `CallTIRInplaceAttrs::inplace_indices` for more details.
 *
 * \return The `arg_ty`, if it can be inferred from the arguments.
 *     Otherwise, std::nullopt.
 */
static ffi::Optional<Type> InferCallTIROutputTypeFromArguments(
    Type func_ty, Type arg_ty, ffi::Optional<Type> packed_ints_ty,
    ffi::Optional<ffi::Array<int64_t>> opt_inplace_indices) {
  auto opt_callee_ty = func_ty.as<FuncType>();
  TVM_FFI_CHECK(opt_callee_ty, TypeError)
      << "The first argument to `R.call_tir` must be a function, "
      << "but instead received argument of type " << func_ty;
  auto callee_ty = opt_callee_ty.value();

  TVM_FFI_CHECK(callee_ty->params.defined(), ValueError)
      << "The first argument to `R.call_tir` must be a function "
      << "with known argument types.  "
      << "However, the first argument was of type " << callee_ty;
  auto callee_params = callee_ty->params.value();

  const TupleTypeNode* args = arg_ty.as<TupleTypeNode>();
  TVM_FFI_CHECK(args, TypeError) << "The second argument to `R.call_tir` must be a tuple, "
                                 << "but instead received expression of type " << arg_ty;

  // R.call_tir expects the PrimFunc to have three groups of arguments.
  //
  // 1. Input arguments that are explicitly provided as Relax arguments.
  // 2. Output tensor arguments.
  // 3. Shape arguments, represented as `T.int64` in the PrimFunc, and
  //    as an optional ShapeExpr argument in the `relax::Call` node.
  //
  // In order to determine the return type of `R.call_tir`, we must
  // identify the PrimFunc arguments that will be in group (2).
  size_t num_input_arguments = args->fields.size();
  size_t num_trailing_int_arguments = 0;
  const ShapeTypeNode* packed_tuple_ty = nullptr;
  if (packed_ints_ty) {
    auto packed_ty = packed_ints_ty.value();
    packed_tuple_ty = packed_ty.as<ShapeTypeNode>();
    TVM_FFI_CHECK(packed_tuple_ty && !packed_tuple_ty->IsUnknownNdim(), TypeError)
        << "The third argument to `R.call_tir`, if present, "
        << "must be a ffi::Shape with known dimensionality.  "
        << "However, the argument received was of type " << packed_ty;
    num_trailing_int_arguments = packed_tuple_ty->ndim;
  } else {
    num_trailing_int_arguments = 0;
  }

  TVM_FFI_CHECK_LE(num_input_arguments + num_trailing_int_arguments, callee_params.size(),
                   ValueError)
      << "R.call_tir attempted to call a function using " << num_input_arguments
      << " input arguments and " << num_trailing_int_arguments << " trailing integer arguments.  "
      << "However, the callee only accepts " << callee_params.size() << " arguments in total.";

  // While Relax can specify a distributed tensor, TIR cannot.  The
  // current implementation does not support determining the output
  // shape for `R.dist.call_tir` calls, as it depends on the lowering
  // of DistIR into regular Relax.
  std::function<bool(Type)> contains_dtensor = [&contains_dtensor](Type ty) -> bool {
    if (ty.as<distributed::DTensorTypeNode>()) {
      return true;
    } else if (auto tuple = ty.as<TupleTypeNode>()) {
      return std::any_of(tuple->fields.begin(), tuple->fields.end(), contains_dtensor);
    } else {
      return false;
    }
  };
  if (contains_dtensor(arg_ty)) {
    return std::nullopt;
  }

  // At this point, the return types are known.  However, the shapes
  // in `callee_params` may contain dynamic shape parameters that are
  // not present in the caller's scope.  The `DeriveCallRetType`
  // utility can infer the value of dynamic parameters in
  // `FuncTypeNode::ret` based on definitions in
  // `FuncTypeNode::params`, inferring the correct values in the
  // caller's scope.
  //
  // Since the callee of `R.call_tir` is provided with output
  // arguments, where `DeriveCallRetType` requires a callee that
  // produces its own outputs, a dummy function signature and
  // arguments are used.

  auto dummy_callee_ty = [&]() -> FuncType {
    ffi::Array<Type> dummy_params(callee_params.begin(),
                                  callee_params.begin() + num_input_arguments);

    for (size_t i = callee_params.size() - num_trailing_int_arguments; i < callee_params.size();
         i++) {
      dummy_params.push_back(callee_params[i]);
    }

    ffi::Array<Type> dummy_ret(callee_params.begin() + num_input_arguments,
                               callee_params.end() - num_trailing_int_arguments);

    if (opt_inplace_indices) {
      // For R.call_tir_inplace, the `inplace_indices` are used to
      // indicate which elements of the `out_ty` will be generated
      // as in-place mutation from an input.  For any in-place
      // mutation, the parameter's Type must be inserted into
      // `out_ty`.
      auto inplace_indices = opt_inplace_indices.value();
      for (size_t i = 0; i < inplace_indices.size(); i++) {
        int64_t inplace_input_index = inplace_indices[i];
        if (inplace_input_index >= 0) {
          dummy_ret.insert(dummy_ret.begin() + i, callee_params[inplace_input_index]);
        }
      }
    }

    auto dummy_out_ty = [&]() -> Type {
      if (dummy_ret.size() == 1) {
        return dummy_ret[0];
      } else {
        return TupleType(dummy_ret);
      }
    }();

    return FuncType(dummy_params, dummy_out_ty);
  }();

  auto dummy_args = [&]() -> ffi::Array<Expr> {
    ffi::Array<Expr> dummy_args =
        args->fields.Map([](const Type& ty) -> Expr { return Var("dummy_leading_arg", ty); });

    for (size_t i = 0; i < num_trailing_int_arguments; i++) {
      TVM_FFI_ICHECK(packed_tuple_ty);
      PrimType dummy_arg_ty = [&]() {
        if (packed_tuple_ty->values) {
          return PrimType(packed_tuple_ty->values.value()[i].dtype());
        } else {
          return PrimType(DataType::Int(64));
        }
      }();
      dummy_args.push_back(Var("dummy_trailing_arg", dummy_arg_ty));
    }

    return dummy_args;
  }();

  auto derived_ret_ty =
      DeriveCallRetType(dummy_callee_ty, Call(Var("dummy_callee", dummy_callee_ty), dummy_args),
                        BlockBuilder::Create(std::nullopt));

  return derived_ret_ty;
}

Type InferTypeCallTIR(const Call& call, const BlockBuilder& ctx) {
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exactly 1 output type.";
  }
  TVM_FFI_ICHECK(call->args[0]->IsInstance<GlobalVarNode>())
      << "R.call_tir expects the first argument to be a GlobalVar referring to a TIR PrimFunc. "
      << "However, the argument " << call->args[0] << " instead has type "
      << call->args[0]->GetTypeKey();

  Type explicit_ty = call->ty_args[0];

  return explicit_ty;
}

Expr NormalizeCallTIR(const BlockBuilder& ctx, Call call) {
  // This function is used for normalization of `relax.call_tir`,
  // along with the variants `relax.call_tir_with_grad` and
  // `relax.call_tir_inplace`.  Therefore, all error messages should
  // be written in terms of `call->op`, and should not explicitly
  // reference the `relax.call_tir` operator.`
  TVM_FFI_ICHECK(call->args.size() == 2 || call->args.size() == 3)
      << "Operation " << call->op << " expects either two arguments [callee, arg_tuple], "
      << "or three arguments [callee, arg_tuple, tir_args], "
      << "but " << call << " has " << call->args.size() << " arguments.";

  auto callee = call->args[0];
  TVM_FFI_ICHECK(callee->ty.as<FuncTypeNode>())
      << "Operation " << call->op << " expects the first argument to be a TIR callee.  "
      << "However, the first argument " << callee << " has type " << callee->ty;

  Expr arg_tuple = call->args[1];

  TVM_FFI_ICHECK(arg_tuple->ty.as<TupleTypeNode>())
      << "Operation " << call->op << " expects the second argument to be a tuple of relax Expr.  "
      << "However, the second argument " << arg_tuple << " has type " << arg_tuple->ty << ".";

  TVM_FFI_ICHECK(arg_tuple.as<TupleNode>() || arg_tuple.as<VarNode>())
      << "Operation " << call->op << " must hold its arguments as an in-line tuple.  "
      << "However, " << call << " has arguments " << arg_tuple
      << ", which is neither an in-line tuple, "
      << "nor a variable binding that may be normalized to an in-line tuple.";

  if (call->args.size() > 2) {
    Expr packed_ints = call->args[2];
    TVM_FFI_ICHECK(packed_ints->ty.as<ShapeTypeNode>())
        << "Operation " << call->op << " expects the optional third argument, "
        << "if present, to be a ffi::Shape.  "
        << "However, the third argument " << packed_ints << " has type " << packed_ints->ty;
  }

  TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1)
      << "R.call_tir should have exactly one `ty_args` parameter, "
      << "which defines the output of the PrimFunc.";

  auto unwrap_binding = [&ctx](Expr expr) -> ffi::Optional<Expr> {
    if (auto var = expr.as<Var>()) {
      if (auto bound_value = ctx->LookupBinding(var.value())) {
        return bound_value.value();
      }
    }
    return std::nullopt;
  };

  Tuple new_arg_tuple = [&]() {
    // No replacement required.  The argument tuple is already
    // provided as an in-line tuple.
    if (auto opt = arg_tuple.as<Tuple>()) {
      return opt.value();
    }

    Expr unwrapped_tuple = arg_tuple;
    while (auto unwrapped = unwrap_binding(unwrapped_tuple)) {
      unwrapped_tuple = unwrapped.value();
    }

    // Preferred replacement.  The argument tuple is provided as a
    // variable, but we know the value bound to that variable.
    if (auto opt = unwrapped_tuple.as<Tuple>()) {
      return opt.value();
    }

    // Fallback case.  The argument tuple is provided as a variable,
    // and we don't know the value bound to that variable.  For
    // example, if a relax function accepted a tuple as an parameter,
    // then provided that same tuple as an argument to call_tir.
    ffi::Array<Expr> tuple_elements;
    size_t num_fields = arg_tuple->ty.as_or_throw<TupleType>()->fields.size();
    for (size_t i = 0; i < num_fields; i++) {
      tuple_elements.push_back(TupleGetItem(arg_tuple, i));
    }
    return Tuple(tuple_elements);
  }();

  if (!new_arg_tuple.same_as(arg_tuple)) {
    auto new_args = call->args;
    new_args.Set(1, new_arg_tuple);
    call.CopyOnWrite()->args = new_args;
  }

  return call;
}

void ValidateCallTIR(Call call) {
  // This function is used for validation of `relax.call_tir`,
  // along with the variants `relax.call_tir_with_grad` and
  // `relax.call_tir_inplace`.  Therefore, all error messages should
  // be written in terms of `call->op`, and should not explicitly
  // reference the `relax.call_tir` operator.`

  auto callee = call->args[0];
  Expr arg_tuple = call->args[1];

  auto packed_int_ty = [&]() -> ffi::Optional<Type> {
    if (call->args.size() <= 2) {
      return std::nullopt;
    } else {
      return GetType(call->args[2]);
    }
  }();

  auto opt_inplace_indices = [&]() -> ffi::Optional<ffi::Array<int64_t>> {
    if (const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>()) {
      return attrs->inplace_indices;
    } else {
      return std::nullopt;
    }
  }();

  Type explicit_ty = call->ty_args[0];
  auto inferred_ty = InferCallTIROutputTypeFromArguments(GetType(callee), GetType(arg_tuple),
                                                         packed_int_ty, opt_inplace_indices);
  if (inferred_ty.defined()) {
    TVM_FFI_CHECK(IsBaseOf(inferred_ty.value(), explicit_ty), TypeError)
        << "The `out_ty` argument for R.call_tir must be compatible with the PrimFunc.  "
        << "However, the PrimFunc's signature implies that the output should be " << inferred_ty
        << ", but the `out_ty` argument was " << explicit_ty;
  }
}

TVM_REGISTER_OP("relax.call_tir")
    .set_num_inputs(3)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("packed_ints", "Expr",
                  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from "
                  "args if unused")
    .set_attr<FInferType>("FInferType", InferTypeCallTIR)
    .set_attr<FNormalize>("FNormalize", NormalizeCallTIR)
    .set_attr<FValidate>("FValidate", ValidateCallTIR)
    .set_attr<bool>("FPurity", true);

Expr MakeCallTIR(Expr func, Tuple args, ffi::Array<TensorType> out_ty_list,
                 ffi::Optional<Expr> packed_ints) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty{nullptr};
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op& op = Op::Get("relax.call_tir");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, {}, {out_ty});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, {}, {out_ty});
  }
  return call;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_tir", MakeCallTIR);
}

// call_tir_with_grad

TVM_REGISTER_OP("relax.call_tir_with_grad")
    .set_num_inputs(3)
    .set_attrs_type<CallTIRWithGradAttrs>()
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("packed_ints", "Expr",
                  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from "
                  "args if unused")
    .set_attr<FInferType>("FInferType", InferTypeCallTIR)
    .set_attr<FNormalize>("FNormalize", NormalizeCallTIR)
    .set_attr<FValidate>("FValidate", ValidateCallTIR)
    .set_attr<bool>("FPurity", true);

Expr MakeCallTIRWithGrad(Expr func, Tuple args, ffi::Array<TensorType> out_ty_list,
                         ffi::String te_grad_name, ffi::Map<ffi::String, ffi::Any> te_grad_kwargs,
                         ffi::Optional<Expr> packed_ints) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir_with_grad should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty{nullptr};
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  ffi::ObjectPtr<CallTIRWithGradAttrs> attrs = ffi::make_object<CallTIRWithGradAttrs>();
  attrs->te_grad_name = te_grad_name;
  attrs->te_grad_kwargs = te_grad_kwargs;

  static const Op& op = Op::Get("relax.call_tir_with_grad");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, Attrs(attrs), {out_ty});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, Attrs(attrs), {out_ty});
  }
  return call;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_tir_with_grad", MakeCallTIRWithGrad);
}

// call_tir_inplace

Expr NormalizeCallTIRInPlace(const BlockBuilder& ctx, Call call) {
  // Apply normalization before error checks.  This allows the error
  // checks to safely require `call->args[1]` to be a Tuple, which
  // may result in an error if performed before normalization.
  call = NormalizeCallTIR(ctx, std::move(call)).as_or_throw<Call>();

  ffi::Array<Type> ty_outputs = [&]() -> ffi::Array<Type> {
    auto out_ty = call->ty_args[0];
    if (auto* tuple_output = out_ty.as<TupleTypeNode>()) {
      return tuple_output->fields;
    } else {
      return {out_ty};
    }
  }();

  // there must be an inplace index for each output
  const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>();
  TVM_FFI_ICHECK(attrs);
  if (attrs->inplace_indices.size() != ty_outputs.size()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "There must be an in-place index specified for each output";
  }

  // check the range for inplace indices, make sure at least one is not -1, ensure they're unique
  size_t num_args = call->args[1].as_or_throw<Tuple>()->fields.size();
  std::unordered_set<int> encountered;
  for (size_t i = 0; i < attrs->inplace_indices.size(); i++) {
    int index = attrs->inplace_indices[i];
    if (index < -1 || index >= static_cast<int>(num_args)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "In-place index " << i << " is out of range (must be between -1 and " << (num_args - 1)
          << ", inclusive, but is " << index << ")";
    }
    if (index != -1) {
      if (encountered.count(index)) {
        TVM_FFI_VISIT_THROW(ValueError, call) << "All in-place indices must be unique, but index "
                                              << index << " appears more than once.";
      }
      encountered.insert(index);
    }
  }
  if (encountered.empty()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "At least one index must have a value other than -1 (or else simply use call_tir)";
  }

  // for safety, we will make sure the output shape for each in-place argument exactly matches the
  // input shape
  // TODO(@slyubomirsky): eventually we will want to handle cases where that is not true
  Tuple call_args = call->args[1].as_or_throw<Tuple>();

  for (size_t i_output = 0; i_output < attrs->inplace_indices.size(); i_output++) {
    auto i_input = attrs->inplace_indices[i_output];
    if (i_input == -1) {
      continue;
    }

    auto ty_output = ty_outputs[i_output];
    auto tinfo_output = ty_output.as<TensorTypeNode>();

    if (!tinfo_output || !tinfo_output->shape.defined() || tinfo_output->IsUnknownDtype()) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The output type for an in-place mutation must be a tensor "
          << "with a defined shape and dtype, "
          << "but output " << i_output << " has type " << ty_output;
    }

    auto ty_input = GetType(call_args->fields[i_input]);
    auto tinfo_input = ty_input.as<TensorTypeNode>();

    if (!tinfo_input ||
        (tinfo_output->IsUnknownDtype() || tinfo_output->dtype != tinfo_input->dtype) ||
        (!tinfo_input->shape.defined() ||
         !CanProveShapeEqual(tinfo_input->shape.value(), tinfo_output->shape.value(),
                             ctx->GetAnalyzer()))) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The input used for an in-place mutation must be "
          << "a tensor with identical shape and dtype as the output.  "
          << "However, output " << i_output << " with type " << ty_output
          << " is specified as an in-place mutation of input " << i_input << " with type "
          << ty_input;
    }
  }

  return call;
}

TVM_REGISTER_OP("relax.call_tir_inplace")
    .set_num_inputs(3)
    .set_attrs_type<CallTIRInplaceAttrs>()
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("packed_ints", "Expr",
                  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from "
                  "args if unused")
    .set_attr<FInferType>("FInferType", InferTypeCallTIR)
    .set_attr<FNormalize>("FNormalize", NormalizeCallTIRInPlace)
    .set_attr<FValidate>("FValidate", ValidateCallTIR)
    // Warning: considered pure, but it has the potential to create visible effects!
    // This should only be used if it has been *checked* that it is safe (no aliases, in-place
    // arguments will no longer be live)
    .set_attr<bool>("FPurity", true);

Expr MakeCallTIRInplace(Expr func, Tuple args, ffi::Array<int64_t> inplace_indices,
                        ffi::Array<TensorType> out_ty_list, ffi::Optional<Expr> packed_ints) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  ffi::ObjectPtr<CallTIRInplaceAttrs> attrs = ffi::make_object<CallTIRInplaceAttrs>();
  attrs->inplace_indices = ffi::Array<int64_t>(inplace_indices.begin(), inplace_indices.end());

  Type out_ty{nullptr};
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op& op = Op::Get("relax.call_tir_inplace");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, Attrs(attrs), {out_ty});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, Attrs(attrs), {out_ty});
  }
  return call;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_tir_inplace", MakeCallTIRInplace);
}

// call_dps_packed

Type InferTypeCallDPSPacked(const Call& call, const BlockBuilder& ctx) {
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exact 1 output type.";
  }
  return call->ty_args[0];
}

TVM_REGISTER_OP("relax.call_dps_packed")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .set_attr<FInferType>("FInferType", InferTypeCallDPSPacked)
    // technically, an impure op could be used with this, but there is
    // little reason to use DPS with an impure op
    .set_attr<bool>("FPurity", true);

Expr MakeCallDPSPacked(Expr func, Tuple args, ffi::Array<TensorType> out_ty_list) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_dps_packed should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty{nullptr};
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op& op = Op::Get("relax.call_dps_packed");
  return Call(op, {func, args}, {}, {out_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_dps_packed", MakeCallDPSPacked);
}

// call_py_func

Type InferTypeCallPyFunc(const Call& call, const BlockBuilder& ctx) {
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exact 1 output type.";
  }
  return call->ty_args[0];
}

void ValidateCallPyFunc(Call call) {
  // Validate that the function name is a string literal
  auto func_name = call->args[0];
  TVM_FFI_ICHECK(func_name->IsInstance<StringImmNode>())
      << "Operation " << call->op << " expects the first argument to be a string literal "
      << "specifying the Python function name. However, the first argument " << func_name
      << " is not a string literal.";

  // Validate that args is a tuple
  Expr arg_tuple = call->args[1];
  TVM_FFI_ICHECK(arg_tuple->ty.as<TupleTypeNode>())
      << "Operation " << call->op << " expects the second argument to be a tuple of relax Expr.  "
      << "However, the second argument " << arg_tuple << " has type " << arg_tuple->ty << ".";

  TVM_FFI_ICHECK(arg_tuple.as<TupleNode>() || arg_tuple.as<VarNode>())
      << "Operation " << call->op << " must hold its arguments as an in-line tuple.  "
      << "However, " << call << " has arguments " << arg_tuple
      << ", which is neither an in-line tuple, "
      << "nor a variable binding that may be normalized to an in-line tuple.";
}

TVM_REGISTER_OP("relax.call_py_func")
    .set_num_inputs(2)
    .add_argument("func_name", "StringImm", "The name of the Python function to call.")
    .add_argument("args", "Tuple", "The input arguments.")
    .set_attr<FInferType>("FInferType", InferTypeCallPyFunc)
    .set_attr<FValidate>("FValidate", ValidateCallPyFunc)
    .set_attr<bool>("FPurity", true);

Expr MakeCallPyFunc(StringImm func_name, Tuple args, ffi::Array<TensorType> out_ty_list) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_py_func should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty{nullptr};
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op& op = Op::Get("relax.call_py_func");
  return Call(op, {func_name, args}, {}, {out_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_py_func", MakeCallPyFunc);
}

// call builtin
Type InferTypeCallBuiltinWithCtx(const Call& call, const BlockBuilder& ctx) {
  if (call->ty_args.size() == 0) {
    // by default return void.
    return TupleType(ffi::Array<Type>());
  } else {
    TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1);
    return call->ty_args[0];
  }
}

TVM_REGISTER_OP("relax.call_builtin_with_ctx")
    .set_num_inputs(4)
    .add_argument("func", "Expr", "The builtin packed func.")
    .add_argument("args", "Tuple", "The input arguments.")
    .set_attr<FInferType>("FInferType", InferTypeCallBuiltinWithCtx)
    // Most builtins are pure, but some are not, like `vm.builtin.attention_kv_cache_append`
    .set_attr<bool>("FPurity", false);

Expr MakeCallBuiltinWithCtx(Expr func, Tuple args, ffi::Array<Type> ty_args) {
  static const Op& op = Op::Get("relax.call_builtin_with_ctx");
  return Call(op, {func, args}, Attrs(), ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_builtin_with_ctx", MakeCallBuiltinWithCtx);
}

TVM_REGISTER_OP("relax.null_value")
    .set_num_inputs(0)
    .set_attr<FInferType>("FInferType", ReturnObjectType)
    .set_attr<bool>("FPurity", true);

Expr MakeCallNullValue() {
  static const Op& op = Op::Get("relax.null_value");
  return Call(op, {}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.null_value", MakeCallNullValue);
}

// print

TVM_REGISTER_OP("relax.print")
    .set_num_inputs(-1)
    .add_argument("vals", "ffi::Array<Expr>",
                  "The first value is Python-style format string to use to print. The others "
                  "are values to print")
    .set_attr<FInferType>("FInferType", ReturnVoidType)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.print")
    .set_attr<bool>("FPurity", false);

Expr MakePrint(ffi::Array<Expr> vals, StringImm format) {
  ffi::Array<Expr> params;
  params.push_back(format);
  for (const auto val : vals) {
    params.push_back(val);
  }
  static const Op& op = Op::Get("relax.print");
  return Call(op, params);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.print", MakePrint);
}

// assert_op

// can't actually name it assert or else Python will consider it a syntax error

Type InferAssertType(const Call& call, const BlockBuilder& ctx) {
  // Ensure that the condition argument is a boolean scalar.
  // Also permitted is a tensor with unknown shape and unknown dtype
  // (checked dynamically in that case). Returns void.
  if (call->args.size() < 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Assert must have at least one argument (the condition).";
  }
  Type arg_ty = GetType(call->args[0]);
  if (!IsBoolType(arg_ty)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "The argument to assert must be a boolean scalar, but received " << arg_ty;
  }
  return ReturnVoidType(call, ctx);
}

TVM_REGISTER_OP("relax.assert_op")
    .set_num_inputs(-1)
    .add_argument("vals", "ffi::Array<Expr>",
                  "The first value is used as the assertion condition. The second value is "
                  "Python-style format string to use for displaying an error message, if the "
                  "assert fails. The others are used as format arguments if there is an error.")
    .set_attr<FInferType>("FInferType", InferAssertType)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.assert_op")
    .set_attr<bool>("FPurity", false);

Expr MakeAssertOp(Expr condition, ffi::Array<Expr> vals, StringImm format) {
  static const Op& op = Op::Get("relax.assert_op");
  ffi::Array<Expr> args = {condition};
  args.push_back(format);
  for (auto val : vals) {
    args.push_back(val);
  }
  return Call(op, args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.assert_op", MakeAssertOp);
}

// make_closure

TVM_REGISTER_OP("relax.make_closure")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The closure.")
    .add_argument("args", "Tuple", "The captured variables.")
    .set_attr<FInferType>("FInferType", ReturnObjectType)
    .set_attr<bool>("FPurity", true);

Expr MakeClosure(Expr func, Tuple args) {
  static const Op& op = Op::Get("relax.make_closure");
  return Call(op, {func, args}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.make_closure", MakeClosure);
}

// invoke_closure

Type InferTypeInvokeClosure(const Call& call, const BlockBuilder& ctx) {
  if (call->ty_args.empty()) {
    return ObjectType();
  } else if (call->ty_args.size() == 1) {
    return call->ty_args[0];
  } else {
    return TupleType(call->ty_args);
  }
}

TVM_REGISTER_OP("relax.invoke_closure")
    .set_num_inputs(2)
    .add_argument("closure", "Expr", "The VMClosure.")
    .add_argument("args", "Tuple", "The captured variables.")
    .set_attr<FInferType>("FInferType", InferTypeInvokeClosure)
    // Not all closures are pure. Use invoke_pure_closure for specifying purity
    .set_attr<bool>("FPurity", false);

Expr InvokeClosure(Expr closure, Tuple args, ffi::Array<Type> ty_args) {
  static const Op& op = Op::Get("relax.invoke_closure");
  return Call(op, {closure, args}, {}, ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.invoke_closure", InvokeClosure);
}

// invoke_pure_closure

TVM_REGISTER_OP("relax.invoke_pure_closure")
    .set_num_inputs(2)
    .add_argument("closure", "Expr", "The VMClosure.")
    .add_argument("args", "Tuple", "The captured variables.")
    .set_attr<FInferType>("FInferType", InferTypeInvokeClosure)
    .set_attr<bool>("FPurity", true);

Expr InvokePureClosure(Expr closure, Tuple args, ffi::Array<Type> ty_args) {
  static const Op& op = Op::Get("relax.invoke_pure_closure");
  return Call(op, {closure, args}, {}, ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.invoke_pure_closure", InvokePureClosure);
}

// shape_of

TVM_REGISTER_OP("relax.shape_of")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input expression")
    .set_attr<FInferType>("FInferType", InferTypeShapeOf)
    .set_attr<bool>("FPurity", true);

Expr MakeShapeOf(Expr expr) {
  static const Op& op = Op::Get("relax.shape_of");
  return Call(op, {expr}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.shape_of", MakeShapeOf);
}

// size

Type InferTypeSize(const Call& call, const BlockBuilder& ctx) {
  auto arg_ty = GetType(call->args[0]);
  auto* tensor_ty = GetType(call->args[0]).as<TensorTypeNode>();
  TVM_FFI_ICHECK(tensor_ty) << "size expects a tensor input, but received " << arg_ty
                            << "; use MatchCast if necessary";
  return TensorType(ShapeExpr(ffi::Array<PrimExpr>{}), DataType::Int(64));
}

TVM_REGISTER_OP("relax.size")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input tensor")
    .set_attr<FInferType>("FInferType", InferTypeSize)
    .set_attr<bool>("FPurity", true);

Expr MakeSize(Expr expr) {
  static const Op& op = Op::Get("relax.size");
  return Call(op, {expr}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.size", MakeSize);
}

// tensor_to_shape

Type ReturnTensorToShapeType(const Call& call, const BlockBuilder& ctx) {
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(call->args[0]->ty.defined());
  const auto* tensor_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty);
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 1)
      << "relax.tensor_to_shape expected argument to be 1-d, "
      << "but " << call << " has argument " << call->args[0] << " with type " << call->args[0]->ty;

  if (tensor_ty->shape.defined()) {
    ShapeExpr shape_expr = tensor_ty->shape.value().as_or_throw<ShapeExpr>();
    const IntImmNode* ndim = shape_expr->values[0].as<IntImmNode>();
    if (ndim) {
      return ShapeType(ndim->value);
    }
  }
  return ShapeType(kUnknownNDim);
}

TVM_REGISTER_OP("relax.tensor_to_shape")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input expression")
    .set_attr<FInferType>("FInferType", ReturnTensorToShapeType)
    .set_attr<bool>("FPurity", true);

Expr MakeTensorToShape(Expr expr) {
  static const Op& op = Op::Get("relax.tensor_to_shape");
  return Call(op, {expr}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.tensor_to_shape", MakeTensorToShape);
}

// shape_to_tensor
Type ReturnShapeToTensorType(const Call& call, const BlockBuilder& ctx) {
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(call->args[0]->ty.defined());
  const auto* ty = GetTypeAs<ShapeTypeNode>(call->args[0]);
  TVM_FFI_ICHECK(ty);
  int32_t ndim = ty->ndim;
  return TensorType(ShapeExpr({PrimExpr(ndim)}), DataType::Int(64));
}

TVM_REGISTER_OP("relax.shape_to_tensor")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input expression")
    .set_attr<FInferType>("FInferType", ReturnShapeToTensorType)
    .set_attr<bool>("FPurity", true);

Expr MakeShapeToTensor(Expr expr) {
  static const Op& op = Op::Get("relax.shape_to_tensor");
  return Call(op, {expr}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.shape_to_tensor", MakeShapeToTensor);
}

// alloc_tensor

Type InferTypeAllocateTensor(const Call& call, const BlockBuilder& ctx) {
  TVM_FFI_ICHECK(call->args[0].as<ShapeExprNode>())
      << "must be ShapeExpr, but got " << call->args[0]->GetTypeKey();
  TVM_FFI_ICHECK(call->args[1].as<DataTypeImmNode>())
      << "must be DataTypeImm, but got " << call->args[1]->GetTypeKey();
  DataType out_dtype;
  if (const auto* dtype_node = call->args[1].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = ffi::GetRef<DataTypeImm>(dtype_node);
    out_dtype = dtype_imm->value;
  }
  int64_t vdevice_index = -1;
  if (auto* prim_value_node = call->args[2].as<PrimValueNode>()) {
    vdevice_index = prim_value_node->value.as<IntImmNode>()->value;
  }
  auto vdevice = GetGlobalVDevice(ctx->GetContextIRModule(), vdevice_index);

  if (vdevice.defined()) {
    return TensorType(call->args[0], out_dtype, vdevice.value());
  }
  return TensorType(call->args[0], out_dtype);
}

TVM_REGISTER_OP("relax.builtin.alloc_tensor")
    .set_num_inputs(4)
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .add_argument("runtime_device_index", "PrimValue",
                  "The device index indicating on which device the tensor is to be "
                  "allocated at runtime. Index -1 is reserved for the host device.")
    .add_argument("storage_scope", "StringImm",
                  "The storage scope of the storage to allocate. Default is global.")
    .set_attr<FInferType>("FInferType", InferTypeAllocateTensor)
    // memory allocation isn't considered a "visible effect" as far as purity is concerned
    .set_attr<bool>("FPurity", true)
    .set_attr<bool>("TAllocator", true);

Expr MakeAllocTensor(Expr shape, DataTypeImm dtype, PrimValue runtime_device_index,
                     StringImm storage_scope) {
  static const Op& op = Op::Get("relax.builtin.alloc_tensor");
  return Call(op, {shape, dtype, runtime_device_index, storage_scope}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.builtin.alloc_tensor", MakeAllocTensor);
}

// memory planning alloc_storage

TVM_REGISTER_OP("relax.memory.alloc_storage")
    .set_num_inputs(4)
    .add_argument("total_space", "Expr", "The total space of the storage to allocate.")
    .add_argument(
        "virtual_device_index", "PrimValue",
        "The virtual device index indicating on which device the storage is to be allocated, "
        "Index -1 is reserved for the host device.")
    .add_argument("storage_scope", "StringImm",
                  "The storage scope of the storage to allocate. Default is global.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .set_attr<FInferType>("FInferType", ReturnObjectType)
    // memory allocation isn't considered a "visible effect" as far as purity is concerned
    .set_attr<bool>("FPurity", true)
    .set_attr<bool>("TAllocator", true);

Expr MakeAllocStorage(Expr size, PrimValue virtual_device_index, StringImm storage_scope,
                      DataTypeImm dtype) {
  static const Op& op = Op::Get("relax.memory.alloc_storage");
  return Call(op, {size, virtual_device_index, storage_scope, dtype}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.alloc_storage", MakeAllocStorage);
}

// memory planning alloc_tensor

Type InferTypeMemAllocTensor(const Call& call, const BlockBuilder& ctx) {
  TVM_FFI_ICHECK(GetTypeAs<ShapeTypeNode>(call->args[2]))
      << "must be a Expr of ShapeType, but got " << call->args[1]->GetTypeKey();
  DataType out_dtype;
  if (const auto* dtype_node = call->args[3].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = ffi::GetRef<DataTypeImm>(dtype_node);
    out_dtype = dtype_imm->value;
  }

  if (call->args.size() == 5) {
    int64_t vdevice_index = -1;
    if (auto* prim_value_node = call->args[4].as<PrimValueNode>()) {
      vdevice_index = prim_value_node->value.as<IntImmNode>()->value;
    }
    auto vdevice = GetGlobalVDevice(ctx->GetContextIRModule(), vdevice_index);
    if (vdevice.defined()) {
      return TensorType(call->args[2], out_dtype, vdevice.value());
    }
  }

  return TensorType(call->args[2], out_dtype);
}

TVM_REGISTER_OP("relax.memory.alloc_tensor")
    .set_num_inputs(5)
    .add_argument("storage", "Expr", "The storage to allocate the tensor to.")
    .add_argument("offset", "PrimValue", "Storage offset to allocate the tensor.")
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .add_argument("runtime_device_index", "PrimValue",
                  "The device index indicating on which device the tensor is to be "
                  "allocated at runtime. Index -1 is reserved for the host device.")
    .set_attr<FInferType>("FInferType", InferTypeMemAllocTensor)
    // memory allocation isn't considered a "visible effect" as far as purity is concerned
    .set_attr<bool>("FPurity", true)
    .set_attr<bool>("TAllocator", true);

Expr MakeMemAllocTensor(Expr storage, PrimValue offset, Expr shape, DataTypeImm dtype,
                        PrimValue virtual_device_index) {
  static const Op& op = Op::Get("relax.memory.alloc_tensor");
  return Call(op, {storage, offset, shape, dtype, virtual_device_index}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "relax.op.memory.alloc_tensor", [](ffi::PackedArgs args, ffi::Any* ret) {
        if (args.size() == 5) {
          *ret = MakeMemAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimValue>(),
                                    args[2].cast<Expr>(), args[3].cast<DataTypeImm>(),
                                    args[4].cast<PrimValue>());
        } else {
          *ret = MakeMemAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimValue>(),
                                    args[2].cast<Expr>(), args[3].cast<DataTypeImm>(),
                                    PrimValue::Int64(0));
        }
      });
}

// memory planning kill_storage

TVM_REGISTER_OP("relax.memory.kill_storage")
    .set_num_inputs(1)
    .add_argument("storage", "Expr", "The storage to be killed.")
    .set_attr<FInferType>("FInferType", ReturnVoidType)
    // We mark this as impure so it wouldn't be removed by "remove_all_unused"
    .set_attr<bool>("FPurity", false);

Expr MakeMemKillStorage(Expr storage) {
  static const Op& op = Op::Get("relax.memory.kill_storage");
  return Call(op, {storage}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.kill_storage", MakeMemKillStorage);
}

// memory planning kill_tensor

TVM_REGISTER_OP("relax.memory.kill_tensor")
    .set_num_inputs(1)
    .add_argument("tensor", "Expr", "The tensor to be killed.")
    .set_attr<FInferType>("FInferType", ReturnVoidType)
    // We mark this as impure so it wouldn't be removed by "remove_all_unused"
    .set_attr<bool>("FPurity", false);

Expr MakeMemKillTensor(Expr tensor) {
  static const Op& op = Op::Get("relax.memory.kill_tensor");
  return Call(op, {tensor}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.kill_tensor", MakeMemKillTensor);
}

// vm alloc_storage

TVM_REGISTER_OP("relax.vm.alloc_storage")
    .set_num_inputs(4)
    .add_argument("size", "Expr", "The size of the storage to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .add_argument("runtime_device_index", "PrimValue",
                  "The device index indicating on which device the tensor is "
                  "to be allocated at runtime.")
    .add_argument("storage_scope", "StringImm",
                  "The storage scope of the storage to allocate. Default is global.")
    .set_attr<FInferType>("FInferType", ReturnObjectType)
    // memory allocation isn't considered a "visible effect" as far as purity is concerned
    .set_attr<bool>("FPurity", true)
    .set_attr<bool>("TAllocator", true);

Expr MakeVMAllocStorage(Expr size, PrimValue runtime_device_index, DataTypeImm dtype,
                        StringImm storage_scope) {
  static const Op& op = Op::Get("relax.vm.alloc_storage");
  return Call(op, {size, runtime_device_index, dtype, storage_scope}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vm.alloc_storage", MakeVMAllocStorage);
}

// vm alloc_tensor

Type InferTypeVMAllocTensor(const Call& call, const BlockBuilder& ctx) {
  DataType out_dtype;
  if (const auto* dtype_node = call->args[3].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = ffi::GetRef<DataTypeImm>(dtype_node);
    out_dtype = dtype_imm->value;
  }
  int64_t vdevice_index = -1;
  if (auto* prim_value_node = call->args[4].as<PrimValueNode>()) {
    vdevice_index = prim_value_node->value.as<IntImmNode>()->value;
  }
  auto vdevice = GetGlobalVDevice(ctx->GetContextIRModule(), vdevice_index);

  if (const auto* output_shape = call->args[2].as<ShapeExprNode>()) {
    return TensorType(ffi::GetRef<Expr>(output_shape), out_dtype, vdevice);
  } else if (const auto* shape_ty = GetTypeAs<ShapeTypeNode>(call->args[2])) {
    if (shape_ty->values.defined()) {
      return TensorType(ShapeExpr(shape_ty->values.value()), out_dtype, vdevice);
    } else {
      return TensorType(out_dtype, shape_ty->ndim, vdevice);
    }
  }
  return TensorType(out_dtype, kUnknownNDim, vdevice);
}

TVM_REGISTER_OP("relax.vm.alloc_tensor")
    .set_num_inputs(5)
    .add_argument("storage", "Expr", "The storage to allocate the tensor to.")
    .add_argument("offset", "PrimValue", "Storage offset to allocate the tensor.")
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .add_argument("runtime_device_index", "PrimValue",
                  "The device index indicating on which device the tensor is "
                  "to be allocated at runtime.")
    .set_attr<FInferType>("FInferType", InferTypeVMAllocTensor)
    // memory allocation isn't considered a "visible effect" as far as purity is concerned
    .set_attr<bool>("FPurity", true)
    .set_attr<bool>("TAllocator", true);

Expr MakeVMAllocTensor(Expr storage, PrimValue offset, Expr shape, DataTypeImm dtype,
                       PrimValue runtime_device_index) {
  static const Op& op = Op::Get("relax.vm.alloc_tensor");
  return Call(op, {storage, offset, shape, dtype, runtime_device_index}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("relax.op.vm.alloc_tensor", [](ffi::PackedArgs args, ffi::Any* ret) {
    if (args.size() == 5) {
      *ret =
          MakeVMAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimValue>(), args[2].cast<Expr>(),
                            args[3].cast<DataTypeImm>(), args[4].cast<PrimValue>());
    } else {
      *ret =
          MakeVMAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimValue>(), args[2].cast<Expr>(),
                            args[3].cast<DataTypeImm>(), PrimValue::Int64(0));
    }
  });
}

// vm kill_object
TVM_REGISTER_OP("relax.vm.kill_object")
    .set_num_inputs(1)
    .add_argument("obj", "Expr", "The object to be killed.")
    .set_attr<FInferType>("FInferType", ReturnVoidType)
    // We mark this as impure so it wouldn't be removed by "remove_all_unused"
    .set_attr<bool>("FPurity", false);

Expr MakeVMKillObject(Expr obj) {
  static const Op& op = Op::Get("relax.vm.kill_object");
  return Call(op, {std::move(obj)}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vm.kill_object", MakeVMKillObject);
}

// vm call_tir_dyn

TVM_REGISTER_OP("relax.vm.call_tir_dyn")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple",
                  "The input arguments (list of tensors and last argument is ShapeExpr)")
    .set_attr<FInferType>("FInferType", ReturnVoidType)
    // "relax.vm.call_tir_dyn" works in an in-place way, which is impure.
    .set_attr<bool>("FPurity", false);

Expr MakeCallTIRDyn(Expr func, Tuple args) {
  static const Op& op = Op::Get("relax.vm.call_tir_dyn");
  return Call(op, {func, args}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vm.call_tir_dyn", MakeCallTIRDyn);
}

// builtin stop_lift_params
Type InferTypeStopLiftParams(const Call& call, const BlockBuilder& ctx) {
  return InferTypeUnaryArith<false>(call, ctx);
}

TVM_REGISTER_OP("relax.builtin.stop_lift_params")
    .set_num_inputs(1)
    .add_argument("x", "Expr", "The input data")
    .set_attr<FInferType>("FInferType", InferTypeStopLiftParams)
    .set_attr<bool>("FPurity", true);

Expr MakeStopLiftParams(Expr x) {
  static const Op& op = Op::Get("relax.builtin.stop_lift_params");
  return Call(op, {x}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.builtin.stop_lift_params", MakeStopLiftParams);
}

// to_vdevice

Type InferToVDeviceType(const Call& call, const BlockBuilder& ctx) {
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(call->args[0]->ty.defined());
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  auto attrs = call->attrs.as<ToVDeviceAttrs>();
  VDevice vdev = attrs->dst_vdevice;
  if (data_ty->shape.defined()) {
    return TensorType(data_ty->shape.value(), data_ty->dtype, vdev, data_ty->span);
  }
  return TensorType(data_ty->dtype, data_ty->ndim, vdev, data_ty->span);
}

TVM_REGISTER_OP("relax.to_vdevice")
    .set_num_inputs(1)
    .set_attrs_type<ToVDeviceAttrs>()
    .add_argument("data", "Expr", "The input expression to be copied")
    .set_attr<FInferType>("FInferType", InferToVDeviceType)
    .set_attr<bool>("FPurity", true);

Expr MakeToVDevice(Expr data, VDevice dst_vdev) {
  static const Op& op = Op::Get("relax.to_vdevice");
  ffi::ObjectPtr<ToVDeviceAttrs> attrs = ffi::make_object<ToVDeviceAttrs>();
  attrs->dst_vdevice = dst_vdev;
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.to_vdevice", MakeToVDevice);
}

// hint_on_device

Type InferHintOnDeviceType(const Call& call, const BlockBuilder& ctx) {
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(call->args[0]->ty.defined());
  TensorType data_ty = GetUnaryInputTensorType(call, ctx);
  return data_ty;
}

TVM_REGISTER_OP("relax.hint_on_device")
    .set_num_inputs(1)
    .set_attrs_type<HintOnDeviceAttrs>()
    .add_argument("data", "Expr", "The input expression")
    .set_attr<FInferType>("FInferType", InferHintOnDeviceType)
    .set_attr<bool>("FPurity", true);

Expr MakeHintOnDevice(Expr data, Device device, ffi::String memory_scope = "global") {
  static const Op& op = Op::Get("relax.hint_on_device");
  ffi::ObjectPtr<HintOnDeviceAttrs> attrs = ffi::make_object<HintOnDeviceAttrs>();
  attrs->device_type = static_cast<int32_t>(device.device_type);
  attrs->index = device.device_id;
  attrs->memory_scope = memory_scope;
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("relax.op.hint_on_device", [](ffi::PackedArgs args, ffi::Any* ret) {
    if (args.size() == 3) {
      *ret = MakeHintOnDevice(args[0].cast<Expr>(), args[1].cast<Device>(),
                              args[2].cast<ffi::String>());
    } else {
      *ret = MakeHintOnDevice(args[0].cast<Expr>(), args[1].cast<Device>());
    }
  });
}

}  // namespace relax
}  // namespace tvm
