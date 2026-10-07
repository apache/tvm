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
#include <tvm/ir/expr_functor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/utils.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>

#include "../transform/utils.h"
#include "call_tir.h"
#include "op_common.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

TVM_FFI_STATIC_INIT_BLOCK() {
  CallTIRPackedAttrs::RegisterReflection();
  CallTIRWithGradAttrs::RegisterReflection();
  CallTIRInplaceAttrs::RegisterReflection();
  CallInplacePackedAttrs::RegisterReflection();
  ToVDeviceAttrs::RegisterReflection();
  HintOnDeviceAttrs::RegisterReflection();
}

bool EqualConstInt(const PrimExpr& lhs, int64_t value) {
  if (const auto* pvalue = lhs.as<IntImmNode>()) {
    return pvalue->value == value;
  }
  return false;
}

bool EqualCheck(const PrimExpr& lhs, const PrimExpr& rhs) {
  PrimExpr diff = lhs - rhs;
  if (const auto* pdiff = diff.as<IntImmNode>()) {
    return pdiff->value == 0;
  }
  tvm::sym::Analyzer ana;
  diff = ana->Simplify(diff);
  if (const auto* pdiff = diff.as<IntImmNode>()) {
    return pdiff->value == 0;
  }
  return false;
}

Type ReturnVoidType(const CallNode*) { return TupleType(ffi::Array<Type>()); }

Type ReturnAnyType(const CallNode*) { return AnyType(); }

Type InferTypeShapeOf(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  // use the Type of the argument
  auto arg_ty = GetType(call->args[0]);
  auto* tensor_ty = GetType(call->args[0]).as<TensorTypeNode>();
  TVM_FFI_ICHECK(tensor_ty) << "shape_of expects a tensor input, but received " << arg_ty
                            << "; use MatchCast if necessary";
  if (tensor_ty->ndim == kUnknownNDim) {
    return ShapeType(kUnknownNDim);
  }
  // if the tensor shape is a Relax var or omitted, do not try to construct a shape expr from it
  if (!tensor_ty->shape.has_value() || tensor_ty->shape.as<VarNode>()) {
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
  if (finfo->derive_func.has_value()) {
    // derive using custom derivation function.
    return finfo->derive_func.value()(call, ctx);
  } else {
    // directly return the normal value.
    return finfo->ret;
  }
}

void ValidateCallPurePacked(const CallNode* call) {
  TVM_FFI_CHECK(call->args.size() >= 1, TypeError)
      << "call_pure_packed expects a function argument";
  for (const ffi::Any* arg = call->args.GetArrayObj()->begin();
       arg != call->args.GetArrayObj()->end(); ++arg) {
    TVM_FFI_CHECK(
        *arg != nullptr && ffi::details::AnyUnsafe::CheckAnyViewStrict<Expr>(ffi::AnyView(*arg)),
        TypeError)
        << "call_pure_packed has an invalid value argument";
  }
  for (const ffi::Any* ty_arg = call->ty_args.GetArrayObj()->begin();
       ty_arg != call->ty_args.GetArrayObj()->end(); ++ty_arg) {
    TVM_FFI_CHECK(*ty_arg != nullptr &&
                      ffi::details::AnyUnsafe::CheckAnyViewStrict<Type>(ffi::AnyView(*ty_arg)),
                  TypeError)
        << "call_pure_packed has an invalid type argument";
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_pure_packed")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateCallPurePacked>())
      .signature(sig::arg("func",
                          "The first argument is the function being called. The rest are the "
                          "arguments to that function."),
                 sig::var_args("args"),
                 sig::var_ty_args("type_args", "Optional type arguments forwarded to the callee."))
      .set_attr<FInferTypeWithBuilder>("relax.FInferTypeWithBuilder", InferTypeCallPurePacked)
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallPurePacked(const Expr& callee, ffi::Array<Expr> args, const Attrs& attrs,
                        ffi::Array<Type> ty_args) {
  static const Op op = Op::Get("relax.call_pure_packed");
  ffi::Array<Expr> call_args = {callee};
  for (auto arg : args) {
    call_args.push_back(arg);
  }
  return Call::Unchecked(Type::Missing(), op, call_args, attrs, ty_args);
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
  Type ret = Type::Missing();
  if (finfo->derive_func.has_value()) {
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

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_inplace_packed")
      .signature(sig::arg("func",
                          "The first argument is the function being called. The rest are the "
                          "arguments to that function."),
                 sig::var_args("args"),
                 sig::var_ty_args("type_args", "Optional type arguments forwarded to the callee."),
                 sig::call_attrs<CallInplacePackedAttrs>())
      .set_attr<FInferTypeWithBuilder>("relax.FInferTypeWithBuilder", InferTypeCallInplacePacked)
      // Warning: considered pure, but it has the potential to create visible effects!
      // This should only be used if it has been *checked* that it is safe (no aliases, in-place
      // arguments will no longer be live) and the user believes the packed func to have no
      // side effects other than modifying the arguments specified as "inplace"
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallInplacePacked(Expr func, ffi::Array<Expr> args, ffi::Array<int64_t> inplace_indices,
                           ffi::Array<Type> ty_args) {
  ffi::ObjectPtr<CallInplacePackedAttrs> attrs = ffi::make_object<CallInplacePackedAttrs>();
  attrs->inplace_indices = ffi::Array<int64_t>(inplace_indices.begin(), inplace_indices.end());

  static const Op op = Op::Get("relax.call_inplace_packed");
  ffi::Array<Expr> call_args = {func};
  call_args.insert(call_args.end(), args.begin(), args.end());
  return Call::Unchecked(Type::Missing(), op, call_args, Attrs(attrs), ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_inplace_packed", MakeCallInplacePacked);
}

// Native TIRx signatures are translated only at explicit Relax call boundaries.
namespace {
Type TIRxPackedValueType(const Type& type, bool is_result) {
  if (is_result && IsVoidType(type)) return VoidType();
  if (const auto* scalar = type.as<PrimTypeNode>()) {
    DLDataType dtype = scalar->dtype;
    bool supported =
        dtype.lanes == 1 &&
        ((dtype.code == kDLBool && dtype.bits == 8) ||
         ((dtype.code == kDLInt || dtype.code == kDLUInt) && dtype.bits <= 64) ||
         (dtype.code == kDLFloat && (dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64)));
    TVM_FFI_CHECK(supported, TypeError)
        << "R.call_tir_packed does not support packed scalar type " << type;
    return type;
  }
  if (type.as<PointerTypeNode>()) return AnyType();
  if (!is_result) {
    if (const auto* tensor = type.as<tirx::TensorTypeNode>()) {
      bool default_layout =
          !tensor->layout.has_value() ||
          ffi::StructuralEqual()(tensor->layout.value(),
                                 tirx::TileLayoutNode::DefaultLayout(tensor->shape));
      TVM_FFI_CHECK(default_layout && tensor->allocated_addr.empty(), TypeError)
          << "A Relax-to-TIRx call cannot implicitly convert tensor layout or allocated address: "
          << type;
      TVM_FFI_CHECK(tensor->storage_scope.empty() || tensor->storage_scope == "global" ||
                        std::string(tensor->storage_scope).rfind("global.", 0) == 0,
                    TypeError)
          << "A Relax-to-TIRx call cannot implicitly convert tensor storage scope: " << type;
      ShapeExpr shape(
          tensor->shape.Map([](PrimExpr dim) { return prim::cast(PrimType::Int(64), dim); }));
      return TensorType(shape, tensor->dtype);
    }
  }
  TVM_FFI_THROW(TypeError) << "R.call_tir_packed does not support native "
                           << (is_result ? "return" : "parameter") << " type " << type;
  TVM_FFI_UNREACHABLE();
}

const tvm::FuncTypeNode* NativeTIRxSignature(const Expr& callee) {
  TVM_FFI_CHECK(callee.as<GlobalVarNode>(), TypeError)
      << "R.call_tir_packed requires a GlobalVar referring to a native TIRx function";
  const auto* signature = callee->ty.as<tvm::FuncTypeNode>();
  TVM_FFI_CHECK(signature, TypeError)
      << "R.call_tir_packed requires a native TIRx function signature, but received " << callee->ty;
  return signature;
}

Tuple InlineTIRxArguments(const BlockBuilder& ctx, const Expr& argument) {
  if (auto tuple = argument.as<Tuple>()) return tuple.value();
  Expr value = argument;
  while (auto var = value.as<Var>()) {
    auto bound = ctx->LookupBinding(var.value());
    if (!bound) break;
    value = bound.value();
    if (auto tuple = value.as<Tuple>()) return tuple.value();
  }
  const auto* type = argument->ty.as<TupleTypeNode>();
  TVM_FFI_CHECK(type, TypeError) << "R.call_tir_packed expects an argument tuple";
  ffi::Array<Expr> fields;
  for (size_t i = 0; i < type->fields.size(); ++i) {
    fields.push_back(TupleGetItem(argument, i));
  }
  return Tuple(fields);
}

// Scalar parameter identities live in the native function, not in FuncType.
// Use that context when available without changing the callee's native signature.
tvm::FuncType ContextualTIRxSignature(const BlockBuilder& ctx, const Expr& callee,
                                      const Tuple& arguments) {
  tvm::FuncType signature = ffi::GetRef<tvm::FuncType>(NativeTIRxSignature(callee));
  IRModule mod = ctx->GetContextIRModule();
  auto it = mod->functions.find(callee.as_or_throw<GlobalVar>());
  if (it == mod->functions.end()) return signature;
  const auto* function = (*it).second.as<tirx::FunctionNode>();
  TVM_FFI_CHECK(function, TypeError) << "A Relax-to-TIRx call must refer to a native TIRx function";
  TVM_FFI_CHECK(ffi::StructuralEqual()(signature, function->ty), TypeError)
      << "The TIRx GlobalVar signature does not match its function";
  auto substitutor = ffi::make_object<tvm::ExprMutator>();
  for (size_t i = 0; i < arguments->fields.size() && i < function->params.size(); ++i) {
    if (!function->params[i]->ty.as<PrimTypeNode>()) continue;
    Expr value = arguments->fields[i];
    while (auto var = value.as<Var>()) {
      auto binding = ctx->LookupBinding(var.value());
      if (!binding) break;
      value = binding.value();
    }
    if (auto prim = value.as<PrimExpr>();
        prim && prim.value().ty() == function->params[i]->ty.as_or_throw<PrimType>()) {
      substitutor->VarRemapSet(function->params[i], prim.value());
    }
  }
  return substitutor->Mutate(signature).as_or_throw<UnchangedOr<tvm::FuncType>>().ValueOrUnchanged(
      signature);
}

void CheckTIRxCarrier(const Type& native, const Type& actual,
                      bool allow_storage_specialization = false) {
  if (native.as<PointerTypeNode>()) {
    TVM_FFI_CHECK(!actual.as<PrimTypeNode>() && !actual.as<StringTypeNode>(), TypeError)
        << "A Relax-to-TIRx pointer argument requires a handle-compatible object or Any, received "
        << actual;
  }
  if (const auto* tensor = native.as<tirx::TensorTypeNode>()) {
    ffi::String native_scope = tensor->storage_scope.empty() ? "global" : tensor->storage_scope;
    const auto* argument = actual.as<TensorTypeNode>();
    ffi::Optional<ffi::String> actual_scope;
    if (argument && argument->vdevice.has_value()) {
      const auto& scope = argument->vdevice.value()->memory_scope;
      actual_scope = scope.empty() ? ffi::String("global") : scope;
    }
    // High-level DPS calls may precede SpecializePrimFuncBasedOnCallSite.
    // The packed bridge is after that phase and requires matching carriers.
    if (native_scope != "global" || (actual_scope && !allow_storage_specialization)) {
      TVM_FFI_CHECK(actual_scope && actual_scope.value() == native_scope, TypeError)
          << "A Relax-to-TIRx call requires matching native and VDevice memory scopes; native "
          << native_scope << ", Relax " << actual;
    }
  }
}

void CheckTIRxArguments(const tvm::FuncType& signature, const ffi::Array<Expr>& arguments,
                        const BlockBuilder& ctx) {
  FuncType adapted = TIRxToRelaxFuncType(signature);
  auto params = adapted->params.value();
  for (size_t i = 0; i < params.size() && i < arguments.size(); ++i) {
    CheckTIRxCarrier(signature->arg_types[i], arguments[i]->ty);
    const auto* expected = params[i].as<TensorTypeNode>();
    const auto* actual = arguments[i]->ty.as<TensorTypeNode>();
    if (!expected || !actual) continue;
    // Unknown Relax metadata defers to the native packed checks.  Keep every
    // known constraint, so partial information never hides a contradiction.
    if (actual->IsUnknownNdim()) {
      params.Set(
          i, TensorType(actual->IsUnknownDtype() ? std::nullopt : expected->dtype, kUnknownNDim));
    } else if (actual->IsUnknownDtype()) {
      params.Set(i, TensorType(expected->shape.value(), std::nullopt));
    }
  }
  adapted = FuncType(params, adapted->ret, false);
  DeriveCallRetType(adapted, Call::Unchecked(Type::Missing(), Var("callee", adapted), arguments),
                    ctx);
}

void CheckFreshTIRxDestination(const tirx::TensorTypeNode* tensor) {
  sym::Analyzer analyzer;
  TVM_FFI_CHECK(!analyzer->CanProve(tensor->elem_offset != 0), TypeError)
      << "R.call_tir allocates fresh destinations with zero element offset";
  if (!tensor->strides.empty()) {
    PrimExpr compact_stride =
        IntImm(tensor->shape.empty() ? PrimType::Int(64)
                                     : tensor->shape.back()->ty.as_or_throw<PrimType>(),
               1);
    for (size_t i = tensor->shape.size(); i > 0; --i) {
      TVM_FFI_CHECK(!analyzer->CanProve(tensor->strides[i - 1] != compact_stride), TypeError)
          << "R.call_tir allocates compact destinations; incompatible native stride at dimension "
          << i - 1;
      compact_stride = compact_stride * tensor->shape[i - 1];
    }
  }
}

void ValidateCallTIRPacked(const CallNode* call) {
  TVM_FFI_CHECK(call->args.size() == 2 && call->ty_args.empty(), TypeError)
      << "R.call_tir_packed expects a callee and an argument tuple, without type arguments";
  TVM_FFI_CHECK(call->attrs.as<CallTIRPackedAttrs>(), TypeError)
      << "R.call_tir_packed requires CallTIRPackedAttrs";
  const auto* native = NativeTIRxSignature(call->args[0]);
  const auto* arguments = call->args[1]->ty.as<TupleTypeNode>();
  TVM_FFI_CHECK(arguments, TypeError) << "R.call_tir_packed expects an argument tuple";
  TVM_FFI_CHECK_EQ(native->arg_types.size(), arguments->fields.size(), TypeError)
      << "R.call_tir_packed requires one argument for each native parameter";
  for (size_t i = 0; i < native->arg_types.size(); ++i) {
    if (native->arg_types[i].as<PointerTypeNode>()) {
      const Type& argument = arguments->fields[i];
      TVM_FFI_CHECK(!argument.as<PrimTypeNode>() && !argument.as<StringTypeNode>(), TypeError)
          << "R.call_tir_packed pointer parameter " << i
          << " requires a handle-compatible object or Any, but received " << argument;
    }
    TVM_FFI_CHECK(!arguments->fields[i].as<distributed::DTensorTypeNode>(), TypeError)
        << "R.call_tir_packed requires distributed tensors to be explicitly lowered";
  }
  ffi::Array<Expr> args =
      arguments->fields.Map([](const Type& type) -> Expr { return Var("argument", type); });
  CheckTIRxArguments(ffi::GetRef<tvm::FuncType>(native), args, BlockBuilder::Create(std::nullopt));
}

Type InferTypeCallTIRPacked(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 2, TypeError)
      << "R.call_tir_packed expects a callee and an argument tuple";
  return TIRxPackedValueType(NativeTIRxSignature(call->args[0])->ret_type, true);
}

Expr NormalizeCallTIRPacked(const BlockBuilder& ctx, Call call) {
  NativeTIRxSignature(call->args[0]);
  Tuple arguments = InlineTIRxArguments(ctx, call->args[1]);
  CheckTIRxArguments(ContextualTIRxSignature(ctx, call->args[0], arguments), arguments->fields,
                     ctx);
  if (!arguments.same_as(call->args[1])) {
    call.CopyOnWrite()->args.Set(1, arguments);
  }
  return call;
}
}  // namespace

FuncType TIRxToRelaxFuncType(const tvm::FuncType& type) {
  return FuncType(
      type->arg_types.Map([](const Type& arg) { return TIRxPackedValueType(arg, false); }),
      TIRxPackedValueType(type->ret_type, true), false);
}

Expr MakeCallTIRPacked(Expr func, Tuple args, bool is_pure) {
  auto attrs = ffi::make_object<CallTIRPackedAttrs>();
  attrs->is_pure = is_pure;
  return Call::Unchecked(Type::Missing(), Op::Get("relax.call_tir_packed"), {func, args},
                         Attrs(attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_tir_packed")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateCallTIRPacked>())
      .signature(sig::arg("func", "The native TIRx function."),
                 sig::arg("args", "Every native argument, in parameter order."),
                 sig::call_attrs<CallTIRPackedAttrs>())
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCallTIRPacked>())
      .set_attr<FNormalize>("FNormalize", NormalizeCallTIRPacked)
      .set_attr<bool>("FPurity", false);
  ffi::reflection::GlobalDef().def("relax.op.call_tir_packed", MakeCallTIRPacked);
}

// call_tir

/* If possible, infer a legal value of `arg_ty`
 *
 * The `R.call_tir` operator and its variants accept an `arg_ty`
 * parameter, which specifies the shape of the tensor or tensors
 * returned by a tirx::Function.  This output shape must be compatible with
 * the shape defined by the tirx::Function's signature.
 *
 * For dynamic shapes, it is not always possible to infer the output
 * of a TIR tirx::Function from its inputs.  For example, a tirx::Function that
 * accepts input buffer `T.Tensor([16], "float32")` and output buffer
 * `T.Tensor([M, N], "float32")` infers the values of `M` and `N` from
 * the shape of the provided output buffer.
 *
 * If the arguments provided are not compatible with the tirx::Function's
 * signature, an error will be raised.  If the arguments are
 * compatible with the tirx::Function's signature, but are not sufficient to
 * determine the output's Type, then `std::nullopt` will be returned.
 *
 * \param func_ty The Type of the TIR callee.
 * \param arg_ty The Type of the argument tuple.
 * \param opt_inplace_indices For `R.call_tir_inplace`, an array of
 *     indices indicating which outputs are constructed from in-place
 *     mutation of the inputs.  See
 *     `CallTIRInplaceAttrs::inplace_indices` for more details.
 *
 * \return The `arg_ty`, if it can be inferred from the arguments.
 *     Otherwise, std::nullopt.
 */
ffi::Optional<Type> InferCallTIROutputTypeFromArguments(
    Type func_ty, Type arg_ty, ffi::Optional<ffi::Array<int64_t>> opt_inplace_indices) {
  auto opt_native_ty = func_ty.as<tvm::FuncType>();
  TVM_FFI_CHECK(opt_native_ty && IsVoidType(opt_native_ty.value()->ret_type), TypeError)
      << "R.call_tir requires a native TIRx function with a void result; "
      << "use R.call_tir_packed for a direct result. Received " << func_ty;
  auto opt_callee_ty = ffi::Optional<FuncType>(TIRxToRelaxFuncType(opt_native_ty.value()));
  TVM_FFI_CHECK(opt_callee_ty, TypeError)
      << "The first argument to `R.call_tir` must be a function, "
      << "but instead received argument of type " << func_ty;
  auto callee_ty = opt_callee_ty.value();

  TVM_FFI_CHECK(callee_ty->params.has_value(), ValueError)
      << "The first argument to `R.call_tir` must be a function "
      << "with known argument types.  "
      << "However, the first argument was of type " << callee_ty;
  auto callee_params = callee_ty->params.value();

  const TupleTypeNode* args = arg_ty.as<TupleTypeNode>();
  TVM_FFI_CHECK(args, TypeError) << "The second argument to `R.call_tir` must be a tuple, "
                                 << "but instead received expression of type " << arg_ty;

  // R.call_tir expects the tirx::Function to have two groups of arguments.
  //
  // 1. Input arguments that are explicitly provided as Relax arguments.
  // 2. Output tensor arguments.
  //
  // In order to determine the return type of `R.call_tir`, we must
  // identify the tirx::Function arguments that will be in group (2).
  size_t num_input_arguments = args->fields.size();
  const auto& native_params = opt_native_ty.value()->arg_types;
  for (size_t i = 0; i < num_input_arguments && i < native_params.size(); ++i) {
    if (!args->fields[i].as<distributed::DTensorTypeNode>()) {
      CheckTIRxCarrier(native_params[i], args->fields[i], true);
    }
  }
  for (size_t i = num_input_arguments; i < native_params.size(); ++i) {
    const auto* destination = native_params[i].as<tirx::TensorTypeNode>();
    TVM_FFI_CHECK(destination, TypeError)
        << "R.call_tir destination parameter " << i << " must have a native tensor type";
    CheckFreshTIRxDestination(destination);
  }

  TVM_FFI_CHECK_LE(args->fields.size(), callee_params.size(), ValueError)
      << "R.call_tir attempted to call a function using " << args->fields.size()
      << " explicit arguments.  "
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
    ffi::Array<Type> dummy_ret(callee_params.begin() + num_input_arguments, callee_params.end());

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

  ffi::Array<Expr> dummy_args =
      args->fields.Map([](const Type& ty) -> Expr { return Var("dummy_arg", ty); });

  Type derived_ret_ty = DeriveCallRetType(
      dummy_callee_ty,
      Call::Unchecked(Type::Missing(), Var("dummy_callee", dummy_callee_ty), dummy_args),
      BlockBuilder::Create(std::nullopt));
  if (derived_ret_ty.as<MissingType>().has_value()) {
    return std::nullopt;
  }

  return derived_ret_ty;
}

void CheckTIRxDestinations(const tvm::FuncType& signature, const Type& arguments,
                           const Type& result,
                           const ffi::Optional<ffi::Array<int64_t>>& inplace_indices) {
  ffi::Array<Type> outputs;
  if (const auto* tuple = result.as<TupleTypeNode>()) {
    outputs = tuple->fields;
  } else {
    outputs.push_back(result);
  }
  size_t destination = arguments.as_or_throw<TupleType>()->fields.size();
  if (inplace_indices) {
    TVM_FFI_CHECK_EQ(inplace_indices.value().size(), outputs.size(), ValueError)
        << "There must be an in-place index specified for each output";
    for (int64_t index : inplace_indices.value()) {
      TVM_FFI_CHECK(index >= -1 && index < static_cast<int64_t>(destination), ValueError)
          << "In-place index is out of range";
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (inplace_indices && inplace_indices.value()[i] >= 0) continue;
    if (destination < signature->arg_types.size() &&
        !outputs[i].as<distributed::DTensorTypeNode>()) {
      CheckTIRxCarrier(signature->arg_types[destination], outputs[i], true);
    }
    ++destination;
  }
}

Type InferTypeCallTIR(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exactly 1 output type.";
  }
  TVM_FFI_ICHECK(call->args[0]->IsInstance<GlobalVarNode>())
      << "R.call_tir expects the first argument to be a GlobalVar referring to a TIR "
         "tirx::Function. "
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
  TVM_FFI_ICHECK_EQ(call->args.size(), 2)
      << "Operation " << call->op << " expects two arguments [callee, arg_tuple], "
      << "but " << call << " has " << call->args.size() << " arguments.";

  auto callee = call->args[0];
  TVM_FFI_ICHECK(callee->ty.as<tvm::FuncTypeNode>())
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

  TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1)
      << "R.call_tir should have exactly one `ty_args` parameter, "
      << "which defines the output of the tirx::Function.";

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

  tvm::FuncType signature = ContextualTIRxSignature(ctx, callee, new_arg_tuple);
  ffi::Optional<ffi::Array<int64_t>> inplace_indices;
  if (const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>()) {
    inplace_indices = attrs->inplace_indices;
  }
  CheckTIRxDestinations(signature, new_arg_tuple->ty, call->ty_args[0], inplace_indices);
  if (auto inferred =
          InferCallTIROutputTypeFromArguments(signature, new_arg_tuple->ty, inplace_indices)) {
    TVM_FFI_CHECK(IsBaseOf(inferred.value(), call->ty_args[0]), TypeError)
        << "R.call_tir out_ty is incompatible with the native function and scalar arguments: "
        << inferred.value() << " versus " << call->ty_args[0];
  }

  if (!new_arg_tuple.same_as(arg_tuple)) {
    auto new_args = call->args;
    new_args.Set(1, new_arg_tuple);
    call.CopyOnWrite()->args = new_args;
  }

  return call;
}

void ValidateCallTIR(const CallNode* call) {
  // This function is used for validation of `relax.call_tir`,
  // along with the variants `relax.call_tir_with_grad` and
  // `relax.call_tir_inplace`.  Therefore, all error messages should
  // be written in terms of `call->op`, and should not explicitly
  // reference the `relax.call_tir` operator.`

  auto callee = call->args[0];
  Expr arg_tuple = call->args[1];

  auto opt_inplace_indices = [&]() -> ffi::Optional<ffi::Array<int64_t>> {
    if (const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>()) {
      return attrs->inplace_indices;
    } else {
      return std::nullopt;
    }
  }();

  Type explicit_ty = call->ty_args[0];
  CheckTIRxDestinations(callee->ty.as_or_throw<tvm::FuncType>(), arg_tuple->ty, explicit_ty,
                        opt_inplace_indices);
  auto inferred_ty =
      InferCallTIROutputTypeFromArguments(GetType(callee), GetType(arg_tuple), opt_inplace_indices);
  if (inferred_ty.has_value()) {
    TVM_FFI_CHECK(IsBaseOf(inferred_ty.value(), explicit_ty), TypeError)
        << "The `out_ty` argument for R.call_tir must be compatible with the tirx::Function.  "
        << "However, the tirx::Function's signature implies that the output should be "
        << inferred_ty << ", but the `out_ty` argument was " << explicit_ty;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_tir")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateCallTIR>())
      .signature(sig::arg("func", "The destination-passing-style function."),
                 sig::arg("args", "The input arguments."),
                 sig::ty_arg("out_type", "The output type."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCallTIR>())
      .set_attr<FNormalize>("FNormalize", NormalizeCallTIR)
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallTIR(Expr func, Tuple args, ffi::Array<TensorType> out_ty_list) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty = Type::Missing();
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op op = Op::Get("relax.call_tir");
  return Call::Unchecked(Type::Missing(), op, {func, args}, {}, {out_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_tir", MakeCallTIR);

  // call_tir_with_grad

  OpDef("relax.call_tir_with_grad")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateCallTIR>())
      .signature(sig::arg("func", "The destination-passing-style function."),
                 sig::arg("args", "The input arguments."),
                 sig::ty_arg("out_type", "The output type."),
                 sig::call_attrs<CallTIRWithGradAttrs>())
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCallTIR>())
      .set_attr<FNormalize>("FNormalize", NormalizeCallTIR)
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallTIRWithGrad(Expr func, Tuple args, ffi::Array<TensorType> out_ty_list,
                         ffi::String te_grad_name, ffi::Map<ffi::String, ffi::Any> te_grad_kwargs) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir_with_grad should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty = Type::Missing();
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  ffi::ObjectPtr<CallTIRWithGradAttrs> attrs = ffi::make_object<CallTIRWithGradAttrs>();
  attrs->te_grad_name = te_grad_name;
  attrs->te_grad_kwargs = te_grad_kwargs;

  static const Op op = Op::Get("relax.call_tir_with_grad");
  return Call::Unchecked(Type::Missing(), op, {func, args}, Attrs(attrs), {out_ty});
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

    if (!tinfo_output || !tinfo_output->shape.has_value() || tinfo_output->IsUnknownDtype()) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The output type for an in-place mutation must be a tensor "
          << "with a defined shape and dtype, "
          << "but output " << i_output << " has type " << ty_output;
    }

    auto ty_input = GetType(call_args->fields[i_input]);
    auto tinfo_input = ty_input.as<TensorTypeNode>();

    if (!tinfo_input ||
        (tinfo_output->IsUnknownDtype() || tinfo_output->dtype != tinfo_input->dtype) ||
        (!tinfo_input->shape.has_value() ||
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

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_tir_inplace")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateCallTIR>())
      .signature(sig::arg("func", "The destination-passing-style function."),
                 sig::arg("args", "The input arguments."),
                 sig::ty_arg("out_type", "The output type."),
                 sig::call_attrs<CallTIRInplaceAttrs>())
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCallTIR>())
      .set_attr<FNormalize>("FNormalize", NormalizeCallTIRInPlace)
      // Warning: considered pure, but it has the potential to create visible effects!
      // This should only be used if it has been *checked* that it is safe (no aliases, in-place
      // arguments will no longer be live)
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallTIRInplace(Expr func, Tuple args, ffi::Array<int64_t> inplace_indices,
                        ffi::Array<TensorType> out_ty_list) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_tir should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  ffi::ObjectPtr<CallTIRInplaceAttrs> attrs = ffi::make_object<CallTIRInplaceAttrs>();
  attrs->inplace_indices = ffi::Array<int64_t>(inplace_indices.begin(), inplace_indices.end());

  Type out_ty = Type::Missing();
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op op = Op::Get("relax.call_tir_inplace");
  return Call::Unchecked(Type::Missing(), op, {func, args}, Attrs(attrs), {out_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_tir_inplace", MakeCallTIRInplace);
}

// call_dps_packed

Type InferTypeCallDPSPacked(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exact 1 output type.";
  }
  return call->ty_args[0];
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_dps_packed")
      .signature(sig::arg("func", "The destination-passing-style function."),
                 sig::arg("args", "The input arguments."),
                 sig::ty_arg("out_type", "The output type."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCallDPSPacked>())
      // technically, an impure op could be used with this, but there is
      // little reason to use DPS with an impure op
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallDPSPacked(Expr func, Tuple args, ffi::Array<TensorType> out_ty_list) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_dps_packed should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty = Type::Missing();
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op op = Op::Get("relax.call_dps_packed");
  return Call::Unchecked(Type::Missing(), op, {func, args}, {}, {out_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_dps_packed", MakeCallDPSPacked);
}

// call_py_func

Type InferTypeCallPyFunc(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exact 1 output type.";
  }
  return call->ty_args[0];
}

void ValidateCallPyFunc(const CallNode* call) {
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
      << "However, the argument tuple is " << arg_tuple << ", which is neither an in-line tuple, "
      << "nor a variable binding that may be normalized to an in-line tuple.";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_py_func")
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateCallPyFunc>())
      .signature(sig::arg("func_name", "The name of the Python function to call."),
                 sig::arg("args", "The input arguments."),
                 sig::ty_arg("out_type", "The output type."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCallPyFunc>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallPyFunc(StringImm func_name, Tuple args, ffi::Array<TensorType> out_ty_list) {
  for (const TensorType& ty : out_ty_list) {
    const auto* shape = ty->shape.as<ShapeExprNode>();
    TVM_FFI_ICHECK(shape != nullptr)
        << "out_ty of call_py_func should have defined ShapeExpr as shape. "
           "However, one given type information is "
        << ty;
  }

  Type out_ty = Type::Missing();
  if (out_ty_list.size() == 1) {
    out_ty = out_ty_list[0];
  } else {
    out_ty = TupleType({out_ty_list.begin(), out_ty_list.end()});
  }

  static const Op op = Op::Get("relax.call_py_func");
  return Call::Unchecked(Type::Missing(), op, {func_name, args}, {}, {out_ty});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_py_func", MakeCallPyFunc);
}

// call builtin
Type InferTypeCallBuiltinWithCtx(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->ty_args.size() == 0) {
    // by default return void.
    return TupleType(ffi::Array<Type>());
  } else {
    TVM_FFI_ICHECK_EQ(call->ty_args.size(), 1);
    return call->ty_args[0];
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_builtin_with_ctx")
      .signature(sig::arg("func", "The builtin packed func."),
                 sig::arg("args", "The input arguments."),
                 sig::var_ty_args("out_type", "Optional output type; omitted for void."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCallBuiltinWithCtx>())
      // Most builtins are pure, but some are not, like `vm.builtin.attention_kv_cache_append`
      .set_attr<bool>("FPurity", false);
}

Expr MakeCallBuiltinWithCtx(Expr func, Tuple args, ffi::Array<Type> ty_args) {
  static const Op op = Op::Get("relax.call_builtin_with_ctx");
  return Call::Unchecked(Type::Missing(), op, {func, args}, Attrs(), ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.call_builtin_with_ctx", MakeCallBuiltinWithCtx);

  OpDef("relax.null_value")
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnAnyType>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeCallNullValue() {
  static const Op op = Op::Get("relax.null_value");
  return Call::Unchecked(Type::Missing(), op, {}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.null_value", MakeCallNullValue);

  // print

  OpDef("relax.print")
      .signature(
          sig::arg("format",
                   "The first value is Python-style format string to use to print. The others "
                   "are values to print"),
          sig::var_args("args"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnVoidType>())
      .set_attr<FCallPacked>("FCallPacked", "relax.run.print")
      .set_attr<bool>("FPurity", false);
}

Expr MakePrint(ffi::Array<Expr> vals, StringImm format) {
  ffi::Array<Expr> params;
  params.push_back(format);
  for (const auto val : vals) {
    params.push_back(val);
  }
  static const Op op = Op::Get("relax.print");
  return Call::Unchecked(Type::Missing(), op, params);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.print", MakePrint);
}

// assert_op

// can't actually name it assert or else Python will consider it a syntax error

Type InferAssertType(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
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
  return ReturnVoidType(call.get());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.assert_op")
      .signature(
          sig::arg("condition",
                   "The first value is used as the assertion condition. The second value is "
                   "Python-style format string to use for displaying an error message, if the "
                   "assert fails. The others are used as format arguments if there is an error."),
          sig::var_args("args"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferAssertType>())
      .set_attr<FCallPacked>("FCallPacked", "relax.run.assert_op")
      .set_attr<bool>("FPurity", false);
}

Expr MakeAssertOp(Expr condition, ffi::Array<Expr> vals, StringImm format) {
  static const Op op = Op::Get("relax.assert_op");
  ffi::Array<Expr> args = {condition};
  args.push_back(format);
  for (auto val : vals) {
    args.push_back(val);
  }
  return Call::Unchecked(Type::Missing(), op, args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.assert_op", MakeAssertOp);

  // make_closure

  OpDef("relax.make_closure")
      .signature(sig::arg("func", "The closure."), sig::arg("args", "The captured variables."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnAnyType>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeClosure(Expr func, Tuple args) {
  static const Op op = Op::Get("relax.make_closure");
  return Call::Unchecked(Type::Missing(), op, {func, args}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.make_closure", MakeClosure);
}

// invoke_closure

Type InferTypeInvokeClosure(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->ty_args.empty()) {
    return AnyType();
  } else if (call->ty_args.size() == 1) {
    return call->ty_args[0];
  } else {
    return TupleType(call->ty_args);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.invoke_closure")
      .signature(sig::arg("closure", "The VMClosure."), sig::arg("args", "The captured variables."),
                 sig::var_ty_args("out_types",
                                  "Zero or more output types; multiple entries form a tuple."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeInvokeClosure>())
      // Not all closures are pure. Use invoke_pure_closure for specifying purity
      .set_attr<bool>("FPurity", false);
}

Expr InvokeClosure(Expr closure, Tuple args, ffi::Array<Type> ty_args) {
  static const Op op = Op::Get("relax.invoke_closure");
  return Call::Unchecked(Type::Missing(), op, {closure, args}, {}, ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.invoke_closure", InvokeClosure);

  // invoke_pure_closure

  OpDef("relax.invoke_pure_closure")
      .signature(sig::arg("closure", "The VMClosure."), sig::arg("args", "The captured variables."),
                 sig::var_ty_args("out_types",
                                  "Zero or more output types; multiple entries form a tuple."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeInvokeClosure>())
      .set_attr<bool>("FPurity", true);
}

Expr InvokePureClosure(Expr closure, Tuple args, ffi::Array<Type> ty_args) {
  static const Op op = Op::Get("relax.invoke_pure_closure");
  return Call::Unchecked(Type::Missing(), op, {closure, args}, {}, ty_args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.invoke_pure_closure", InvokePureClosure);

  // shape_of

  OpDef("relax.shape_of")
      .signature(sig::arg("input", "The input expression"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeShapeOf>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeShapeOf(Expr expr) {
  static const Op op = Op::Get("relax.shape_of");
  return Call::Unchecked(Type::Missing(), op, {expr}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.shape_of", MakeShapeOf);
}

// size

Type InferTypeSize(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  auto arg_ty = GetType(call->args[0]);
  auto* tensor_ty = GetType(call->args[0]).as<TensorTypeNode>();
  TVM_FFI_ICHECK(tensor_ty) << "size expects a tensor input, but received " << arg_ty
                            << "; use MatchCast if necessary";
  return TensorType(ShapeExpr(ffi::Array<PrimExpr>{}), PrimType::Int(64));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.size")
      .signature(sig::arg("input", "The input tensor"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeSize>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeSize(Expr expr) {
  static const Op op = Op::Get("relax.size");
  return Call::Unchecked(Type::Missing(), op, {expr}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.size", MakeSize);
}

// tensor_to_shape

Type ReturnTensorToShapeType(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(!call->args[0]->ty.as<MissingType>().has_value());
  const auto* tensor_ty = GetTypeAs<TensorTypeNode>(call->args[0]);
  TVM_FFI_ICHECK(tensor_ty);
  TVM_FFI_ICHECK_EQ(tensor_ty->ndim, 1)
      << "relax.tensor_to_shape expected argument to be 1-d, "
      << "but " << call << " has argument " << call->args[0] << " with type " << call->args[0]->ty;

  if (tensor_ty->shape.has_value()) {
    ShapeExpr shape_expr = tensor_ty->shape.value().as_or_throw<ShapeExpr>();
    const IntImmNode* ndim = shape_expr->values[0].as<IntImmNode>();
    if (ndim) {
      return ShapeType(ndim->value.as<int>().value());
    }
  }
  return ShapeType(kUnknownNDim);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.tensor_to_shape")
      .signature(sig::arg("input", "The input expression"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnTensorToShapeType>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeTensorToShape(Expr expr) {
  static const Op op = Op::Get("relax.tensor_to_shape");
  return Call::Unchecked(Type::Missing(), op, {expr}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.tensor_to_shape", MakeTensorToShape);
}

// shape_to_tensor
Type ReturnShapeToTensorType(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(!call->args[0]->ty.as<MissingType>().has_value());
  const auto* ty = GetTypeAs<ShapeTypeNode>(call->args[0]);
  TVM_FFI_ICHECK(ty);
  int32_t ndim = ty->ndim;
  return TensorType(ShapeExpr({PrimExpr(ndim)}), PrimType::Int(64));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.shape_to_tensor")
      .signature(sig::arg("input", "The input expression"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnShapeToTensorType>())
      .set_attr<FCallPacked>("FCallPacked", "relax.run.shape_to_tensor")
      .set_attr<bool>("FPurity", true);
}

Expr MakeShapeToTensor(Expr expr) {
  static const Op op = Op::Get("relax.shape_to_tensor");
  return Call::Unchecked(Type::Missing(), op, {expr}, {}, {});
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
  PrimType out_dtype = PrimType::Void();
  if (const auto* dtype_node = call->args[1].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = ffi::GetRef<DataTypeImm>(dtype_node);
    out_dtype = PrimType(dtype_imm->value);
  }
  int64_t vdevice_index = -1;
  if (const auto* int_imm = call->args[2].as<IntImmNode>()) {
    vdevice_index = int_imm->value.as<int>().value();
  }
  auto vdevice = GetGlobalVDevice(ctx->GetContextIRModule(), vdevice_index);

  if (vdevice.has_value()) {
    return TensorType(call->args[0], out_dtype, vdevice.value());
  }
  return TensorType(call->args[0], out_dtype);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.builtin.alloc_tensor")
      .signature(
          sig::arg("shape", "The shape of the tensor to allocate."),
          sig::arg("dtype", "The dtype of the tensor to allocate."),
          sig::arg<IntExpr>("runtime_device_index",
                            "The device index indicating on which device the tensor is to be "
                            "allocated at runtime. Index -1 is reserved for the host device."),
          sig::arg("storage_scope",
                   "The storage scope of the storage to allocate. Default is global."),
          sig::var_ty_args("out_type", "Optional output type used by allocation rewrites."))
      .set_attr<FInferTypeWithBuilder>("relax.FInferTypeWithBuilder", InferTypeAllocateTensor)
      // memory allocation isn't considered a "visible effect" as far as purity is concerned
      .set_attr<bool>("FPurity", true)
      .set_attr<bool>("TAllocator", true);
}

Expr MakeAllocTensor(Expr shape, DataTypeImm dtype, PrimExpr runtime_device_index,
                     StringImm storage_scope) {
  static const Op op = Op::Get("relax.builtin.alloc_tensor");
  return Call::Unchecked(Type::Missing(), op, {shape, dtype, runtime_device_index, storage_scope},
                         Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.builtin.alloc_tensor", MakeAllocTensor);

  // memory planning alloc_storage

  OpDef("relax.memory.alloc_storage")
      .signature(
          sig::arg("total_space", "The total space of the storage to allocate."),
          sig::arg(
              "virtual_device_index",
              "The virtual device index indicating on which device the storage is to be allocated, "
              "Index -1 is reserved for the host device."),
          sig::arg("storage_scope",
                   "The storage scope of the storage to allocate. Default is global."),
          sig::arg("dtype", "The dtype of the tensor to allocate."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnAnyType>())
      // memory allocation isn't considered a "visible effect" as far as purity is concerned
      .set_attr<bool>("FPurity", true)
      .set_attr<bool>("TAllocator", true);
}

Expr MakeAllocStorage(Expr size, PrimExpr virtual_device_index, StringImm storage_scope,
                      DataTypeImm dtype) {
  static const Op op = Op::Get("relax.memory.alloc_storage");
  return Call::Unchecked(Type::Missing(), op, {size, virtual_device_index, storage_scope, dtype},
                         Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.alloc_storage", MakeAllocStorage);
}

// memory planning alloc_tensor

Type InferTypeMemAllocTensor(const Call& call, const BlockBuilder& ctx) {
  TVM_FFI_ICHECK(GetTypeAs<ShapeTypeNode>(call->args[2]))
      << "must be a Expr of ShapeType, but got " << call->args[1]->GetTypeKey();
  PrimType out_dtype = PrimType::Void();
  if (const auto* dtype_node = call->args[3].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = ffi::GetRef<DataTypeImm>(dtype_node);
    out_dtype = PrimType(dtype_imm->value);
  }

  if (call->args.size() == 5) {
    int64_t vdevice_index = -1;
    if (const auto* int_imm = call->args[4].as<IntImmNode>()) {
      vdevice_index = int_imm->value.as<int>().value();
    }
    auto vdevice = GetGlobalVDevice(ctx->GetContextIRModule(), vdevice_index);
    if (vdevice.has_value()) {
      return TensorType(call->args[2], out_dtype, vdevice.value());
    }
  }

  return TensorType(call->args[2], out_dtype);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.memory.alloc_tensor")
      .signature(
          sig::arg("storage", "The storage to allocate the tensor to."),
          sig::arg<IntExpr>("offset", "Storage offset to allocate the tensor."),
          sig::arg("shape", "The shape of the tensor to allocate."),
          sig::arg("dtype", "The dtype of the tensor to allocate."),
          sig::arg<IntExpr>("runtime_device_index",
                            "The device index indicating on which device the tensor is to be "
                            "allocated at runtime. Index -1 is reserved for the host device."))
      .set_attr<FInferTypeWithBuilder>("relax.FInferTypeWithBuilder", InferTypeMemAllocTensor)
      // memory allocation isn't considered a "visible effect" as far as purity is concerned
      .set_attr<bool>("FPurity", true)
      .set_attr<bool>("TAllocator", true);
}

Expr MakeMemAllocTensor(Expr storage, PrimExpr offset, Expr shape, DataTypeImm dtype,
                        PrimExpr virtual_device_index) {
  static const Op op = Op::Get("relax.memory.alloc_tensor");
  return Call::Unchecked(Type::Missing(), op, {storage, offset, shape, dtype, virtual_device_index},
                         Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "relax.op.memory.alloc_tensor", [](ffi::PackedArgs args, ffi::Any* ret) {
        if (args.size() == 5) {
          *ret = MakeMemAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimExpr>(),
                                    args[2].cast<Expr>(), args[3].cast<DataTypeImm>(),
                                    args[4].cast<PrimExpr>());
        } else {
          *ret = MakeMemAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimExpr>(),
                                    args[2].cast<Expr>(), args[3].cast<DataTypeImm>(),
                                    IntImm::Int64(0));
        }
      });

  // memory planning kill_storage

  OpDef("relax.memory.kill_storage")
      .signature(sig::arg("storage", "The storage to be killed."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnVoidType>())
      // We mark this as impure so it wouldn't be removed by "remove_all_unused"
      .set_attr<bool>("FPurity", false);
}

Expr MakeMemKillStorage(Expr storage) {
  static const Op op = Op::Get("relax.memory.kill_storage");
  return Call::Unchecked(Type::Missing(), op, {storage}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.kill_storage", MakeMemKillStorage);

  // memory planning kill_tensor

  OpDef("relax.memory.kill_tensor")
      .signature(sig::arg("tensor", "The tensor to be killed."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnVoidType>())
      // We mark this as impure so it wouldn't be removed by "remove_all_unused"
      .set_attr<bool>("FPurity", false);
}

Expr MakeMemKillTensor(Expr tensor) {
  static const Op op = Op::Get("relax.memory.kill_tensor");
  return Call::Unchecked(Type::Missing(), op, {tensor}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.kill_tensor", MakeMemKillTensor);

  // vm alloc_storage

  OpDef("relax.vm.alloc_storage")
      .signature(sig::arg("size", "The size of the storage to allocate."),
                 sig::arg<IntExpr>("runtime_device_index",
                                   "The device index indicating on which device the tensor is "
                                   "to be allocated at runtime."),
                 sig::arg("dtype", "The dtype of the tensor to allocate."),
                 sig::arg("storage_scope",
                          "The storage scope of the storage to allocate. Default is global."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnAnyType>())
      // memory allocation isn't considered a "visible effect" as far as purity is concerned
      .set_attr<bool>("FPurity", true)
      .set_attr<bool>("TAllocator", true);
}

Expr MakeVMAllocStorage(Expr size, PrimExpr runtime_device_index, DataTypeImm dtype,
                        StringImm storage_scope) {
  static const Op op = Op::Get("relax.vm.alloc_storage");
  return Call::Unchecked(Type::Missing(), op, {size, runtime_device_index, dtype, storage_scope},
                         Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vm.alloc_storage", MakeVMAllocStorage);
}

// vm alloc_tensor

Type InferTypeVMAllocTensor(const Call& call, const BlockBuilder& ctx) {
  PrimType out_dtype = PrimType::Void();
  if (const auto* dtype_node = call->args[3].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = ffi::GetRef<DataTypeImm>(dtype_node);
    out_dtype = PrimType(dtype_imm->value);
  }
  int64_t vdevice_index = -1;
  if (const auto* int_imm = call->args[4].as<IntImmNode>()) {
    vdevice_index = int_imm->value.as<int>().value();
  }
  auto vdevice = GetGlobalVDevice(ctx->GetContextIRModule(), vdevice_index);

  if (const auto* output_shape = call->args[2].as<ShapeExprNode>()) {
    return TensorType(ffi::GetRef<Expr>(output_shape), out_dtype, vdevice);
  } else if (const auto* shape_ty = GetTypeAs<ShapeTypeNode>(call->args[2])) {
    if (shape_ty->values.has_value()) {
      return TensorType(ShapeExpr(shape_ty->values.value()), out_dtype, vdevice);
    } else {
      return TensorType(out_dtype, shape_ty->ndim, vdevice);
    }
  }
  return TensorType(out_dtype, kUnknownNDim, vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.vm.alloc_tensor")
      .signature(sig::arg("storage", "The storage to allocate the tensor to."),
                 sig::arg<IntExpr>("offset", "Storage offset to allocate the tensor."),
                 sig::arg("shape", "The shape of the tensor to allocate."),
                 sig::arg("dtype", "The dtype of the tensor to allocate."),
                 sig::arg<IntExpr>("runtime_device_index",
                                   "The device index indicating on which device the tensor is "
                                   "to be allocated at runtime."))
      .set_attr<FInferTypeWithBuilder>("relax.FInferTypeWithBuilder", InferTypeVMAllocTensor)
      // memory allocation isn't considered a "visible effect" as far as purity is concerned
      .set_attr<bool>("FPurity", true)
      .set_attr<bool>("TAllocator", true);
}

Expr MakeVMAllocTensor(Expr storage, PrimExpr offset, Expr shape, DataTypeImm dtype,
                       PrimExpr runtime_device_index) {
  static const Op op = Op::Get("relax.vm.alloc_tensor");
  return Call::Unchecked(Type::Missing(), op, {storage, offset, shape, dtype, runtime_device_index},
                         Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("relax.op.vm.alloc_tensor", [](ffi::PackedArgs args, ffi::Any* ret) {
    if (args.size() == 5) {
      *ret = MakeVMAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimExpr>(), args[2].cast<Expr>(),
                               args[3].cast<DataTypeImm>(), args[4].cast<PrimExpr>());
    } else {
      *ret = MakeVMAllocTensor(args[0].cast<Expr>(), args[1].cast<PrimExpr>(), args[2].cast<Expr>(),
                               args[3].cast<DataTypeImm>(), IntImm::Int64(0));
    }
  });

  // vm kill_object

  OpDef("relax.vm.kill_object")
      .signature(sig::arg("obj", "The object to be killed."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnVoidType>())
      // We mark this as impure so it wouldn't be removed by "remove_all_unused"
      .set_attr<bool>("FPurity", false);
}

Expr MakeVMKillObject(Expr obj) {
  static const Op op = Op::Get("relax.vm.kill_object");
  return Call::Unchecked(Type::Missing(), op, {std::move(obj)}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vm.kill_object", MakeVMKillObject);

  // vm call_tir_dyn

  OpDef("relax.vm.call_tir_dyn")
      .signature(
          sig::arg("func", "The destination-passing-style function."),
          sig::arg("args", "The input arguments (list of tensors and last argument is ShapeExpr)"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&ReturnVoidType>())
      // "relax.vm.call_tir_dyn" works in an in-place way, which is impure.
      .set_attr<bool>("FPurity", false);
}

Expr MakeCallTIRDyn(Expr func, Tuple args) {
  static const Op op = Op::Get("relax.vm.call_tir_dyn");
  return Call::Unchecked(Type::Missing(), op, {func, args}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.vm.call_tir_dyn", MakeCallTIRDyn);
}

// builtin stop_lift_params
Type InferTypeStopLiftParams(const CallNode* call) {
  return InferTypeUnaryArithContextFree<false>(call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.builtin.stop_lift_params")
      .signature(sig::arg("x", "The input data"),
                 sig::var_ty_args("out_type", "Optional output type."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeStopLiftParams>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeStopLiftParams(Expr x) {
  static const Op op = Op::Get("relax.builtin.stop_lift_params");
  return Call::Unchecked(Type::Missing(), op, {x}, Attrs(), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.builtin.stop_lift_params", MakeStopLiftParams);
}

// to_vdevice

Type InferToVDeviceType(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(!call->args[0]->ty.as<MissingType>().has_value());
  TensorType data_ty = GetUnaryInputTensorType(call);
  auto attrs = call->attrs.as<ToVDeviceAttrs>();
  VDevice vdev = attrs->dst_vdevice;
  if (data_ty->shape.has_value()) {
    return TensorType(data_ty->shape.value(), data_ty->dtype, vdev, data_ty->span);
  }
  return TensorType(data_ty->dtype, data_ty->ndim, vdev, data_ty->span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.to_vdevice")
      .signature(sig::arg("data", "The input expression to be copied"),
                 sig::call_attrs<ToVDeviceAttrs>())
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferToVDeviceType>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeToVDevice(Expr data, VDevice dst_vdev) {
  static const Op op = Op::Get("relax.to_vdevice");
  ffi::ObjectPtr<ToVDeviceAttrs> attrs = ffi::make_object<ToVDeviceAttrs>();
  attrs->dst_vdevice = dst_vdev;
  return Call::Unchecked(Type::Missing(), op, {data}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.to_vdevice", MakeToVDevice);
}

// hint_on_device

Type InferHintOnDeviceType(const CallNode* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TVM_FFI_ICHECK(call->args.size() == 1);
  TVM_FFI_ICHECK(!call->args[0]->ty.as<MissingType>().has_value());
  TensorType data_ty = GetUnaryInputTensorType(call);
  return data_ty;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.hint_on_device")
      .signature(sig::arg("data", "The input expression"), sig::call_attrs<HintOnDeviceAttrs>())
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferHintOnDeviceType>())
      .set_attr<bool>("FPurity", true);
}

Expr MakeHintOnDevice(Expr data, Device device, ffi::String memory_scope = "global") {
  static const Op op = Op::Get("relax.hint_on_device");
  ffi::ObjectPtr<HintOnDeviceAttrs> attrs = ffi::make_object<HintOnDeviceAttrs>();
  attrs->device_type = static_cast<int32_t>(device.device_type);
  attrs->index = device.device_id;
  attrs->memory_scope = memory_scope;
  return Call::Unchecked(Type::Missing(), op, {data}, Attrs(attrs), {});
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
