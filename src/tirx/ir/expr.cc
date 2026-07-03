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
 * \file expr.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>

#include "../../support/str_escape.h"
#include "buffer_common.h"

namespace tvm {
namespace tirx {

namespace {
// File-local helper: returns the vscale multiplier if `lanes` is of the form
// `multiplier * vscale()` or `vscale() * multiplier`, nullopt otherwise.
std::optional<int> ExtractVscaleFactor(const PrimExpr& lanes) {
  auto is_vscale = [](const PrimExpr& e) -> bool {
    if (const auto* call = e.as<CallNode>()) {
      return call->op.same_as(tirx::builtin::vscale());
    }
    return false;
  };
  if (const auto* mul = lanes.as<MulNode>()) {
    if (const auto* imm = mul->a.as<IntImmNode>(); imm && is_vscale(mul->b)) {
      return static_cast<int>(imm->value);
    }
    if (const auto* imm = mul->b.as<IntImmNode>(); imm && is_vscale(mul->a)) {
      return static_cast<int>(imm->value);
    }
  }
  return std::nullopt;
}

int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

TVM_FFI_INLINE const PrimTypeNode* GetPrimTypeNode(const PrimExpr& expr) {
  // Avoid PrimExpr::ty() ObjectRef materialization in expression constructor hot paths.
  const auto* node = expr.get();
  TVM_FFI_DCHECK(node != nullptr);
  TVM_FFI_DCHECK(!node->ExprNode::ty.IsMissing());
  const auto* prim_ty = node->ExprNode::ty.as<PrimTypeNode>();
  TVM_FFI_DCHECK(prim_ty != nullptr);
  return prim_ty;
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  VarNode::RegisterReflection();
  IterVarNode::RegisterReflection();
  StringImmNode::RegisterReflection();
  CastNode::RegisterReflection();
  AddNode::RegisterReflection();
  SubNode::RegisterReflection();
  MulNode::RegisterReflection();
  DivNode::RegisterReflection();
  ModNode::RegisterReflection();
  FloorDivNode::RegisterReflection();
  FloorModNode::RegisterReflection();
  MinNode::RegisterReflection();
  MaxNode::RegisterReflection();
  EQNode::RegisterReflection();
  NENode::RegisterReflection();
  LTNode::RegisterReflection();
  LENode::RegisterReflection();
  GTNode::RegisterReflection();
  GENode::RegisterReflection();
  AndNode::RegisterReflection();
  OrNode::RegisterReflection();
  NotNode::RegisterReflection();
  SelectNode::RegisterReflection();
  BufferLoadNode::RegisterReflection();
  ProducerLoadNode::RegisterReflection();
  RampNode::RegisterReflection();
  BroadcastNode::RegisterReflection();
  LetNode::RegisterReflection();
  ShuffleNode::RegisterReflection();
  CommReducerNode::RegisterReflection();
  ReduceNode::RegisterReflection();
}

/* \brief Convert an object to a PrimExpr
 *
 * All conversions to a PrimExpr are performed as part of the FFI,
 * when calling a function that accepts a PrimExpr as an argument.  If
 * a function must normalize to a PrimExpr (e.g. before accessing the
 * `expr.dtype` field), this function allows the FFI conversions to be
 * explicitly invoked.
 */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.convert",
                        [](ffi::Variant<PrimExpr, ffi::Array<PrimExpr>> expr) { return expr; });
  // Note: kRepr for VarNode is registered via TVM_REGISTER_SCRIPT_AS_REPR in
  // src/script/printer/tirx/expr.cc (-> ReprPrintTIR which delegates to TVMScriptPrinter).
}

#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                                        \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                 \
    using T = Name::ContainerType;                                                \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                 \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                 \
    const PrimTypeNode* a_ty = GetPrimTypeNode(a);                                \
    const PrimTypeNode* b_ty = GetPrimTypeNode(b);                                \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                          \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>();                               \
    node->ExprNode::ty = a.get()->ExprNode::ty;                                   \
    node->a = std::move(a);                                                       \
    node->b = std::move(b);                                                       \
    node->span = std::move(span);                                                 \
    data_ = std::move(node);                                                      \
  }

#define TVM_DEFINE_CMPOP_CONSTRUCTOR(Name)                                        \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                 \
    using T = Name::ContainerType;                                                \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                 \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                 \
    const PrimTypeNode* a_ty = GetPrimTypeNode(a);                                \
    const PrimTypeNode* b_ty = GetPrimTypeNode(b);                                \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                          \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>();                               \
    node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});     \
    node->a = std::move(a);                                                       \
    node->b = std::move(b);                                                       \
    node->span = std::move(span);                                                 \
    data_ = std::move(node);                                                      \
  }

// Var
Var::Var(ffi::String name_hint, PrimType dtype, Span span) {
  auto n = ffi::make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->ExprNode::ty = dtype;
  n->span = std::move(span);
  data_ = std::move(n);
}

Var::Var(ffi::String name_hint, Type type_annotation, Span span) {
  auto n = ffi::make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->ExprNode::ty = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var Var::copy_with_name(const ffi::String& name) const {
  const VarNode* node = get();
  ffi::ObjectPtr<VarNode> new_ptr = ffi::make_object<VarNode>(*node);
  new_ptr->name_hint = name;
  return Var(new_ptr);
}

Var Var::copy_with_suffix(const ffi::String& suffix) const {
  return this->copy_with_name(get()->name_hint + suffix);
}

Var Var::copy_with_dtype(PrimType dtype) const {
  const VarNode* node = get();
  ffi::ObjectPtr<VarNode> new_ptr = ffi::make_object<VarNode>(*node);
  new_ptr->ExprNode::ty = dtype;
  return Var(new_ptr);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Var", [](ffi::String name_hint, ffi::AnyView type, Span span) {
    if (type.as<Type>()) {
      return Var(name_hint, type.cast<Type>(), span);
    } else {
      return Var(name_hint, type.cast<PrimType>(), span);
    }
  });
}

// IterVar
IterVar::IterVar(Range dom, Var var, IterVarType t, ffi::String thread_tag, Span span) {
  ffi::ObjectPtr<IterVarNode> n = ffi::make_object<IterVarNode>();
  if (dom.defined() && dom->extent.defined()) {
    PrimType extent_ty = dom->extent.ty();
    PrimType var_ty = var.ty();
    TVM_FFI_ICHECK(extent_ty.code() == DLDataTypeCode::kDLInt)
        << "The dtype of the domain of an IterVar must be an integer type. However, the domain's "
           "dtype is "
        << extent_ty->dtype;
    TVM_FFI_ICHECK(extent_ty == var_ty)
        << "The dtype of the extent of an IterVar (" << extent_ty->dtype
        << ") must match its associated Var's dtype (" << var_ty->dtype << ")";
  }
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.IterVar", [](Range dom, Var var, int iter_type, ffi::String thread_tag, Span span) {
        return IterVar(dom, var, static_cast<IterVarType>(iter_type), thread_tag, span);
      });
}

// StringImm
StringImm::StringImm(ffi::String value, Span span) {
  ffi::ObjectPtr<StringImmNode> node = ffi::make_object<StringImmNode>();
  node->ExprNode::ty = PrimType::Handle();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.StringImm",
                        [](ffi::String value, Span span) { return StringImm(value, span); });
}

// Cast
Cast::Cast(PrimType value_ty, PrimExpr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  PrimType value_expr_ty = value.ty();
  TVM_FFI_ICHECK_EQ(value_ty->dtype.lanes, value_expr_ty->dtype.lanes);
  ffi::ObjectPtr<CastNode> node = ffi::make_object<CastNode>();
  node->ExprNode::ty = std::move(value_ty);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Cast", [](PrimType dtype, PrimExpr value, Span span) {
    return Cast(dtype, value, span);
  });
}

// Add
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Add",
                        [](PrimExpr a, PrimExpr b, Span span) { return Add(a, b, span); });
}

// Sub
TVM_DEFINE_BINOP_CONSTRUCTOR(Sub);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Sub",
                        [](PrimExpr a, PrimExpr b, Span span) { return Sub(a, b, span); });
}

// Mul
TVM_DEFINE_BINOP_CONSTRUCTOR(Mul);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Mul",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mul(a, b, span); });
}

// Div
TVM_DEFINE_BINOP_CONSTRUCTOR(Div);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Div",
                        [](PrimExpr a, PrimExpr b, Span span) { return Div(a, b, span); });
}

// Mod
TVM_DEFINE_BINOP_CONSTRUCTOR(Mod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Mod",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mod(a, b, span); });
}

// FloorDiv
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorDiv);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.FloorDiv",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorDiv(a, b, span); });
}

// FloorMod
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorMod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.FloorMod",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorMod(a, b, span); });
}

// Min
TVM_DEFINE_BINOP_CONSTRUCTOR(Min);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Min",
                        [](PrimExpr a, PrimExpr b, Span span) { return Min(a, b, span); });
}

// Max
TVM_DEFINE_BINOP_CONSTRUCTOR(Max);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Max",
                        [](PrimExpr a, PrimExpr b, Span span) { return Max(a, b, span); });
}

// EQ
TVM_DEFINE_CMPOP_CONSTRUCTOR(EQ);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.EQ",
                        [](PrimExpr a, PrimExpr b, Span span) { return EQ(a, b, span); });
}

// NE
TVM_DEFINE_CMPOP_CONSTRUCTOR(NE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.NE",
                        [](PrimExpr a, PrimExpr b, Span span) { return NE(a, b, span); });
}

// LT
TVM_DEFINE_CMPOP_CONSTRUCTOR(LT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.LT",
                        [](PrimExpr a, PrimExpr b, Span span) { return LT(a, b, span); });
}

// LE
TVM_DEFINE_CMPOP_CONSTRUCTOR(LE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.LE",
                        [](PrimExpr a, PrimExpr b, Span span) { return LE(a, b, span); });
}

// GT
TVM_DEFINE_CMPOP_CONSTRUCTOR(GT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.GT",
                        [](PrimExpr a, PrimExpr b, Span span) { return GT(a, b, span); });
}

// GE
TVM_DEFINE_CMPOP_CONSTRUCTOR(GE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.GE",
                        [](PrimExpr a, PrimExpr b, Span span) { return GE(a, b, span); });
}

// And
And::And(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<AndNode> node = ffi::make_object<AndNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.And",
                        [](PrimExpr a, PrimExpr b, Span span) { return And(a, b, span); });
}

// Or
Or::Or(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<OrNode> node = ffi::make_object<OrNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Or",
                        [](PrimExpr a, PrimExpr b, Span span) { return Or(a, b, span); });
}

// Not
Not::Not(PrimExpr a, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  PrimType a_ty = a.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));

  ffi::ObjectPtr<NotNode> node = ffi::make_object<NotNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Not", [](PrimExpr a, Span span) { return Not(a, span); });
}

// Select
Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  TVM_FFI_CHECK(condition.defined(), ValueError) << "condition is undefined";
  TVM_FFI_CHECK(true_value.defined(), ValueError) << "true_value is undefined";
  TVM_FFI_CHECK(false_value.defined(), ValueError) << "true_value is undefined";
  PrimType condition_ty = condition.ty();
  PrimType true_ty = true_value.ty();
  PrimType false_ty = false_value.ty();
  TVM_FFI_ICHECK(condition_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(GetLanesOrVScaleFactor(condition_ty) == GetLanesOrVScaleFactor(true_ty) ||
                 condition_ty.IsScalar());
  TVM_FFI_CHECK(false_ty == true_ty, TypeError)
      << "mismatched types. "
      << "False type: " << false_ty->dtype << "; True type: " << true_ty->dtype;

  ffi::ObjectPtr<SelectNode> node = ffi::make_object<SelectNode>();
  node->ExprNode::ty = true_ty;
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.Select", [](PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
        return Select(condition, true_value, false_value, span);
      });
}

// Ramp
Ramp::Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(base.defined());
  TVM_FFI_ICHECK(stride.defined());
  PrimType base_ty = base.ty();
  PrimType stride_ty = stride.ty();
  TVM_FFI_ICHECK(base_ty.IsScalar());
  TVM_FFI_ICHECK(stride_ty.IsScalar());
  if (stride_ty != base_ty) {
    stride = cast(base_ty, stride);
  }

  ffi::ObjectPtr<RampNode> node = ffi::make_object<RampNode>();
  auto* lanes_as_int = lanes.as<IntImmNode>();
  if (lanes_as_int) {
    int lanes = static_cast<int>(lanes_as_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->ExprNode::ty = base_ty.WithLanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->ExprNode::ty =
        PrimType::ScalableVector(base_ty.code(), base_ty.bits(), vscale_factor.value());
    lanes = Mul(Call(PrimType::Int(32), tirx::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                vscale_factor.value());
    node->lanes = lanes;
  }
  node->base = base;
  node->stride = stride;
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Ramp", [](PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
    return Ramp(base, stride, lanes, span);
  });
}

// Broadcast
Broadcast::Broadcast(PrimExpr value, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(value.defined());
  PrimType value_ty = value.ty();
  TVM_FFI_ICHECK(value_ty.IsScalar());

  ffi::ObjectPtr<BroadcastNode> node = ffi::make_object<BroadcastNode>();
  auto* lanes_int = lanes.as<IntImmNode>();
  if (lanes_int) {
    int lanes = static_cast<int>(lanes_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->ExprNode::ty = value_ty.WithLanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->ExprNode::ty =
        PrimType::ScalableVector(value_ty.code(), value_ty.bits(), vscale_factor.value());
    lanes = Mul(Call(PrimType::Int(32), tirx::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                vscale_factor.value());
    node->lanes = lanes;
  }
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = node;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Broadcast", [](PrimExpr value, PrimExpr lanes, Span span) {
    return Broadcast(value, lanes, span);
  });
}

// Let
Let::Let(Var var, PrimExpr value, PrimExpr body, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(body.defined());
  TVM_FFI_ICHECK(value.ty() == var.ty());

  ffi::ObjectPtr<LetNode> node = ffi::make_object<LetNode>();
  node->ExprNode::ty = body.ty();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Let", [](Var var, PrimExpr value, PrimExpr body, Span span) {
    return Let(var, value, body, span);
  });
}

// Shuffle
Shuffle::Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0U);
  TVM_FFI_ICHECK_NE(indices.size(), 0U);

  PrimType base_type = vectors[0].ty().WithLanes(1);
  int total_lanes = 0;

  for (PrimExpr val : vectors) {
    PrimType val_ty = val.ty();
    TVM_FFI_ICHECK(val_ty.WithLanes(1)->dtype == base_type->dtype);
    total_lanes += val_ty.lanes();
  }
  TVM_FFI_ICHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  ffi::ObjectPtr<ShuffleNode> node = ffi::make_object<ShuffleNode>();
  node->ExprNode::ty = base_type.WithLanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = node;
}

PrimExpr Shuffle::Concat(ffi::Array<PrimExpr> vectors, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  ffi::Array<PrimExpr> indices;
  int index = 0;
  for (const PrimExpr& e : vectors) {
    for (int i = 0; i < e.ty().lanes(); ++i) {
      indices.push_back(IntImm::Int32(index++));
    }
  }
  return Shuffle(vectors, indices, span);
}

PrimExpr Shuffle::ExtractElement(PrimExpr vector, int index, Span span) {
  return Shuffle({vector}, {IntImm::Int32(index)}, span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Shuffle",
                        [](ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
                          return Shuffle(vectors, indices, span);
                        });
}

// CommReducer
CommReducer::CommReducer(ffi::Array<Var> lhs, ffi::Array<Var> rhs, ffi::Array<PrimExpr> result,
                         ffi::Array<PrimExpr> identity_element, Span span) {
  size_t n_group = result.size();
  TVM_FFI_CHECK_EQ(lhs.size(), n_group, ValueError)
      << "The number of vars in `lhs` must equal to the "
         "number of elements in `results`";
  TVM_FFI_CHECK_EQ(rhs.size(), n_group, ValueError)
      << "The number of vars in `rhs` must equal to the "
         "number of elements in `results`";
  TVM_FFI_CHECK_EQ(identity_element.size(), n_group, ValueError)
      << "The number of identities must equal to the number of elements in `results`";

  // Change the dtype of input vars to adapt to the dtype of identities
  ffi::ArrayObj* p_lhs = lhs.CopyOnWrite();
  ffi::ArrayObj* p_rhs = rhs.CopyOnWrite();
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  var_map.reserve(n_group * 2);
  for (int i = 0; i < static_cast<int>(n_group); ++i) {
    PrimType dtype = identity_element[i].ty();
    Var l = lhs[i].copy_with_dtype(dtype);
    Var r = rhs[i].copy_with_dtype(dtype);
    var_map[lhs[i].get()] = l;
    var_map[rhs[i].get()] = r;

    p_lhs->SetItem(i, l);
    p_rhs->SetItem(i, r);
  }

  ffi::ArrayObj* p_result = result.CopyOnWrite();
  for (int i = 0; i < static_cast<int>(n_group); ++i) {
    p_result->SetItem(i, Substitute(result[i], var_map));
  }

  auto node = ffi::make_object<CommReducerNode>();
  node->lhs = lhs;
  node->rhs = rhs;
  node->result = result;
  node->identity_element = identity_element;
  node->span = std::move(span);
  data_ = std::move(node);
}

ffi::Array<PrimExpr> CommReducerNode::operator()(ffi::Array<PrimExpr> a,
                                                 ffi::Array<PrimExpr> b) const {
  TVM_FFI_ICHECK_EQ(a.size(), b.size());
  TVM_FFI_ICHECK_EQ(lhs.size(), a.size());
  TVM_FFI_ICHECK_EQ(rhs.size(), b.size());
  ffi::Map<Var, PrimExpr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  return Substitute(this->result, value_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.CommReducer",
           [](ffi::Array<Var> lhs, ffi::Array<Var> rhs, ffi::Array<PrimExpr> result,
              ffi::Array<PrimExpr> identity_element,
              Span span) { return CommReducer(lhs, rhs, result, identity_element, span); })
      .def_method("tirx.CommReducerCombine", &tirx::CommReducerNode::operator());
}

// Reduce
Reduce::Reduce(CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
               PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
  for (size_t i = 0; i < axis.size(); ++i) {
    TVM_FFI_ICHECK_EQ(axis[i]->iter_type, kCommReduce)
        << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = IntImm::Bool(true);
  }
  auto n = ffi::make_object<ReduceNode>();
  TVM_FFI_ICHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    TVM_FFI_ICHECK(axis[i].defined());
  }
  if (!init.empty()) {
    TVM_FFI_ICHECK_EQ(init.size(), source.size()) << "Number of inits should match number of exprs";
    for (size_t i = 0; i < init.size(); i++) {
      TVM_FFI_ICHECK(init[i].defined()) << "Init value must be defined";
      TVM_FFI_ICHECK(init[i]->IsInstance<ProducerLoadNode>() || init[i]->IsInstance<IntImmNode>() ||
                     init[i]->IsInstance<FloatImmNode>())
          << "init can only be a IntImm, FloatImm or ProducerLoad, "
          << "but received " << init[i] << " of type " << init[i]->GetTypeKey();
    }
  }
  n->ExprNode::ty = source[value_index].ty();
  n->combiner = std::move(combiner);
  n->source = std::move(source);
  n->init = std::move(init);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.Reduce", [](CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
                        PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
        return Reduce(combiner, source, axis, condition, value_index, init, span);
      });
}

// BufferLoad
void BufferLoadNode::LegalizeDType() {
  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].ty().IsScalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  if (indices.empty()) {
    this->ExprNode::ty = buffer->dtype;
  } else {
    PrimType index_ty = indices.back().ty();
    int16_t buffer_encoded_lanes = static_cast<int16_t>(buffer->dtype->dtype.lanes);
    bool is_buffer_dtype_scalable = buffer_encoded_lanes < -1;
    bool is_index_scalable = index_ty.IsScalableVector();

    TVM_FFI_ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
        << "Index dtype and buffer dtype can't both be scalable.";

    if (is_index_scalable) {
      this->ExprNode::ty =
          PrimType::ScalableVector(buffer->dtype.code(), buffer->dtype.bits(),
                                   index_ty.VScaleFactor() * buffer->dtype.lanes());
    } else if (is_buffer_dtype_scalable) {
      this->ExprNode::ty = PrimType::ScalableVector(buffer->dtype.code(), buffer->dtype.bits(),
                                                    -buffer_encoded_lanes * index_ty.lanes());
    } else {
      this->ExprNode::ty = buffer->dtype.WithLanes(index_ty.lanes() * buffer->dtype.lanes());
    }
  }
}

BufferLoad::BufferLoad(Buffer buffer, ffi::Array<PrimExpr> indices,
                       ffi::Optional<PrimExpr> predicate, Span span) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  if (predicate.defined()) {
    PrimType predicate_ty = predicate.value().ty();
    bool is_index_scalable = indices.empty() ? false : indices.back().ty().IsScalableVector();
    bool is_predicate_scalable = predicate_ty.IsScalableVector();
    TVM_FFI_ICHECK_EQ(is_index_scalable, is_predicate_scalable)
        << "Predicate mask dtype and load indices must both be scalable.";

    int16_t buffer_encoded_lanes = static_cast<int16_t>(buffer->dtype->dtype.lanes);
    int buffer_lanes = buffer_encoded_lanes < -1 ? -buffer_encoded_lanes : buffer_encoded_lanes;
    int index_lanes = indices.empty() ? 1 : GetLanesOrVScaleFactor(indices.back().ty());
    int predicate_lanes = GetLanesOrVScaleFactor(predicate_ty);
    TVM_FFI_ICHECK_EQ(index_lanes * buffer_lanes, predicate_lanes)
        << "Got a predicate mask with " << predicate_lanes
        << " lanes, but trying to load a vector with " << index_lanes
        << " lanes. The number of lanes must match.";

    TVM_FFI_ICHECK(predicate_ty.MatchesCode(DLDataTypeCode::kDLBool) ||
                   predicate_ty.MatchesElementType(DLDataTypeCode::kDLUInt, 1))
        << "Predicate mask elements must be boolean values, but got " << predicate_ty->dtype << ".";
  }

  ffi::ObjectPtr<BufferLoadNode> node = ffi::make_object<BufferLoadNode>();
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  node->predicate = std::move(predicate);
  node->span = std::move(span);
  node->LegalizeDType();
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.BufferLoad", [](Buffer buffer, ffi::Array<PrimExpr> indices,
                                              ffi::Optional<PrimExpr> predicate, Span span) {
    return BufferLoad(buffer, indices, predicate, span);
  });
}

// ProducerLoad
ProducerLoad::ProducerLoad(DataProducer producer, ffi::Array<PrimExpr> indices, Span span) {
  ffi::ObjectPtr<ProducerLoadNode> node = ffi::make_object<ProducerLoadNode>();
  node->ExprNode::ty = producer->GetDataType();
  node->producer = std::move(producer);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ProducerLoad",
                        [](DataProducer producer, ffi::Array<PrimExpr> indices, Span span) {
                          return ProducerLoad(producer, indices, span);
                        });
}

}  // namespace tirx
}  // namespace tvm
