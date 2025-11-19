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
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <optional>

#include "../../arith/scalable_expression.h"
#include "../../support/str_escape.h"
#include "buffer_common.h"

namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK() {
  VarNode::RegisterReflection();
  SizeVarNode::RegisterReflection();
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
  CallNode::RegisterReflection();
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
  refl::GlobalDef().def("tir.convert",
                        [](ffi::Variant<PrimExpr, ffi::Array<PrimExpr>> expr) { return expr; });
}

#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                                                   \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                            \
    using T = Name::ContainerType;                                                           \
    ICHECK(a.defined()) << "ValueError: a is undefined\n";                                   \
    ICHECK(b.defined()) << "ValueError: b is undefined\n";                                   \
    CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types. " << a.dtype() << " vs. " \
                                  << b.dtype() << "\n";                                      \
    ObjectPtr<T> node = ffi::make_object<T>();                                               \
    node->dtype = a.dtype();                                                                 \
    node->a = std::move(a);                                                                  \
    node->b = std::move(b);                                                                  \
    node->span = std::move(span);                                                            \
    data_ = std::move(node);                                                                 \
  }

#define TVM_DEFINE_CMPOP_CONSTRUCTOR(Name)                                                   \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                            \
    using T = Name::ContainerType;                                                           \
    ICHECK(a.defined()) << "ValueError: a is undefined\n";                                   \
    ICHECK(b.defined()) << "ValueError: b is undefined\n";                                   \
    CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types. " << a.dtype() << " vs. " \
                                  << b.dtype() << "\n";                                      \
    ObjectPtr<T> node = ffi::make_object<T>();                                               \
    DataType a_dtype = a.dtype();                                                            \
    node->dtype =                                                                            \
        DataType::Bool(a_dtype.get_lanes_or_vscale_factor(), a_dtype.is_scalable_vector());  \
    node->a = std::move(a);                                                                  \
    node->b = std::move(b);                                                                  \
    node->span = std::move(span);                                                            \
    data_ = std::move(node);                                                                 \
  }

// Var
Var::Var(ffi::String name_hint, DataType dtype, Span span) {
  auto n = ffi::make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->type_annotation = GetTypeFromRuntimeDataType(dtype);
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var::Var(ffi::String name_hint, Type type_annotation, Span span) {
  auto n = ffi::make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = GetRuntimeDataType(type_annotation);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var Var::copy_with_name(const ffi::String& name) const {
  const VarNode* node = get();
  ObjectPtr<VarNode> new_ptr;
  if (auto* ptr = this->as<SizeVarNode>()) {
    new_ptr = ffi::make_object<SizeVarNode>(*ptr);
  } else {
    new_ptr = ffi::make_object<VarNode>(*node);
  }
  new_ptr->name_hint = name;
  return Var(new_ptr);
}

Var Var::copy_with_suffix(const ffi::String& suffix) const {
  return this->copy_with_name(get()->name_hint + suffix);
}

Var Var::copy_with_dtype(DataType dtype) const {
  const VarNode* node = get();
  ObjectPtr<VarNode> new_ptr;
  if (auto* ptr = this->as<SizeVarNode>()) {
    new_ptr = ffi::make_object<SizeVarNode>(*ptr);
  } else {
    new_ptr = ffi::make_object<VarNode>(*node);
  }
  new_ptr->type_annotation = GetTypeFromRuntimeDataType(dtype);
  new_ptr->dtype = std::move(dtype);
  return Var(new_ptr);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Var", [](ffi::String name_hint, ffi::AnyView type, Span span) {
    if (type.as<Type>()) {
      return Var(name_hint, type.cast<Type>(), span);
    } else {
      return Var(name_hint, type.cast<DataType>(), span);
    }
  });
}

// SizeVar
SizeVar::SizeVar(ffi::String name_hint, DataType dtype, Span span) {
  auto n = ffi::make_object<SizeVarNode>();
  n->name_hint = std::move(name_hint);
  n->type_annotation = GetTypeFromRuntimeDataType(dtype);
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

SizeVar::SizeVar(ffi::String name_hint, Type type_annotation, Span span) {
  auto n = ffi::make_object<SizeVarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = GetRuntimeDataType(type_annotation);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.SizeVar",
                        [](ffi::String s, DataType t, Span span) { return SizeVar(s, t, span); });
}

// IterVar
IterVar::IterVar(Range dom, Var var, IterVarType t, ffi::String thread_tag, Span span) {
  ObjectPtr<IterVarNode> n = ffi::make_object<IterVarNode>();
  if (dom.defined() && dom->extent.defined()) {
    CHECK(dom->extent.dtype().is_int())
        << "The dtype of the domain of an IterVar must be an integer type. However, the domain's "
           "dtype is "
        << dom->extent.dtype();
    CHECK_EQ(dom->extent.dtype(), var.dtype())
        << "The dtype of the extent of an IterVar (" << dom->extent.dtype()
        << ") must match its associated Var's dtype (" << var.dtype() << ")";
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
      "tir.IterVar", [](Range dom, Var var, int iter_type, ffi::String thread_tag, Span span) {
        return IterVar(dom, var, static_cast<IterVarType>(iter_type), thread_tag, span);
      });
}

// StringImm
StringImm::StringImm(ffi::String value, Span span) {
  ObjectPtr<StringImmNode> node = ffi::make_object<StringImmNode>();
  node->dtype = DataType::Handle();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.StringImm",
                        [](ffi::String value, Span span) { return StringImm(value, span); });
}

// Cast
Cast::Cast(DataType t, PrimExpr value, Span span) {
  ICHECK(value.defined());
  ICHECK_EQ(t.get_lanes_or_vscale_factor(), value.dtype().get_lanes_or_vscale_factor());
  ICHECK(t.is_scalable_vector() == value.dtype().is_scalable_vector());
  ObjectPtr<CastNode> node = ffi::make_object<CastNode>();
  node->dtype = t;
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Cast", [](DataType dtype, PrimExpr value, Span span) {
    return Cast(dtype, value, span);
  });
}

// Add
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Add",
                        [](PrimExpr a, PrimExpr b, Span span) { return Add(a, b, span); });
}

// Sub
TVM_DEFINE_BINOP_CONSTRUCTOR(Sub);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Sub",
                        [](PrimExpr a, PrimExpr b, Span span) { return Sub(a, b, span); });
}

// Mul
TVM_DEFINE_BINOP_CONSTRUCTOR(Mul);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Mul",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mul(a, b, span); });
}

// Div
TVM_DEFINE_BINOP_CONSTRUCTOR(Div);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Div",
                        [](PrimExpr a, PrimExpr b, Span span) { return Div(a, b, span); });
}

// Mod
TVM_DEFINE_BINOP_CONSTRUCTOR(Mod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Mod",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mod(a, b, span); });
}

// FloorDiv
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorDiv);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.FloorDiv",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorDiv(a, b, span); });
}

// FloorMod
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorMod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.FloorMod",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorMod(a, b, span); });
}

// Min
TVM_DEFINE_BINOP_CONSTRUCTOR(Min);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Min",
                        [](PrimExpr a, PrimExpr b, Span span) { return Min(a, b, span); });
}

// Max
TVM_DEFINE_BINOP_CONSTRUCTOR(Max);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Max",
                        [](PrimExpr a, PrimExpr b, Span span) { return Max(a, b, span); });
}

// EQ
TVM_DEFINE_CMPOP_CONSTRUCTOR(EQ);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.EQ", [](PrimExpr a, PrimExpr b, Span span) { return EQ(a, b, span); });
}

// NE
TVM_DEFINE_CMPOP_CONSTRUCTOR(NE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.NE", [](PrimExpr a, PrimExpr b, Span span) { return NE(a, b, span); });
}

// LT
TVM_DEFINE_CMPOP_CONSTRUCTOR(LT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.LT", [](PrimExpr a, PrimExpr b, Span span) { return LT(a, b, span); });
}

// LE
TVM_DEFINE_CMPOP_CONSTRUCTOR(LE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.LE", [](PrimExpr a, PrimExpr b, Span span) { return LE(a, b, span); });
}

// GT
TVM_DEFINE_CMPOP_CONSTRUCTOR(GT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.GT", [](PrimExpr a, PrimExpr b, Span span) { return GT(a, b, span); });
}

// GE
TVM_DEFINE_CMPOP_CONSTRUCTOR(GE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.GE", [](PrimExpr a, PrimExpr b, Span span) { return GE(a, b, span); });
}

// And
And::And(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.defined()) << "ValueError: a is undefined";
  ICHECK(b.defined()) << "ValueError: b is undefined";
  ICHECK(a.dtype().is_bool());
  ICHECK(b.dtype().is_bool());
  ICHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types";

  ObjectPtr<AndNode> node = ffi::make_object<AndNode>();
  node->dtype =
      DataType::Bool(a.dtype().get_lanes_or_vscale_factor(), a.dtype().is_scalable_vector());
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.And",
                        [](PrimExpr a, PrimExpr b, Span span) { return And(a, b, span); });
}

// Or
Or::Or(PrimExpr a, PrimExpr b, Span span) {
  ICHECK(a.defined()) << "ValueError: a is undefined";
  ICHECK(b.defined()) << "ValueError: b is undefined";
  ICHECK(a.dtype().is_bool());
  ICHECK(b.dtype().is_bool());
  ICHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types";

  ObjectPtr<OrNode> node = ffi::make_object<OrNode>();
  node->dtype =
      DataType::Bool(a.dtype().get_lanes_or_vscale_factor(), a.dtype().is_scalable_vector());
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Or", [](PrimExpr a, PrimExpr b, Span span) { return Or(a, b, span); });
}

// Not
Not::Not(PrimExpr a, Span span) {
  ICHECK(a.defined()) << "ValueError: a is undefined";
  ICHECK(a.dtype().is_bool());

  ObjectPtr<NotNode> node = ffi::make_object<NotNode>();
  DataType a_dtype = a.dtype();
  node->dtype = DataType::Bool(a_dtype.get_lanes_or_vscale_factor(), a_dtype.is_scalable_vector());
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Not", [](PrimExpr a, Span span) { return Not(a, span); });
}

// Select
Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  ICHECK(condition.defined()) << "ValueError: condition is undefined";
  ICHECK(true_value.defined()) << "ValueError: true_value is undefined";
  ICHECK(false_value.defined()) << "ValueError: true_value is undefined";
  ICHECK(condition.dtype().is_bool());
  ICHECK(condition.dtype().get_lanes_or_vscale_factor() ==
             true_value.dtype().get_lanes_or_vscale_factor() ||
         condition.dtype().is_scalar());
  ICHECK(false_value.dtype() == true_value.dtype())
      << "TypeError: mismatched types. "
      << "False type: " << false_value.dtype() << "; True type: " << true_value.dtype();

  ObjectPtr<SelectNode> node = ffi::make_object<SelectNode>();
  node->dtype = true_value.dtype();
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.Select", [](PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
        return Select(condition, true_value, false_value, span);
      });
}

// Ramp
Ramp::Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
  ICHECK(base.defined());
  ICHECK(stride.defined());
  ICHECK(base.dtype().is_scalar());
  ICHECK(stride.dtype().is_scalar());
  if (stride.dtype() != base.dtype()) {
    stride = cast(base.dtype(), stride);
  }

  ObjectPtr<RampNode> node = ffi::make_object<RampNode>();
  auto* lanes_as_int = lanes.as<IntImmNode>();
  if (lanes_as_int) {
    int lanes = static_cast<int>(lanes_as_int->value);
    ICHECK_GT(lanes, 1);
    node->dtype = base.dtype().with_lanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = arith::ExtractVscaleFactor(lanes);
    ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->dtype = base.dtype().with_scalable_vscale_factor(vscale_factor.value());
    lanes = Mul(Call(DataType::Int(32), tir::builtin::vscale(), {}), vscale_factor.value());
    node->lanes = lanes;
  }
  node->base = base;
  node->stride = stride;
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Ramp", [](PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
    return Ramp(base, stride, lanes, span);
  });
}

// Broadcast
Broadcast::Broadcast(PrimExpr value, PrimExpr lanes, Span span) {
  ICHECK(value.defined());
  ICHECK(value.dtype().is_scalar());

  ObjectPtr<BroadcastNode> node = ffi::make_object<BroadcastNode>();
  auto* lanes_int = lanes.as<IntImmNode>();
  if (lanes_int) {
    int lanes = static_cast<int>(lanes_int->value);
    ICHECK_GT(lanes, 1);
    node->dtype = value.dtype().with_lanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = arith::ExtractVscaleFactor(lanes);
    ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->dtype = value.dtype().with_scalable_vscale_factor(vscale_factor.value());
    lanes = Mul(Call(DataType::Int(32), tir::builtin::vscale(), {}), vscale_factor.value());
    node->lanes = lanes;
  }
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = node;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Broadcast", [](PrimExpr value, PrimExpr lanes, Span span) {
    return Broadcast(value, lanes, span);
  });
}

// Let
Let::Let(Var var, PrimExpr value, PrimExpr body, Span span) {
  ICHECK(value.defined());
  ICHECK(body.defined());
  ICHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetNode> node = ffi::make_object<LetNode>();
  node->dtype = body.dtype();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Let", [](Var var, PrimExpr value, PrimExpr body, Span span) {
    return Let(var, value, body, span);
  });
}

// Call
Call::Call(DataType dtype, RelaxExpr op, ffi::Array<PrimExpr> args, Span span) {
  for (size_t i = 0; i < args.size(); ++i) {
    ICHECK(args[i].defined()) << "arg " << i << " is not defined()";
  }

  ObjectPtr<CallNode> node = ffi::make_object<CallNode>();
  node->dtype = dtype;
  node->op = std::move(op);
  node->args = std::move(args);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tir.Call",
      [](ffi::Optional<DataType> dtype, RelaxExpr op,
         ffi::Array<ffi::Variant<ffi::String, DLDataType, IterVar, BufferRegion, PrimExpr>> args,
         Span span) {
        ffi::Array<PrimExpr> prim_expr_args;
        for (const auto& it : args) {
          if (auto opt_str = it.as<ffi::String>()) {
            prim_expr_args.push_back(StringImm(opt_str.value()));
          } else if (auto opt_dtype = it.as<DLDataType>()) {
            prim_expr_args.push_back(StringImm(ffi::DLDataTypeToString(opt_dtype.value())));
          } else if (const auto* iter_var = it.as<IterVarNode>()) {
            prim_expr_args.push_back(iter_var->var);
          } else if (const auto* br = it.as<BufferRegionNode>()) {
            ffi::Array<PrimExpr> indices;
            for (Range r : br->region) {
              if (is_one(r->extent)) {
                indices.push_back(r->min);
              } else if (r->extent.as<IntImmNode>()) {
                indices.push_back(tir::Ramp(r->min, make_const(r->min->dtype, 1), r->extent));
              } else {
                LOG(FATAL) << "ValueError: Cannot convert to BufferLoad: "
                           << ffi::GetRef<BufferRegion>(br);
              }
            }
            prim_expr_args.push_back(BufferLoad(br->buffer, indices));
          } else {
            prim_expr_args.push_back(Downcast<PrimExpr>(it));
          }
        }
        return Call(dtype.value_or(DataType::Void()), op, prim_expr_args, span);
      });
}

// Shuffle
Shuffle::Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
  ICHECK_NE(vectors.size(), 0U);
  ICHECK_NE(indices.size(), 0U);

  DataType base_type = vectors[0].dtype().element_of();
  int total_lanes = 0;

  for (PrimExpr val : vectors) {
    ICHECK(val.dtype().element_of() == base_type);
    total_lanes += val.dtype().lanes();
  }
  ICHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  ObjectPtr<ShuffleNode> node = ffi::make_object<ShuffleNode>();
  node->dtype = base_type.with_lanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = node;
}

PrimExpr Shuffle::Concat(ffi::Array<PrimExpr> vectors, Span span) {
  ICHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  ffi::Array<PrimExpr> indices;
  int index = 0;
  for (const PrimExpr& e : vectors) {
    for (int i = 0; i < e.dtype().lanes(); ++i) {
      indices.push_back(IntImm(DataType::Int(32), index++));
    }
  }
  return Shuffle(vectors, indices, span);
}

PrimExpr Shuffle::ExtractElement(PrimExpr vector, int index, Span span) {
  return Shuffle({vector}, {Integer(index)}, span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.Shuffle",
                        [](ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
                          return Shuffle(vectors, indices, span);
                        });
}

// CommReducer
CommReducer::CommReducer(ffi::Array<Var> lhs, ffi::Array<Var> rhs, ffi::Array<PrimExpr> result,
                         ffi::Array<PrimExpr> identity_element, Span span) {
  size_t n_group = result.size();
  CHECK_EQ(lhs.size(), n_group) << "ValueError: The number of vars in `lhs` must equal to the "
                                   "number of elements in `results`";
  CHECK_EQ(rhs.size(), n_group) << "ValueError: The number of vars in `rhs` must equal to the "
                                   "number of elements in `results`";
  CHECK_EQ(identity_element.size(), n_group)
      << "ValueError: The number of identities must equal to the number of elements in `results`";

  // Change the dtype of input vars to adapt to the dtype of identities
  ffi::ArrayObj* p_lhs = lhs.CopyOnWrite();
  ffi::ArrayObj* p_rhs = rhs.CopyOnWrite();
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  var_map.reserve(n_group * 2);
  for (int i = 0; i < static_cast<int>(n_group); ++i) {
    DataType dtype = identity_element[i].dtype();
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
  ICHECK_EQ(a.size(), b.size());
  ICHECK_EQ(lhs.size(), a.size());
  ICHECK_EQ(rhs.size(), b.size());
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
      .def("tir.CommReducer",
           [](ffi::Array<Var> lhs, ffi::Array<Var> rhs, ffi::Array<PrimExpr> result,
              ffi::Array<PrimExpr> identity_element,
              Span span) { return CommReducer(lhs, rhs, result, identity_element, span); })
      .def_method("tir.CommReducerCombine", &tir::CommReducerNode::operator());
}

// Reduce
Reduce::Reduce(CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
               PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
  for (size_t i = 0; i < axis.size(); ++i) {
    ICHECK_EQ(axis[i]->iter_type, kCommReduce) << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = ffi::make_object<ReduceNode>();
  ICHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    ICHECK(axis[i].defined());
  }
  if (!init.empty()) {
    ICHECK_EQ(init.size(), source.size()) << "Number of inits should match number of exprs";
    for (size_t i = 0; i < init.size(); i++) {
      ICHECK(init[i].defined()) << "Init value must be defined";
      ICHECK(init[i]->IsInstance<ProducerLoadNode>() || init[i]->IsInstance<IntImmNode>() ||
             init[i]->IsInstance<FloatImmNode>())
          << "init can only be a IntImm, FloatImm or ProducerLoad, "
          << "but received " << init[i] << " of type " << init[i]->GetTypeKey();
    }
  }
  n->dtype = source[value_index].dtype();
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
      "tir.Reduce", [](CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
                       PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
        return Reduce(combiner, source, axis, condition, value_index, init, span);
      });
}

// BufferLoad
void BufferLoadNode::LegalizeDType() {
  // for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
  //   ICHECK(indices[i].dtype().is_scalar())
  //       << "Only the last index of a buffer access may be a vector type.";
  // }

  if (indices.empty()) {
    this->dtype = buffer->dtype;
  } else {
    auto index_dtype = indices.back().dtype();
    bool is_buffer_dtype_scalable = buffer->dtype.is_scalable_vector();
    bool is_index_scalable = index_dtype.is_scalable_vector();

    ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
        << "Index dtype and buffer dtype can't both be scalable.";

    if (is_index_scalable) {
      this->dtype = buffer->dtype.with_scalable_vscale_factor(index_dtype.vscale_factor() *
                                                              buffer->dtype.lanes());
    } else if (is_buffer_dtype_scalable) {
      this->dtype = buffer->dtype.with_scalable_vscale_factor(buffer->dtype.vscale_factor() *
                                                              index_dtype.lanes());
    } else {
      this->dtype = buffer->dtype.with_lanes(index_dtype.lanes() * buffer->dtype.lanes());
    }
  }
}

BufferLoad::BufferLoad(Buffer buffer, ffi::Array<PrimExpr> indices,
                       ffi::Optional<PrimExpr> predicate, Span span) {
  ICHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  if (predicate.defined()) {
    DataType predicate_dtype = predicate.value().dtype();

    bool is_index_scalable = indices.empty() ? false : indices.back().dtype().is_scalable_vector();
    bool is_predicate_scalable = predicate_dtype.is_scalable_vector();
    ICHECK_EQ(is_index_scalable, is_predicate_scalable)
        << "Predicate mask dtype and load indices must both be scalable.";

    int buffer_lanes = buffer->dtype.get_lanes_or_vscale_factor();
    int index_lanes = indices.empty() ? 1 : indices.back().dtype().get_lanes_or_vscale_factor();
    int predicate_lanes = predicate_dtype.get_lanes_or_vscale_factor();
    ICHECK_EQ(index_lanes * buffer_lanes, predicate_lanes)
        << "Got a predicate mask with " << predicate_lanes
        << " lanes, but trying to load a vector with " << index_lanes
        << " lanes. The number of lanes must match.";

    DataType predicate_element_dtype = predicate_dtype.element_of();
    ICHECK(predicate_element_dtype.is_predicate_dtype())
        << "Predicate mask elements must be boolean values, but got " << predicate_element_dtype
        << ".";
  }

  ObjectPtr<BufferLoadNode> node = ffi::make_object<BufferLoadNode>();
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  node->predicate = std::move(predicate);
  node->span = std::move(span);
  node->LegalizeDType();
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.BufferLoad", [](Buffer buffer, ffi::Array<PrimExpr> indices,
                                             ffi::Optional<PrimExpr> predicate, Span span) {
    return BufferLoad(buffer, indices, predicate, span);
  });
}

// ProducerLoad
ProducerLoad::ProducerLoad(DataProducer producer, ffi::Array<PrimExpr> indices, Span span) {
  ObjectPtr<ProducerLoadNode> node = ffi::make_object<ProducerLoadNode>();
  node->dtype = producer->GetDataType();
  node->producer = std::move(producer);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.ProducerLoad",
                        [](DataProducer producer, ffi::Array<PrimExpr> indices, Span span) {
                          return ProducerLoad(producer, indices, span);
                        });
}

}  // namespace tir
}  // namespace tvm
