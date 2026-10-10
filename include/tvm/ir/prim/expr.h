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
 * \file tvm/ir/prim/expr.h
 * \brief TIR expressions.
 */
// Acknowledgement: Many low-level IR nodes originate from Halide.
#ifndef TVM_IR_PRIM_EXPR_H_
#define TVM_IR_PRIM_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/object_functor.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/runtime/base.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace prim {

using IntImmNode = tvm::IntImmNode;
using FloatImmNode = tvm::FloatImmNode;

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class CastNode : public ExprNode {
 public:
  explicit CastNode(PrimExpr value) : value(std::move(value)) {}
  explicit CastNode(ffi::UnsafeInit tag) : value(tag) {}

  /*! \brief Original data type. */
  PrimExpr value;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CastNode>().def_ro("value", &CastNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Cast", CastNode, ExprNode);
};

/*!
 * \brief Managed reference to CastNode
 * \sa CastNode
 */
class Cast : public PrimExpr {
 public:
  TVM_DLL Cast(PrimType value_ty, PrimExpr value, Location loc = UnknownLoc());
  explicit Cast(ffi::ObjectPtr<CastNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Cast, PrimExpr, CastNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CastNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class BinaryOpNode : public ExprNode {
 public:
  BinaryOpNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
  explicit BinaryOpNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}

  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  }
  static const constexpr int _type_child_slots [[maybe_unused]] = 0;
  static const constexpr bool _type_final [[maybe_unused]] = true;
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprNode);
};

/*! \brief a + b */
class AddNode : public BinaryOpNode<AddNode> {
 public:
  using BinaryOpNode<AddNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.Add";
};

/*!
 * \brief Managed reference to AddNode
 * \sa AddNode
 */
class Add : public PrimExpr {
 public:
  TVM_DLL Add(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit Add(ffi::ObjectPtr<AddNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Add, PrimExpr, AddNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AddNode);
};

/*! \brief a << b */
class LShiftNode : public BinaryOpNode<LShiftNode> {
 public:
  using BinaryOpNode<LShiftNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.LShift";
};

/*!
 * \brief Managed reference to LShiftNode
 * \sa LShiftNode
 */
class LShift : public PrimExpr {
 public:
  TVM_DLL LShift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit LShift(ffi::ObjectPtr<LShiftNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LShift, PrimExpr, LShiftNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LShiftNode);
};

/*! \brief a >> b */
class RShiftNode : public BinaryOpNode<RShiftNode> {
 public:
  using BinaryOpNode<RShiftNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.RShift";
};

/*!
 * \brief Managed reference to RShiftNode
 * \sa RShiftNode
 */
class RShift : public PrimExpr {
 public:
  TVM_DLL RShift(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit RShift(ffi::ObjectPtr<RShiftNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RShift, PrimExpr, RShiftNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RShiftNode);
};

/*! \brief a & b */
class BitwiseAndNode : public BinaryOpNode<BitwiseAndNode> {
 public:
  using BinaryOpNode<BitwiseAndNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.BitwiseAnd";
};

/*!
 * \brief Managed reference to BitwiseAndNode
 * \sa BitwiseAndNode
 */
class BitwiseAnd : public PrimExpr {
 public:
  TVM_DLL BitwiseAnd(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit BitwiseAnd(ffi::ObjectPtr<BitwiseAndNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseAnd, PrimExpr, BitwiseAndNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseAndNode);
};

/*! \brief a | b */
class BitwiseOrNode : public BinaryOpNode<BitwiseOrNode> {
 public:
  using BinaryOpNode<BitwiseOrNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.BitwiseOr";
};

/*!
 * \brief Managed reference to BitwiseOrNode
 * \sa BitwiseOrNode
 */
class BitwiseOr : public PrimExpr {
 public:
  TVM_DLL BitwiseOr(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit BitwiseOr(ffi::ObjectPtr<BitwiseOrNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseOr, PrimExpr, BitwiseOrNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseOrNode);
};

/*! \brief a ^ b */
class BitwiseXorNode : public BinaryOpNode<BitwiseXorNode> {
 public:
  using BinaryOpNode<BitwiseXorNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.BitwiseXor";
};

/*!
 * \brief Managed reference to BitwiseXorNode
 * \sa BitwiseXorNode
 */
class BitwiseXor : public PrimExpr {
 public:
  TVM_DLL BitwiseXor(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit BitwiseXor(ffi::ObjectPtr<BitwiseXorNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseXor, PrimExpr, BitwiseXorNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseXorNode);
};

/*! \brief a - b */
class SubNode : public BinaryOpNode<SubNode> {
 public:
  using BinaryOpNode<SubNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.Sub";
};

/*!
 * \brief Managed reference to SubNode
 * \sa SubNode
 */
class Sub : public PrimExpr {
 public:
  TVM_DLL Sub(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());

  explicit Sub(ffi::ObjectPtr<SubNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Sub, PrimExpr, SubNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SubNode);
};

/*! \brief a * b */
class MulNode : public BinaryOpNode<MulNode> {
 public:
  using BinaryOpNode<MulNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.Mul";
};

/*!
 * \brief Managed reference to MulNode
 * \sa MulNode
 */
class Mul : public PrimExpr {
 public:
  TVM_DLL Mul(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit Mul(ffi::ObjectPtr<MulNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Mul, PrimExpr, MulNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MulNode);
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class DivNode : public BinaryOpNode<DivNode> {
 public:
  using BinaryOpNode<DivNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.Div";
};

/*!
 * \brief Managed reference to DivNode
 * \sa DivNode
 */
class Div : public PrimExpr {
 public:
  TVM_DLL Div(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit Div(ffi::ObjectPtr<DivNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Div, PrimExpr, DivNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DivNode);
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class ModNode : public BinaryOpNode<ModNode> {
 public:
  using BinaryOpNode<ModNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.Mod";
};

/*!
 * \brief Managed reference to ModNode
 * \sa ModNode
 */
class Mod : public PrimExpr {
 public:
  TVM_DLL Mod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit Mod(ffi::ObjectPtr<ModNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Mod, PrimExpr, ModNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ModNode);
};

/*! \brief Floor division, floor(a/b) */
class FloorDivNode : public BinaryOpNode<FloorDivNode> {
 public:
  using BinaryOpNode<FloorDivNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.FloorDiv";
};

/*!
 * \brief Managed reference to FloorDivNode
 * \sa FloorDivNode
 */
class FloorDiv : public PrimExpr {
 public:
  TVM_DLL FloorDiv(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit FloorDiv(ffi::ObjectPtr<FloorDivNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloorDiv, PrimExpr, FloorDivNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorDivNode);
};

/*! \brief The remainder of the floordiv */
class FloorModNode : public BinaryOpNode<FloorModNode> {
 public:
  using BinaryOpNode<FloorModNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.FloorMod";
};

/*!
 * \brief Managed reference to FloorModNode
 * \sa FloorModNode
 */
class FloorMod : public PrimExpr {
 public:
  TVM_DLL FloorMod(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit FloorMod(ffi::ObjectPtr<FloorModNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FloorMod, PrimExpr, FloorModNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorModNode);
};

/*! \brief min(a, b) */
class MinNode : public BinaryOpNode<MinNode> {
 public:
  using BinaryOpNode<MinNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.Min";
};

/*!
 * \brief Managed reference to MinNode
 * \sa MinNode
 */
class Min : public PrimExpr {
 public:
  TVM_DLL Min(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit Min(ffi::ObjectPtr<MinNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Min, PrimExpr, MinNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MinNode);
};

/*! \brief max(a, b) */
class MaxNode : public BinaryOpNode<MaxNode> {
 public:
  using BinaryOpNode<MaxNode>::BinaryOpNode;
  static constexpr const char* _type_key = "prim.Max";
};

/*!
 * \brief Managed reference to MaxNode
 * \sa MaxNode
 */
class Max : public PrimExpr {
 public:
  TVM_DLL Max(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit Max(ffi::ObjectPtr<MaxNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Max, PrimExpr, MaxNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MaxNode);
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template <typename T>
class CmpOpNode : public ExprNode {
 public:
  CmpOpNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
  explicit CmpOpNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}

  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<T>().def_ro("a", &T::a).def_ro("b", &T::b);
  }
  static const constexpr int _type_child_slots [[maybe_unused]] = 0;
  static const constexpr bool _type_final [[maybe_unused]] = true;
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, ExprNode);
};

/*! \brief a == b */
class EQNode : public CmpOpNode<EQNode> {
 public:
  using CmpOpNode<EQNode>::CmpOpNode;
  static constexpr const char* _type_key = "prim.EQ";
};

/*!
 * \brief Managed reference to EQNode
 * \sa EQNode
 */
class EQ : public PrimExpr {
 public:
  TVM_DLL EQ(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit EQ(ffi::ObjectPtr<EQNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EQ, PrimExpr, EQNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EQNode);
};

/*! \brief a != b */
class NENode : public CmpOpNode<NENode> {
 public:
  using CmpOpNode<NENode>::CmpOpNode;
  static constexpr const char* _type_key = "prim.NE";
};

/*!
 * \brief Managed reference to NENode
 * \sa NENode
 */
class NE : public PrimExpr {
 public:
  TVM_DLL NE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit NE(ffi::ObjectPtr<NENode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(NE, PrimExpr, NENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NENode);
};

/*! \brief a < b */
class LTNode : public CmpOpNode<LTNode> {
 public:
  using CmpOpNode<LTNode>::CmpOpNode;
  static constexpr const char* _type_key = "prim.LT";
};

/*!
 * \brief Managed reference to LTNode
 * \sa LTNode
 */
class LT : public PrimExpr {
 public:
  TVM_DLL LT(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit LT(ffi::ObjectPtr<LTNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LT, PrimExpr, LTNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LTNode);
};

/*! \brief a <= b */
struct LENode : public CmpOpNode<LENode> {
 public:
  using CmpOpNode<LENode>::CmpOpNode;
  static constexpr const char* _type_key = "prim.LE";
};

/*!
 * \brief Managed reference to LENode
 * \sa LENode
 */
class LE : public PrimExpr {
 public:
  TVM_DLL LE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit LE(ffi::ObjectPtr<LENode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LE, PrimExpr, LENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LENode);
};

/*! \brief a > b */
class GTNode : public CmpOpNode<GTNode> {
 public:
  using CmpOpNode<GTNode>::CmpOpNode;
  static constexpr const char* _type_key = "prim.GT";
};

/*!
 * \brief Managed reference to GTNode
 * \sa GTNode
 */
class GT : public PrimExpr {
 public:
  TVM_DLL GT(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit GT(ffi::ObjectPtr<GTNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GT, PrimExpr, GTNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GTNode);
};

/*! \brief a >= b */
class GENode : public CmpOpNode<GENode> {
 public:
  using CmpOpNode<GENode>::CmpOpNode;
  static constexpr const char* _type_key = "prim.GE";
};

/*!
 * \brief Managed reference to GENode
 * \sa GENode
 */
class GE : public PrimExpr {
 public:
  TVM_DLL GE(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit GE(ffi::ObjectPtr<GENode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(GE, PrimExpr, GENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GENode);
};

/*! \brief a && b */
class AndNode : public ExprNode {
 public:
  AndNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
  explicit AndNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}

  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AndNode>().def_ro("a", &AndNode::a).def_ro("b", &AndNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.And", AndNode, ExprNode);
};

/*!
 * \brief Managed reference to AndNode
 * \sa AndNode
 */
class And : public PrimExpr {
 public:
  TVM_DLL And(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit And(ffi::ObjectPtr<AndNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(And, PrimExpr, AndNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AndNode);
};

/*! \brief a || b */
class OrNode : public ExprNode {
 public:
  OrNode(PrimExpr a, PrimExpr b) : a(std::move(a)), b(std::move(b)) {}
  explicit OrNode(ffi::UnsafeInit tag) : a(tag), b(tag) {}

  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrNode>().def_ro("a", &OrNode::a).def_ro("b", &OrNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Or", OrNode, ExprNode);
};

/*!
 * \brief Managed reference to OrNode
 * \sa OrNode
 */
class Or : public PrimExpr {
 public:
  TVM_DLL Or(PrimExpr a, PrimExpr b, Location loc = UnknownLoc());
  explicit Or(ffi::ObjectPtr<OrNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Or, PrimExpr, OrNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OrNode);
};

/*! \brief !a */
class NotNode : public ExprNode {
 public:
  explicit NotNode(PrimExpr a) : a(std::move(a)) {}
  explicit NotNode(ffi::UnsafeInit tag) : a(tag) {}

  /*! \brief The input operand. */
  PrimExpr a;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NotNode>().def_ro("a", &NotNode::a);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Not", NotNode, ExprNode);
};

/*!
 * \brief Managed reference to NotNode
 * \sa NotNode
 */
class Not : public PrimExpr {
 public:
  TVM_DLL Not(PrimExpr a, Location loc = UnknownLoc());
  explicit Not(ffi::ObjectPtr<NotNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Not, PrimExpr, NotNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NotNode);
};

/*! \brief ~a */
class BitwiseNotNode : public ExprNode {
 public:
  explicit BitwiseNotNode(PrimExpr a) : a(std::move(a)) {}
  explicit BitwiseNotNode(ffi::UnsafeInit tag) : a(tag) {}

  /*! \brief The input operand. */
  PrimExpr a;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BitwiseNotNode>().def_ro("a", &BitwiseNotNode::a);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.BitwiseNot", BitwiseNotNode, ExprNode);
};

/*!
 * \brief Managed reference to BitwiseNotNode
 * \sa BitwiseNotNode
 */
class BitwiseNot : public PrimExpr {
 public:
  TVM_DLL BitwiseNot(PrimExpr a, Location loc = UnknownLoc());
  explicit BitwiseNot(ffi::ObjectPtr<BitwiseNotNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BitwiseNot, PrimExpr, BitwiseNotNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BitwiseNotNode);
};

/*!
 * \brief return true_value if condition is true, otherwise return false_value.
 * \note Both true_value and false_value could be evaluated
 *       regardless of the condition value.
 *       Do not use it to guard against out of bound access,
 *       please use if_then_else instead.
 */
class SelectNode : public ExprNode {
 public:
  SelectNode(PrimExpr condition, PrimExpr true_value, PrimExpr false_value)
      : condition(std::move(condition)),
        true_value(std::move(true_value)),
        false_value(std::move(false_value)) {}
  explicit SelectNode(ffi::UnsafeInit tag) : condition(tag), true_value(tag), false_value(tag) {}

  /*! \brief The condition */
  PrimExpr condition;
  /*! \brief value to be returned when condition is true. */
  PrimExpr true_value;
  /*! \brief value to be returned when condition is false. */
  PrimExpr false_value;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SelectNode>()
        .def_ro("condition", &SelectNode::condition)
        .def_ro("true_value", &SelectNode::true_value)
        .def_ro("false_value", &SelectNode::false_value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Select", SelectNode, ExprNode);
};

/*!
 * \brief Managed reference to SelectNode
 * \sa SelectNode
 */
class Select : public PrimExpr {
 public:
  TVM_DLL Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value,
                 Location loc = UnknownLoc());

  explicit Select(ffi::ObjectPtr<SelectNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Select, PrimExpr, SelectNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SelectNode);
};

/*!
 * \brief Let binding. Bind var to value then evaluate body.
 */
class LetNode : public ExprNode {
 public:
  LetNode(Var var, PrimExpr value, PrimExpr body)
      : var(std::move(var)), value(std::move(value)), body(std::move(body)) {}
  explicit LetNode(ffi::UnsafeInit tag) : var(tag), value(tag), body(tag) {}

  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be binded. */
  PrimExpr value;
  /*! \brief The result expression. */
  PrimExpr body;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LetNode>()
        .def_ro("var", &LetNode::var, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("value", &LetNode::value)
        .def_ro("body", &LetNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("prim.Let", LetNode, ExprNode);
};

/*!
 * \brief Managed reference to LetNode
 * \sa LetNode
 */
class Let : public PrimExpr {
 public:
  TVM_DLL Let(Var var, PrimExpr value, PrimExpr body, Location loc = UnknownLoc());
  explicit Let(ffi::ObjectPtr<LetNode> node) : PrimExpr(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Let, PrimExpr, LetNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LetNode);
};

/*
 * \brief Template function to convert Map to unordered_map
 *  Sometimes useful for API gluing when internal uses unordered_map
 * \param dmap The container map
 * \return The corresponding unordered_map.
 * \tparam K the key of the Map.
 * \tparam V the value of the Map.
 */
template <typename K, typename V>
inline std::unordered_map<K, V> as_unordered_map(const ffi::Map<K, V>& dmap) {
  std::unordered_map<K, V> ret;
  for (auto kv : dmap) {
    ret[kv.first] = kv.second;
  }
  return ret;
}

/*!
 * \brief Compare two expressions recursively and check if they are equal
 *        to each other without var remapping.
 *
 *  This function does not remap variable bindings, it will not
 *  return true for (let x = 1 in x + 1) vs (let y = 1 in y + 1), unless x.same_as(y).
 *
 *  Use StructuralEqual for such cases.
 *
 *  Due to the restriction of not remapping variables, this function can run
 *  faster than StructuralEqual and can be used as a utility function during arithmetic
 *  simplifications.
 *
 * \sa StructuralEqual
 */
struct ExprDeepEqual {
 public:
  TVM_DLL bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const;
};

}  // namespace prim

namespace ffi {

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::CastNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::AddNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseNotNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseXorNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseOrNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::BitwiseAndNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::RShiftNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::LShiftNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::SubNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::MulNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::DivNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::ModNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::FloorDivNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::FloorModNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::MinNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::MaxNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::EQNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::NENode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::LTNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::LENode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::GTNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::GENode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::AndNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::OrNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::NotNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::SelectNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::RampNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::BroadcastNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::LetNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::ShuffleNode> = true;
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_IR_PRIM_EXPR_H_
