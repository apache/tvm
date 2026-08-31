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
#include <tvm/ir/node_functor.h>
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

/*! \brief ffi::String constants, only used in asserts. */
class StringImmNode : public ExprNode {
 public:
  /*! \brief The constant value content. */
  ffi::String value;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StringImmNode>().def_ro("value", &StringImmNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.StringImm", StringImmNode, ExprNode);
};

/*!
 * \brief Managed reference to StringImmNode.
 * \sa StringImmNode
 */
class StringImm : public PrimExpr {
 public:
  TVM_DLL StringImm(ffi::String value, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StringImm, PrimExpr, StringImmNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(StringImmNode);
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class CastNode : public ExprNode {
 public:
  /*! \brief Original data type. */
  PrimExpr value;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CastNode>().def_ro("value", &CastNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Cast", CastNode, ExprNode);
};

/*!
 * \brief Managed reference to CastNode
 * \sa CastNode
 */
class Cast : public PrimExpr {
 public:
  TVM_DLL Cast(PrimType value_ty, PrimExpr value, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Cast, PrimExpr, CastNode);
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
  static constexpr const char* _type_key = "ir.prim.Add";
};

/*!
 * \brief Managed reference to AddNode
 * \sa AddNode
 */
class Add : public PrimExpr {
 public:
  TVM_DLL Add(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Add, PrimExpr, AddNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AddNode);
};

/*! \brief a - b */
class SubNode : public BinaryOpNode<SubNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.Sub";
};

/*!
 * \brief Managed reference to SubNode
 * \sa SubNode
 */
class Sub : public PrimExpr {
 public:
  TVM_DLL Sub(PrimExpr a, PrimExpr b, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Sub, PrimExpr, SubNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SubNode);
};

/*! \brief a * b */
class MulNode : public BinaryOpNode<MulNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.Mul";
};

/*!
 * \brief Managed reference to MulNode
 * \sa MulNode
 */
class Mul : public PrimExpr {
 public:
  TVM_DLL Mul(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Mul, PrimExpr, MulNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MulNode);
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class DivNode : public BinaryOpNode<DivNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.Div";
};

/*!
 * \brief Managed reference to DivNode
 * \sa DivNode
 */
class Div : public PrimExpr {
 public:
  TVM_DLL Div(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Div, PrimExpr, DivNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DivNode);
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class ModNode : public BinaryOpNode<ModNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.Mod";
};

/*!
 * \brief Managed reference to ModNode
 * \sa ModNode
 */
class Mod : public PrimExpr {
 public:
  TVM_DLL Mod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Mod, PrimExpr, ModNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ModNode);
};

/*! \brief Floor division, floor(a/b) */
class FloorDivNode : public BinaryOpNode<FloorDivNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.FloorDiv";
};

/*!
 * \brief Managed reference to FloorDivNode
 * \sa FloorDivNode
 */
class FloorDiv : public PrimExpr {
 public:
  TVM_DLL FloorDiv(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloorDiv, PrimExpr, FloorDivNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorDivNode);
};

/*! \brief The remainder of the floordiv */
class FloorModNode : public BinaryOpNode<FloorModNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.FloorMod";
};

/*!
 * \brief Managed reference to FloorModNode
 * \sa FloorModNode
 */
class FloorMod : public PrimExpr {
 public:
  TVM_DLL FloorMod(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloorMod, PrimExpr, FloorModNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloorModNode);
};

/*! \brief min(a, b) */
class MinNode : public BinaryOpNode<MinNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.Min";
};

/*!
 * \brief Managed reference to MinNode
 * \sa MinNode
 */
class Min : public PrimExpr {
 public:
  TVM_DLL Min(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Min, PrimExpr, MinNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MinNode);
};

/*! \brief max(a, b) */
class MaxNode : public BinaryOpNode<MaxNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.Max";
};

/*!
 * \brief Managed reference to MaxNode
 * \sa MaxNode
 */
class Max : public PrimExpr {
 public:
  TVM_DLL Max(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Max, PrimExpr, MaxNode);
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
  static constexpr const char* _type_key = "ir.prim.EQ";
};

/*!
 * \brief Managed reference to EQNode
 * \sa EQNode
 */
class EQ : public PrimExpr {
 public:
  TVM_DLL EQ(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(EQ, PrimExpr, EQNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EQNode);
};

/*! \brief a != b */
class NENode : public CmpOpNode<NENode> {
 public:
  static constexpr const char* _type_key = "ir.prim.NE";
};

/*!
 * \brief Managed reference to NENode
 * \sa NENode
 */
class NE : public PrimExpr {
 public:
  TVM_DLL NE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(NE, PrimExpr, NENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NENode);
};

/*! \brief a < b */
class LTNode : public CmpOpNode<LTNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.LT";
};

/*!
 * \brief Managed reference to LTNode
 * \sa LTNode
 */
class LT : public PrimExpr {
 public:
  TVM_DLL LT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LT, PrimExpr, LTNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LTNode);
};

/*! \brief a <= b */
struct LENode : public CmpOpNode<LENode> {
 public:
  static constexpr const char* _type_key = "ir.prim.LE";
};

/*!
 * \brief Managed reference to LENode
 * \sa LENode
 */
class LE : public PrimExpr {
 public:
  TVM_DLL LE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LE, PrimExpr, LENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LENode);
};

/*! \brief a > b */
class GTNode : public CmpOpNode<GTNode> {
 public:
  static constexpr const char* _type_key = "ir.prim.GT";
};

/*!
 * \brief Managed reference to GTNode
 * \sa GTNode
 */
class GT : public PrimExpr {
 public:
  TVM_DLL GT(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GT, PrimExpr, GTNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GTNode);
};

/*! \brief a >= b */
class GENode : public CmpOpNode<GENode> {
 public:
  static constexpr const char* _type_key = "ir.prim.GE";
};

/*!
 * \brief Managed reference to GENode
 * \sa GENode
 */
class GE : public PrimExpr {
 public:
  TVM_DLL GE(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GE, PrimExpr, GENode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GENode);
};

/*! \brief a && b */
class AndNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AndNode>().def_ro("a", &AndNode::a).def_ro("b", &AndNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.And", AndNode, ExprNode);
};

/*!
 * \brief Managed reference to AndNode
 * \sa AndNode
 */
class And : public PrimExpr {
 public:
  TVM_DLL And(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(And, PrimExpr, AndNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AndNode);
};

/*! \brief a || b */
class OrNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  PrimExpr a;
  /*! \brief The right operand. */
  PrimExpr b;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrNode>().def_ro("a", &OrNode::a).def_ro("b", &OrNode::b);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Or", OrNode, ExprNode);
};

/*!
 * \brief Managed reference to OrNode
 * \sa OrNode
 */
class Or : public PrimExpr {
 public:
  TVM_DLL Or(PrimExpr a, PrimExpr b, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Or, PrimExpr, OrNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(OrNode);
};

/*! \brief !a */
class NotNode : public ExprNode {
 public:
  /*! \brief The input operand. */
  PrimExpr a;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<NotNode>().def_ro("a", &NotNode::a);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Not", NotNode, ExprNode);
};

/*!
 * \brief Managed reference to NotNode
 * \sa NotNode
 */
class Not : public PrimExpr {
 public:
  TVM_DLL Not(PrimExpr a, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Not, PrimExpr, NotNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(NotNode);
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Select", SelectNode, ExprNode);
};

/*!
 * \brief Managed reference to SelectNode
 * \sa SelectNode
 */
class Select : public PrimExpr {
 public:
  TVM_DLL Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Select, PrimExpr, SelectNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SelectNode);
};

/*!
 * \brief Let binding. Bind var to value then evaluate body.
 */
class LetNode : public ExprNode {
 public:
  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be binded. */
  PrimExpr value;
  /*! \brief The result expression. */
  PrimExpr body;
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LetNode>()
        // TODO(tqchen): use SEqHashDefNonRecursive after the next pypi tvm-ffi release
        .def_ro("var", &LetNode::var, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("value", &LetNode::value)
        .def_ro("body", &LetNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.prim.Let", LetNode, ExprNode);
};

/*!
 * \brief Managed reference to LetNode
 * \sa LetNode
 */
class Let : public PrimExpr {
 public:
  TVM_DLL Let(Var var, PrimExpr value, PrimExpr body, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Let, PrimExpr, LetNode);
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
}  // namespace prim

namespace ffi {

template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::StringImmNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::CastNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, prim::AddNode> = true;
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
template <>
inline constexpr bool use_default_type_traits_v<tvm::prim::StringImm> = false;

template <>
struct TypeTraits<tvm::prim::StringImm>
    : public ObjectRefWithFallbackTraitsBase<tvm::prim::StringImm, ffi::String> {
  TVM_FFI_INLINE static tvm::prim::StringImm ConvertFallbackValue(ffi::String value) {
    return tvm::prim::StringImm(value);
  }
};
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_IR_PRIM_EXPR_H_
