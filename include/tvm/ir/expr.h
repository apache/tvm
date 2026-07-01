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
 * \file tvm/ir/expr.h
 * \brief Base expr nodes in TVM.
 */
#ifndef TVM_IR_EXPR_H_
#define TVM_IR_EXPR_H_

#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/base_expr.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/source_map.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

namespace tvm {

// Forward-declare VirtualDevice to avoid circular imports.
class VirtualDevice;

/*!
 * \brief add operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator+(PrimExpr a, PrimExpr b);

/*!
 * \brief subtraction operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator-(PrimExpr a, PrimExpr b);

/*!
 * \brief negation.
 *
 * \param a input.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator-(PrimExpr a);

/*!
 * \brief multiplication operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator*(PrimExpr a, PrimExpr b);

/*!
 * \brief division operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator/(PrimExpr a, PrimExpr b);

/*!
 * \brief left shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<<(PrimExpr a, PrimExpr b);

/*!
 * \brief right shift operator
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>>(PrimExpr a, PrimExpr b);

/*!
 * \brief greater
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>(PrimExpr a, PrimExpr b);

/*!
 * \brief greater_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator>=(PrimExpr a, PrimExpr b);

/*!
 * \brief less
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<(PrimExpr a, PrimExpr b);

/*!
 * \brief less_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator<=(PrimExpr a, PrimExpr b);

/*!
 * \brief equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator==(PrimExpr a, PrimExpr b);

/*!
 * \brief not_equal
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator!=(PrimExpr a, PrimExpr b);

/*!
 * \brief and
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator&&(PrimExpr a, PrimExpr b);

/*!
 * \brief or
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator||(PrimExpr a, PrimExpr b);

/*!
 * \brief not
 *
 * \param a left operand
 * \return The result expression.
 * \note This operator does eager constant folding.
 */
TVM_DLL PrimExpr operator!(PrimExpr a);

/*!
 * \brief take bitwise and of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator&(PrimExpr a, PrimExpr b);

/*!
 * \brief take bitwise or of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator|(PrimExpr a, PrimExpr b);

/*!
 * \brief take bitwise xor of two values
 *
 * \param a left operand
 * \param b right operand
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator^(PrimExpr a, PrimExpr b);

/*!
 * \brief take bitwise negation of two values
 *
 * \param a the input expression.
 * \return The result expression.
 * \note this function does eager constant folding for
 *       index types(int32, int64) when possible.
 */
TVM_DLL PrimExpr operator~(PrimExpr a);

class GlobalVar;
/*!
 * \brief Global variable that lives in the top-level module.
 *
 * A GlobalVar only refers to function definitions.
 * This is used to enable recursive calls between function.
 *
 * \sa GlobalVarNode
 */
class GlobalVarNode : public ExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  ffi::String name_hint;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GlobalVarNode>().def_ro("name_hint", &GlobalVarNode::name_hint);
  }

  bool SEqual(const GlobalVarNode* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
    return equal(name_hint, other->name_hint, false, "name_hint");
  }

  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
    return hash(name_hint, init_hash, false);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GlobalVar", GlobalVarNode, ExprNode);
};

/*!
 * \brief Managed reference to GlobalVarNode.
 * \sa GlobalVarNode
 */
class GlobalVar : public Expr {
 public:
  TVM_DLL explicit GlobalVar(ffi::String name_hint, Span span = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GlobalVar, Expr, GlobalVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalVarNode);
};

/*!
 * \brief Call corresponds to callable invocation.
 */
class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator/function being invoked.
   *
   * It can be an Op, a GlobalVar, a local function value, or another callable
   * expression.
   */
  Expr op;

  /*! \brief The arguments of the call. */
  ffi::Array<Expr> args;

  /*! \brief The additional attributes. */
  Attrs attrs;

  /*! \brief The type information arguments passed to the callee. */
  ffi::Array<Type> ty_args;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CallNode>()
        .def_ro("op", &CallNode::op)
        .def_ro("args", &CallNode::args)
        .def_ro("attrs", &CallNode::attrs)
        .def_ro("ty_args", &CallNode::ty_args);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Call", CallNode, ExprNode);
};

/*!
 * \brief Managed reference to CallNode.
 */
class Call : public Expr {
 public:
  TVM_DLL Call(Type ret_ty, Expr op, ffi::Array<Expr> args, Attrs attrs = Attrs(),
               ffi::Array<Type> ty_args = ffi::Array<Type>(), Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Call, Expr, CallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
};

/*!
 * \brief Constant integer literals in the program.
 * \sa IntImm
 */
class IntImmNode : public ExprNode {
 public:
  /*! \brief the Internal value. */
  int64_t value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntImmNode>().def_ro("value", &IntImmNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IntImm", IntImmNode, ExprNode);
};

/*!
 * \brief Managed reference class to IntImmNode.
 *
 * \sa IntImmNode
 */
class IntImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param value_ty The primitive type of the value.
   * \param value The internal value.
   * \param span The location of this object in the source code.
   */
  TVM_DLL IntImm(PrimType value_ty, int64_t value, Span span = Span());

  /*!
   * \brief Construct a scalar boolean constant.
   * \param value The boolean value.
   * \param span The location of this object in the source code.
   */
  static IntImm Bool(bool value, Span span = Span()) {
    return IntImm(PrimType::Bool(), value, span);
  }

  /*!
   * \brief Construct a scalar int32 constant.
   * \param value The integer value.
   * \param span The location of this object in the source code.
   */
  static IntImm Int32(int64_t value, Span span = Span()) {
    return IntImm(PrimType::Int(32), value, span);
  }

  /*!
   * \brief Construct a scalar int64 constant.
   * \param value The integer value.
   * \param span The location of this object in the source code.
   */
  static IntImm Int64(int64_t value, Span span = Span()) {
    return IntImm(PrimType::Int(64), value, span);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntImm, PrimExpr, IntImmNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IntImmNode);
};

/*!
 * \brief Constant floating point literals in the program.
 * \sa FloatImm
 */
class FloatImmNode : public ExprNode {
 public:
  /*! \brief The constant value content. */
  double value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FloatImmNode>().def_ro("value", &FloatImmNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FloatImm", FloatImmNode, ExprNode);
};

/*!
 * \brief Managed reference class to FloatImmNode.
 *
 * \sa FloatImmNode
 */
class FloatImm : public PrimExpr {
 public:
  /*!
   * \brief Constructor.
   * \param value_ty The primitive type of the value.
   * \param value The internal value.
   * \param span The location in the source code.
   */
  TVM_DLL FloatImm(PrimType value_ty, double value, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloatImm, PrimExpr, FloatImmNode);
  static constexpr bool _type_container_is_exact = true;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloatImmNode);
};

/*! \brief range over one dimension */
class RangeNode : public ffi::Object {
 public:
  /*! \brief beginning of the node */
  PrimExpr min;
  /*! \brief the extend of range */
  PrimExpr extent;
  /*! \brief the location of this range in the source */
  mutable Span span;
  /*! \brief constructor */
  RangeNode() {}
  RangeNode(PrimExpr min, PrimExpr extent, Span span = Span())
      : min(min), extent(extent), span(span) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RangeNode>()
        .def_ro("min", &RangeNode::min)
        .def_ro("extent", &RangeNode::extent)
        .def_ro("span", &RangeNode::span, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Range", RangeNode, ffi::Object);
};

/*! \brief Range container  */
class Range : public ffi::ObjectRef {
 public:
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   * \param span The location of the Range in the source.
   */
  TVM_DLL Range(PrimExpr begin, PrimExpr end, Span span = Span());
  /*!
   * \brief construct a new range with min and extent
   *  The corresponding constructor is removed,
   *  because that is counter convention of tradition meaning
   *  of range(begin, end)
   *
   * \param min The minimum range.
   * \param extent The extent of the range.
   * \param span The location of the Range in the source.
   */
  TVM_DLL static Range FromMinExtent(PrimExpr min, PrimExpr extent, Span span = Span());
  // declare range.
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Range, ffi::ObjectRef, RangeNode);
};

namespace ffi {
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, IntImmNode> = true;
template <>
inline constexpr bool object_ref_contains_v<PrimExpr, FloatImmNode> = true;

// Type traits to enable automatic conversion into IntImm, Integer, and Bool
// when called through the FFI
template <>
inline constexpr bool use_default_type_traits_v<IntImm> = false;

// specialize to enable implicit conversion from const char*
template <>
struct TypeTraits<IntImm> : public ObjectRefWithFallbackTraitsBase<IntImm, int64_t> {
  TVM_FFI_INLINE static IntImm ConvertFallbackValue(int64_t value) {
    auto value_ty =
        (value > std::numeric_limits<int>::max() || value < std::numeric_limits<int>::min())
            ? PrimType::Int(64)
            : PrimType::Int(32);
    return IntImm(value_ty, value);
  }
};

template <>
inline constexpr bool use_default_type_traits_v<FloatImm> = false;

template <>
struct TypeTraits<FloatImm> : public ObjectRefWithFallbackTraitsBase<FloatImm, double> {
  TVM_FFI_INLINE static FloatImm ConvertFallbackValue(double value) {
    return FloatImm(PrimType::Float(32), value);
  }
};
}  // namespace ffi
}  // namespace tvm

/* \brief Allow tvm.GLobalVar as key in STL tables
 *
 * For most IR expressions, it would be ambiguous whether the
 * expression should follow reference equality or structural equality.
 * This is not the case for variables, which do not contain nested
 * internal structure, and are frequently used as keys in lookup
 * tables.
 *
 * Providing `std::hash` and `std::equal_to` specializations for
 * `tvm::GlobalVar` allows it to be used as a key in STL tables.  For
 * other IR expressions, the user must specify the type of equality
 * used (e.g. `std::unordered_set<T, StructuralHash, StructuralEqual>`
 * or `std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>`).
 */
template <>
struct std::hash<tvm::GlobalVar> {
  std::size_t operator()(const tvm::GlobalVar& var) const { return tvm::ffi::ObjectPtrHash()(var); }
};

template <>
struct std::equal_to<tvm::GlobalVar> {
  bool operator()(const tvm::GlobalVar& var_a, const tvm::GlobalVar& var_b) const {
    return tvm::ffi::ObjectPtrEqual()(var_a, var_b);
  }
};
#endif  // TVM_IR_EXPR_H_
