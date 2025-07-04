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

#include <tvm/ffi/reflection/reflection.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/source_map.h>
#include <tvm/ir/type.h>
#include <tvm/node/node.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>

namespace tvm {

using tvm::String;

// Forward-declare VirtualDevice to avoid circular imports.
class VirtualDevice;

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class BaseExprNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BaseExprNode>().def_ro("span", &BaseExprNode::span, refl::DefaultValue(Span()));
  }

  static constexpr const char* _type_key = "ir.BaseExpr";

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 64;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseExprNode, Object);
};

/*!
 * \brief Managed reference to BaseExprNode.
 * \sa BaseExprNode
 */
class BaseExpr : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseExpr, ObjectRef, BaseExprNode);
};

/*!
 * \brief Base node of all primitive expressions.
 *
 *  A primitive expression deals with low-level
 *  POD data types and handles without
 *  doing life-cycle management for objects.
 *
 *  PrimExpr is used in the low-level code
 *  optimizations and integer analysis.
 *
 * \sa PrimExpr
 */
class PrimExprNode : public BaseExprNode {
 public:
  /*!
   * \brief The runtime data type of the primitive expression.
   *
   * runtime::DataType(dtype) provides coarse grained type information
   * during compile time and runtime. It is eagerly built in
   * PrimExpr expression construction and can be used for
   * quick type checking.
   *
   * dtype is sufficient to decide the Type of the PrimExpr
   * when it corresponds to POD value types such as i32.
   *
   * When dtype is DataType::Handle(), the expression could corresponds to
   * a more fine-grained Type, and we can get the type by running lazy type inference.
   */
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimExprNode>().def_ro("dtype", &PrimExprNode::dtype);
  }

  TVM_OBJECT_ENABLE_SCRIPT_PRINTER();

  static constexpr const char* _type_key = "ir.PrimExpr";
  static constexpr const uint32_t _type_child_slots = 40;
  TVM_DECLARE_BASE_OBJECT_INFO(PrimExprNode, BaseExprNode);
};

/*!
 * \brief Reference to PrimExprNode.
 * \sa PrimExprNode
 */
class PrimExpr : public BaseExpr {
 public:
  /*!
   * \brief construct from integer.
   * \param value The value to be constructed.
   */
  TVM_DLL PrimExpr(int32_t value);  // NOLINT(*)
  /*!
   * \brief construct from float.
   * \param value The value to be constructed.
   */
  TVM_DLL PrimExpr(float value);  // NOLINT(*)

  /*! \return the data type of this expression. */
  DataType dtype() const { return static_cast<const PrimExprNode*>(get())->dtype; }

  TVM_DEFINE_OBJECT_REF_METHODS(PrimExpr, BaseExpr, PrimExprNode);

  /*!
   * \brief construct from string to form a StringImm.
   * \param value The value to be constructed.
   */
  TVM_DLL static PrimExpr ConvertFallbackValue(String value);  // NOLINT(*)
};

/*!
 * \brief Base class for other IR constructs that can be converted to PrimExpr.
 * This is useful for the FFI to convert the expressions to PrimExpr.
 * \sa PrimExpr
 */
class PrimExprConvertibleNode : public Object {
 public:
  virtual ~PrimExprConvertibleNode() {}
  virtual PrimExpr ToPrimExpr() const = 0;

  static constexpr const char* _type_key = "ir.PrimExprConvertible";
  TVM_DECLARE_BASE_OBJECT_INFO(PrimExprConvertibleNode, Object);
};

/*!
 * \brief Managed reference to PrimExprConvertibleNode.
 * \sa PrimExprConvertibleNode
 */
class PrimExprConvertible : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PrimExprConvertible, ObjectRef, PrimExprConvertibleNode);
};

namespace ffi {
// define automatic conversion from bool, int64_t, double, String to PrimExpr
// These functions are declared early to avoid circular dependency
template <>
inline constexpr bool use_default_type_traits_v<PrimExpr> = false;

template <>
struct TypeTraits<PrimExpr>
    : public ObjectRefWithFallbackTraitsBase<PrimExpr, StrictBool, int64_t, double, String,
                                             PrimExprConvertible> {
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(StrictBool value);
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(int64_t value);
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(double value);
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(String value) {
    return PrimExpr::ConvertFallbackValue(value);
  }
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(PrimExprConvertible value) {
    return value->ToPrimExpr();
  }
};
}  // namespace ffi

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

/*!
 * \brief Base node of all non-primitive expressions.
 *
 * RelaxExpr supports tensor and functions as first class citizen.
 * The life-cycle of the corresponding
 * objects are implicitly managed by the language.
 *
 * \sa RelaxExpr
 */
class RelaxExprNode : public BaseExprNode {
 public:
  /*!
   * \brief Stores the result of structure information of the
   *        expression that encapsulate both static shape and
   *        runtime information such as shape.
   */
  mutable Optional<ObjectRef> struct_info_ = Optional<ObjectRef>();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RelaxExprNode>().def_ro("struct_info_", &RelaxExprNode::struct_info_);
  }

  static constexpr const char* _type_key = "ir.RelaxExpr";
  static constexpr const uint32_t _type_child_slots = 22;
  TVM_DECLARE_BASE_OBJECT_INFO(RelaxExprNode, BaseExprNode);
};

/*!
 * \brief Managed reference to RelaxExprNode.
 * \sa RelaxExprNode
 */
class RelaxExpr : public BaseExpr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RelaxExpr, BaseExpr, RelaxExprNode);
};

class GlobalVar;
/*!
 * \brief Global variable that lives in the top-level module.
 *
 * A GlobalVar only refers to function definitions.
 * This is used to enable recursive calls between function.
 *
 * \sa GlobalVarNode
 */
class GlobalVarNode : public RelaxExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  String name_hint;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GlobalVarNode>().def_ro("name_hint", &GlobalVarNode::name_hint);
  }

  bool SEqualReduce(const GlobalVarNode* other, SEqualReducer equal) const {
    // name matters for global var.
    return equal(name_hint, other->name_hint) && equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name_hint);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "ir.GlobalVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(GlobalVarNode, RelaxExprNode);
};

/*!
 * \brief Managed reference to GlobalVarNode.
 * \sa GlobalVarNode
 */
class GlobalVar : public RelaxExpr {
 public:
  TVM_DLL explicit GlobalVar(String name_hint, Span span = {});

  TVM_DEFINE_OBJECT_REF_METHODS(GlobalVar, RelaxExpr, GlobalVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalVarNode);
};

/*!
 * \brief Constant integer literals in the program.
 * \sa IntImm
 */
class IntImmNode : public PrimExprNode {
 public:
  /*! \brief the Internal value. */
  int64_t value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntImmNode>().def_ro("value", &IntImmNode::value);
  }

  bool SEqualReduce(const IntImmNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.IntImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(IntImmNode, PrimExprNode);
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
   * \param dtype The data type of the value.
   * \param value The internal value.
   * \param span The location of this object in the source code.
   */
  TVM_DLL IntImm(DataType dtype, int64_t value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(IntImm, PrimExpr, IntImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IntImmNode);
};

/*!
 * \brief Constant floating point literals in the program.
 * \sa FloatImm
 */
class FloatImmNode : public PrimExprNode {
 public:
  /*! \brief The constant value content. */
  double value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FloatImmNode>().def_ro("value", &FloatImmNode::value);
  }

  bool SEqualReduce(const FloatImmNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "ir.FloatImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(FloatImmNode, PrimExprNode);
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
   * \param dtype The data type of the value.
   * \param value The internal value.
   * \param span The location in the source code.
   */
  TVM_DLL FloatImm(DataType dtype, double value, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(FloatImm, PrimExpr, FloatImmNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FloatImmNode);
};

/*!
 * \brief Boolean constant.
 *
 *  This reference type is useful to add additional compile-time
 *  type checks and helper functions for Integer equal comparisons.
 */
class Bool : public IntImm {
 public:
  explicit Bool(bool value, Span span = Span()) : IntImm(DataType::Bool(), value, span) {}
  Bool operator!() const { return Bool((*this)->value == 0); }
  operator bool() const { return (*this)->value != 0; }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Bool, IntImm, IntImmNode);
};

// Overload operators to make sure we have the most fine grained types.
inline Bool operator||(const Bool& a, bool b) { return Bool(a.operator bool() || b); }
inline Bool operator||(bool a, const Bool& b) { return Bool(a || b.operator bool()); }
inline Bool operator||(const Bool& a, const Bool& b) {
  return Bool(a.operator bool() || b.operator bool());
}
inline Bool operator&&(const Bool& a, bool b) { return Bool(a.operator bool() && b); }
inline Bool operator&&(bool a, const Bool& b) { return Bool(a && b.operator bool()); }
inline Bool operator&&(const Bool& a, const Bool& b) {
  return Bool(a.operator bool() && b.operator bool());
}

inline bool operator==(const Bool& a, bool b) { return a.operator bool() == b; }
inline bool operator==(bool a, const Bool& b) { return a == b.operator bool(); }
inline bool operator==(const Bool& a, const Bool& b) {
  return a.operator bool() == b.operator bool();
}

/*!
 * \brief Container of constant int that adds more constructors.
 *
 * This is used to store and automate type check
 * attributes that must be constant integer.
 *
 * \sa IntImm
 */
class Integer : public IntImm {
 public:
  Integer() {}
  /*!
   * \brief constructor from node.
   */
  explicit Integer(ObjectPtr<Object> node) : IntImm(node) {}
  /*!
   * \brief Construct integer from int value.
   */
  Integer(int value, Span span = Span()) : IntImm(DataType::Int(32), value, span) {}  // NOLINT(*)
  /*!
   * \brief Construct integer from int imm.
   * \param other The other value.
   */
  Integer(IntImm other) : IntImm(std::move(other)) {}  // NOLINT(*)
  /*!
   * \brief Constructor from enum
   * \tparam Enum The enum type.
   * \param value The enum value.
   */
  template <typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
  explicit Integer(Enum value) : Integer(static_cast<int>(value)) {
    static_assert(std::is_same<int, typename std::underlying_type<Enum>::type>::value,
                  "declare enum to be enum int to use visitor");
  }
  /*!
   * \brief Assign an expression to integer.
   * \param other another expression.
   */
  Integer& operator=(const IntImm& other) {
    data_ = ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(other);
    return *this;
  }
  /*!
   * \brief convert to int64_t
   */
  int64_t IntValue() const {
    ICHECK(data_ != nullptr) << " Trying to reference a null Integer";
    return (*this)->value;
  }
  // comparators
  Bool operator==(int other) const {
    if (data_ == nullptr) return Bool(false);
    return Bool((*this)->value == other);
  }
  Bool operator!=(int other) const { return !(*this == other); }
  template <typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
  Bool operator==(Enum other) const {
    return *this == static_cast<int>(other);
  }
  template <typename Enum, typename = typename std::enable_if<std::is_enum<Enum>::value>::type>
  Bool operator!=(Enum other) const {
    return *this != static_cast<int>(other);
  }
};

/*! \brief range over one dimension */
class RangeNode : public Object {
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
        .def_ro("extent", &RangeNode::extent);
  }

  bool SEqualReduce(const RangeNode* other, SEqualReducer equal) const {
    return equal(min, other->min) && equal(extent, other->extent);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(min);
    hash_reduce(extent);
  }

  static constexpr const char* _type_key = "ir.Range";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(RangeNode, Object);
};

/*! \brief Range container  */
class Range : public ObjectRef {
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
  static Range FromMinExtent(PrimExpr min, PrimExpr extent, Span span = Span());
  // declare range.
  TVM_DEFINE_OBJECT_REF_METHODS(Range, ObjectRef, RangeNode);
};

namespace ffi {
// Type traits to enable automatic conversion into IntImm, Integer, and Bool
// when called through the FFI
template <>
inline constexpr bool use_default_type_traits_v<IntImm> = false;

// specialize to enable implicit conversion from const char*
template <>
struct TypeTraits<IntImm> : public ObjectRefWithFallbackTraitsBase<IntImm, int64_t> {
  TVM_FFI_INLINE static IntImm ConvertFallbackValue(int64_t value) {
    auto dtype =
        (value > std::numeric_limits<int>::max() || value < std::numeric_limits<int>::min())
            ? DataType::Int(64)
            : DataType::Int(32);
    return IntImm(dtype, value);
  }
};

template <>
inline constexpr bool use_default_type_traits_v<Integer> = false;

template <>
struct TypeTraits<Integer> : public ObjectRefWithFallbackTraitsBase<Integer, int64_t> {
  TVM_FFI_INLINE static Integer ConvertFallbackValue(int64_t value) { return Integer(value); }
};

template <>
inline constexpr bool use_default_type_traits_v<FloatImm> = false;

template <>
struct TypeTraits<FloatImm> : public ObjectRefWithFallbackTraitsBase<FloatImm, double> {
  TVM_FFI_INLINE static FloatImm ConvertFallbackValue(double value) {
    return FloatImm(runtime::DataType::Float(32), value);
  }
};

template <>
inline constexpr bool use_default_type_traits_v<Bool> = false;

template <>
struct TypeTraits<Bool> : public ObjectRefWithFallbackTraitsBase<Bool, int64_t> {
  TVM_FFI_INLINE static Bool ConvertFallbackValue(int64_t value) { return Bool(value != 0); }
};

// define automatic conversion from bool, int64_t, double to PrimExpr
TVM_FFI_INLINE PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(StrictBool value) {
  return IntImm(DataType::Bool(), value, Span());
}

TVM_FFI_INLINE PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(int64_t value) {
  return TypeTraits<IntImm>::ConvertFallbackValue(value);
}

TVM_FFI_INLINE PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(double value) {
  return TypeTraits<FloatImm>::ConvertFallbackValue(value);
}
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
 * or `std::unordered_set<T, ObjectPtrHash, ObjectPtrEqual>`).
 */
template <>
struct std::hash<tvm::GlobalVar> {
  std::size_t operator()(const tvm::GlobalVar& var) const {
    return tvm::runtime::ObjectPtrHash()(var);
  }
};

template <>
struct std::equal_to<tvm::GlobalVar> {
  bool operator()(const tvm::GlobalVar& var_a, const tvm::GlobalVar& var_b) const {
    return tvm::runtime::ObjectPtrEqual()(var_a, var_b);
  }
};
#endif  // TVM_IR_EXPR_H_
