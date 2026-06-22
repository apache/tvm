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

#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/source_map.h>
#include <tvm/runtime/data_type.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>

namespace tvm {

// Forward-declare VirtualDevice to avoid circular imports.
class VirtualDevice;

/*!
 * \brief Type is the base type of all types.
 *
 * TVM's type system contains following subclasses:
 *
 * - PrimType: type of primitive type values used in the low-level IR.
 * - FuncType: type of a function.
 * - TensorType: type of certain Tensor values in the expression.
 *
 * There are also advanced types to support generic(polymorphic types).
 * \sa Type
 */
class TypeNode : public ffi::Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // span do not participate in structural equal and hash.
    refl::ObjectDef<TypeNode>().def_ro("span", &TypeNode::span, refl::DefaultValue(Span()),
                                       refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  static constexpr const uint32_t _type_child_slots = 14;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Type", TypeNode, ffi::Object);
};

/*!
 * \brief Managed reference to TypeNode.
 * \sa TypeNode
 */
class Type : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Type, ffi::ObjectRef, TypeNode);
};

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class BaseExprNode : public ffi::Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  /*!
   * \brief The deduced or annotated type of the expression.
   *
   * This field is intentionally nullable because type information may
   * be populated by later analysis passes instead of expression
   * constructors.
   */
  mutable Type ty;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // span and ty do not participate in structural equal and hash.
    refl::ObjectDef<BaseExprNode>()
        .def_ro("span", &BaseExprNode::span, refl::DefaultValue(Span()),
                refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("ty", &BaseExprNode::ty, refl::DefaultValue(Type()),
                refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  static constexpr const uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.BaseExpr", BaseExprNode, ffi::Object);
};

/*!
 * \brief Managed reference to BaseExprNode.
 * \sa BaseExprNode
 */
class BaseExpr : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BaseExpr, ffi::ObjectRef, BaseExprNode);
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

  static constexpr const uint32_t _type_child_slots = 40;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.PrimExpr", PrimExprNode, BaseExprNode);
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

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimExpr, BaseExpr, PrimExprNode);

  /*!
   * \brief construct from string to form a StringImm.
   * \param value The value to be constructed.
   */
  TVM_DLL static PrimExpr ConvertFallbackValue(ffi::String value);  // NOLINT(*)
};

/*!
 * \brief Base class for other IR constructs that can be converted to PrimExpr.
 * This is useful for the FFI to convert the expressions to PrimExpr.
 * \sa PrimExpr
 */
class PrimExprConvertibleNode : public ffi::Object {
 public:
  virtual ~PrimExprConvertibleNode() {}
  virtual PrimExpr ToPrimExpr() const = 0;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.PrimExprConvertible", PrimExprConvertibleNode, ffi::Object);
};

/*!
 * \brief Managed reference to PrimExprConvertibleNode.
 * \sa PrimExprConvertibleNode
 */
class PrimExprConvertible : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimExprConvertible, ffi::ObjectRef,
                                             PrimExprConvertibleNode);
};

namespace ffi {
// define automatic conversion from bool, int64_t, double, ffi::String to PrimExpr
// These functions are declared early to avoid circular dependency
template <>
inline constexpr bool use_default_type_traits_v<PrimExpr> = false;

template <>
struct TypeTraits<PrimExpr>
    : public ObjectRefWithFallbackTraitsBase<PrimExpr, StrictBool, int64_t, double, ffi::String,
                                             PrimExprConvertible> {
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(StrictBool value);
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(int64_t value);
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(double value);
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(ffi::String value) {
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
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RelaxExprNode>();
  }

  static constexpr const uint32_t _type_child_slots = 22;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.RelaxExpr", RelaxExprNode, BaseExprNode);
};

/*!
 * \brief Managed reference to RelaxExprNode.
 * \sa RelaxExprNode
 */
class RelaxExpr : public BaseExpr {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(RelaxExpr, BaseExpr, RelaxExprNode);
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.GlobalVar", GlobalVarNode, RelaxExprNode);
};

/*!
 * \brief Managed reference to GlobalVarNode.
 * \sa GlobalVarNode
 */
class GlobalVar : public RelaxExpr {
 public:
  TVM_DLL explicit GlobalVar(ffi::String name_hint, Span span = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GlobalVar, RelaxExpr, GlobalVarNode);
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IntImm", IntImmNode, PrimExprNode);
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

  /*!
   * \brief Construct a scalar boolean constant.
   * \param value The boolean value.
   * \param span The location of this object in the source code.
   */
  static IntImm Bool(bool value, Span span = Span()) {
    return IntImm(DataType::Bool(), value, span);
  }

  /*!
   * \brief Construct a scalar int32 constant.
   * \param value The integer value.
   * \param span The location of this object in the source code.
   */
  static IntImm Int32(int64_t value, Span span = Span()) {
    return IntImm(DataType::Int(32), value, span);
  }

  /*!
   * \brief Construct a scalar int64 constant.
   * \param value The integer value.
   * \param span The location of this object in the source code.
   */
  static IntImm Int64(int64_t value, Span span = Span()) {
    return IntImm(DataType::Int(64), value, span);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntImm, PrimExpr, IntImmNode);
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FloatImm", FloatImmNode, PrimExprNode);
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

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FloatImm, PrimExpr, FloatImmNode);
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
inline constexpr bool use_default_type_traits_v<FloatImm> = false;

template <>
struct TypeTraits<FloatImm> : public ObjectRefWithFallbackTraitsBase<FloatImm, double> {
  TVM_FFI_INLINE static FloatImm ConvertFallbackValue(double value) {
    return FloatImm(runtime::DataType::Float(32), value);
  }
};

// define automatic conversion from bool, int64_t, double to PrimExpr
TVM_FFI_INLINE PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(StrictBool value) {
  return IntImm::Bool(value);
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
