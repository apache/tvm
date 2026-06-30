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
 * \file tvm/ir/base_expr.h
 * \brief Base expression and primitive type nodes.
 */
#ifndef TVM_IR_BASE_EXPR_H_
#define TVM_IR_BASE_EXPR_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/source_map.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace tvm {

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
  /*! \brief Sentinel for a type that has not been populated yet. */
  TVM_DLL static Type Missing();

  /*! \return whether this is the missing-type sentinel. */
  TVM_DLL bool IsMissing() const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Type, ffi::ObjectRef, TypeNode);
};

/*!
 * \brief Primitive data types used in the low-level IR.
 *
 * PrimType represents POD-values and handles that are
 * not automatically managed by the runtime.
 *
 * \sa PrimType
 */
class PrimTypeNode final : public TypeNode {
 public:
  /*!
   * \brief The raw DLPack dtype represented by this primitive type.
   */
  DLDataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimTypeNode>().def_ro("dtype", &PrimTypeNode::dtype);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PrimType", PrimTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PrimTypeNode.
 * \sa PrimTypeNode
 */
class PrimType final : public Type {
 public:
  /*!
   * \brief Construct from a raw DLPack dtype.
   * \param dtype The corresponding DLPack dtype.
   */
  TVM_DLL explicit PrimType(DLDataType dtype);

  /*!
   * \brief Construct from DLPack dtype fields.
   * \param code The DLPack dtype code.
   * \param bits The scalar bit width.
   * \param lanes The fixed lane count.
   */
  TVM_DLL PrimType(DLDataTypeCode code, int bits, int lanes = 1);

  /*! \brief Construct a signed integer type with fixed lanes. */
  TVM_DLL static PrimType Int(int bits, int lanes = 1);
  /*! \brief Construct an unsigned integer type with fixed lanes. */
  TVM_DLL static PrimType UInt(int bits, int lanes = 1);
  /*! \brief Construct a floating-point type with fixed lanes. */
  TVM_DLL static PrimType Float(int bits, int lanes = 1);
  /*! \brief Construct a bfloat type with fixed lanes. */
  TVM_DLL static PrimType BFloat(int bits, int lanes = 1);
  /*! \brief Construct a boolean type with fixed lanes. */
  TVM_DLL static PrimType Bool(int lanes = 1);
  /*! \brief Construct an opaque handle type. */
  TVM_DLL static PrimType Handle(int bits = 64, int lanes = 1);
  /*! \brief Construct the void sentinel type, encoded as handle(0, 0). */
  TVM_DLL static PrimType Void();
  /*!
   * \brief Construct a scalable vector type.
   * \param code The DLPack dtype code.
   * \param bits The scalar bit width.
   * \param lanes The positive vscale factor to encode in the DLPack lane field.
   */
  TVM_DLL static PrimType ScalableVector(DLDataTypeCode code, int bits, int lanes);

  /*! \return The DLPack dtype code. */
  TVM_FFI_INLINE DLDataTypeCode code() const {
    return static_cast<DLDataTypeCode>(static_cast<int>(get()->dtype.code));
  }

  /*! \return The scalar bit width. */
  TVM_FFI_INLINE int32_t bits() const { return get()->dtype.bits; }

  /*!
   * \return The fixed lane count.
   * \note Throws on scalable vector types, where the encoded lane field stores a vscale factor.
   */
  TVM_FFI_INLINE int32_t lanes() const {
    int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
    if (TVM_FFI_PREDICT_FALSE(encoded_lanes < 0)) {
      TVM_FFI_THROW(InternalError)
          << "Can't fetch the lanes of a scalable vector at a compile time.";
    }
    return encoded_lanes;
  }

  /*!
   * \brief Check the scalar element code and bit width.
   * \note Lane count and scalable-vector encoding are intentionally ignored.
   */
  TVM_FFI_INLINE bool MatchesElementType(DLDataTypeCode code, int bits) const {
    DLDataType dtype = get()->dtype;
    return dtype.code == static_cast<uint8_t>(code) && dtype.bits == bits;
  }

  /*!
   * \brief Check whether the dtype code matches any of the provided DLPack codes.
   * \note Bit width and lanes are intentionally ignored.
   */
  template <typename... Codes>
  TVM_FFI_INLINE bool MatchesCode(Codes... codes) const {
    uint8_t dtype_code = get()->dtype.code;
    return ((dtype_code == static_cast<uint8_t>(codes)) || ...);
  }

  /*! \brief Whether this type is a scalar, excluding fixed and scalable vectors. */
  TVM_FFI_INLINE bool IsScalar() const {
    int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
    return encoded_lanes == 1;
  }

  /*! \brief Whether this type is the void sentinel `handle(0, 0)`. */
  TVM_FFI_INLINE bool IsVoid() const {
    DLDataType dtype = get()->dtype;
    return dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLOpaqueHandle) && dtype.bits == 0 &&
           static_cast<int16_t>(dtype.lanes) == 0;
  }

  /*! \brief Whether this type is an opaque handle, excluding the void sentinel. */
  TVM_FFI_INLINE bool IsHandle() const {
    return this->code() == DLDataTypeCode::kDLOpaqueHandle && !this->IsVoid();
  }

  /*! \brief Whether this type is a scalable vector. */
  TVM_FFI_INLINE bool IsScalableVector() const {
    return static_cast<int16_t>(get()->dtype.lanes) < -1;
  }

  /*! \brief Whether this type is a fixed-length vector. */
  TVM_FFI_INLINE bool IsFixedLengthVector() const {
    return static_cast<int16_t>(get()->dtype.lanes) > 1;
  }

  /*!
   * \brief Return the number of bytes needed to store one value of this type.
   *
   * This uses the same packed sub-byte dtype sizing rule as runtime tensors.
   * Scalable vector types have no compile-time storage size and are rejected.
   */
  TVM_FFI_INLINE size_t StorageBytes() const {
    DLDataType dtype = get()->dtype;
    int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
    if (TVM_FFI_PREDICT_FALSE(encoded_lanes < 0)) {
      TVM_FFI_THROW(InternalError)
          << "Cannot compute compile-time storage bytes for non-fixed vector type " << dtype;
    }
    return static_cast<size_t>(
        (static_cast<uint64_t>(dtype.bits) * static_cast<uint64_t>(dtype.lanes) + 7) / 8);
  }

  /*! \brief Return the same type with a different dtype code, preserving bits and lanes. */
  TVM_FFI_INLINE PrimType WithCode(DLDataTypeCode code) const {
    DLDataType dtype = get()->dtype;
    int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
    if (encoded_lanes < -1) {
      return ScalableVector(code, dtype.bits, -encoded_lanes);
    }
    return PrimType(code, dtype.bits, encoded_lanes);
  }

  /*! \brief Return the same type with a different scalar bit width, preserving code and lanes. */
  TVM_FFI_INLINE PrimType WithBits(int bits) const {
    DLDataType dtype = get()->dtype;
    int16_t encoded_lanes = static_cast<int16_t>(dtype.lanes);
    if (encoded_lanes < -1) {
      return ScalableVector(this->code(), bits, -encoded_lanes);
    }
    return PrimType(this->code(), bits, encoded_lanes);
  }

  /*! \brief Return the same scalar element type with a fixed lane count. */
  TVM_FFI_INLINE PrimType WithLanes(int lanes) const {
    return PrimType(this->code(), this->bits(), lanes);
  }

  /*! \return The vscale factor encoded in a scalable vector type. */
  TVM_FFI_INLINE int32_t VScaleFactor() const {
    int16_t encoded_lanes = static_cast<int16_t>(get()->dtype.lanes);
    if (encoded_lanes >= -1) {
      TVM_FFI_THROW(InternalError) << "A fixed length vector doesn't have a vscale factor.";
    }
    return -encoded_lanes;
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimType, Type, PrimTypeNode);
};

inline bool operator==(const PrimType& lhs, const PrimType& rhs) {
  return lhs->dtype == rhs->dtype;
}

inline bool operator!=(const PrimType& lhs, const PrimType& rhs) { return !(lhs == rhs); }

/*!
 * \brief Base type of all the expressions.
 * \sa Expr
 */
class ExprNode : public ffi::Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  /*!
   * \brief The deduced or annotated type of the expression.
   *
   * Type::Missing() denotes type information that will be populated by
   * later analysis passes instead of expression constructors.
   */
  mutable Type ty = Type::Missing();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    // span and ty do not participate in structural equal and hash.
    refl::ObjectDef<ExprNode>()
        .def_ro("span", &ExprNode::span, refl::DefaultValue(Span()),
                refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("ty", &ExprNode::ty, refl::DefaultValue(Type::Missing()),
                refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  static constexpr const uint32_t _type_child_slots = 64;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Expr", ExprNode, ffi::Object);
};

/*!
 * \brief Managed reference to ExprNode.
 * \sa ExprNode
 */
class Expr : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Expr, ffi::ObjectRef, ExprNode);
};

class Call;

/*!
 * \brief Typed reference/view over an expression whose result type is a
 * specific Type subtype.
 * \tparam ExpectedType The expected expression result type.
 */
template <typename ExpectedType>
class TypedExpr : public Expr {
 public:
  /*! \return the typed result of this expression. */
  ExpectedType ty() const {
    const auto* node = get();
    TVM_FFI_DCHECK(node != nullptr);
    const auto* ty_node = node->ExprNode::ty.template as<typename ExpectedType::ContainerType>();
    TVM_FFI_DCHECK(ty_node != nullptr);
    return ffi::GetRef<ExpectedType>(ty_node);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TypedExpr, Expr, ExprNode);
  static constexpr bool _type_container_is_exact = false;
};

/*!
 * \brief Typed reference/view over any Expr whose `ExprNode::ty` is PrimType.
 *
 * PrimExpr is a type category rather than a dedicated runtime node category.
 * It can contain intrinsic primitive nodes such as IntImmNode and FloatImmNode,
 * or a general ExprNode such as CallNode, when that expression's `ty` field is
 * a PrimType. This keeps primitive-only APIs explicit while allowing shared
 * Expr nodes for cross-dialect values with richer result types when needed.
 */
class PrimExpr : public TypedExpr<PrimType> {
 public:
  using TypedExpr<PrimType>::ty;

  /*!
   * \brief Construct from a call after checking that its result type is
   * PrimType.
   * \param call The call to view as a primitive expression.
   */
  TVM_DLL PrimExpr(Call call);  // NOLINT(*)

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

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimExpr, TypedExpr<PrimType>, ExprNode);
  static constexpr bool _type_container_is_exact = false;

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
template <>
inline constexpr bool use_default_type_traits_v<PrimType> = false;

template <>
struct TypeTraits<PrimType> : public ObjectRefWithFallbackTraitsBase<PrimType, DLDataType> {
  TVM_FFI_INLINE static PrimType ConvertFallbackValue(DLDataType dtype) { return PrimType(dtype); }
};

template <typename ExpectedType>
inline constexpr bool use_default_type_traits_v<TypedExpr<ExpectedType>> = false;

template <typename ExpectedType>
struct TypeTraits<TypedExpr<ExpectedType>>
    : public ObjectRefTypeTraitsBase<TypedExpr<ExpectedType>> {
  using Base = ObjectRefTypeTraitsBase<TypedExpr<ExpectedType>>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeSchema;
  using Base::TypeStr;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return TypedExpr<ExpectedType>::_type_is_nullable;
    }
    if (src->type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
        !details::IsObjectInstance<ExprNode>(src->type_index)) {
      return false;
    }
    const auto* expr = static_cast<const ExprNode*>(
        details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
    return details::AnyUnsafe::CheckAnyStrict<ExpectedType>(expr->ty);
  }

  TVM_FFI_INLINE static std::optional<TypedExpr<ExpectedType>> TryCastFromAnyView(
      const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return details::ObjectUnsafe::ObjectRefFromObjectPtr<TypedExpr<ExpectedType>>(nullptr);
      }
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<TypedExpr<ExpectedType>>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<ExprNode>(src->v_obj));
    }
    return std::nullopt;
  }
};

template <>
inline constexpr bool use_default_type_traits_v<PrimExpr> = false;

template <typename ObjectRefType, typename ExpectedType, typename... FallbackTypes>
struct TypedExprWithFallbackTraitsBase
    : public ObjectRefWithFallbackTraitsBase<ObjectRefType, FallbackTypes...> {
  using Base = ObjectRefWithFallbackTraitsBase<ObjectRefType, FallbackTypes...>;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<TypedExpr<ExpectedType>>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static std::optional<ObjectRefType> TryCastFromAnyView(const TVMFFIAny* src) {
    if (TypeTraits<TypedExpr<ExpectedType>>::TryCastFromAnyView(src)) {
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<ObjectRefType>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<ExprNode>(src->v_obj));
    }
    return Base::template TryFallbackTypes<FallbackTypes...>(src);
  }
};

// define automatic conversion from bool, int64_t, double, ffi::String to PrimExpr
// These functions are declared early to avoid circular dependency
template <>
struct TypeTraits<PrimExpr>
    : public TypedExprWithFallbackTraitsBase<PrimExpr, PrimType, StrictBool, int64_t, double,
                                             ffi::String, PrimExprConvertible> {
  using Base = TypedExprWithFallbackTraitsBase<PrimExpr, PrimType, StrictBool, int64_t, double,
                                               ffi::String, PrimExprConvertible>;
  using Base::CheckAnyStrict;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TryCastFromAnyView;
  using Base::TypeSchema;
  using Base::TypeStr;

  TVM_DLL static PrimExpr ConvertFallbackValue(StrictBool value);
  TVM_DLL static PrimExpr ConvertFallbackValue(int64_t value);
  TVM_DLL static PrimExpr ConvertFallbackValue(double value);
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(ffi::String value) {
    return PrimExpr::ConvertFallbackValue(value);
  }
  TVM_FFI_INLINE static PrimExpr ConvertFallbackValue(PrimExprConvertible value) {
    return value->ToPrimExpr();
  }
};

template <>
inline constexpr bool use_default_type_traits_v<Expr> = false;

// Allow generic Expr arguments to use the primitive-literal conversions
// already defined by PrimExpr.
template <>
struct TypeTraits<Expr> : public ObjectRefWithFallbackTraitsBase<Expr, PrimExpr> {
  TVM_FFI_INLINE static Expr ConvertFallbackValue(PrimExpr value) { return value; }
};
}  // namespace ffi

}  // namespace tvm

#endif  // TVM_IR_BASE_EXPR_H_
