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
 * \file tvm/tirx/expr.h
 * \brief TIRx tensor expressions and construction helpers.
 */
#ifndef TVM_TIRX_EXPR_H_
#define TVM_TIRX_EXPR_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/type.h>

namespace tvm {
// forward declare shared statement.
class Stmt;

namespace tirx {

/*!
 * \brief Checked zero-state view over an ordinary VarNode with TensorType.
 *
 * TensorVar does not introduce a runtime object or a second identity.  It
 * safely widens to Var, and get() returns the underlying VarNode used by
 * identity-sensitive maps.  operator-> exposes the same VarNode, while ty()
 * and the metadata accessors expose its immutable TensorType access contract.
 */
class TensorVar : public Var {
 public:
  /*! \brief Construct a fresh tensor variable from an explicit TensorType. */
  TVM_TIRX_DLL explicit TensorVar(ffi::String name, TensorType type, Location loc = UnknownLoc());

  /*! \brief Create a checked tensor view over an existing ordinary Var. */
  explicit TensorVar(Var var) : Var(std::move(var)) {
    TVM_FFI_ICHECK(get() != nullptr && get()->ty.as<TensorTypeNode>())
        << "Expected a non-null Var with TensorType";
  }

  /*! \brief Return the tensor type carried by the ordinary variable. */
  TensorType ty() const { return get()->ty.as_or_throw<TensorType>(); }

  /*! \brief Return the shape of the tensor. */
  const ffi::Array<PrimExpr>& shape() const { return ty()->shape; }

  /*! \brief Return the primitive element type. */
  PrimType dtype() const { return ty()->dtype; }

  /*! \brief Return explicit strides, or an empty array for a compact tensor. */
  const ffi::Array<PrimExpr>& strides() const { return ty()->strides; }

  /*! \brief Return the offset in tensor elements. */
  const PrimExpr& elem_offset() const { return ty()->elem_offset; }

  /*! \brief Project the physical pointer established by the definition site. */
  Expr data() const { return Call(ty()->DataPtrType(), tensor_data_ptr_op(), {*this}); }

  /*! \brief Return the storage scope associated with this tensor. */
  runtime::MemoryScope scope() const { return ty()->scope; }

  /*!
   * \brief Return true if the tensor is a scalar.
   * \param alloc_or_decl Whether to consider alloc_scalar and decl_scalar as scalar. True for
   * alloc_scalar, False for decl_scalar.
   */
  TVM_TIRX_DLL bool IsScalar(bool alloc_or_decl = true) const;

  explicit TensorVar(ffi::ObjectPtr<VarNode> node) : TensorVar(Var(std::move(node))) {}
  explicit TensorVar(ffi::UnsafeInit tag) : Var(tag) {}
  TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TensorVar);

  [[maybe_unused]] static constexpr bool _type_is_nullable = false;
  static constexpr bool _type_container_is_exact = false;
  using ContainerType = VarNode;
};

// Preserve ObjectRef-style identity comparison for exact TensorVar operands.
// Comparisons widened to Var or Expr remain unavailable; use same_as for identity.
inline bool operator==(const TensorVar& lhs, const TensorVar& rhs) { return lhs.same_as(rhs); }

inline bool operator!=(const TensorVar& lhs, const TensorVar& rhs) { return !lhs.same_as(rhs); }

}  // namespace tirx
}  // namespace tvm

namespace tvm::ffi {

template <>
inline constexpr bool use_default_type_traits_v<tirx::TensorVar> = false;

template <>
struct TypeTraits<tirx::TensorVar> : public ObjectRefTypeTraitsBase<tirx::TensorVar> {
  using Base = ObjectRefTypeTraitsBase<tirx::TensorVar>;
  using Base::CopyFromAnyViewAfterCheck;
  using Base::CopyToAnyView;
  using Base::GetMismatchTypeInfo;
  using Base::MoveFromAnyAfterCheck;
  using Base::MoveToAny;
  using Base::TypeSchema;
  using Base::TypeStr;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return false;
    }
    if (src->type_index != tvm::VarNode::RuntimeTypeIndex()) {
      return false;
    }
    const auto* var = static_cast<const tvm::VarNode*>(
        details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
    return details::AnyUnsafe::CheckAnyStrict<tirx::TensorType>(var->ExprNode::ty);
  }

  TVM_FFI_INLINE static std::optional<tirx::TensorVar> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) {
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<tirx::TensorVar>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<tvm::VarNode>(src->v_obj));
    }
    return std::nullopt;
  }
};

}  // namespace tvm::ffi

#endif  // TVM_TIRX_EXPR_H_
