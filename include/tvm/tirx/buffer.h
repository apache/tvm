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
 * \file tvm/tirx/buffer.h
 * \brief Symbolic n-dimensional array, to represent a memory buffer.
 */
#ifndef TVM_TIRX_BUFFER_H_
#define TVM_TIRX_BUFFER_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/var.h>

#include <string>

namespace tvm {
namespace tirx {

#ifndef TVM_INDEX_DEFAULT_I64
#define TVM_INDEX_DEFAULT_I64 1
#endif
/*! \brief if TVM_INDEX_DEFAULT_I64 is set, return int64, otherwise return int32 */
inline PrimType DefaultIndexPrimType() {
#if TVM_INDEX_DEFAULT_I64
  static const PrimType default_index_ty = PrimType::Int(64);
#else
  static const PrimType default_index_ty = PrimType::Int(32);
#endif
  return default_index_ty;
}

inline DLDataType DefaultIndexType() {
#if TVM_INDEX_DEFAULT_I64
  return DLDataType{kDLInt, 64, 1};
#else
  return DLDataType{kDLInt, 32, 1};
#endif
}

// forward declare Stmt
class Stmt;

/*!
 * \brief Structural type of a TIRx buffer variable.
 *
 * A buffer value is an ordinary VarNode whose ExprNode::ty is BufferType.
 * BufferType owns the immutable access contract.  The physical pointer is
 * deliberately not stored here; it is obtained with buffer_data(BufferVar)
 * and is bound by the surrounding buffer definition.
 */
class BufferTypeNode : public TypeNode {
 public:
  /*! \brief dtype in the content of the tensor */
  PrimType dtype = PrimType::Void();
  /*! \brief Storage scope/address space of the buffer. */
  ffi::String storage_scope;
  /*! \brief The type of the buffer prior to flattening
   *
   * This contains the shape as it is accessed by
   * BufferLoad/BufferStore nodes, and used by the low-level code
   * generators.
   */
  ffi::Array<PrimExpr> shape;
  /*!
   * \brief The strides of each dimension
   *  This can be an empty array, indicating array is contiguous
   */
  ffi::Array<PrimExpr> strides;
  /*! \brief The offset in terms of number of dtype elements (including lanes) */
  PrimExpr elem_offset;
  /*! \brief Alignment requirement of data pointer in bytes. */
  int data_alignment;
  /*!
   * \brief Factor of elem_offset field,
   *  elem_offset is guaranteed to be multiple of offset_factor.
   */
  int offset_factor;
  /*! \brief The layout of the buffer */
  ffi::Optional<Layout> layout;

  /*! \brief The allocated address of the buffer.
   * The address might be multi-dimensional based on its scope.
   * For example, trn.psum takes 2D address, representing (bank, offset).
   */
  ffi::Array<PrimExpr> allocated_addr;

  /*! \brief constructor */
  BufferTypeNode() {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BufferTypeNode>()
        .def_ro("dtype", &BufferTypeNode::dtype)
        .def_ro("storage_scope", &BufferTypeNode::storage_scope)
        // TODO(tqchen): use SEqHashDefNonRecursive after the next pypi tvm-ffi release
        .def_ro("shape", &BufferTypeNode::shape, refl::AttachFieldFlag::SEqHashDefRecursive())
        // TODO(tqchen): use SEqHashDefNonRecursive after the next pypi tvm-ffi release
        .def_ro("strides", &BufferTypeNode::strides, refl::AttachFieldFlag::SEqHashDefRecursive())
        // TODO(tqchen): use SEqHashDefNonRecursive after the next pypi tvm-ffi release
        .def_ro("elem_offset", &BufferTypeNode::elem_offset,
                refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("data_alignment", &BufferTypeNode::data_alignment)
        .def_ro("offset_factor", &BufferTypeNode::offset_factor)
        .def_ro("layout", &BufferTypeNode::layout)
        .def_ro("allocated_addr", &BufferTypeNode::allocated_addr);
  }

  /*! \return preferred index type for this buffer node */
  DLDataType DefaultIndexType() const {
    return shape.size() != 0 ? shape[0].ty()->dtype : tvm::tirx::DefaultIndexType();
  }

  /*! \return primitive element type for compiler-side uses. */
  PrimType ElementType() const { return dtype; }

  /*! \return type of the physical pointer projected by buffer_data. */
  PointerType DataPointerType() const { return PointerType(dtype, storage_scope); }

  /*! \brief Determine the offset in the buffer of the given index.
   *
   * Returns the buffer offset, in number of elements of type dtype,
   * without adjusting for number of lanes.  (e.g. The number of
   * float16x4 elements in a buffer of type float16x4.)
   *
   * \param index The index to be accessed.
   * \param inner Ignore the elem_offset, return inner offset only
   */
  ffi::Array<PrimExpr> ElemOffset(ffi::Array<PrimExpr> index, bool inner = false) const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.BufferType", BufferTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to BufferTypeNode.
 */
class BufferType : public Type {
 public:
  TVM_DLL BufferType(ffi::String storage_scope, PrimType dtype, ffi::Array<PrimExpr> shape,
                     ffi::Array<PrimExpr> strides, PrimExpr elem_offset, int data_alignment,
                     int offset_factor, ffi::Optional<Layout> layout = std::nullopt,
                     ffi::Array<PrimExpr> allocated_addr = {}, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BufferType, Type, BufferTypeNode);

  explicit BufferType(ffi::ObjectPtr<BufferTypeNode> n) : Type(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }
};

/*!
 * \brief Checked zero-state view over an ordinary VarNode with BufferType.
 *
 * BufferVar does not introduce a runtime object or a second identity.  It
 * safely widens to Var, and get() returns the underlying VarNode used by
 * identity-sensitive maps.  operator-> exposes the immutable BufferType
 * access contract for concise compiler-side metadata access.
 */
class BufferVar : public Var {
 public:
  /*! \brief Construct a fresh buffer variable from an explicit BufferType. */
  TVM_DLL explicit BufferVar(ffi::String name, BufferType type, Span span = Span());

  /*! \brief Create a checked buffer view over an existing ordinary Var. */
  explicit BufferVar(Var var) : Var(std::move(var)) {
    TVM_FFI_ICHECK(get() != nullptr && get()->ty.as<BufferTypeNode>())
        << "Expected a non-null Var with BufferType";
  }

  /*! \brief Return the ordinary variable view over the same identity. */
  Var var() const { return ffi::GetRef<Var>(get()); }

  /*! \brief Return the buffer type carried by the ordinary variable. */
  BufferType type() const { return get()->ty.as_or_throw<BufferType>(); }

  /*! \brief Return the buffer's diagnostic name. */
  const ffi::String& name() const { return get()->name; }

  /*! \brief Return the source span carried by the ordinary Var. */
  const Span& span() const { return get()->span; }

  /*! \brief Project the physical pointer established by the definition site. */
  TVM_DLL Expr data() const;

  /*!
   * \brief Return a new buffer that is equivalent with current one
   *  but always add stride field.
   * \return The strided version of the buffer.
   */
  TVM_DLL BufferVar MakeStrideView() const;
  /*!
   * \brief Make a new symbolic buffer representing a slice of the buffer.
   * \param begins The beginning position of each dimension.
   * \param extents The extent of each dimension.
   * \note This function will make target buffer as compact as possible.
   *  If stride is not needed in the slice, it won't be presented
   * \return the result buffer.
   */
  TVM_DLL BufferVar MakeSlice(ffi::Array<PrimExpr> begins, ffi::Array<PrimExpr> extents) const;
  /*!
   * \brief Get access ptr to the entire buffer.
   * \param access_mask The access mask
   * \param ptr_type The type of the pointer.
   * \param content_lanes The number of lanes for the (data) type.
   * \param offset The offset of ptr.
   * \param input_extent The extent of ptr.
   */
  TVM_DLL Expr access_ptr(int access_mask, PointerType ptr_type = PointerType::VoidPointerTy(),
                          int content_lanes = 1, PrimExpr offset = IntImm::Int32(0),
                          ffi::Optional<PrimExpr> input_extent = std::nullopt) const;
  /*!
   * \brief Create an Expr that does a vector load at begin index.
   * \param begin The beginning index
   * \param dtype The data type to be loaded.
   */
  TVM_DLL PrimExpr vload(ffi::Array<PrimExpr> begin, PrimType dtype) const;
  /*!
   * \brief Create a Stmt that does a vector store at begin index.
   * \param begin The beginning index
   * \param value The value to be stored.
   */
  TVM_DLL Stmt vstore(ffi::Array<PrimExpr> begin, PrimExpr value) const;

  /*!
   * \brief Get a flattened version of the buffer.
   *
   * If flattening changes the type, the result is a fresh BufferVar.  Callers
   * that use it as a view over this buffer must bind the returned variable with
   * `DeclBuffer(flattened, this->data())`.
   */
  BufferVar GetFlattenedBuffer() const;

  /*! \brief Determine the offset in the buffer of the given index.
   *
   * Returns the buffer offset, in number of elements of type dtype,
   * without adjusting for number of lanes.  (e.g. The number of
   * float16x4 elements in a buffer of type float16x4.)
   */
  ffi::Array<PrimExpr> OffsetOf(ffi::Array<PrimExpr> index) const;

  /*!
   * \brief Get the buffer_offset op for the given index.
   * \param index The index to be accessed.
   * \return The buffer_offset op.
   */
  PrimExpr OffsetOf_p(const ffi::Array<PrimExpr>& indices) const;

  /*!
   * \brief Return the storage scope associated with this buffer.
   */
  TVM_DLL ffi::String scope() const;

  /*!
   * \brief Return a new buffer with the allocated address.
   */
  TVM_DLL BufferVar with_allocated_addr(ffi::Array<PrimExpr> allocated_addr) const;

  /*!
   * \brief Return true if the buffer is a scalar.
   * \param alloc_or_decl Whether to consider alloc_scalar and decl_scalar as scalar. True for
   * alloc_scalar, False for decl_scalar.
   */
  TVM_DLL bool IsScalar(bool alloc_or_decl = true) const;

  /*!
   * \brief Return a new buffer with the dtype.
   */
  TVM_DLL BufferVar with_dtype(PrimType dtype) const;

  /*! \return primitive element type for compiler-side uses. */
  PrimType ElementType() const { return (*this)->ElementType(); }

  /*! \return type of the physical pointer projected by buffer_data. */
  PointerType DataPointerType() const { return (*this)->DataPointerType(); }

  BufferVar() = default;
  explicit BufferVar(ffi::ObjectPtr<VarNode> n) : Var(std::move(n)) {}
  explicit BufferVar(ffi::UnsafeInit tag) : Var(tag) {}
  TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(BufferVar);

  const BufferTypeNode* operator->() const {
    const auto* var_node = static_cast<const VarNode*>(data_.get());
    TVM_FFI_ICHECK(var_node != nullptr);
    const auto* type_node = var_node->ty.as<BufferTypeNode>();
    TVM_FFI_ICHECK(type_node != nullptr)
        << "Expected a Var with BufferType, but " << var_node->name << " has type " << var_node->ty;
    return type_node;
  }

  const VarNode* get() const { return static_cast<const VarNode*>(data_.get()); }

  [[maybe_unused]] static constexpr bool _type_is_nullable = false;
  static constexpr bool _type_container_is_exact = false;
  using ContainerType = VarNode;
};

// Preserve ObjectRef-style identity comparison for exact BufferVar operands.
// Comparisons widened to Var or Expr continue to build symbolic expressions.
inline bool operator==(const BufferVar& lhs, const BufferVar& rhs) { return lhs.same_as(rhs); }

inline bool operator!=(const BufferVar& lhs, const BufferVar& rhs) { return !lhs.same_as(rhs); }

/*! \brief Recover a checked buffer view from an ordinary VarNode pointer. */
inline BufferVar GetBufferVar(const VarNode* var) { return BufferVar(ffi::GetRef<Var>(var)); }

inline ffi::ObjectPtr<BufferTypeNode> CopyBufferType(const BufferVar& var) {
  return ffi::make_object<BufferTypeNode>(*var.operator->());
}

inline BufferVar RebuildBufferVar(const BufferVar& var, ffi::ObjectPtr<BufferTypeNode> type,
                                  ffi::Optional<ffi::String> name = std::nullopt) {
  return BufferVar(name.value_or(var.name()), BufferType(std::move(type)), var.span());
}

/*!
 * \brief Construct a new buffer given shape, and dtype.
 * \param shape The shape of the buffer,
 * \param dtype The content data type.
 * \param name The name of the buffer
 * \param storage_scope The storage scope associated with this buffer
 * \param span The location of this object in the source code.
 * \return The created buffer.
 * \sa BufferVar for complete constructor.
 */
TVM_DLL BufferVar decl_buffer(ffi::Array<PrimExpr> shape, PrimType dtype = PrimType::Float(32),
                              ffi::String name = "buffer", ffi::String storage_scope = "",
                              Span span = Span());

/*!
 * \brief Creates a TIR buffer for the provided parameters.
 * \param shape shape of the buffer
 * \param dtype data type
 * \param name buffer name
 * \param data_alignment alignment requirement of data pointer in bytes
 * \param offset_factor Factor of elem_offset field, elem_offset is guaranteed to be
 *                      multiple of offset_factor
                        User can specify data_alignment and offset_factor to be 0
 *                      A default value will be picked.
 * \param memory_scope memory scope of the buffer
 */
TVM_DLL tirx::BufferVar BufferWithOffsetAlignment(ffi::Array<PrimExpr> shape, PrimType dtype,
                                                  std::string name, int data_alignment,
                                                  int offset_factor, std::string memory_scope = "");
}  // namespace tirx
}  // namespace tvm

namespace tvm::ffi {

template <>
inline constexpr bool use_default_type_traits_v<tirx::BufferVar> = false;

template <>
struct TypeTraits<tirx::BufferVar> : public ObjectRefTypeTraitsBase<tirx::BufferVar> {
  using Base = ObjectRefTypeTraitsBase<tirx::BufferVar>;
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
    if (src->type_index != tirx::VarNode::RuntimeTypeIndex()) {
      return false;
    }
    const auto* var = static_cast<const tirx::VarNode*>(
        details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj).get());
    return details::AnyUnsafe::CheckAnyStrict<tirx::BufferType>(var->ExprNode::ty);
  }

  TVM_FFI_INLINE static std::optional<tirx::BufferVar> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) {
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<tirx::BufferVar>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<tirx::VarNode>(src->v_obj));
    }
    return std::nullopt;
  }
};

}  // namespace tvm::ffi

#endif  // TVM_TIR_BUFFER_H_
