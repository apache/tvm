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
 * \file tvm/tirx/type.h
 * \brief Types specific to TIRX.
 */
#ifndef TVM_TIRX_TYPE_H_
#define TVM_TIRX_TYPE_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/layout.h>

#include <optional>

namespace tvm::tirx {

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

/*!
 * \brief Structural type of a TIRx tensor variable.
 *
 * A tensor value is an ordinary VarNode whose ExprNode::ty is TensorType.
 * TensorType owns the immutable access contract.  The physical pointer is
 * deliberately not stored here; it is obtained with buffer_data(BufferVar)
 * and is bound by the surrounding buffer definition.
 */
class TensorTypeNode : public TypeNode {
 public:
  explicit TensorTypeNode(ffi::UnsafeInit tag) : elem_offset(tag) {}

  explicit TensorTypeNode(PrimExpr elem_offset) : elem_offset(std::move(elem_offset)) {}

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

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorTypeNode>()
        .def_ro("dtype", &TensorTypeNode::dtype)
        .def_ro("storage_scope", &TensorTypeNode::storage_scope)
        // TODO(tqchen): use SEqHashDefSimple after the next pypi tvm-ffi release
        .def_ro("shape", &TensorTypeNode::shape, refl::AttachFieldFlag::SEqHashDefPattern())
        // TODO(tqchen): use SEqHashDefSimple after the next pypi tvm-ffi release
        .def_ro("strides", &TensorTypeNode::strides, refl::AttachFieldFlag::SEqHashDefPattern())
        // TODO(tqchen): use SEqHashDefSimple after the next pypi tvm-ffi release
        .def_ro("elem_offset", &TensorTypeNode::elem_offset,
                refl::AttachFieldFlag::SEqHashDefPattern())
        .def_ro("data_alignment", &TensorTypeNode::data_alignment)
        .def_ro("offset_factor", &TensorTypeNode::offset_factor)
        .def_ro("layout", &TensorTypeNode::layout)
        .def_ro("allocated_addr", &TensorTypeNode::allocated_addr);
  }

  /*! \return preferred index type for this buffer node */
  DLDataType DefaultIndexType() const {
    return shape.size() != 0 ? shape[0].ty()->dtype : tvm::tirx::DefaultIndexType();
  }

  /*! \return primitive element type for compiler-side uses. */
  PrimType ElementType() const { return dtype; }

  /*! \return type of the physical pointer projected by buffer_data. */
  PointerType DataPointerType() const { return PointerType(dtype, storage_scope); }

  /*! \brief Whether this type supports scalar buffer syntax. */
  TVM_DLL bool IsScalar(bool alloc_or_decl = true) const;

  /*! \brief Return the constant element count, or nullopt for symbolic extents or overflow. */
  TVM_DLL std::optional<int64_t> ConstantAllocationSize() const;

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

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TensorType", TensorTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TensorTypeNode.
 */
class TensorType : public Type {
 public:
  TVM_DLL TensorType(ffi::String storage_scope, PrimType dtype, ffi::Array<PrimExpr> shape,
                     ffi::Array<PrimExpr> strides, ffi::Optional<PrimExpr> elem_offset,
                     int data_alignment, int offset_factor,
                     ffi::Optional<Layout> layout = std::nullopt,
                     ffi::Array<PrimExpr> allocated_addr = {}, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorType, Type, TensorTypeNode);

  explicit TensorType(ffi::ObjectPtr<TensorTypeNode> n) : Type(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }
};

/*! \brief The type of a multi-dimensional buffer region expression. */
class BufferRegionTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BufferRegionTypeNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.BufferRegionType", BufferRegionTypeNode, TypeNode);
};

/*! \brief Managed reference to BufferRegionTypeNode. */
class BufferRegionType : public Type {
 public:
  TVM_DLL BufferRegionType();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BufferRegionType, Type, BufferRegionTypeNode);
};

/*!
 * \brief The type of tensor map.
 * \sa TensorMapType
 */
class TensorMapTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorMapTypeNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TensorMapType", TensorMapTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TensorMapTypeNode.
 * \sa TensorMapTypeNode
 */
class TensorMapType : public Type {
 public:
  TVM_DLL TensorMapType(Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorMapType, Type, TensorMapTypeNode);
};

}  // namespace tvm::tirx
#endif  // TVM_TIRX_TYPE_H_
