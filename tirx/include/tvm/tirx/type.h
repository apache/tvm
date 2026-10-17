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
#include <tvm/ir/prim/op.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tirx/layout.h>

#include <optional>

namespace tvm::tirx {

#ifndef TVM_INDEX_DEFAULT_I64
#define TVM_INDEX_DEFAULT_I64 1
#endif
/*! \brief if TVM_INDEX_DEFAULT_I64 is set, return int64, otherwise return int32 */
inline PrimType DefaultIndexType() {
#if TVM_INDEX_DEFAULT_I64
  static const PrimType default_index_ty = PrimType::Int(64);
#else
  static const PrimType default_index_ty = PrimType::Int(32);
#endif
  return default_index_ty;
}

/*!
 * \brief Structural type of a TIRx tensor variable.
 *
 * A tensor value is an ordinary VarNode whose ExprNode::ty is TensorType.
 * TensorType owns the immutable access contract.  The physical pointer is
 * deliberately not stored here; it is obtained with tensor_data_ptr(TensorVar)
 * and is bound by the surrounding tensor definition.
 */
class TensorTypeNode : public TypeNode {
 public:
  explicit TensorTypeNode(ffi::UnsafeInit) {}
  TensorTypeNode() = default;

  /*! \brief dtype in the content of the tensor */
  PrimType dtype = PrimType::Void();
  /*! \brief Storage scope/address space of the tensor. */
  runtime::MemoryScope scope;
  /*! \brief The type of the tensor prior to flattening
   *
   * This contains the shape as it is accessed by
   * TensorLoad/TensorStore nodes, and used by the low-level code
   * generators.
   */
  ffi::Array<PrimExpr> shape;
  /*! \brief Alignment requirement of data pointer in bytes. */
  int data_alignment;
  /*! \brief The layout of the tensor */
  ffi::Optional<Layout> layout;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorTypeNode>()
        .def_ro("shape", &TensorTypeNode::shape, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("dtype", &TensorTypeNode::dtype)
        .def_ro("scope", &TensorTypeNode::scope, refl::DefaultValue(runtime::MemoryScope::Global()))
        .def_ro("data_alignment", &TensorTypeNode::data_alignment,
                refl::DefaultValue(static_cast<int>(runtime::kAllocAlignment)))
        .def_ro("layout", &TensorTypeNode::layout, refl::DefaultValue(nullptr));
  }

  /*! \return preferred index type for this tensor node */
  PrimType DefaultIndexType() const {
    return !shape.empty() ? shape[0].ty() : tvm::tirx::DefaultIndexType();
  }

  /*! \return type of the physical pointer projected by tensor_data_ptr. */
  PtrType DataPtrType() const { return PtrType(dtype, scope); }

  /*! \brief Whether this type supports scalar tensor syntax. */
  TVM_TIRX_DLL bool IsScalar(bool alloc_or_decl = true) const;

  /*! \brief Return the constant element count, or nullopt for symbolic extents or overflow. */
  TVM_TIRX_DLL std::optional<int64_t> ConstantAllocationSize() const;

  /*! \brief Determine the offset in the tensor of the given index.
   *
   * Returns the tensor offset, in number of elements of type dtype,
   * without adjusting for number of lanes.  (e.g. The number of
   * float16x4 elements in a tensor of type float16x4.)
   *
   * \param index The index to be accessed.
   */
  TVM_TIRX_DLL ffi::Array<PrimExpr> ElemOffset(ffi::Array<PrimExpr> index) const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TensorType", TensorTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TensorTypeNode.
 */
class TensorType : public Type {
 public:
  TVM_TIRX_DLL TensorType(ffi::Array<PrimExpr> shape, PrimType dtype = PrimType::Float(32),
                          runtime::MemoryScope scope = {},
                          int data_alignment = 0, ffi::Optional<Layout> layout = std::nullopt,
                          Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorType, Type, TensorTypeNode);

  explicit TensorType(ffi::ObjectPtr<TensorTypeNode> n) : Type(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }
};

/*!
 * \brief Match a tensor's shape-aligned strided memory representation.
 *
 * Returns (shape, strides, element offset). An absent layout yields contiguous
 * row-major strides and zero offset. A TileLayout must have one memory iterator
 * per shape dimension, no replicas, and only a memory-axis offset. Other layouts
 * or mismatched extents return nullopt. Strides are always actual expressions;
 * an empty stride array means rank zero, not an implicit compact contract.
 *
 * ExprDeepEqual preserves variable identity without remapping or symbolic proofs,
 * avoiding the overhead and broader equivalences of general structural matching.
 */
inline ffi::Optional<ffi::Tuple<ffi::Array<PrimExpr>, ffi::Array<PrimExpr>, PrimExpr>>
MatchShapeStridePattern(const TensorType& tensor) {
  ffi::Array<PrimExpr> strides;
  PrimExpr offset = IntImm(DefaultIndexType(), 0);
  if (!tensor->layout.has_value()) {
    PrimExpr stride = IntImm(tensor->shape.empty() ? DefaultIndexType() : tensor->shape[0].ty(), 1);
    strides = ffi::Array<PrimExpr>(tensor->shape.size(), stride);
    for (size_t i = tensor->shape.size(); i > 0; --i) {
      strides.Set(i - 1, stride);
      stride = stride * tensor->shape[i - 1];
    }
  } else {
    const auto* tile = tensor->layout.as<TileLayoutNode>();
    if (!tile || !tile->replica.empty() || tile->shard.size() != tensor->shape.size()) {
      return std::nullopt;
    }
    prim::ExprDeepEqual equal;
    static const Axis memory_axis = Axis::Get("m");
    for (size_t i = 0; i < tile->shard.size(); ++i) {
      if (tile->shard[i]->axis != memory_axis || !equal(tile->shard[i]->extent, tensor->shape[i]))
        return std::nullopt;
      strides.push_back(tile->shard[i]->stride);
    }
    for (const auto& [axis, value] : tile->offset) {
      if (axis != memory_axis) return std::nullopt;
      offset = value;
    }
  }
  return ffi::Tuple(tensor->shape, std::move(strides), std::move(offset));
}

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
  TVM_TIRX_DLL TensorMapType(Location loc = UnknownLoc());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorMapType, Type, TensorMapTypeNode);
};

}  // namespace tvm::tirx
#endif  // TVM_TIRX_TYPE_H_
