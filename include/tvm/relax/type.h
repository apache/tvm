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
 * \file tvm/relax/type.h
 * \brief Relax Types.
 */
#ifndef TVM_RELAX_TYPE_H_
#define TVM_RELAX_TYPE_H_

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/ir/type.h>
#include <tvm/tir/expr.h>

#include <string>

namespace tvm {
namespace relax {

/*! \brief Indicates the number of dimensions of a tensor is unknown at compile time. */
static constexpr int kUnknownNDim = -1;

class ShapeTypeNode : public TypeNode {
 public:
  /*! \brief size of the shape. */
  int ndim;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShapeTypeNode>().def_ro("ndim", &ShapeTypeNode::ndim);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ShapeType", ShapeTypeNode, TypeNode);
};

class ShapeType : public Type {
 public:
  // TODO(relax-team): remove the default value later.
  TVM_DLL ShapeType(int ndim = kUnknownNDim, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ShapeType, Type, ShapeTypeNode);
};

/*!
 * \brief Dynamic version of TensorType
 *
 * Use relax::TensorStructInfo for more detailed (possibly dynamic) shape constrains
 */
class TensorTypeNode : public TypeNode {
 public:
  /*!
   * \brief The number of dimensions of the tensor, use -1 to denote tensor with unknown number of
   * dimensions.
   */
  int ndim;
  /*! \brief The content data type, use void to denote the dtype is unknown. */
  DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorTypeNode>()
        .def_ro("ndim", &TensorTypeNode::ndim)
        .def_ro("dtype", &TensorTypeNode::dtype);
  }

  inline bool IsUnknownNdim() const { return ndim == kUnknownNDim; }

  inline bool IsUnknownDtype() const { return dtype.is_void(); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.DynTensorType", TensorTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TensorTypeNode.
 * \sa TensorTypeNode.
 */
class TensorType : public Type {
 public:
  /*!
   * \brief Constructor.
   * \param ndim The number of dimensions of the tensor.
   * \param dtype The runtime dtype of the tensor's elements.
   * \param span The span.
   */
  TVM_DLL TensorType(int ndim, DataType dtype, Span span = Span());

  /*!
   * \brief Create a TensorType with unknown ndim.
   */
  TVM_DLL static TensorType CreateUnknownNDim(DataType dtype, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorType, Type, TensorTypeNode);
};

using TensorTypeNode = TensorTypeNode;
using TensorType = TensorType;

class ObjectTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ObjectTypeNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ObjectType", ObjectTypeNode, TypeNode);
};

class ObjectType : public Type {
 public:
  TVM_DLL ObjectType(Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ObjectType, Type, ObjectTypeNode);
};

class PackedFuncTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PackedFuncTypeNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.PackedFuncType", PackedFuncTypeNode, TypeNode);
};

class PackedFuncType : public Type {
 public:
  TVM_DLL PackedFuncType(Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PackedFuncType, Type, PackedFuncTypeNode);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TYPE_H_
