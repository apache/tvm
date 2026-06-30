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
 * \file tvm/ir/type.h
 * \brief IR/AST nodes for TVM types shared across IR variants.
 */
#ifndef TVM_IR_TYPE_H_
#define TVM_IR_TYPE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/base_expr.h>
#include <tvm/ir/source_map.h>

#include <string>

namespace tvm {

/*!
 * \brief Low-level raw pointer type.
 *
 *  PointerType represents type hints in the TIR to be
 *  passed to the final code generator.
 *
 *  PointerType should not occur in the high-level analysis.
 *
 * \sa PointerType
 */
class PointerTypeNode : public TypeNode {
 public:
  /*!
   * \brief The type of the element which the pointer points to.
   */
  Type element_type = PrimType::Void();
  /*!
   * \brief The storage scope of the pointer
   */
  ffi::String storage_scope;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PointerTypeNode>()
        .def_ro("element_type", &PointerTypeNode::element_type)
        .def_ro("storage_scope", &PointerTypeNode::storage_scope);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.PointerType", PointerTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PointerTypeNode.
 * \sa PointerTypeNode
 */
class PointerType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param element_type The type of the element which the pointer points to.
   * \param storage_scope The storage scope into which the pointer addresses
   */
  TVM_DLL explicit PointerType(Type element_type, ffi::String storage_scope = "");

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PointerType, Type, PointerTypeNode);
};

/*!
 * \brief The type of tuple values.
 * \sa TupleType
 */
class TupleTypeNode : public TypeNode {
 public:
  /*! \brief The type of each field in the tuple. */
  ffi::Array<Type> fields;

  TupleTypeNode() {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleTypeNode>().def_ro("fields", &TupleTypeNode::fields);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TupleType", TupleTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TupleTypeNode.
 * \sa TupleTypeNode.
 */
class TupleType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param fields Fields in the tuple.
   * \param span The span of the type.
   */
  TVM_DLL explicit TupleType(ffi::Array<Type> fields, Span span = Span());

  /*!
   * \brief Create an empty tuple type that constains nothing.
   * \return A empty tuple type.
   */
  TVM_DLL TupleType static Empty();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TupleType, Type, TupleTypeNode);
};

/*!
 * \return a type that represents void.
 */
inline Type VoidType() { return TupleType::Empty(); }

/*!
 * \brief Check whether the tyep represents void.
 * \return The check result.
 */
inline bool IsVoidType(const Type& type) {
  auto* n = type.as<TupleTypeNode>();
  return n && n->fields.size() == 0;
}

/*!
 * \brief Function type.
 *
 * We support polymorphic function type.
 * This can be roughly viewed as template function in C++.
 *
 * \sa FuncType, TypeConstraint
 */
class FuncTypeNode : public TypeNode {
 public:
  /*! \brief type type of arguments */
  ffi::Array<Type> arg_types;
  /*! \brief The type of return value. */
  Type ret_type = VoidType();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FuncTypeNode>()
        .def_ro("arg_types", &FuncTypeNode::arg_types)
        .def_ro("ret_type", &FuncTypeNode::ret_type);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.FuncType", FuncTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to FuncTypeNode.
 * \sa FuncTypeNode
 */
class FuncType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param arg_types The types of the arguments.
   * \param ret_type The type of the return value.
   * \param span The span information.
   * \sa FuncTypeNode for more docs about these fields.
   */
  TVM_DLL FuncType(ffi::Array<Type> arg_types, Type ret_type, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FuncType, Type, FuncTypeNode);
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorMapType", TensorMapTypeNode, TypeNode);
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

}  // namespace tvm
#endif  // TVM_IR_TYPE_H_
