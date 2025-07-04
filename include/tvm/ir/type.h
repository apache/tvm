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
 * \brief IR/AST nodes for the unified type system in TVM.
 *
 * We use TVM's type system as the unified type system
 * throughout the stack.
 *
 * This file contains types that are common across IR variants.
 *
 * ## Relation between Type and runtime::DataType
 *
 * Besides Type, we also store a dtype field in the low-level PrimExpr.
 * runtime::DataType(dtype) provides coarse grained type information
 * during compile time and runtime. It is eagerly built in
 * low-level expression construction and can be used for
 * quick type checking in the low-level IR.
 * For example, when an Expr's dtype is int32,
 * we know for sure that its type is also int32.
 *
 * On the other hand, Type provides more fine grained information.
 * For example, a low level expression can have DataType::Handle() as
 * its dtype and MemRef[float32] as its type.
 * Types are usually lazily constructed via type checking,
 * so they may not readily be available during IR construction.
 *
 * The unified Type serves as a common bridge across IR dialects.
 * For example, we require all the functions to have a type signature,
 * which allow us to build cross dialect function calls.
 */
#ifndef TVM_IR_TYPE_H_
#define TVM_IR_TYPE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/reflection.h>
#include <tvm/ir/source_map.h>
#include <tvm/node/node.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>

#include <string>

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
class TypeNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static constexpr const char* _type_key = "ir.Type";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 14;
  TVM_DECLARE_BASE_OBJECT_INFO(TypeNode, Object);
};

/*!
 * \brief Managed reference to TypeNode.
 * \sa TypeNode
 */
class Type : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Type, ObjectRef, TypeNode);
};

/*!
 * \brief Primitive data types used in the low-level IR.
 *
 * PrimType represents POD-values and handles that are
 * not automatically managed by the runtime.
 *
 * \sa PrimType
 */
class PrimTypeNode : public TypeNode {
 public:
  /*!
   * \brief The corresponding dtype field.
   */
  runtime::DataType dtype;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimTypeNode>().def_ro("dtype", &PrimTypeNode::dtype);
  }

  bool SEqualReduce(const PrimTypeNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(dtype); }

  static constexpr const char* _type_key = "ir.PrimType";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimTypeNode, TypeNode);
};

/*
 * \brief Managed reference to PrimTypeNode.
 * \sa PrimTypeNode
 */
class PrimType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param dtype The corresponding dtype.
   * \param span The span
   */
  TVM_DLL explicit PrimType(runtime::DataType dtype, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(PrimType, Type, PrimTypeNode);
};

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
  Type element_type;
  /*!
   * \brief The storage scope of the pointer
   */
  String storage_scope;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PointerTypeNode>()
        .def_ro("element_type", &PointerTypeNode::element_type)
        .def_ro("storage_scope", &PointerTypeNode::storage_scope);
  }

  bool SEqualReduce(const PointerTypeNode* other, SEqualReducer equal) const {
    // Make "global" equal to ""
    String lhs_scope = storage_scope.empty() ? "global" : storage_scope;
    String rhs_scope = other->storage_scope.empty() ? "global" : other->storage_scope;
    return equal(element_type, other->element_type) && equal(lhs_scope, rhs_scope);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(element_type);
    // Make "global" equal to ""
    hash_reduce(storage_scope.empty() ? "global" : storage_scope);
  }

  static constexpr const char* _type_key = "ir.PointerType";
  TVM_DECLARE_FINAL_OBJECT_INFO(PointerTypeNode, TypeNode);
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
  TVM_DLL explicit PointerType(Type element_type, String storage_scope = "");

  TVM_DEFINE_OBJECT_REF_METHODS(PointerType, Type, PointerTypeNode);
};

/*!
 * \brief The type of tuple values.
 * \sa TupleType
 */
class TupleTypeNode : public TypeNode {
 public:
  /*! \brief The type of each field in the tuple. */
  Array<Type> fields;

  TupleTypeNode() {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleTypeNode>()
        .def_ro("fields", &TupleTypeNode::fields)
        .def_ro("span", &TupleTypeNode::span);
  }

  bool SEqualReduce(const TupleTypeNode* other, SEqualReducer equal) const {
    return equal(fields, other->fields);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(fields); }

  static constexpr const char* _type_key = "ir.TupleType";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleTypeNode, TypeNode);
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
  TVM_DLL explicit TupleType(Array<Type> fields, Span span = Span());

  /*!
   * \brief Create an empty tuple type that constains nothing.
   * \return A empty tuple type.
   */
  TVM_DLL TupleType static Empty();

  TVM_DEFINE_OBJECT_REF_METHODS(TupleType, Type, TupleTypeNode);
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
  Array<Type> arg_types;
  /*! \brief The type of return value. */
  Type ret_type;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FuncTypeNode>()
        .def_ro("arg_types", &FuncTypeNode::arg_types)
        .def_ro("ret_type", &FuncTypeNode::ret_type)
        .def_ro("span", &FuncTypeNode::span);
  }

  bool SEqualReduce(const FuncTypeNode* other, SEqualReducer equal) const {
    // type params first as they defines type vars.
    return equal(arg_types, other->arg_types) && equal(ret_type, other->ret_type);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(arg_types);
    hash_reduce(ret_type);
  }

  static constexpr const char* _type_key = "ir.FuncType";
  TVM_DECLARE_FINAL_OBJECT_INFO(FuncTypeNode, TypeNode);
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
  TVM_DLL FuncType(Array<Type> arg_types, Type ret_type, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(FuncType, Type, FuncTypeNode);
};

/*!
 * \brief The type of tensor map.
 * \sa TensorMapType
 */
class TensorMapTypeNode : public TypeNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorMapTypeNode>().def_ro("span", &TensorMapTypeNode::span);
  }

  bool SEqualReduce(const TensorMapTypeNode* other, SEqualReducer equal) const {
    return equal(span, other->span);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(span); }

  static constexpr const char* _type_key = "ir.TensorMapType";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorMapTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to TensorMapTypeNode.
 * \sa TensorMapTypeNode
 */
class TensorMapType : public Type {
 public:
  TVM_DLL TensorMapType(Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS_WITHOUT_DEFAULT_CONSTRUCTOR(TensorMapType, Type, TensorMapTypeNode);
};

}  // namespace tvm
#endif  // TVM_IR_TYPE_H_
