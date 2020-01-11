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
 * \file tvm/relay/type.h
 * \brief Relay typed AST nodes.
 */
#ifndef TVM_RELAY_TYPE_H_
#define TVM_RELAY_TYPE_H_

#include <tvm/ir/type.h>
#include <tvm/ir/type_relation.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>
#include <tvm/node/env_func.h>

#include <tvm/ir.h>
#include <string>

#include "base.h"
#include "../attrs.h"

namespace tvm {
namespace relay {

using Any = tvm::ir::AnyNode;
using Kind = TypeKind;
using Type = tvm::Type;
using TypeNode = tvm::TypeNode;
using TypeVar = tvm::TypeVar;
using TypeVarNode = tvm::TypeVarNode;
using GlobalTypeVar = tvm::GlobalTypeVar;
using GlobalTypeVarNode = tvm::GlobalTypeVarNode;
using TypeConstraint = tvm::TypeConstraint;
using TypeConstraintNode = tvm::TypeConstraintNode;
using FuncType = tvm::FuncType;
using FuncTypeNode = tvm::FuncTypeNode;
using TypeRelation = tvm::TypeRelation;
using TypeRelationNode = tvm::TypeRelationNode;
using TypeRelationFn = tvm::TypeRelationFn;
using TypeReporter = tvm::TypeReporter;
using TypeReporterNode = tvm::TypeReporterNode;

/*!
 * \brief Base of all Tensor types
 *  This container can hold TensorType or GenericTensorType.
 */
class BaseTensorTypeNode : public TypeNode {
 public:
  static constexpr const char* _type_key = "relay.BaseTensorType";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseTensorTypeNode, TypeNode);
};

class BaseTensorType : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseTensorType, Type, BaseTensorTypeNode);
};

/*!
 * \brief This is the most commonly used type in relay.
 *  TensorType have a fixed dimension, data type.
 *
 *  The elements of shape can be either IntImm(constant integer),
 *  or any symbolic integer expression.
 *  The symbolic integer allows generic shape inference in certain cases.
 * \sa TensorTypeNode The container class of TensorType.
 */
class TensorType;
/*! \brief TensorType container node */
class TensorTypeNode : public BaseTensorTypeNode {
 public:
  /*!
   * \brief The shape of the tensor,
   *  represented by IndexExpr(tvm::Expr).
   */
  Array<IndexExpr> shape;
  /*! \brief The content data type */
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("span", &span);
  }

  /*! \brief Return product of elements in the shape.
   *  \return (d1 * d_2 ... * d_n) if shape is (d_1, d_2, ..., d_n) and 1 if shape size is zero.
   */
  TVM_DLL IndexExpr Size() const;

  TVM_DLL static TensorType make(Array<IndexExpr> shape, DataType dtype);

  /*! \brief Construct an scalar containing elements of dtype.  */
  TVM_DLL static TensorType Scalar(DataType dtype);

  static constexpr const char* _type_key = "relay.TensorType";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorTypeNode, BaseTensorTypeNode);
};

class TensorType : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TensorType, Type, TensorTypeNode);
};

/*!
 * \brief Type application.
 */
class TypeCall;
/*! \brief TypeCall container node */
class TypeCallNode : public TypeNode {
 public:
  /*!
   * \brief The type-level function (ADT that takes type params).
   */
  Type func;
  /*! \brief The arguments. */
  tvm::Array<Type> args;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("args", &args);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeCall make(Type func, tvm::Array<Type> args);

  static constexpr const char* _type_key = "relay.TypeCall";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeCallNode, TypeNode);
};

class TypeCall : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TypeCall, Type, TypeCallNode);
};

/*!
 * \brief IncompleteType.
 * This is intermediate values that is used during type inference.
 *
 * If we view the type relations as "computational graph of types",
 * then IncompleteType represents intermediate values of the graph,
 * TypeVar represents the input to the graph.
 */
class IncompleteType;

/*! \brief IncompleteType container node */
class IncompleteTypeNode : public TypeNode {
 public:
  Kind kind;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  TVM_DLL static IncompleteType make(Kind kind);

  static constexpr const char* _type_key = "relay.IncompleteType";
  TVM_DECLARE_FINAL_OBJECT_INFO(IncompleteTypeNode, TypeNode);
};

class IncompleteType : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(IncompleteType, Type, IncompleteTypeNode);
};

/*!
 * \brief The type of tuple values.
 */
class TupleType;
/*!
 * \brief TupleType container.
 */
class TupleTypeNode : public TypeNode {
 public:
  /*! \brief The type of each field in the tuple. */
  tvm::Array<Type> fields;

  TupleTypeNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
  }

  TVM_DLL static TupleType make(tvm::Array<Type> fields);

  static constexpr const char* _type_key = "relay.TupleType";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleTypeNode, TypeNode);
};

class TupleType : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TupleType, Type, TupleTypeNode);
};

/*!
 * \brief The type of reference values.
 */
class RefType;
/*!
 * \brief Reference Type in relay.
 */
class RefTypeNode : public TypeNode {
 public:
  /*! \brief The type of value in the Reference. */
  Type value;

  RefTypeNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("span", &span);
  }

  TVM_DLL static RefType make(Type value);

  static constexpr const char* _type_key = "relay.RefType";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefTypeNode, TypeNode);
};

class RefType : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RefType, Type, RefTypeNode);
};

// The following fields contains advanced typing
// Only keep the class name and reserved for future usage.
class GenericTensorType;
// stores a DataType.
class GenericDataType;
// stores a DataType.
class GenericShape;

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TYPE_H_
