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
 * We use Relay's type system as the unified type system
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

#include <tvm/runtime/object.h>
#include <tvm/runtime/data_type.h>
#include <tvm/node/node.h>
#include <tvm/node/container.h>
#include <tvm/ir/span.h>
#include <string>

namespace tvm {

/*!
 * \brief Type is the base type of all types.
 *
 * Relay's type system contains following subclasses:
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

  static constexpr const char* _type_key = "relay.Type";
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
  }

  static constexpr const char* _type_key = "relay.PrimType";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to PrimTypeNode.
 * \sa PrimTypeNode
 */
class PrimType : public Type {
 public:
  /*!
   * \brief Constructor
   * \param dtype The corresponding dtype.
   */
  TVM_DLL PrimType(runtime::DataType dtype);

  TVM_DEFINE_OBJECT_REF_METHODS(PrimType, Type, PrimTypeNode);
};

/*! \brief Possible kinds of TypeVars. */
enum TypeKind : int {
  kType = 0,
  /*! \brief Template variable in shape expression. */
  kShapeVar = 1,
  kBaseType = 2,
  kConstraint = 4,
  kAdtHandle = 5,
  kTypeData = 6
};

/*!
 * \brief Type parameter in the function.
 *  This can be viewed as template parameter in c++ template function.
 *
 * For example, in the following pesudo code,
 * the TypeVar of f is TypeVar(kind=kShapeVar, var=n).
 * This function can take in a Tensor with shape=(3, 3) and
 * returns a Tensor with shape=(9,)
 *
 * \code
 *
 *  template<i32 n>
 *  f(x : Tensor[i32, (n, n)]) -> Tensor[i32, (n * n)]
 *
 * \endcode
 * \sa TypeVar, TypeKind
 */
class TypeVarNode : public TypeNode {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  std::string name_hint;
  /*! \brief The kind of type parameter */
  TypeKind kind;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.TypeVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeVarNode, TypeNode);
};

/*!
 * \brief Managed reference to TypeVarNode
 * \sa TypeVarNode
 */
class TypeVar : public Type {
 public:
  /*!
   * \brief Constructor
   * \param name_hint The name of the type var.
   * \param kind The kind of the type var.
   */
  TVM_DLL TypeVar(std::string name_hint, TypeKind kind);

  TVM_DEFINE_OBJECT_REF_METHODS(TypeVar, Type, TypeVarNode);
};

/*!
 * \brief A global type variable that is used for defining new types or type aliases.
 * \sa GlobalTypeVar
 */
class GlobalTypeVarNode : public TypeNode {
 public:
  /*!
   * \brief The name of the variable,
   *  this only acts as a hint to the user,
   *  and is not used for equality.
   */
  std::string name_hint;
  /*! \brief The kind of type parameter */
  TypeKind kind;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("kind", &kind);
  }

  static constexpr const char* _type_key = "relay.GlobalTypeVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(GlobalTypeVarNode, TypeNode);
};

/*!
 * \brief Managed reference to GlobalTypeVarNode
 * \sa GlobalTypeVarNode
 */
class GlobalTypeVar : public Type {
 public:
  /*!
   * \brief Constructor
   * \param name_hint The name of the type var.
   * \param kind The kind of the type var.
   */
  TVM_DLL GlobalTypeVar(std::string name_hint, TypeKind kind);

  TVM_DEFINE_OBJECT_REF_METHODS(GlobalTypeVar, Type, GlobalTypeVarNode);
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.TupleType";
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
   */
  TVM_DLL explicit TupleType(Array<Type> fields);

  /*!
   * \brief Create an empty tuple type that constains nothing.
   * \return A empty tuple type.
   */
  TVM_DLL TupleType static Empty();

  TVM_DEFINE_OBJECT_REF_METHODS(TupleType, Type, TupleTypeNode);
};

/*!
 * \brief Potential Constraints in a function.
 * \sa TypeConstraint
 */
class TypeConstraintNode : public TypeNode {
 public:
  static constexpr const char* _type_key = "relay.TypeConstraint";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeConstraintNode, TypeNode);
};

/*!
 * \brief Managed reference to TypeConstraintNode.
 * \sa TypeConstraintNode, TypeRelation
 */
class TypeConstraint : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TypeConstraint, Type, TypeConstraintNode);
};

/*!
 * \brief Function type.
 *
 * We support polymorphic function type.
 * This can be roughly viewed as template function in C++.
 *
 * \sa FuncType, TypeVar, TypeConstraint
 */
class FuncTypeNode : public TypeNode {
 public:
  /*! \brief type type of arguments */
  Array<Type> arg_types;
  /*! \brief The type of return value. */
  Type ret_type;
  // The following fields are used in polymorphic(template) functions
  // For normal functions, the following two fields will be empty.
  /*! \brief The type parameters of the function */
  Array<TypeVar> type_params;
  /*!
   * \brief potential constraint the type need to obey
   * \note this field is reserved for futher purposes.
   */
  Array<TypeConstraint> type_constraints;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("arg_types", &arg_types);
    v->Visit("ret_type", &ret_type);
    v->Visit("type_params", &type_params);
    v->Visit("type_constraints", &type_constraints);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.FuncType";
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
   * \param type_params The type parameters.
   * \param type_constraints The type constraints.
   * \sa FuncTypeNode for more docs about these fields.
   */
  TVM_DLL FuncType(Array<Type> arg_types,
                   Type ret_type,
                   Array<TypeVar> type_params,
                   Array<TypeConstraint> type_constraints);

  TVM_DEFINE_OBJECT_REF_METHODS(FuncType, Type, FuncTypeNode);
};

/*!
 * \brief Intermediate values that is used to indicate incomplete type
 *         during type inference.
 *
 * If we view the type relations as "computational graph of types",
 * then IncompleteType represents intermediate values of the graph,
 * TypeVar represents the input to the graph.
 *
 * \sa IncompleteType
 */
class IncompleteTypeNode : public TypeNode {
 public:
  /*! \brief kind of the type. */
  TypeKind kind;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.IncompleteType";
  TVM_DECLARE_FINAL_OBJECT_INFO(IncompleteTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to IncompleteTypeNode.
 * \sa IncompleteTypeNode
 */
class IncompleteType : public Type {
 public:
  /*!
   * \brief Constructor.
   * \param kind kind of the type.
   */
  TVM_DLL explicit IncompleteType(TypeKind kind);

  TVM_DEFINE_OBJECT_REF_METHODS(IncompleteType, Type, IncompleteTypeNode);
};


/*!
 * \brief Reference Type High-level Relay IR.
 *
 * \sa RelayRefType.
 */
class RelayRefTypeNode : public TypeNode {
 public:
  /*! \brief The type of value in the Reference. */
  Type value;

  RelayRefTypeNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("span", &span);
  }

  static constexpr const char* _type_key = "relay.RefType";
  TVM_DECLARE_FINAL_OBJECT_INFO(RelayRefTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to RelayRefTypeNode.
 * \sa RelayRefTypeNode.
 */
class RelayRefType : public Type {
 public:
  TVM_DLL explicit RelayRefType(Type value);
  TVM_DEFINE_OBJECT_REF_METHODS(RelayRefType, Type, RelayRefTypeNode);
};
}  // namespace tvm
#endif  // TVM_IR_TYPE_H_
