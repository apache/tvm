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
#include <tvm/node/node.h>
#include <tvm/node/env_func.h>
#include <tvm/node/container.h>
#include <tvm/ir/span.h>
#include <string>

namespace tvm {

/*! \brief Base type of all the types. */
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
 * \brief Type is the base type of all types.
 *
 * Relay's type system contains following two key concepts:
 *
 * - PrimitiveType: type of primitive type values used in the low-level IR.
 * - TensorType: type of certain Tensor values in the expression.
 * - FunctionType: the type of the function.
 *
 * There are also advanced types to support generic(polymorphic types),
 * which can be ignored when first reading the code base.
 */
class Type : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Type, ObjectRef, TypeNode);
};

/*! \brief Possible kinds of TypeVars. */
enum TypeKind : int {
  kType = 0,
  /*! \brief Template variable in shape expression. */
  kShapeVar = 1,
  kBaseType = 2,
  kShape = 3,
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
 * \sa TypeVarNode The actual container class of TypeVar
 */
class TypeVar;
/*! \brief TypeVar container node */
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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeVar make(std::string name, TypeKind kind);

  static constexpr const char* _type_key = "relay.TypeVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeVarNode, TypeNode);
};

class TypeVar : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TypeVar, Type, TypeVarNode);
};

/*!
 * \brief A global type variable that is used for defining new types or type aliases.
 */
class GlobalTypeVar;
/*! \brief GlobalTypeVar container node */
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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("kind", &kind);
  }

  TVM_DLL static GlobalTypeVar make(std::string name, TypeKind kind);

  static constexpr const char* _type_key = "relay.GlobalTypeVar";
  TVM_DECLARE_FINAL_OBJECT_INFO(GlobalTypeVarNode, TypeNode);
};

class GlobalTypeVar : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(GlobalTypeVar, Type, GlobalTypeVarNode);
};

/*!
 * \brief Potential Constraints in the type.
 * \note This is reserved for future use.
 */
class TypeConstraint;
/*! \brief TypeConstraint container node. */
class TypeConstraintNode : public TypeNode {
 public:
  static constexpr const char* _type_key = "relay.TypeConstraint";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeConstraintNode, TypeNode);
};

class TypeConstraint : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TypeConstraint, Type, TypeConstraintNode);
};

class FuncType;
/*!
 * \brief Function type in Relay.
 *
 * Relay support polymorphic function type.
 * This can be roughly viewed as template function in C++.
 *
 * \sa TypeVar, TypeConstraint
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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("arg_types", &arg_types);
    v->Visit("ret_type", &ret_type);
    v->Visit("type_params", &type_params);
    v->Visit("type_constraints", &type_constraints);
    v->Visit("span", &span);
  }

  TVM_DLL static FuncType make(Array<Type> arg_types,
                               Type ret_type,
                               Array<TypeVar> type_params,
                               Array<TypeConstraint> type_constraints);

  static constexpr const char* _type_key = "relay.FuncType";
  TVM_DECLARE_FINAL_OBJECT_INFO(FuncTypeNode, TypeNode);
};

class FuncType : public Type {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FuncType, Type, FuncTypeNode);
};

}  // namespace tvm
#endif  // TVM_IR_TYPE_H_
