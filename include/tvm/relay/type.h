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

#include <tvm/api_registry.h>
#include <tvm/ir.h>
#include <tvm/node/node.h>
#include <string>

#include "base.h"
#include "../attrs.h"

namespace tvm {
namespace relay {

using Any = tvm::ir::Any;

/*! \brief Base type of the Relay type hiearchy. */
class TypeNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Type";
  TVM_DECLARE_BASE_NODE_INFO(TypeNode, Node);
};

/*!
 * \brief Type is the base type of relay type hiearchy.
 *
 * Relay's type system contains following two key concepts:
 *
 * - TensorType: type of certain Tensor values in the expression.
 * - FunctionType: the type of the function.
 *
 * There are also advanced types to support generic(polymorphic types),
 * which can be ignored when first reading the code base.
 */
class Type : public NodeRef {
 public:
  Type() {}
  explicit Type(NodePtr<tvm::Node> p) : NodeRef(p) {}

  using ContainerType = TypeNode;
};

/*!
 * \brief Base of all Tensor types
 *  This container can hold TensorType or GenericTensorType.
 */
class BaseTensorTypeNode : public TypeNode {
 public:
  static constexpr const char* _type_key = "relay.BaseTensorType";
  TVM_DECLARE_BASE_NODE_INFO(BaseTensorTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(BaseTensorType, BaseTensorTypeNode, Type);

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

  void VisitAttrs(tvm::AttrVisitor* v) final {
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
  TVM_DECLARE_NODE_TYPE_INFO(TensorTypeNode, BaseTensorTypeNode);
};

RELAY_DEFINE_NODE_REF(TensorType, TensorTypeNode, Type);

/*! \brief Possible kinds of Type. */
enum Kind : int {
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
   * \brief The variable itself is only meaningful when
   *  kind is ShapeVar, otherwise, we only use the name.
   */
  tvm::Var var;
  /*! \brief The kind of type parameter */
  Kind kind;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeVar make(std::string name, Kind kind);

  static constexpr const char* _type_key = "relay.TypeVar";
  TVM_DECLARE_NODE_TYPE_INFO(TypeVarNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TypeVar, TypeVarNode, Type);

/*!
 * \brief A global type variable that is used for defining new types or type aliases.
 */
class GlobalTypeVar;
/*! \brief GlobalTypeVar container node */
class GlobalTypeVarNode : public TypeNode {
 public:
  /*!
   * \brief The variable itself is only meaningful when
   *  kind is ShapeVar; otherwise, we only use the name.
   */
  tvm::Var var;
  /*! \brief The kind of type parameter */
  Kind kind;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  TVM_DLL static GlobalTypeVar make(std::string name, Kind kind);

  static constexpr const char* _type_key = "relay.GlobalTypeVar";
  TVM_DECLARE_NODE_TYPE_INFO(GlobalTypeVarNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(GlobalTypeVar, GlobalTypeVarNode, Type);

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

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("args", &args);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeCall make(Type func, tvm::Array<Type> args);

  static constexpr const char* _type_key = "relay.TypeCall";
  TVM_DECLARE_NODE_TYPE_INFO(TypeCallNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TypeCall, TypeCallNode, Type);

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

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  TVM_DLL static IncompleteType make(Kind kind);

  static constexpr const char* _type_key = "relay.IncompleteType";
  TVM_DECLARE_NODE_TYPE_INFO(IncompleteTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(IncompleteType, IncompleteTypeNode, Type);

/*!
 * \brief Potential Constraints in the type.
 * \note This is reserved for future use.
 */
class TypeConstraint;
/*! \brief TypeConstraint container node. */
class TypeConstraintNode : public TypeNode {
 public:
  static constexpr const char* _type_key = "relay.TypeConstraint";
  TVM_DECLARE_BASE_NODE_INFO(TypeConstraintNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TypeConstraint, TypeConstraintNode, Type);

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
  tvm::Array<Type> arg_types;
  /*! \brief The type of return value. */
  Type ret_type;
  // The following fields are used in polymorphic(template) functions
  // For normal functions, the following two fields will be empty.
  /*! \brief The type parameters of the function */
  tvm::Array<TypeVar> type_params;
  /*!
   * \brief potential constraint the type need to obey
   * \note this field is reserved for futher purposes.
   */
  tvm::Array<TypeConstraint> type_constraints;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("arg_types", &arg_types);
    v->Visit("ret_type", &ret_type);
    v->Visit("type_params", &type_params);
    v->Visit("type_constraints", &type_constraints);
    v->Visit("span", &span);
  }

  TVM_DLL static FuncType make(tvm::Array<Type> arg_types,
                               Type ret_type,
                               tvm::Array<TypeVar> type_params,
                               tvm::Array<TypeConstraint> type_constraints);

  static constexpr const char* _type_key = "relay.FuncType";
  TVM_DECLARE_NODE_TYPE_INFO(FuncTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(FuncType, FuncTypeNode, Type);

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

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
  }

  TVM_DLL static TupleType make(tvm::Array<Type> fields);

  static constexpr const char* _type_key = "relay.TupleType";
  TVM_DECLARE_NODE_TYPE_INFO(TupleTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TupleType, TupleTypeNode, Type);

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

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("value", &value);
    v->Visit("span", &span);
  }

  TVM_DLL static RefType make(Type value);

  static constexpr const char* _type_key = "relay.RefType";
  TVM_DECLARE_NODE_TYPE_INFO(RefTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(RefType, RefTypeNode, Type);

class TypeReporter;

/*!
 * \brief reporter that reports back to the
 *  type resolution information.
 */
class TypeReporterNode : public Node {
 public:
  /*!
   * \brief Create a type equality constraint.
   *
   *  The "assign direction" acts as a hint to the solver
   *  showing that it is more likely to resolve dst by src.
   *  But it is possible for the solver to resolve src by dst as well.
   */
  TVM_DLL virtual void Assign(const Type& dst, const Type& src) = 0;

  /*!
   * \brief assert shape expression comparison.
   * \note Use assert only if any of the condition input is symbolic.
   * \param cond The condition of operation.
   * \return false if assertation can be proven to have failed
   *      true if solver can still proceed.
   */
  TVM_DLL virtual bool Assert(const IndexExpr& cond)= 0;
  /*!
   * \brief assert shape expression equals each other.
   * \param lhs The left operand.
   * \param rhs The right operand.
   * \return false if assertation can be proven to have failed
   *      true if solver can still proceed.
   */
  TVM_DLL virtual bool AssertEQ(const IndexExpr& lhs, const IndexExpr& rhs) = 0;

  /*!
   * \brief Set the location at which to report unification errors.
   * \param ref The program node to report the error.
   */
  TVM_DLL virtual void SetLocation(const NodeRef& ref) = 0;

  // solver is not serializable.
  void VisitAttrs(tvm::AttrVisitor* v) final {}

  static constexpr const char* _type_key = "relay.TypeReporter";
  TVM_DECLARE_NODE_TYPE_INFO(TypeReporterNode, Node);
};

/*!
 * \brief Container class of TypeReporter.
 * \sa TypeReporterNode
 */
class TypeReporter : public NodeRef {
 public:
  TypeReporter() {}
  explicit TypeReporter(::tvm::NodePtr<::tvm::Node> n) : NodeRef(n) {
  }
  TypeReporterNode* operator->() const {
    return static_cast<TypeReporterNode*>(node_.get());
  }
  using ContainerType = TypeReporterNode;
};

/*!
 * \brief User defined type constraint function.
 *
 * If the input type information can be used to fully decide
 * the IncompleteTypes, then the function should call
 * reporter.Assign to report the new types, and return true.
 * Otherwise, the function should return false.
 *
 * \param args The arguments to the relation.
 *   The types are stored in the form of
 *   [input_type_0, input_type_1, ... input_type_n,
 *    output_type_0, output_type_1, ... output_type_m]
 *
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return false if This relation cannot be resolved.
 *   true if this relation has been resolved.
 */
using TypeRelationFn =
    TypedEnvFunc<bool(const Array<Type>& args,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter)>;

/*!
 * \brief User defined type relation, is an input-output relation on types.
 */
class TypeRelation;
/*!
 * \brief TypeRelation container.
 * \note This node is not directly serializable.
 * The type function need to be lookedup in the module.
 */
class TypeRelationNode : public TypeConstraintNode {
 public:
  /*!
   * \brief The function on input and output variables which
   *  this is not directly serializable,
   *  need to be looked-up in the module.
   */
  TypeRelationFn func;
  /*! \brief The type arguments to the type function. */
  tvm::Array<Type> args;
  /*! \brief Number of inputs arguments */
  int num_inputs;
  /*! \brief Attributes to the relation function */
  Attrs attrs;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("args", &args);
    v->Visit("num_inputs", &num_inputs);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeRelation make(TypeRelationFn func,
                                   Array<Type> args,
                                   int num_args,
                                   Attrs attrs);

  static constexpr const char* _type_key = "relay.TypeRelation";
  TVM_DECLARE_NODE_TYPE_INFO(TypeRelationNode, TypeConstraintNode);
};

RELAY_DEFINE_NODE_REF(TypeRelation, TypeRelationNode, TypeConstraint);

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
