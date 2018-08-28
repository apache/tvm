/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/type.h
 * \brief Relay typed AST nodes.
 */
#ifndef TVM_RELAY_TYPE_H_
#define TVM_RELAY_TYPE_H_

#include <tvm/api_registry.h>
#include <tvm/ir.h>
#include <tvm/node.h>
#include <string>

#include "./base.h"

namespace tvm {
namespace relay {

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
  explicit Type(std::shared_ptr<tvm::Node> p) : NodeRef(p) {}

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
   *  represented by ShapeExpr(tvm::Expr).
   */
  Array<ShapeExpr> shape;
  /*! \brief The content data type */
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("shape", &shape);
    v->Visit("dtype", &dtype);
    v->Visit("span", &span);
  }

  TVM_DLL static TensorType make(Array<ShapeExpr> shape, DataType dtype);

  /*! \brief Constructing an unsigned integer type */
  TVM_DLL static TensorType Int(int bits, int lanes = 1);

  /*! \brief Constructing an unsigned integer type */
  TVM_DLL static TensorType UInt(int bits, int lanes = 1);

  /*! \brief Construct a floating-point type */
  TVM_DLL static TensorType Float(int bits, int lanes = 1);

  /*! \brief Construct a boolean type */
  TVM_DLL static TensorType Bool(int lanes = 1);

  static constexpr const char* _type_key = "relay.TensorType";
  TVM_DECLARE_NODE_TYPE_INFO(TensorTypeNode, BaseTensorTypeNode);
};

RELAY_DEFINE_NODE_REF(TensorType, TensorTypeNode, Type);

/*!
 * \brief Type parameter in the function.
 *  This can be viewed as template parameter in c++ template function.
 *
 * For example, in the following pesudo code,
 * the TypeParam of f is TypeParam(kind=kShapeVar, var=n).
 * This function can take in a Tensor with shape=(3, 3) and
 * returns a Tensor with shape=(9,)
 *
 * \code
 *
 *  template<i32 n>
 *  f(x : Tensor[i32, (n, n)]) -> Tensor[i32, (n * n)]
 *
 * \endcode
 * \sa TypeParamNode The actual container class of TypeParam
 */
class TypeParam;
/*! \brief TypeParam container node */
class TypeParamNode : public TypeNode {
 public:
  /*! \brief possible kinds of TypeParam */
  enum Kind : int {
    /*! \brief template variable in shape expression */
    kShapeVar = 0,
    kShape    = 1,
    kBaseType = 2,
    kType     = 3,
  };
  /*!
   * \brief The variable
   *  The variable itself is only meaningful when
   *  kind is ShapeVar, otherwise, we can only use the name.
   */
  tvm::Var var;
  /*! \brief The kind of type parameter */
  Kind kind;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("kind", &kind);
    v->Visit("span", &span);
  }

  TVM_DLL static TypeParam make(std::string name, Kind kind);

  static constexpr const char* _type_key = "relay.TypeParam";
  TVM_DECLARE_NODE_TYPE_INFO(TypeParamNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TypeParam, TypeParamNode, Type);

/*!
 * \brief Potential Constraints in the type.
 * \note This is reserved for future use.
 */
class TypeConstraint;
/*! \brief TypeConstraint container node. */
class TypeConstraintNode : public Node {
 public:
  static constexpr const char* _type_key = "relay.TypeConstraint";
  TVM_DECLARE_BASE_NODE_INFO(TypeConstraintNode, Node);
};

RELAY_DEFINE_NODE_REF(TypeConstraint, TypeConstraintNode, NodeRef);

class FuncType;
/*!
 * \brief Function type in Relay.
 *
 * Relay support polymorphic function type.
 * This can be roughly viewed as template function in C++.
 *
 * \sa TypeParam, TypeConstraint
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
  tvm::Array<TypeParam> type_params;
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

  TVM_DLL static FuncType make(tvm::Array<Type> arg_types, Type ret_type,
                                tvm::Array<TypeParam> type_params,
                                tvm::Array<TypeConstraint> type_constraints);

  static constexpr const char* _type_key = "relay.FuncType";
  TVM_DECLARE_NODE_TYPE_INFO(FuncTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(FuncType, FuncTypeNode, Type);

using TypeRelationFn = runtime::TypedPackedFunc<Array<Type>(const Array<Type>&, int)>;

/*!
 * \brief Opaque type relation, is an input-output relation on types.
 */
class TypeRelation;
/*!
 * \brief TypeRelation container.
 * \note This node is not directly serializable.
 * The type function need to be lookedup in the environment.
 */
class TypeRelationNode : public RelayNode {
 public:
  /*! \brief The name of the function */
  std::string name;
  /*! \brief Number of input type arguments, can be -1, which means VarArgs */
  int num_args;
  /*!
   * \brief The function on input and output variables which
   *  this is not directly serializable,
   *  need to be looked-up in the environment.
   */
  TypeRelationFn func_;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("num_args", &num_args);
  }

  TVM_DLL static TypeRelation make(std::string name, int num_args, TypeRelationFn func_);

  static constexpr const char* _type_key = "relay.TypeRelation";
  TVM_DECLARE_NODE_TYPE_INFO(TypeRelationNode, RelayNode);
};

RELAY_DEFINE_NODE_REF(TypeRelation, TypeRelationNode, Type);

/*!
 * \brief Call a type function with some number of arguments.
 */
class TypeCall;
/*!
 * \brief TypeCall container.
 */
class TypeCallNode : public TypeNode {
 public:
  /*! \brief The type function to be called. */
  Type func;
  
  /*! \brief The type arguments to the type function. */
  tvm::Array<Type> args;

  TypeCallNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("args", &args);
  }

  TVM_DLL static TypeCall make(Type func, tvm::Array<Type> args);

  static constexpr const char* _type_key = "relay.TypeCall";
  TVM_DECLARE_NODE_TYPE_INFO(TypeCallNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TypeCall, TypeCallNode, Type);

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
  }

  TVM_DLL static TupleType make(tvm::Array<Type> fields);

  static constexpr const char* _type_key = "relay.TypeTuple";
  TVM_DECLARE_NODE_TYPE_INFO(TupleTypeNode, TypeNode);
};

RELAY_DEFINE_NODE_REF(TupleType, TupleTypeNode, Type);

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
