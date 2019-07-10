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
 * \file tvm/relay/expr.h
 * \brief Relay expression language.
 */
#ifndef TVM_RELAY_EXPR_H_
#define TVM_RELAY_EXPR_H_

#include <tvm/attrs.h>
#include <string>
#include <functional>
#include "./base.h"
#include "./type.h"

namespace tvm {
namespace relay {

/*!
 * \brief A Relay expression.
 */
class Expr;
/*!
 * \brief Base type of the Relay expression hiearchy.
 */
class ExprNode : public RelayNode {
 public:
  /*!
   * \brief Stores the result of type inference(type checking).
   *
   * \note This can be undefined before type inference.
   *       This value is discarded during serialization.
   */
  mutable Type checked_type_ = Type(nullptr);
  /*!
   * \return The checked_type
   */
  const Type& checked_type() const;
  /*!
   * \brief Check if the inferred(checked) type of the Expr
   *  is backed by a TTypeNode and return it.
   *
   * \note This function will thrown an error if the node type
   *       of this Expr is not TTypeNode.
   *
   * \return The corresponding TTypeNode pointer.
   * \tparam The specific TypeNode we look for.
   */
  template<typename TTypeNode>
  inline const TTypeNode* type_as() const;

  static constexpr const char* _type_key = "relay.Expr";
  TVM_DECLARE_BASE_NODE_INFO(ExprNode, RelayNode);
};

RELAY_DEFINE_NODE_REF(Expr, ExprNode, NodeRef);

/*!
 * \brief Constant tensor, backed by an NDArray on the cpu(0) device.
 *
 * \note Scalar constants are represented by rank-0 const tensor.
 *  Constant folding are handled uniformly via Tensor types.
 */
class Constant;
/*!
 * \brief Constant tensor type.
 */
class ConstantNode : public ExprNode {
 public:
  /*! \brief The data of the tensor */
  runtime::NDArray data;

  /*! \return The corresponding tensor type of the data */
  TensorType tensor_type() const;

  /*! \return Whether it is scalar(rank-0 tensor) */
  bool is_scalar() const {
    return data->ndim == 0;
  }

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Constant make(runtime::NDArray data);

  static constexpr const char* _type_key = "relay.Constant";
  TVM_DECLARE_NODE_TYPE_INFO(ConstantNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Constant, ConstantNode, Expr);

/*! \brief Tuple of multiple Exprs */
class Tuple;
/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<relay::Expr> fields;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Tuple make(tvm::Array<relay::Expr> fields);

  static constexpr const char* _type_key = "relay.Tuple";
  TVM_DECLARE_NODE_TYPE_INFO(TupleNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Tuple, TupleNode, Expr);

/*!
 * \brief Local variables used in the let expression.
 *
 * Its semantics are similar to tvm.Var node used in TVM's low level
 * tensor expression language.
 *
 * \note Each Var is bind only once and is immutable.
 */
class Var;
/*! \brief Container for Var */
class VarNode : public ExprNode {
 public:
  /*!
   * \brief The unique identifier of the Var.
   *
   * vid will be preserved for the same Var during type inference
   * and other rewritings, while the VarNode might be recreated
   * to attach additional information.
   * This property can be used to keep track of parameter Var
   * information across passes.
   */
  Id vid;
  /*!
   * \brief type annotaion of the variable.
   * This field records user provided type annotation of the Var.
   * This field is optional and can be None.
   */
  Type type_annotation;

  /*! \return The name hint of the variable */
  const std::string& name_hint() const {
    return vid->name_hint;
  }

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("vid", &vid);
    v->Visit("type_annotation", &type_annotation);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Var make(std::string name_hint,
                          Type type_annotation);

  TVM_DLL static Var make(Id vid,
                          Type type_annotation);

  static constexpr const char* _type_key = "relay.Var";
  TVM_DECLARE_NODE_TYPE_INFO(VarNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Var, VarNode, Expr);

/*!
 * \brief Global variable that leaves in the top-level module.
 * This is used to enable recursive calls between function.
 *
 * \note A GlobalVar may only point to functions.
 */
class GlobalVar;
/*! \brief A GlobalId from the node's current type to target type. */
class GlobalVarNode : public ExprNode {
 public:
  /*! \brief The name of the variable, this only acts as a hint. */
  std::string name_hint;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name_hint", &name_hint);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static GlobalVar make(std::string name_hint);

  static constexpr const char* _type_key = "relay.GlobalVar";
  TVM_DECLARE_NODE_TYPE_INFO(GlobalVarNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(GlobalVar, GlobalVarNode, Expr);

/*!
 * \brief Function (subgraph in computational graph)
 */
class Function;
/*! \brief Function container */
class FunctionNode : public ExprNode {
 public:
  /*! \brief Function parameters */
  tvm::Array<Var> params;
  /*!
   * \brief
   * The expression which represents the computation of the function,
   * the expression may reference the parameters, and the type of it
   * or sub-expressions may reference the type variables.
   */
  Expr body;
  /*! \brief User annotated return type of the function. */
  Type ret_type;
  /*!
   * \brief Type parameters of the function.
   *  Enables the function to vary its type based on these.
   *  This corresponds to template paramaters in c++'s terminology.
   *
   * \note This can be usually empty for non-polymorphic functions.
   */
  tvm::Array<TypeVar> type_params;

  /*!
   * \brief The attributes which store metadata about functions.
   */
  tvm::Attrs attrs;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("params", &params);
    v->Visit("body", &body);
    v->Visit("ret_type", &ret_type);
    v->Visit("type_params", &type_params);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  /*!
   * \brief Return the derived function annotation of this expression.
   *
   * \return The function type annotation.
   * \note The function type annotation can contain IncompleteType.
   */
  TVM_DLL FuncType func_type_annotation() const;

  /*!
   * \brief Check whether the function is a primitive function.
   *
   * \return Whether the function is primitive or not.
   */
  bool IsPrimitive() const;

  TVM_DLL static Function make(tvm::Array<Var> params,
                               Expr body,
                               Type ret_type,
                               tvm::Array<TypeVar> ty_params,
                               tvm::Attrs attrs = Attrs());

  static constexpr const char* _type_key = "relay.Function";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Function, FunctionNode, Expr);


TVM_DLL NodeRef FunctionGetAttr(const Function& func, const std::string& key);
TVM_DLL Function FunctionSetAttr(const Function& func, const std::string& key, const NodeRef& data);


/*!
 * \brief Call corresponds to operator invocation.
 *  Corresponds to the operator in computational graph terminology.
 */
class Call;
/*! \brief Call container. */
class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be relay::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  Expr op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<relay::Expr> args;

  /*! \brief The additional attributes */
  Attrs attrs;

  /*!
   * \brief The type arguments passed to polymorphic(template) function.
   *
   * This is the advance feature that is only used when the function is
   * polymorphic. It is safe to be ignored in most cases. For example, in the
   * following code, the type_args of addone call is [int].
   *
   * \code
   *
   * template<typename T>
   * T addone(T a) { return a + 1; }
   *
   * void main() {
   *   int x = addone<int>(10);
   * }
   *
   * \endcode
   */
  tvm::Array<Type> type_args;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("attrs", &attrs);
    v->Visit("type_args", &type_args);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Call make(Expr op,
                           Array<Expr> args,
                           Attrs attrs = Attrs(),
                           Array<Type> type_args = Array<Type>());

  static constexpr const char* _type_key = "relay.Call";
  TVM_DECLARE_NODE_TYPE_INFO(CallNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Call, CallNode, Expr);

/*!
 * \brief Let binding that binds a local var and optionally a type annotation.
 *
 * \note Let is useful to transform the program to be A-normal form.
 *  where each of the expression corresponds to a let binding.
 *
 *  For developers who are familar with the computational graph.
 *  Each of the let can be viewed as a operator node in the computational graph.
 *  Traversing the list of let bindings is similar to running
 * PostDFS-order(topo-order) traversal on the computational graph.
 */
class Let;
/*! \brief A binding of a sub-network. */
class LetNode : public ExprNode {
 public:
  /*! \brief The variable we bind to */
  Var var;
  /*! \brief The value we bind var to */
  Expr value;
  /*! \brief The body of the let binding */
  Expr body;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Let make(Var var, Expr value, Expr body);

  static constexpr const char* _type_key = "relay.Let";
  TVM_DECLARE_NODE_TYPE_INFO(LetNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(Let, LetNode, Expr);

/*!
 * \brief Condition expression
 *
 * Unlike traditional statement `if`s, the if evalutes
 * to the result of the branch taken.
 *
 * let x = if (true) { 1 } else { 0 }; // x is 1
 * let y = if (false) { 1 } else { 0 }; // y is 0
 *
 * \note This is similar to C's ternary operator.
 */
class If;
/*! \brief container of If */
class IfNode : public ExprNode {
 public:
  /*! \brief The condition */
  Expr cond;
  /*! \brief The expression evaluated when condition is true. */
  Expr true_branch;
  /*! \brief The expression evaluated when condition is false */
  Expr false_branch;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("cond", &cond);
    v->Visit("true_branch", &true_branch);
    v->Visit("false_branch", &false_branch);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static If make(Expr cond, Expr true_branch, Expr false_branch);

  static constexpr const char* _type_key = "relay.If";
  TVM_DECLARE_NODE_TYPE_INFO(IfNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(If, IfNode, Expr);

/*! \brief Get index-th field out of a tuple. */
class TupleGetItem;
class TupleGetItemNode : public ExprNode {
 public:
  /*! \brief The tuple Expression */
  Expr tuple;
  /*! \brief which value to get */
  int index;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("tuple_value", &tuple);
    v->Visit("index", &index);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static TupleGetItem make(Expr tuple, int index);

  static constexpr const char* _type_key = "relay.TupleGetItem";
  TVM_DECLARE_NODE_TYPE_INFO(TupleGetItemNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(TupleGetItem, TupleGetItemNode, Expr);

/*! \brief Create a new Reference out of initial value. */
class RefCreate;
class RefCreateNode : public ExprNode {
 public:
  /*! \brief The initial value of the Reference. */
  Expr value;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static RefCreate make(Expr value);

  static constexpr const char* _type_key = "relay.RefCreate";
  TVM_DECLARE_NODE_TYPE_INFO(RefCreateNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(RefCreate, RefCreateNode, Expr);

/*! \brief Get value out of Reference. */
class RefRead;
class RefReadNode : public ExprNode {
 public:
  /*! \brief The Reference Expression. */
  Expr ref;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("ref", &ref);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static RefRead make(Expr ref);

  static constexpr const char* _type_key = "relay.RefRead";
  TVM_DECLARE_NODE_TYPE_INFO(RefReadNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(RefRead, RefReadNode, Expr);

/*! \brief Set value of Reference. The whole expression evaluates to an Empty Tuple. */
class RefWrite;
class RefWriteNode : public ExprNode {
 public:
  /*! \brief The Reference Expression. */
  Expr ref;
  /*! \brief The value to write into. */
  Expr value;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("ref", &ref);
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static RefWrite make(Expr ref, Expr value);

  static constexpr const char* _type_key = "relay.RefWrite";
  TVM_DECLARE_NODE_TYPE_INFO(RefWriteNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(RefWrite, RefWriteNode, Expr);

/*!
 * \brief Base class of the temporary expression.
 *
 * TempExprs are pass specific expression that can be
 * useful to define intermediate result in the
 * rewriting pass such as layout or type transformation.
 *
 * Subclass TempExprNode allows us to pattern match on
 * specific kind of TempExpr and use them for expression rewriting.
 *
 * TempExpr should only be used within a pass,
 */
class TempExprNode : public ExprNode {
 public:
  /*!
   * \brief Convert the expression to a normal(non-temp) Expr.
   * \return The corresponding normal(non-temp) expression.
   */
  virtual Expr Realize() const = 0;

  static constexpr const char* _type_key = "relay.TempExpr";
  TVM_DECLARE_BASE_NODE_INFO(TempExprNode, ExprNode);
};

RELAY_DEFINE_NODE_REF(TempExpr, TempExprNode, Expr);

// implementataions
inline const Type& ExprNode::checked_type() const {
  CHECK(checked_type_.defined()) << "internal error: the type checker has "
    "not populated the checked_type "
    "field for "
                                 << GetRef<Expr>(this);
  return this->checked_type_;
}

template<typename TTypeNode>
inline const TTypeNode* ExprNode::type_as() const {
  static_assert(std::is_base_of<TypeNode, TTypeNode>::value,
                "TType must be a special case of type");
  CHECK(checked_type_.defined())
      << "Type inference for this Expr has not completed. Try to call infer_type pass.";
  const TTypeNode* node = checked_type_.as<TTypeNode>();
  CHECK(node != nullptr)
      << "Expected type to be " << TTypeNode::_type_key
      << ", but get " << checked_type_->type_key();
  return node;
}

/*! \brief Pretty print a Relay node, producing a fragment of the Relay text format. */
std::string PrettyPrint(const NodeRef& node);

/*!
 * \brief Render the node as a string in the Relay text format.
 * \param node The node to be rendered.
 * \param show_meta_data Whether to print meta data section.
 * \param annotate An optional callback function for attaching
 *        additional comment block to an expr.
 * \return The text representation.
 */
std::string AsText(const NodeRef& node,
                   bool show_meta_data = true,
                   runtime::TypedPackedFunc<std::string(Expr)> annotate = nullptr);
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_H_
