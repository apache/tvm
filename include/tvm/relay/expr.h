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

#include <tvm/ir/attrs.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <string>
#include <functional>
#include "./base.h"
#include "./type.h"

namespace tvm {
namespace relay {

using Expr = tvm::RelayExpr;
using ExprNode = tvm::RelayExprNode;
using BaseFunc = tvm::BaseFunc;
using BaseFuncNode = tvm::BaseFuncNode;
using GlobalVar = tvm::GlobalVar;
using GlobalVarNode = tvm::GlobalVarNode;
using tvm::PrettyPrint;

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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Constant make(runtime::NDArray data);

  static constexpr const char* _type_key = "relay.Constant";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantNode, ExprNode);
};

class Constant : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Constant, RelayExpr, ConstantNode);
};

/*! \brief Tuple of multiple Exprs */
class Tuple;
/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<relay::Expr> fields;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Tuple make(tvm::Array<relay::Expr> fields);

  static constexpr const char* _type_key = "relay.Tuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleNode, ExprNode);
};

class Tuple : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Tuple, RelayExpr, TupleNode);
};

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

  void VisitAttrs(tvm::AttrVisitor* v) {
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
  TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, ExprNode);
};

class Var : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Var, RelayExpr, VarNode);
};

/*!
 * \brief Function (subgraph in computational graph)
 */
class Function;
/*! \brief Function container */
class FunctionNode : public BaseFuncNode {
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

  void VisitAttrs(tvm::AttrVisitor* v) {
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

  /*!
   * \brief Check whether the function is marked as inline.
   *
   * \return Whether the function should be inlined or not.
   */
  bool IsMarkedInline() const;

  /*!
   * \brief Check whether the function should use the TVM default compiler to build, or
   * use other compilers.
   *
   * \return Whether the function will be compiled using the default compiler
   * (e.g. those are used in the TVM stack).
   */
  bool UseDefaultCompiler() const;

  TVM_DLL static Function make(tvm::Array<Var> params,
                               Expr body,
                               Type ret_type,
                               tvm::Array<TypeVar> ty_params,
                               tvm::Attrs attrs = Attrs());

  /*!
   * \brief Attach the function's parameters to its attributes for use in analysis.
   * \return The function with its parameters attached.
   */
  Function SetParams(const tvm::Map<Var, Constant>& parameters) const;

  /*!
   * \brief Retrieve the function's parameters.
   *
   * \return The function's parameter.
   */
  tvm::Map<Var, Constant> GetParams() const;

  static constexpr const char* _type_key = "relay.Function";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionNode, BaseFuncNode);
};

class Function : public BaseFunc {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Function, BaseFunc, FunctionNode);
};


TVM_DLL ObjectRef FunctionGetAttr(const Function& func, const std::string& key);
TVM_DLL Function FunctionSetAttr(const Function& func,
                                 const std::string& key,
                                 const ObjectRef& data);

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

  void VisitAttrs(tvm::AttrVisitor* v) {
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
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, ExprNode);
};

class Call : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Call, RelayExpr, CallNode);
};

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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static Let make(Var var, Expr value, Expr body);

  static constexpr const char* _type_key = "relay.Let";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetNode, ExprNode);
};

class Let : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Let, RelayExpr, LetNode);
};

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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("cond", &cond);
    v->Visit("true_branch", &true_branch);
    v->Visit("false_branch", &false_branch);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static If make(Expr cond, Expr true_branch, Expr false_branch);

  static constexpr const char* _type_key = "relay.If";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfNode, ExprNode);
};

class If : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(If, RelayExpr, IfNode);
};

/*! \brief Get index-th field out of a tuple. */
class TupleGetItem;
class TupleGetItemNode : public ExprNode {
 public:
  /*! \brief The tuple Expression */
  Expr tuple;
  /*! \brief which value to get */
  int index;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tuple_value", &tuple);
    v->Visit("index", &index);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static TupleGetItem make(Expr tuple, int index);

  static constexpr const char* _type_key = "relay.TupleGetItem";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemNode, ExprNode);
};

class TupleGetItem : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItem, RelayExpr, TupleGetItemNode);
};

/*! \brief Create a new Reference out of initial value. */
class RefCreate;
class RefCreateNode : public ExprNode {
 public:
  /*! \brief The initial value of the Reference. */
  Expr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static RefCreate make(Expr value);

  static constexpr const char* _type_key = "relay.RefCreate";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefCreateNode, ExprNode);
};

class RefCreate : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RefCreate, RelayExpr, RefCreateNode);
};

/*! \brief Get value out of Reference. */
class RefRead;
class RefReadNode : public ExprNode {
 public:
  /*! \brief The Reference Expression. */
  Expr ref;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ref", &ref);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static RefRead make(Expr ref);

  static constexpr const char* _type_key = "relay.RefRead";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefReadNode, ExprNode);
};

class RefRead : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RefRead, RelayExpr, RefReadNode);
};
/*! \brief Set value of Reference. The whole expression evaluates to an Empty Tuple. */
class RefWrite;
class RefWriteNode : public ExprNode {
 public:
  /*! \brief The Reference Expression. */
  Expr ref;
  /*! \brief The value to write into. */
  Expr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ref", &ref);
    v->Visit("value", &value);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
  }

  TVM_DLL static RefWrite make(Expr ref, Expr value);

  static constexpr const char* _type_key = "relay.RefWrite";
  TVM_DECLARE_FINAL_OBJECT_INFO(RefWriteNode, ExprNode);
};

class RefWrite : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RefWrite, RelayExpr, RefWriteNode);
};

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
  /*! \brief virtual destructor */
  virtual ~TempExprNode() {}
  /*!
   * \brief Convert the expression to a normal(non-temp) Expr.
   * \return The corresponding normal(non-temp) expression.
   */
  virtual Expr Realize() const = 0;

  static constexpr const char* _type_key = "relay.TempExpr";
  TVM_DECLARE_BASE_OBJECT_INFO(TempExprNode, ExprNode);
};

class TempExpr : public Expr {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TempExpr, RelayExpr, TempExprNode);
};


/*! \brief namespace of the attributes that are attached to a function. */
namespace attr {
/*! \brief Mark the function as a primitive function. */
constexpr const char* kPrimitive = "Primitive";
/*!
 * \brief Indicate the compiler that should be used for builing this function.
 * When this is unset or set to "default", the default compilation pipeline will be used.
 */
constexpr const char* kCompiler = "Compiler";
/*! \brief Indicate if the function is a closure. */
constexpr const char* kClosure = "Closure";
/*! \brief Store a Var to parameter/Constant mapping on a Function. */
constexpr const char* kParams = "__params__";
/*! \brief Store the unique external symbol for external compilers. */
constexpr const char* kExternalSymbol = "ExternalSymbol";
/*! \brief Mark if the function should be avoided being optimized. */
constexpr const char* kSkipOptimization = "SkipOptimization";
/*! \brief Treat the function as a composite operator. */
constexpr const char* kComposite = "Composite";
/*! \brief Mark the function to be inlined. */
constexpr const char* kInline = "Inline";
}  // namespace attr

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_EXPR_H_
