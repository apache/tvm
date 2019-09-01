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
 * \file tvm/ir.h
 * \brief Additional high level nodes in the IR
 */
// Acknowledgement: Most low-level IR nodes originate from Halide.

#ifndef TVM_IR_H_
#define TVM_IR_H_

#include <type_traits>
#include <string>
#include <vector>
#include <utility>
#include "base.h"
#include "expr.h"
#include "runtime/util.h"

namespace tvm {
namespace ir {

using IntImm = tvm::IntImm;
using Variable = tvm::Variable;

/*! \brief constant unsigned integer. */
class UIntImm : public ExprNode {
 public:
  /*! \brief The constant value content. */
  uint64_t value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("value", &value);
  }

  TVM_DLL static Expr make(Type t, uint64_t value);

  static constexpr const char* _type_key = "UIntImm";
  TVM_DECLARE_NODE_TYPE_INFO(UIntImm, ExprNode);
};

/*! \brief Floating point constants. */
class FloatImm : public ExprNode {
 public:
  /*! \brief The constant value content. */
  double value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("value", &value);
  }

  TVM_DLL static Expr make(Type t, double value);

  static constexpr const char* _type_key = "FloatImm";
  TVM_DECLARE_NODE_TYPE_INFO(FloatImm, ExprNode);
};

/*! \brief String constants, only used in asserts. */
class StringImm : public ExprNode {
 public:
  /*! \brief The constant value content. */
  std::string value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("value", &value);
  }

  TVM_DLL Expr static make(std::string value);

  static constexpr const char* _type_key = "StringImm";
  TVM_DECLARE_NODE_TYPE_INFO(StringImm, ExprNode);
};

/*!
 * \brief Cast value from one data type to another.
 * \note The lanes of value should keep fixed.
 */
class Cast : public ExprNode {
 public:
  /*! \brief Original data type. */
  Expr value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("value", &value);
  }

  TVM_DLL static Expr make(Type t, Expr v);

  static constexpr const char* _type_key = "Cast";
  TVM_DECLARE_NODE_TYPE_INFO(Cast, ExprNode);
};

/*!
 * \brief Base template to implement binary ops.
 * \tparam T The type of the child class.
 */
template<typename T>
class BinaryOpNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  Expr a;
  /*! \brief The right operand. */
  Expr b;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &(this->type));
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  static Expr make(Expr a, Expr b) {
    CHECK(a.defined()) << "ValueError: a is undefined\n";
    CHECK(b.defined()) << "ValueError: b is undefined\n";
    CHECK(a.type() == b.type()) << "TypeError: mismatched types\n";
    NodePtr<T> node = make_node<T>();
    node->type = a.type();
    node->a = std::move(a);
    node->b = std::move(b);
    return Expr(node);
  }

  TVM_DECLARE_NODE_TYPE_INFO(T, ExprNode);
};

/*! \brief a + b */
class Add : public BinaryOpNode<Add> {
 public:
  static constexpr const char* _type_key = "Add";
};

/*! \brief a - b */
class Sub : public BinaryOpNode<Sub> {
 public:
  static constexpr const char* _type_key = "Sub";
};

/*! \brief a * b */
class Mul : public BinaryOpNode<Mul> {
 public:
  static constexpr const char* _type_key = "Mul";
};

/*!
 * \brief a / b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class Div : public BinaryOpNode<Div> {
 public:
  static constexpr const char* _type_key = "Div";
};

/*!
 * \brief a % b in the C semnatics.
 * \note For integer division, C standard uses trunc div.
 */
class Mod : public BinaryOpNode<Mod> {
 public:
  static constexpr const char* _type_key = "Mod";
};

/*! \brief Floor division, floor(a/b) */
class FloorDiv : public BinaryOpNode<FloorDiv> {
 public:
  static constexpr const char* _type_key = "FloorDiv";
};

/*! \brief The remainder of the floordiv */
class FloorMod : public BinaryOpNode<FloorMod> {
 public:
  static constexpr const char* _type_key = "FloorMod";
};

/*! \brief min(a, b) */
class Min : public BinaryOpNode<Min> {
 public:
  static constexpr const char* _type_key = "Min";
};

/*! \brief max(a, b) */
class Max : public BinaryOpNode<Max> {
 public:
  static constexpr const char* _type_key = "Max";
};

/*!
 * \brief Base template to implement comparison ops.
 * \tparam T The type of the child class.
 */
template<typename T>
class CmpOpNode : public ExprNode {
 public:
  /*! \brief The left operand. */
  Expr a;
  /*! \brief The right operand. */
  Expr b;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &(this->type));
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  static Expr make(Expr a, Expr b) {
    CHECK(a.defined()) << "ValueError: a is undefined\n";
    CHECK(b.defined()) << "ValueError: b is undefined\n";
    CHECK(a.type() == b.type()) << "TypeError: mismatched types\n";
    NodePtr<T> node = make_node<T>();
    node->type = Bool(a.type().lanes());
    node->a = std::move(a);
    node->b = std::move(b);
    return Expr(node);
  }

  TVM_DECLARE_NODE_TYPE_INFO(T, ExprNode);
};

/*! \brief a == b */
class EQ : public CmpOpNode<EQ> {
 public:
  static constexpr const char* _type_key = "EQ";
};

/*! \brief a != b */
class NE : public CmpOpNode<NE> {
 public:
  static constexpr const char* _type_key = "NE";
};

/*! \brief a < b */
class LT : public CmpOpNode<LT> {
 public:
  static constexpr const char* _type_key = "LT";
};

/*! \brief a <= b */
struct LE : public CmpOpNode<LE> {
 public:
  static constexpr const char* _type_key = "LE";
};

/*! \brief a > b */
class GT : public CmpOpNode<GT> {
 public:
  static constexpr const char* _type_key = "GT";
};

/*! \brief a >= b */
class GE : public CmpOpNode<GE> {
 public:
  static constexpr const char* _type_key = "GE";
};

/*! \brief a && b */
class And : public ExprNode {
 public:
  /*! \brief The left operand. */
  Expr a;
  /*! \brief The right operand. */
  Expr b;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &(this->type));
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  TVM_DLL static Expr make(Expr a, Expr b);

  static constexpr const char* _type_key = "And";
  TVM_DECLARE_NODE_TYPE_INFO(And, ExprNode);
};

/*! \brief a || b */
class Or : public ExprNode {
 public:
  /*! \brief The left operand. */
  Expr a;
  /*! \brief The right operand. */
  Expr b;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  TVM_DLL static Expr make(Expr a, Expr b);

  static constexpr const char* _type_key = "Or";
  TVM_DECLARE_NODE_TYPE_INFO(Or, ExprNode);
};

/*! \brief !a */
class Not : public ExprNode {
 public:
  /*! \brief The input operand. */
  Expr a;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("a", &a);
  }

  TVM_DLL static Expr make(Expr a);

  static constexpr const char* _type_key = "Not";
  TVM_DECLARE_NODE_TYPE_INFO(Not, ExprNode);
};

/*!
 * \brief return true_value if condition is true, otherwise return false_value.
 * \note Both true_value and false_value could be evaluated
 *       regardless of the condition value.
 *       Do not use it to guard against out of bound access,
 *       please use if_then_else instead.
 */
class Select : public ExprNode {
 public:
  /*! \brief The condition */
  Expr condition;
  /*! \brief value to be returned when condition is true. */
  Expr true_value;
  /*! \brief value to be returned when condition is false. */
  Expr false_value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("condition", &condition);
    v->Visit("true_value", &true_value);
    v->Visit("false_value", &false_value);
  }

  TVM_DLL static Expr make(Expr condition, Expr true_value, Expr false_value);

  static constexpr const char* _type_key = "Select";
  TVM_DECLARE_NODE_TYPE_INFO(Select, ExprNode);
};

/*!
 * \brief Load the value from buffer_var.
 *
 *  Equivalent to ((DType*)buffer_var)[index]
 *  where DType is the type specified by type().element_of().
 *
 *  For example, if type = float32x3, then the load will corresponds to
 *
 * \code
 *
 *  auto buffer = static_cast<float*>(buffer_var);
 *  auto loaded_val = float32x3(buffer[index.v0], buffer[index.v1], buffer[index.v2]);
 *
 * \endcode
 */
class Load : public ExprNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The index locations to be loaded. */
  Expr index;
  /*! \brief The predicate to mask which lanes would be loaded. */
  Expr predicate;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("buffer_var", &buffer_var);
    v->Visit("index", &index);
    v->Visit("predicate", &predicate);
  }

  TVM_DLL static Expr make(Type type, Var buffer_var, Expr index, Expr predicate);

  static constexpr const char* _type_key = "Load";
  TVM_DECLARE_NODE_TYPE_INFO(Load, ExprNode);
};

/*!
 * \brief Construct a vector with lanes elements
 *        where its i-th element equals base + i * stride.
 *  This is useful to construct a index for a continuous vector load.
 *
 *  Examples:
 *  - ramp(0, 1, 3) = [0, 1, 2]
 *  - ramp(1, 2, 4) = [1, 3, 5, 7]
 */
class Ramp : public ExprNode {
 public:
  /*! \brief The base value. */
  Expr base;
  /*! \brief The stride of each step. */
  Expr stride;
  /*! \brief Total number of lanes. */
  int lanes;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("base", &base);
    v->Visit("stride", &stride);
    v->Visit("lanes", &lanes);
  }

  TVM_DLL static Expr make(Expr base, Expr stride, int lanes);

  static constexpr const char* _type_key = "Ramp";
  TVM_DECLARE_NODE_TYPE_INFO(Ramp, ExprNode);
};

/*! \brief Create a vector where all the elements are value. */
class Broadcast : public ExprNode {
 public:
  /*! \brief The base value. */
  Expr value;
  /*! \brief The numerb of lanes. */
  int lanes;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("value", &value);
    v->Visit("lanes", &lanes);
  }

  TVM_DLL static Expr make(Expr value, int lanes);

  static constexpr const char* _type_key = "Broadcast";
  TVM_DECLARE_NODE_TYPE_INFO(Broadcast, ExprNode);
};

/*!
 * \brief Let binding. Bind var to value then evaluate body.
 */
class Let : public ExprNode {
 public:
  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be binded. */
  Expr value;
  /*! \brief The result expression. */
  Expr body;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  TVM_DLL static Expr make(Var var, Expr value, Expr body);

  static constexpr const char* _type_key = "Let";
  TVM_DECLARE_NODE_TYPE_INFO(Let, ExprNode);
};

// Call node, represent a function call or a multi-dimensional array load.
//
// TODO(tvm-team):
// Refactor call with more explicit property registrations.
// rather than calling a string symbol.
// We should move most information into function itself and remove name.

/*! \brief Base node of internal functions. */
class FunctionBaseNode : public Node {
 public:
  /*! \return the name of the function */
  virtual const std::string& func_name() const = 0;
  /*! \return the number of outputs of this function */
  virtual int num_outputs() const = 0;
};

/*! \brief reference to a function */
class FunctionRef : public NodeRef {
 public:
  TVM_DEFINE_NODE_REF_METHODS(FunctionRef, NodeRef, FunctionBaseNode);
};

/*!
 * \brief Call node.
 */
class Call : public ExprNode {
 public:
  /*! \brief Possible types of calls. */
  enum CallType : int {
    /*! \brief Extern "C" function. */
    Extern = 0,
    /*! \brief Extern CXX function. */
    ExternCPlusPlus = 1,
    /*! \brief Extern "C" without side-effect. */
    PureExtern = 2,
    /*! \brief Halide-style call, evaluates func(args). */
    Halide = 3,
    /*! \brief Intrinsic functions. */
    Intrinsic = 4,
    /*! \brief Intrinsic functions that are pure. */
    PureIntrinsic = 5
  };
  /*! \brief The name of the function/intrinsic. */
  std::string name;
  /*! \brief The arguments. */
  Array<Expr> args;
  /*! \brief Type of calls. */
  CallType call_type;
  /*! \brief The function to be called. */
  FunctionRef func;
  /*! \brief The output value index if func's value is a tuple. */
  int value_index{0};

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("call_type", &call_type);
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
  }

  TVM_DLL static Expr make(Type type,
                           std::string name,
                           Array<Expr> args,
                           CallType call_type,
                           FunctionRef func = FunctionRef(),
                           int value_index = 0);

  /*! \return Whether call node is pure. */
  bool is_pure() const {
    return (call_type == PureExtern ||
            call_type == PureIntrinsic ||
            call_type == Halide);
  }

  /*!
   * \return Whether call node corresponds to a defined intrinsic.
   * \param intrin_name The name of the intrinsic.
   */
  bool is_intrinsic(const char* intrin_name) const {
    return
        ((call_type == Intrinsic ||
          call_type == PureIntrinsic) &&
         name == intrin_name);
  }

  static constexpr const char* _type_key = "Call";
  TVM_DECLARE_NODE_TYPE_INFO(Call, ExprNode);

  // Build-in intrinsics
  static constexpr const char* reinterpret = "reinterpret";
  static constexpr const char* bitwise_and = "bitwise_and";
  static constexpr const char* bitwise_not = "bitwise_not";
  static constexpr const char* bitwise_xor = "bitwise_xor";
  static constexpr const char* bitwise_or = "bitwise_or";
  static constexpr const char* shift_left = "shift_left";
  static constexpr const char* shift_right = "shift_right";
  static constexpr const char* popcount = "popcount";
  static constexpr const char* likely = "likely";
  static constexpr const char* glsl_texture_store = "glsl_texture_store";
  static constexpr const char* prefetch = "prefetch";
};

/*!
 * \brief Shuffle instruction.
 *  vec = concat(vectors)
 *  result = (vec[indices[0]], vec[indices[1]] ...)
 */
class Shuffle : public ExprNode {
 public:
  /*! \brief the input vectors. */
  Array<Expr> vectors;
  /*! \brief The indices of each element. */
  Array<Expr> indices;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("vectors", &vectors);
    v->Visit("indices", &indices);
  }

  TVM_DLL static Expr make(Array<Expr> vectors, Array<Expr> indices);
  TVM_DLL static Expr make_concat(Array<Expr> vectors);
  TVM_DLL static Expr make_extract_element(Expr vector, int index);

  static constexpr const char* _type_key = "Shuffle";
  TVM_DECLARE_NODE_TYPE_INFO(Shuffle, ExprNode);
};

// Reduce operator
class CommReducerNode;

class CommReducer : public NodeRef {
 public:
  CommReducer() {}
  explicit CommReducer(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const CommReducerNode* get() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const CommReducerNode* operator->() const;
  /*! \brief type indicate the container type */
  using ContainerType = CommReducerNode;
};

/*!
 * \brief A commutative reducer node to represent a commutative
 *  binary operator with identity element
 */
class CommReducerNode : public Node {
 public:
  /*! \brief The left argument of reducer */
  Array<Var> lhs;
  /*! \brief The right argument of reducer */
  Array<Var> rhs;
  /*! \brief The result of reducer */
  Array<Expr> result;
  /*!
   * \brief The identity element of reducer, which leaves other
   *  elements unchanged when combined with it, with respect to
   *  the binary operation of this reducer uses.
   */
  Array<Expr> identity_element;
  /*! \brief Function call operator to combine a and b */
  Array<Expr> operator()(Array<Expr> a, Array<Expr> b) const;
  /*! \brief construct CommReducer from args, result and identity_element */
  TVM_DLL static CommReducer make(Array<Var> lhs,
                                  Array<Var> rhs,
                                  Array<Expr> result,
                                  Array<Expr> identity_element);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("result", &result);
    v->Visit("identity_element", &identity_element);
  }

  static constexpr const char* _type_key = "CommReducer";
  TVM_DECLARE_NODE_TYPE_INFO(CommReducerNode, Node);
};

inline const CommReducerNode* CommReducer::get() const {
  return static_cast<CommReducerNode*>(node_.get());
}
inline const CommReducerNode* CommReducer::operator->() const {
  return static_cast<CommReducerNode*>(node_.get());
}

/*! \brief Reduction operator operator */
class Reduce : public ExprNode {
 public:
  /*! \brief The commutative combiner */
  CommReducer combiner;
  /*! \brief The source operand */
  Array<Expr> source;
  /*! \brief The reduction axis */
  Array<IterVar> axis;
  /*!
   * \brief Predicate on the reduction
   *  Only add the body to reduction if condition is true.
   */
  Expr condition;
  /*! \brief the index of this reduce node */
  int value_index;

  /*! \brief construct expr from op and rdom */
  TVM_DLL static Expr make(CommReducer combiner,
                           Array<Expr> src,
                           Array<IterVar> rdom,
                           Expr condition,
                           int value_index);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("combiner", &combiner);
    v->Visit("source", &source);
    v->Visit("axis", &axis);
    v->Visit("condition", &condition);
    v->Visit("value_index", &value_index);
  }

  static constexpr const char* _type_key = "Reduce";
  TVM_DECLARE_NODE_TYPE_INFO(Reduce, ExprNode);
};

/*! \brief Any shape. */
class Any : public ExprNode {
 public:
  void VisitAttrs(AttrVisitor* v) final {}
  /*! \brief Convert to var. */
  Var ToVar() const {
    return Variable::make(Int(32), "any_dim");
  }

  TVM_DLL static Expr make();

  static constexpr const char* _type_key = "Any";
  TVM_DECLARE_NODE_TYPE_INFO(Any, ExprNode);
};

// Statements
/*!
 * \brief Let binding, bind var to value, then run body.
 */
class LetStmt : public StmtNode {
 public:
  /*! \brief The variable. */
  Var var;
  /*! \brief The value to be binded. */
  Expr value;
  /*! \brief The body block. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  TVM_DLL static Stmt make(Var var, Expr value, Stmt body);

  static constexpr const char* _type_key = "LetStmt";
  TVM_DECLARE_NODE_TYPE_INFO(LetStmt, StmtNode);
};

/*!
 * \brief Define certain auxiliary attribute for the body to be a symbolic value.
 *  This provide auxiliary information for IR passes that transforms body.
 *
 *  In terms of effect, this is equivalent to Block(Evaluate(value), body).
 *
 *  Examples of possible usage:
 *    - Bound of function, variables.
 *    - Hint which block corresponds to a parallel region.
 */
class AttrStmt : public StmtNode {
 public:
  /*! \brief this is attribute about certain node */
  NodeRef node;
  /*! \brief the type key of the attribute */
  std::string attr_key;
  /*! \brief The attribute value, value is well defined at current scope. */
  Expr value;
  /*! \brief The body statement to be executed */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("node", &node);
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
    v->Visit("body", &body);
  }

  TVM_DLL static Stmt make(NodeRef node,
                           std::string type_key,
                           Expr value,
                           Stmt body);

  static constexpr const char* _type_key = "AttrStmt";
  TVM_DECLARE_NODE_TYPE_INFO(AttrStmt, StmtNode);
};

/*!
 * \brief Assert condition, if an error occurs, return the error message.
 */
class AssertStmt : public StmtNode {
 public:
  /*! \brief Condition to be checked. */
  Expr condition;
  /*! \brief Error message when assertion failed. */
  Expr message;
  /*!
   * \brief Body which this assertion holds true.
   *  Will be executed after the assertion.
   */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("condition", &condition);
    v->Visit("message", &message);
    v->Visit("body", &body);
  }

  TVM_DLL static Stmt make(Expr condition, Expr message, Stmt body);

  static constexpr const char* _type_key = "AssertStmt";
  TVM_DECLARE_NODE_TYPE_INFO(AssertStmt, StmtNode);
};

// TODO(tvm-team): consider consolidate with AttrStmt.
/*! \brief annotation node of producer/consumer relation. */
class ProducerConsumer : public StmtNode {
 public:
  /*! \brief The corresponding tensor. */
  FunctionRef func;
  /*! \brief Whether the relation is producer. */
  bool is_producer;
  /*! \brief Body to be executed. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("is_producer", &is_producer);
    v->Visit("body", &body);
  }

  TVM_DLL static Stmt make(FunctionRef func, bool is_producer, Stmt body);

  static constexpr const char* _type_key = "ProducerConsumer";
  TVM_DECLARE_NODE_TYPE_INFO(ProducerConsumer, StmtNode);
};

/*!
 * \brief Store value to the buffer.
 *
 *  Equivalent to ((DType*)buffer_var)[index] = value.
 *  where DType is the type specified by type().element_of().
 *
 *  For example, if type = float32x3, then the load will corresponds to
 *
 * \code
 *
 *  auto buffer = static_cast<float*>(buffer_var);
 *  buffer[index.v0] = value.v0;
 *  buffer[index.v1] = value.v1;
 *  buffer[index.v2] = value.v2;
 *
 * \endcode
 * \sa Load
 */
class Store : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The value to be stored. */
  Expr value;
  /*! \brief The index locations to be stored. */
  Expr index;
  /*! \brief The predicate to mask which lanes would be stored. */
  Expr predicate;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("value", &value);
    v->Visit("index", &index);
    v->Visit("predicate", &predicate);
  }

  TVM_DLL static Stmt make(Var buffer_var,
                           Expr value,
                           Expr index,
                           Expr predicate);

  static constexpr const char* _type_key = "Store";
  TVM_DECLARE_NODE_TYPE_INFO(Store, StmtNode);
};

/*!
 * \brief Store value into mult-dimensional array defined by func.
 */
class Provide : public StmtNode {
 public:
  /*! \brief The function to be updated. */
  FunctionRef func;
  /*! \brief The output value index if func's value is a tuple. */
  int value_index{0};
  /*! \brief The value to be stored. */
  Expr value;
  /*! \brief The index arguments of the function. */
  Array<Expr> args;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
    v->Visit("value", &value);
    v->Visit("args", &args);
  }

  TVM_DLL static Stmt make(FunctionRef func,
                           int value_index,
                           Expr value,
                           Array<Expr> args);

  static constexpr const char* _type_key = "Provide";
  TVM_DECLARE_NODE_TYPE_INFO(Provide, StmtNode);
};

/*!
 * \brief Allocate a buffer that can be used in body.
 */
class Allocate : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The type of the buffer. */
  DataType type;
  /*! \brief The extents of the buffer. */
  Array<Expr> extents;
  /*! \brief Only allocate buffer when condition is satisfied. */
  Expr condition;
  /*! \brief The body to be executed. */
  Stmt body;
  // The following two fields are deprecated
  // kept for backward compatibility and will be refactored later.
  Expr new_expr;
  std::string free_function;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("buffer_var", &buffer_var);
    v->Visit("dtype", &type);
    v->Visit("extents", &extents);
    v->Visit("condition", &condition);
    v->Visit("body", &body);
  }

  TVM_DLL static Stmt make(Var buffer_var,
                           DataType type,
                           Array<Expr> extents,
                           Expr condition,
                           Stmt body,
                           Expr new_expr = Expr(),
                           std::string free_function = std::string());

  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \return The result.
   */
  int32_t constant_allocation_size() const {
    return constant_allocation_size(extents);
  }
  /*!
   * \brief If the buffer size is constant, return the size.
   *        Otherwise return 0.
   * \param extents The extents of the buffer.
   * \return The result.
   */
  TVM_DLL static int32_t constant_allocation_size(
      const Array<Expr>& extents);

  static constexpr const char* _type_key = "Allocate";
  TVM_DECLARE_NODE_TYPE_INFO(Allocate, StmtNode);
};

/*! \brief Free the resources in the buffer before the scope ends. */
class Free : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("buffer_var", &buffer_var);
  }

  TVM_DLL static Stmt make(Var buffer_var);

  static constexpr const char* _type_key = "Free";
  TVM_DECLARE_NODE_TYPE_INFO(Free, StmtNode);
};

/*!
 * \brief Annotate the bounds where func need to be written and read in body.
 *  We will need to allocate space for the corresponding regions.
 */
class Realize : public StmtNode {
 public:
  /*! \brief The function to be realized. */
  FunctionRef func;
  /*! \brief The output value index if func's value is a tuple. */
  int value_index;
  /*! \brief The data type of the array. */
  DataType type;
  /*! \brief Bounds to be realized. */
  Region bounds;
  /*! \brief Only realize if condition holds. */
  Expr condition;
  /*! \brief The body of realization. */
  Stmt body;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
    v->Visit("dtype", &type);
    v->Visit("bounds", &bounds);
    v->Visit("condition", &condition);
    v->Visit("body", &body);
  }

  TVM_DLL static Stmt make(FunctionRef func,
                           int value_index,
                           DataType type,
                           Region bounds,
                           Expr condition,
                           Stmt body);

  static constexpr const char* _type_key = "Realize";
  TVM_DECLARE_NODE_TYPE_INFO(Realize, StmtNode);
};

/*!
 * \brief A sequence of statements.
 */
class Block : public StmtNode {
 public:
  /*! \brief The first statement. */
  Stmt first;
  /*! \brief The restof statments. */
  Stmt rest;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("first", &first);
    v->Visit("rest", &rest);
  }

  TVM_DLL static Stmt make(Stmt first, Stmt rest);
  TVM_DLL static Stmt make(const std::vector<Stmt> &stmts);

  static constexpr const char* _type_key = "Block";
  TVM_DECLARE_NODE_TYPE_INFO(Block, StmtNode);
};

/*!
 * \brief IfThenElse statment.
 */
class IfThenElse : public StmtNode {
 public:
  /*! \brief The condition. */
  Expr condition;
  /*! \brief The branch to be executed when condition is true. */
  Stmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  Stmt else_case;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("condition", &condition);
    v->Visit("then_case", &then_case);
    v->Visit("else_case", &else_case);
  }

  TVM_DLL static Stmt make(Expr condition, Stmt then_case, Stmt else_case = Stmt());

  static constexpr const char* _type_key = "IfThenElse";
  TVM_DECLARE_NODE_TYPE_INFO(IfThenElse, StmtNode);
};

/*!
 * \brief Evaluates an expression.
 *  This is mostly used for putting a Call node into Stmt.
 *
 *  If value do not have side-effect, this node can be safely removed.
 */
class Evaluate : public StmtNode {
 public:
  /*! \brief The expression to be evaluated. */
  Expr value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("value", &value);
  }

  TVM_DLL static Stmt make(Expr v);

  static constexpr const char* _type_key = "Evaluate";
  TVM_DECLARE_NODE_TYPE_INFO(Evaluate, StmtNode);
};

/*! \brief Additional annotation of for loop. */
enum class ForType : int {
  /*! \brief serial execution. */
  Serial = 0,
  /*! \brief parallel execution on CPU. */
  Parallel = 1,
  /*! \brief Vector SIMD loop annotaion. */
  Vectorized = 2,
  /*! \brief Unroll annotation. */
  Unrolled = 3
};

// Kevice api of for loop
// kept for backward compatibility
// consider refactor and remove later.
enum class DeviceAPI: int {
  None = 0
};

/*!
 * \brief A for loop, with poissible type annotations.
 *
 * \code
 *
 *  for (loop_var = min; loop_var < min + extent; ++loop_var) {
 *    // body
 *  }
 * \endcode
 */
class For : public StmtNode {
 public:
  /*! \brief The loop variable. */
  Var loop_var;
  /*! \brief The minimum value of iteration. */
  Expr min;
  /*! \brief The extent of the iteration. */
  Expr extent;
  /*! \brief The type of the for loop. */
  ForType for_type;
  /*!
   * \brief Deprecated, reserved for backward compatibility.
   *  Consider refactor and remove later.
   */
  DeviceAPI device_api;
  /*! \brief The body of the for loop. */
  Stmt body;

  TVM_DLL static Stmt make(Var loop_var,
                           Expr min,
                           Expr extent,
                           ForType for_type,
                           DeviceAPI device_api,
                           Stmt body);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("loop_var", &loop_var);
    v->Visit("min", &min);
    v->Visit("extent", &extent);
    v->Visit("for_type", &for_type);
    v->Visit("device_api", &device_api);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "For";
  TVM_DECLARE_NODE_TYPE_INFO(For, StmtNode);
};

/*!
 * \brief A prefetch hint of func.
 */
class Prefetch : public StmtNode {
 public:
  /*! \brief The function to be prefetched. */
  FunctionRef func;
  /*! \brief The output value index if func's value is a tuple. */
  int value_index;
  /*! \brief The data type of the array. */
  DataType type;
  /*! \brief Bounds to be prefetched. */
  Region bounds;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("value_index", &value_index);
    v->Visit("type", &type);
    v->Visit("bounds", &bounds);
  }

  TVM_DLL static Stmt make(FunctionRef func,
                           int value_index,
                           DataType type,
                           Region bounds);

  static constexpr const char* _type_key = "Prefetch";
  TVM_DECLARE_NODE_TYPE_INFO(Prefetch, StmtNode);
};

/*!
 * \brief Auxiliary data structure used in IR Pass to indicate a tensor.
 */
struct TensorKey {
  FunctionRef f;
  int value_index;

  inline bool operator==(const TensorKey& other) const {
    return f == other.f && value_index == other.value_index;
  }
  inline std::string GetName() const {
    if (f->num_outputs() == 1) return f->func_name();
    std::ostringstream os;
    os << f->func_name() << ".v" << value_index;
    return os.str();
  }
};

/*! \brief namespace of possible attribute sin AttrStmt.attr_key */
namespace attr {
// The above attr does not pass to ir stage.
/*! \brief Mark launching extent of thread, used by device API. */
constexpr const char* thread_extent = "thread_extent";
/*! \brief Mark launching of a virtual thread. */
constexpr const char* virtual_thread = "virtual_thread";
/*! \brief Mark region is processed by a co-proccesor */
constexpr const char* coproc_scope = "coproc_scope";
/*!
 * \brief Mark region creates coprocessor micro ops,
 *  can be reused if corresponding variable is independent.
 */
constexpr const char* coproc_uop_scope = "coproc_uop_scope";
/*! \brief Mark the scope as volatile access for certain handle. */
constexpr const char* volatile_scope = "volatile_scope";
/*!
 * \brief Mark the scope as generated by extern primitive.
 *  such scope can contain arbitrary ir program and we need to be careful
 *  when make certain assumptions about the structure of the program.
 */
constexpr const char* extern_scope = "extern_scope";
/*!
 * \brief Mark the scope as when computation start to happen
 *  This can hint some code generator to create a new function for compute.
 */
constexpr const char* compute_scope = "compute_scope";
/*! \brief Mark storage scope of buffers */
constexpr const char* storage_scope = "storage_scope";
/*! \brief Mark storage alignement requirement of buffers */
constexpr const char* storage_alignment = "storage_alignment";
/*! \brief Mark storage scope of realization */
constexpr const char* realize_scope = "realize_scope";
/*! \brief The allocation context for global malloc in host. */
constexpr const char* device_context_id = "device_context_id";
/*! \brief The device type. */
constexpr const char* device_context_type = "device_context_type";
/*! \brief Mark of loop scope */
constexpr const char* loop_scope = "loop_scope";
/*! \brief Mark of reduce scope */
constexpr const char* reduce_scope = "reduce_scope";
/*! \brief Mark region is guarded by the pragma extension */
constexpr const char* pragma_scope_prefix = "pragma_";
/*! \brief Import llvm source or file into the final code gen module */
constexpr const char* pragma_import_llvm = "pragma_import_llvm";
/*!
 * \brief Mark of prefetch scope, value=offset,
 *  run prefetch of Tensor on the current loop scope
 */
constexpr const char* prefetch_scope = "prefetch_scope";
/*!
 * \brief Marks production of double buffer data
 */
constexpr const char* double_buffer_scope = "double_buffer_scope";
/*!
 * \brief Marks region used by double buffer write
 */
constexpr const char* double_buffer_write = "double_buffer_write";
/*! \brief Mark of scan update scope */
constexpr const char* scan_update_scope = "scan_update_scope";
/*! \brief Mark of scan init scope */
constexpr const char* scan_init_scope = "scan_init_scope";
/*!
 * \brief Mark alignment of buffer dimension
 *  stmt.node is Tensor
 *  stmt.value is tvm_tuple(dim, align, offset)
 *  This gives hint to require stride of dim to be k * align + offset.
 */
constexpr const char* buffer_dim_align = "buffer_dim_align";
/*! \brief Mark stores/loads with theirs bounds.  */
constexpr const char* buffer_bound = "buffer_bound";
/*!
 * \brief Bind the buffer specification to the region of the op
 *  When this scope occurs, the stmt.node is a Array<NodeRef> = [buffer, tensor]
 *  stmt.value is a tvm_tuple(min0, extent0, min1, extent1, ...).
 *  The scope represents that we need to bind the storage region of tensor to buffer.
 *  This will affect replacement of some variables inside the scope that
 *  corresponds to field of buffer to be the actual expressions of tensor during
 *  storage flattening phase.
 */
constexpr const char* buffer_bind_scope = "buffer_bind_scope";
// Pipeline related attributes
/*! \brief channel read scope */
constexpr const char* channel_read_scope = "channel_read_scope";
/*! \brief Advance step of channel after end of scope */
constexpr const char* channel_read_advance = "channel_read_advance";
/*! \brief channel write scope */
constexpr const char* channel_write_scope = "channel_write_scope";
/*! \brief Advance step of channel after end of scope */
constexpr const char* channel_write_advance = "channel_write_advance";
/*! \brief pipeline stage scope, implies always execution */
constexpr const char* pipeline_stage_scope = "pipeline_stage_scope";
/*! \brief pipeline execution scope, implies the scope can be pipelined. */
constexpr const char* pipeline_exec_scope = "pipeline_exec_scope";
/*!
 * \brief Mark that this stage is an OpenGL shader. Since OpenGL shader only
 * allows writing out to one element of the output texture, the Provide node
 * gets translated to a special Call::glsl_texture_store statement instead of a
 * Store statement.
 */
constexpr const char* opengl_stage_scope = "opengl_stage_scope";

/*!
 * \brief Mark that it is in the device scope.
 */
constexpr const char* device_scope = "device_scope";

/*!
 * \brief Check if attr_key is a pragma key extension
 * \param attr_key The attr key to be compared
 * \return true if it is a pragma key
 */
inline bool IsPragmaKey(const std::string& attr_key) {
  return attr_key.compare(0, 7, "pragma_") == 0;
}

}  // namespace attr

/*! \brief namespace of TVM Intrinsic functions */
namespace intrinsic {
/*!
 * \brief See pesudo code
 *
 *  Handle tvm_address_of(Load *op) {
 *     return &op->buffer_var[index];
 *  }
 */
constexpr const char* tvm_address_of = "tvm_address_of";
/*!
 * \brief Same as select, used for unsafe memory access.
 *
 *  Type tvm_if_then_else(cond, a, b) {
 *    return cond ? a : b;
 *  }
 */
constexpr const char* tvm_if_then_else = "tvm_if_then_else";
/*!
 * \brief Get head access address with memory access pattern info.
 *
 *  This operator also marks range of the memory access
 *  The offset and extent are in unit of the DType(including vectorization factor).
 *  rw_mask is a bit_mask setting whether the access is a read(1) or write(2).
 *  The access is assume to happen in the current expression.
 *
 *  PtrType tvm_access_ptr(Expr dtype, DType* data,
 *                         int offset, int extent,
 *                         int rw_mask) {
 *    // DType == dtype.type();
 *    return &data[offset];
 *  }
 */
constexpr const char* tvm_access_ptr = "tvm_access_ptr";
/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 *  can be used to cache function local static resources.
 */
constexpr const char* tvm_static_handle = "tvm_static_handle";
/*!
 * \brief Return a unique context id, used for hint of workspace separation.
 *  Different context id ganrantees not having overlapping workspace.
 */
constexpr const char* tvm_context_id = "tvm_context_id";
/*!
 * \brief tvm_tuple is not an actual function and cannot codegen.
 *  It is used to represent tuple structure in value field of AttrStmt,
 *  for the sake of giving hint to optimization.
 *
 *  Handle tvm_tuple(value0, value1, ..., value_n);
 */
constexpr const char* tvm_tuple = "tvm_tuple";
/*!
 * \brief See pesudo code
 *
 *  Type tvm_struct_get(StructType* arr, int index, int field_id) {
 *     return arr[index]->field;
 *  }
 * \sa TVMStructFieldKind
 */
constexpr const char* tvm_struct_get = "tvm_struct_get";
/*!
 * \brief See pesudo code
 *
 *  Handle tvm_struct_set(StructType* arr, int index, int field_id, value) {
 *     arr[index]->field = value;
 *  }
 * \sa TVMStructFieldKind
 */
constexpr const char* tvm_struct_set = "tvm_struct_set";
/*!
 * \brief See pesudo code
 *
 *  bool tvm_handle_is_null(void* handle) {
 *     return handle == nullptr
 *  }
 */
constexpr const char* tvm_handle_is_null = "tvm_handle_is_null";
/*!
 * \brief See pesudo code
 *
 *  void tvm_throw_last_error() {
 *    throw TVMGetLastError();
 *  }
 */
constexpr const char* tvm_throw_last_error = "tvm_throw_last_error";
/*!
 * \brief See pesudo code
 *
 *  dtype in {shape, array, arg_value, arg_tcode}
 *
 *  Handle tvm_stack_alloca(string dtype, int num) {
 *     return new on stack dtype[num];
 *  }
 */
constexpr const char* tvm_stack_alloca = "tvm_stack_alloca";
/*!
 * \brief Allocate a shape tuple on stack, return the handle.
 *
 *  Handle tvm_stack_make_shape(list args) {
 *     ret = alloca stack int64_t[len(args)];
 *     for i in range(len(args)):
 *        ret[i] = args[i]
 *     return &ret[0];
 *  }
 */
constexpr const char* tvm_stack_make_shape = "tvm_stack_make_shape";
/*!
 * \brief Allocate a NDArray(DLTensor) on stack, return the handle.
 *
 *  Type tvm_stack_make_array(Expr data,
 *                            Expr shape,
 *                            Expr strides,
 *                            Expr ndim,
 *                            Expr dtype,
 *                            Expr elem_offset) {
 *     ret = alloca stack DLTensor();
 *     ret->data = data;
 *     ret->shape = shape;
 *     ret->strides = strides != 0 ? strides : nullptr;
 *     ret->ndim = ndim;
 *     ret->dtype = dtype.type();
 *     ret->byte_offset = elem_offset * sizeof(dtype);
 *     return ret;
 *  }
 */
constexpr const char* tvm_stack_make_array = "tvm_stack_make_array";
/*!
 * \brief See pesudo code
 *
 *  int tvm_call_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     return 0;
 *  }
 */
constexpr const char* tvm_call_packed = "tvm_call_packed";
/*!
 * \brief See pesudo code
 *
 *  int tvm_call_trace_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     return 0;
 *  }
 */
constexpr const char *tvm_call_trace_packed = "tvm_call_trace_packed";
/*!
 * \brief See pesudo code
 *  Mark the content as thread local context, can get optimized
 *  by only call the call once at thread start.
 *
 *  Do not allow nesting(getting a thread context from another).
 *
 *  Handle tvm_thread_context(Expr call) {
 *     return call;
 *  }
 */
constexpr const char* tvm_thread_context = "tvm_thread_context";
/*!
 * \brief Lowered version of call packed, the space of value and
 *  type codes are explicitly allocated.
 *
 *  int tvm_call_packed_lowered(name,
 *                              TVMValue* value_stack,
 *                              int* tcode_stack,
 *                              int begin,
 *                              int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
constexpr const char* tvm_call_packed_lowered = "tvm_call_packed_lowered";
/*!
 * \brief Lowered version of trace intrinsic, the space of value and
 *  type codes are explicitly allocated. The return value is the
 *  (end - 1) value on the stack.
 *
 *  int tvm_call_trace_packed_lowered(name,
 *                                    TVMValue* value_stack,
 *                                    int* tcode_stack,
 *                                    int begin,
 *                                    int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
constexpr const char *tvm_call_trace_packed_lowered =
    "tvm_call_trace_packed_lowered";
/*!
 * \brief See pseudo code
 *
 *  int tvm_storage_sync(std::string storage_scope) {
 *     __sync(storage_scope);
 *     return 0;
 *  }
 */
constexpr const char* tvm_storage_sync = "tvm_storage_sync";
/*!
 * \brief See pseudo code
 *
 *  Type tvm_warp_shuffle(Type value, warp_id) {
 *     return (value passed in by warp indicated by warp_id);
 *  }
 */
constexpr const char* tvm_warp_shuffle = "tvm_warp_shuffle";
/*!
 * \brief Initialize the global barrier.
 *  Call this at beginning of kernel that need global barrier.
 */
constexpr const char* tvm_global_barrier_kinit = "tvm_global_barrier_kinit";
/*!
 * \brief See pesudo code
 *
 *  void tvm_thread_allreduce(UIntImm size, Expr source0, ..., Expr cond,
 *                            Var reduce_temp0, .., Var thread_idx1, ...) {
 *     // constraint by the other thread_idx remain the same.
 *     // reduce_temp is used to save intermediate result.
 *     reduce_temp0, ... = reduce(combiner, source0, ..., cond
 *       over [thread_idx1, thread_idx2] passed by any caller)
 *  }
 */
constexpr const char* tvm_thread_allreduce = "tvm_thread_allreduce";

}   // namespace intrinsic

/*!
 * \brief Create a type annotation expression
 * \param dtype The data type
 * \return Expr a expression with dtype.
 */
inline Expr TypeAnnotation(Type dtype) {
  return ir::Call::make(dtype,
                        "type_annotation", {},
                        ir::Call::PureIntrinsic);
}

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, ForType for_type);

}  // namespace ir
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::ir::TensorKey> {
  std::size_t operator()(const ::tvm::ir::TensorKey& k) const {
    size_t lhs = k.f.hash();
    size_t rhs = static_cast<size_t>(k.value_index);
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std

#endif  // TVM_IR_H_
