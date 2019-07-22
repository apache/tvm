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
 * \file tvm/expr.h
 * \brief The Expr and related elements in DataFlow construction.
 */
#ifndef TVM_EXPR_H_
#define TVM_EXPR_H_

#include <string>
#include <algorithm>
#include <unordered_map>
#include "base.h"
#include "dtype.h"
#include "node/container.h"
#include "node/ir_functor.h"
#include "runtime/c_runtime_api.h"

namespace tvm {

/*! \brief Base node of all expressions. */
class ExprNode : public Node {
 public:
  /*! \brief The data type of the expression. */
  DataType type;

  static constexpr const char* _type_key = "Expr";
  TVM_DECLARE_BASE_NODE_INFO(ExprNode, Node);
};

/*! \brief Container of all expressions. */
class Expr : public NodeRef {
 public:
  Expr() {}
  explicit Expr(NodePtr<Node> ptr) : NodeRef(ptr) {}
  /*!
   * \brief construct from integer.
   * \param value The value to be constructed.
   */
  TVM_DLL Expr(int32_t value);  // NOLINT(*)
  /*!
   * \brief construct from float.
   * \param value The value to be constructed.
   */
  TVM_DLL Expr(float value);  // NOLINT(*)
  /*!
   * \brief construct from string.
   * \param str The value to be constructed.
   */
  TVM_DLL Expr(std::string str);  // NOLINT(*)

  /*! \return the data type of this expression. */
  DataType type() const {
    return static_cast<const ExprNode*>(get())->type;
  }

  /*! \brief type indicate the container type */
  using ContainerType = ExprNode;
};

/*! \brief Base node of all statements. */
class StmtNode : public Node {
 public:
  static constexpr const char* _type_key = "Stmt";
  TVM_DECLARE_BASE_NODE_INFO(StmtNode, Node);
};

/*! \brief Container of all statements */
class Stmt : public NodeRef {
 public:
  TVM_DEFINE_NODE_REF_METHODS(Stmt, NodeRef, StmtNode);
};

class Var;
/*!
 * \brief A variable node in the IR.
 *
 * A vraible is uniquely identified by its address.
 *
 * Each variable is only binded once in the following nodes:
 * - Allocate
 * - For
 * - Let
 * - LetStmt
 */
class Variable : public ExprNode {
 public:
  /*!
   * \brief The hint to the variable name.
   * \note Each variable is uniquely identified by its address.
   */
  std::string name_hint;

  static Var make(DataType dtype, std::string name_hint);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("name", &name_hint);
  }

  static constexpr const char* _type_key = "Variable";
  TVM_DECLARE_NODE_TYPE_INFO(Variable, ExprNode);
};

/*! \brief a named variable in TVM */
class Var : public Expr {
 public:
  explicit Var(NodePtr<Node> n) : Expr(n) {}
  TVM_DLL explicit Var(std::string name_hint = "v",
                       Type t = Int(32));
  /*!
   * \brief Make a new copy of var with same type, append suffix
   * \param suffix The suffix to be appended.
   * \return the new Var copy
   */
  Var copy_with_suffix(const std::string& suffix) const {
    return Var((*this)->name_hint + suffix, (*this)->type);
  }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const Variable* operator->() const {
    return get();
  }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const Variable* get() const {
    return static_cast<Variable*>(node_.get());
  }
  /*! \brief type indicate the container type */
  using ContainerType = Variable;
};

// Backward compatibility, will be removed later.
using VarExpr = Var;
using BaseExprNode = ExprNode;
using ExprHash = NodeHash;
using ExprEqual = NodeEqual;

class Integer;
/*! \brief ExprNode: constant integer. */
class IntImm : public ExprNode {
 public:
  /*! \brief the Internal value. */
  int64_t value;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("value", &value);
  }

  TVM_DLL static Integer make(DataType t, int64_t value);

  static constexpr const char* _type_key = "IntImm";
  TVM_DECLARE_NODE_TYPE_INFO(IntImm, ExprNode);
};

/*!
 * \brief Container of constant integer (IntImm).
 *
 * This is used to store and automate type check
 * attributes that must be constant integer.
 */
class Integer : public Expr {
 public:
  Integer() : Expr() {}
  /*!
   * \brief constructor from node.
   */
  explicit Integer(NodePtr<Node> node) : Expr(node) {}
  /*!
   * \brief Construct integer from int value.
   */
  Integer(int value) : Expr(value) {}  // NOLINT(*)
  /*!
   * \brief Assign an expression to integer.
   * \param other another expression.
   */
  Integer& operator=(const Integer& other) {
    node_ = other.node_;
    return *this;
  }
  /*!
   * \brief Get pointer to the internal value.
   * \return the content of the integer.
   */
  const IntImm* operator->() const {
    return static_cast<const IntImm*>(node_.get());
  }
  /*!
   * \brief convert to int64_t
   */
  operator int64_t() const {
    CHECK(node_ != nullptr)
        << " Trying to reference a null Integer";
    return (*this)->value;
  }
  /*! \brief type indicate the container type */
  using ContainerType = IntImm;
};

/*! \brief range over one dimension */
class RangeNode : public Node {
 public:
  /*! \brief beginning of the node */
  Expr min;
  /*! \brief the extend of range */
  Expr extent;
  /*! \brief constructor */
  RangeNode() {}
  RangeNode(Expr min, Expr extent) : min(min), extent(extent) {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("min", &min);
    v->Visit("extent", &extent);
  }

  static constexpr const char* _type_key = "Range";
  TVM_DECLARE_NODE_TYPE_INFO(RangeNode, Node);
};

/*! \brief Range constainer  */
class Range : public NodeRef {
 public:
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   */
  TVM_DLL Range(Expr begin, Expr end);
  /*!
   * \brief construct a new range with min and extent
   *  The corresponding constructor is removed,
   *  because that is counter convention of tradition meaning
   *  of range(begin, end)
   *
   * \param min The minimum range.
   * \param extent The extent of the range.
   */
  static Range make_by_min_extent(Expr min, Expr extent);
  // declare range.
  TVM_DEFINE_NODE_REF_METHODS(Range, NodeRef, RangeNode);
};

/*! \brief container class of iteration variable. */
class IterVarNode;

using Region = Array<Range>;

/*!
 * \brief Type of iteration variable.
 *  Each IterVar have a specific type.
 *
 *  The type of iter var can be overriden via
 *  stage.iter_var_attrs given they are compatible.
 */
enum IterVarType : int {
  /*!
   * \brief Data parallel iteration.
   *  This normally corresponds to axis of Tensor.
   *  Allow all IterVar manipulations.
   *
   * \note This does not mean the loop
   *  have to be executed in parallel fashion.
   */
  kDataPar = 0,
  /*!
   * \brief The IterVar itself is a thread-index
   *  of a fixed thread launching group.
   *  Note that this is already assumed to be paralellized.
   *
   *  Disallow: split/fuse/vectorize/parallel
   */
  kThreadIndex = 1,
  /*!
   * \brief Communicative reduction.
   *  Cannot be directly parallelized.
   *
   *  Disallow: parallel/vectorize
   */
  kCommReduce = 2,
  /*!
   * \brief Serial loops with loop carry dependency,
   *  the iteration must execute in order.
   *  Cannot be re-ordered.
   *
   *  Disallow: reorder/parallel/vectorize
   */
  kOrdered = 3,
  /*!
   * \brief IterVar is opaque,
   *
   *  May not corresponds to any generated loop
   *  Disallow all IterVar manipulations and compute_at
   *
   * \note This is usually used to implement composite op
   *  or external op, where the
   */
  kOpaque = 4,
  // The following are possible additional
  // types that are provided during schedule
  /*!
   * \brief The execution is unrolled.
   */
  kUnrolled = 5,
  /*!
   * \brief The loop is vectorized.
   */
  kVectorized = 6,
  /*!
   * \brief The loop is parallelized.
   */
  kParallelized = 7,
  /*!
   * \brief Marks boundary of tensorization intrinsic.
   */
  kTensorized = 8
};

/*!
 * \brief Iteration Variable,
 *  represents an iteration over an integer interval.
 */
class IterVar : public NodeRef {
 public:
  // construct a new iter var without a domain
  IterVar() {}
  // construct from shared ptr.
  explicit IterVar(NodePtr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IterVarNode* operator->() const;
  /*!
   * \return the corresponding var in the IterVar.
   */
  inline operator Expr() const;
  /*! \brief specify container node */
  using ContainerType = IterVarNode;
};

/*!
 * \brief Create a new IterVar that represents an axis in thread.
 *
 * \param dom Optional, domain of the thread axis.
 * \param tag The thread tag of the axis.
 */
TVM_DLL IterVar thread_axis(Range dom, std::string tag);

/*!
 * \brief Create a new IterVar for reduction operations.
 *
 * \param dom The domain of the reduction axis.
 * \param name The name of the reduction axis.
 */
TVM_DLL IterVar reduce_axis(Range dom, std::string name = "rv");

using Domain = Array<Range>;

/*!
 * \brief Dump the node to stderr, used for debug purposes.
 * \param node The input node
 */
TVM_DLL void Dump(const NodeRef& node);

// definition of Node.
/*!
 * \brief An iteration variable representing an iteration
 *  over a one dimensional interval.
 */
class IterVarNode : public Node {
 public:
  /*!
   * \brief the domain of iteration, if known, can be None
   *  For the intermediate schedule node, before schedule.
   */
  Range dom;
  /*! \brief The looping variable */
  Var var;
  /*! \brief The type of the IterVar */
  IterVarType iter_type;
  /*!
   * \brief additional tag on the iteration variable,
   *  set this if this is binded already to a known thread tag.
   */
  std::string thread_tag;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dom", &dom);
    v->Visit("var", &var);
    v->Visit("iter_type", &iter_type);
    v->Visit("thread_tag", &thread_tag);
  }

  TVM_DLL static IterVar make(Range dom, Var var,
                              IterVarType iter_type,
                              std::string thread_tag = "");

  static constexpr const char* _type_key = "IterVar";
  TVM_DECLARE_NODE_TYPE_INFO(IterVarNode, Node);
};

// inline implementations
inline const IterVarNode* IterVar::operator->() const {
  return static_cast<const IterVarNode*>(node_.get());
}

inline IterVar::operator Expr() const {
  return (*this)->var;
}

inline const char* IterVarType2String(IterVarType t) {
  switch (t) {
    case kDataPar: return "DataPar";
    case kThreadIndex: return "ThreadIndex";
    case kCommReduce: return "CommReduce";
    case kOrdered: return "Ordered";
    case kOpaque: return "Opaque";
    case kUnrolled: return "Unrolled";
    case kVectorized: return "Vectorized";
    case kParallelized: return "Parallelized";
    case kTensorized: return "Tensorized";
  }
  return "Unknown";
}

/*!
 * \brief Construct a new Var expression
 * \param name_hint The name hint for the expression
 * \param t The type of the expression
 */
TVM_DLL Var var(std::string name_hint, Type t = Int(32));

/*
 * \brief Template function to convert Map to unordered_map
 *  Sometimes useful for API gluing when internal uses unordered_map
 * \param dmap The container map
 * \return The corresponding unordered_map.
 * \tparam K the key of the Map.
 * \tparam V the value of the Map.
 */
template<typename K, typename V>
inline std::unordered_map<K, V> as_unordered_map(const Map<K, V>& dmap) {
  std::unordered_map<K, V> ret;
  for (auto kv : dmap) {
    ret[kv.first] = kv.second;
  }
  return ret;
}

// Printer infra.
/*! \brief A Pretty printer class to print the IR. */
class IRPrinter {
 public:
  /*! \brief The output stream */
  std::ostream& stream;
  /*! \brief The indentation level. */
  int indent{0};
  explicit IRPrinter(std::ostream& stream)  // NOLINT(*)
      : stream(stream) {}

  /*! \brief The node to be printed. */
  TVM_DLL void Print(const NodeRef& node);
  /*! \brief Print indent to the stream */
  TVM_DLL void PrintIndent();
  // Allow registration to be printer.
  using FType = IRFunctor<void(const NodeRef&, IRPrinter *)>;
  TVM_DLL static FType& vtable();
};

// default print function for all nodes
inline std::ostream& operator<<(std::ostream& os, const NodeRef& n) {  // NOLINT(*)
  IRPrinter(os).Print(n);
  return os;
}
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::IterVar> {
  std::size_t operator()(const ::tvm::IterVar& k) const {
    return k.hash();
  }
};
}
#endif  // TVM_EXPR_H_
