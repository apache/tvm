/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr.h
 * \brief The Expr and related elements in DataFlow construction.
 */
#ifndef TVM_EXPR_H_
#define TVM_EXPR_H_

#include <ir/Expr.h>
#include <ir/IRPrinter.h>
#include <ir/IROperator.h>
#include <string>
#include <algorithm>
#include "./base.h"

namespace tvm {

using Halide::Type;
using Halide::Float;
using Halide::Int;
using Halide::UInt;
using Halide::Handle;

using Halide::Expr;
using Halide::VarExpr;
using Halide::IR::FunctionRef;
using Halide::IR::FunctionBaseNode;
using Halide::Internal::Stmt;
using Halide::Internal::IRPrinter;
using Halide::Internal::Variable;

/*! \brief a named variable in TVM */
class Var : public Halide::VarExpr {
 public:
  explicit Var(const std::string& name_hint = "v",
               Type t = Int(32)) : VarExpr(name_hint, t) {}

  explicit Var(std::shared_ptr<Node> n) : VarExpr(n) {}

  /*! \brief type indicate the container type */
  using ContainerType = Variable;
};


/*! \brief container class of iteration variable. */
class IterVarNode;

/*!
 * \brief same as Halide::IR::Range
 *  except it provide an constructor with (begin, end)
 *
 *  \note Traditional Halide's Range have a constructor with
 *   (begin, extent), which does not match the convention in e.g. python.
 *   We decided to correct it by removing the constructor in HalideIR,
 *   and add it back in TVM's range.
 */
class Range : public Halide::IR::Range {
 public:
  /*! \brief constructor */
  Range() {}
  explicit Range(std::shared_ptr<Node> n) : Halide::IR::Range(n) {}
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   */
  Range(Expr begin, Expr end);

  static Range make_with_min_extent(Expr min, Expr extent);
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
  explicit IterVar(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief construction of iteration variable.
   * \param dom The iteration domain.
   * \param var_name The name of iteration variable.
   * \param thread_tag The additional tag to indicate whether the var is binded to fixed-thread.
   */
  explicit IterVar(Range dom, std::string var_name = "i", std::string thread_tag = "");
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

using Domain = Array<Range>;

// functions
using Halide::cast;
using Halide::min;
using Halide::max;
using Halide::abs;
using Halide::select;

/*!
 * \brief sum of of source expression over rdom
 * \param source The source expression.
 * \param rdom List of iteration variables that will be used for reduction.
 */
Expr sum(Expr source, Array<IterVar> rdom);

/*!
 * \brief max of of source expression over rdom
 * \param source The source expression.
 * \param rdom List of iteration variables that will be used for reduction.
 */
Expr max(Expr source, Array<IterVar> rdom);

/*!
 * \brief max of of source expression over rdom
 * \param source The source expression.
 * \param rdom List of iteration variables that will be used for reduction.
 */
Expr min(Expr source, Array<IterVar> rdom);


// print functions for expr
std::ostream& operator<<(std::ostream& os, const NodeRef& n);  // NOLINT(*)

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
  /*!
   * \brief additional tag on the iteration variable,
   *  set this if this is binded already to a known thread tag.
   */
  std::string thread_tag;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dom", &dom);
    v->Visit("var", &var);
    v->Visit("thread_tag", &thread_tag);
  }

  static IterVar make(Range dom, Var var, std::string thread_tag);

  static constexpr const char* _type_key = "IterVar";
  TVM_DECLARE_NODE_TYPE_INFO(IterVarNode);
};

// inline implementations
inline const IterVarNode* IterVar::operator->() const {
  return static_cast<const IterVarNode*>(node_.get());
}

inline IterVar::operator Expr() const {
  return (*this)->var;
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
