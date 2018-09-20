/*!
 *  Copyright (c) 2017 by Contributors
 * \file int_set_internal.h
 * \brief Implementations of integer set
 */
#ifndef TVM_ARITHMETIC_INT_SET_INTERNAL_H_
#define TVM_ARITHMETIC_INT_SET_INTERNAL_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>

namespace tvm {
namespace arith {

using HalideIR::Internal::Interval;

/*! \brief Set of continuous interval */
struct IntervalSet : public IntSetNode {
  /*! \brief the internal interval*/
  Interval i;

  static IntSet make(Interval i) {
    NodePtr<IntervalSet> n =
        make_node<IntervalSet>();
    n->i = i;
    return IntSet(n);
  }
  static IntSet make(Expr min, Expr max) {
    NodePtr<IntervalSet> n =
        make_node<IntervalSet>();
    n->i.min = min;
    n->i.max = max;
    return IntSet(n);
  }

  static constexpr const char* _type_key = "IntervalSet";
  TVM_DECLARE_NODE_TYPE_INFO(IntervalSet, IntSetNode);
};

/*!
 * \brief set represented by strided integers
 *  Reserved for cases where strided access is supported.
 */
struct StrideSet : public IntSetNode {
  /*! \brief the base inetrval */
  Interval base;
  /*! \brief additional extents in positive number */
  Array<Expr> extents;
  /*! \brief additional strides in positive number */
  Array<Expr> strides;

  static constexpr const char* _type_key = "StrideSet";
  TVM_DECLARE_NODE_TYPE_INFO(StrideSet, IntSetNode);
};

/*!
 * \brief Set represented by range of ModularEntry.
 *  Used for front-end modular analysis.
 */
struct ModularSet : public IntSetNode {
  /*! \brief Internal modular entry */
  ModularEntry e;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("base", &(e.base));
    v->Visit("coeff", &(e.coeff));
  }
  static constexpr const char* _type_key = "ModularSet";
  TVM_DECLARE_NODE_TYPE_INFO(ModularSet, IntSetNode);
};


}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITHMETIC_INT_SET_INTERNAL_H_
