/*!
 *  Copyright (c) 2017 by Contributors
 * \file canonical.h
 * \brief Internal canonicalized expression simplification engine.
 */
#ifndef TVM_ARITHMETIC_CANONICAL_H_
#define TVM_ARITHMETIC_CANONICAL_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>

namespace tvm {
namespace arith {

/*!
 * \brief A stateful CanonicalEngine over SSA.
 *
 *  Simplify and CSE with canonicalization expressions.
 *  Each call's result will get cached, so next call will
 *  simply return the cached result.
 */
class Canonical {
 public:
  /*! \brief constructor */
  explicit Canonical(Map<Var, Range> var_range);
  /*!
   * \brief simplify expression e.
   * \param expr The expression to be simplified.
   */
  Expr Simplify(Expr expr);
  /*!
   * \brief simplify stmt.
   * \param stmt The stmt to be simplified.
   */
  Stmt Simplify(Stmt expr);
  /*!
   * \brief Set range and level variable
   * \param v The variable
   * \param r The range of the variable, can be undefined.
   * \param level The scope level of the variable,
   *  affect the order of formula in communicative ops.
   */
  void SetRange(Var v, Range r, int level);

  class Internal;
 private:
  // Internal pointer
  std::shared_ptr<Internal> ptr_;
};


}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITHMETIC_CANONICAL_H_
