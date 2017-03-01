/*!
 *  Copyright (c) 2017 by Contributors
 * \file modular.h
 * \brief Modular integer set analysis
 */
#ifndef TVM_ARITHMETIC_MODULAR_H_
#define TVM_ARITHMETIC_MODULAR_H_

#include <tvm/expr.h>
#include "./int_set.h"

namespace tvm {
namespace arith {

/*!
 * \brief Range of a linear integer function.
 *  Use to do specify the possible index values.
 *
 *  set = { base + coeff * x | x \in Z }
 *
 *  When coeff != 0, it can also be written as
 *  set = { n | n % coeff == base }
 *
 *  This is useful to decide if the index is dividable by certain value.
 *  For example, if index = 0 + 4 x, then we know it can be divided by 4.
 */
struct ModularEntry {
  /*! \brief The base */
  int base;
  /*! \brief linear co-efficient */
  int coeff;

  /*! \return entry represent everything */
  static ModularEntry everything() {
    // always safe to set 0 + x, so it can be everything.
    ModularEntry e;
    e.base = 0; e.coeff = 1;
    return e;
  }
};

/*!
 * \brief Evaluate the expression with modular analysis
 * \param e The expression to be evaluated.
 * \param mod_map Map of modular statistics of known variables.
 * \return The ModularEntry covering all possible value of e.
 */
ModularEntry EvalModular(
    const Expr& e,
    const std::unordered_map<const Variable*, ModularEntry>& mod_map);
/*!
 * \brief Same as EvalModular, used by front-end.
 * \param e The expression to be evaluated.
 * \param mod_map Map of modular statistics of known variables.
 * \return A ModularSet covering all possible value of e.
 */
IntSet EvalModular(const Expr& e,
                   const Map<Var, IntSet>& mod_map);
}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_MODULAR_H_
