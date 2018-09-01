/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/alpha_eq.h
 * \brief Check expressions and types for structural equivalence.
 */
#ifndef TVM_RELAY_ALPHA_EQ_H_
#define TVM_RELAY_ALPHA_EQ_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

/*! \brief Compare two expressions for structural equivalence.

    This comparsion operator respects scoping and compares
    expressions without regard to variable choice.

    For example: `let x = 1 in x` is equal to `let y = 1 in y`.

    See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
    for more details.

    \param e1 The left hand expression.
    \param e2 The right hand expression.

    \return true if equal, otherwise false

*/
bool AlphaEqual(const Expr& e1, const Expr& e2);

/*! \brief Compare two types for structural equivalence.

    This comparsion operator respects scoping and compares
    expressions without regard to variable choice.

    For example: `forall s, Tensor[f32, s]` is equal to
    `forall w, Tensor[f32, w]`.

    See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
    for more details.

    \param t1 The left hand type.
    \param t2 The right hand type.

    \return true if equal, otherwise false

*/
bool AlphaEqual(const Type& t1, const Type& t2);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ALPHA_EQ_H_
