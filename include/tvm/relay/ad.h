/*!
 *  Copyright (c) 2018 by Contributors
 * \file ad.h
 * \brief Common api shared by all the automatic differentiation(AD) algorithm.
 */
#ifndef TVM_RELAY_AD_H_
#define TVM_RELAY_AD_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/environment.h>

namespace tvm {
namespace relay {

/*! What is automatic differentiation(AD) and why is it important?
 * By AD, we roughly mean, given a term which denote some mathematical function,
 * derive a term which denote the derivative of that mathematical function.
 * Such a method can be compile-time, which is a macro on completely known function.
 * Formally speaking, such requirement mean that the input function is a closed expression -
 * that is, it only refer to local variable that is it's parameter, or defined inside it.
 * Every top level definition satisfy this criteria.
 * AD can also be run-time, which mean it is merely a function term of AD : (Float[] -> Float[]) -> (Float[] -> Float[]).
 * In relay we currently only support compile-time AD, but it should be enough for a lot of use case.
 *
 * In deep learning, the most common way to train a deep neural network is by gradient descend or some of it's variant.
 * Such optimization method require us to input the gradient of neural network, which can be obtained easily using AD.
 * In fact, back propagation is essentially reverse-mode automatic differentiation, a kind of AD!
 */

/*! In relay, automatic differentiation(AD) is a macro,
 *  that transform closed expr(expr without free variable/free type variable) of type
 *  (x0, x1, x2, ...) -> Float[] to
 *  (x0, x1, x2, ...) -> (Float[], (x0, x1,  x2, ...)),
 *  When x0, x1, x2... are Float.
 *  WithGradientType will take the type of input, and produce the type of output.
 *  There are multiple implementation of AD in relay, with different characteristic.
 *  However, they all transform the input expr according to WithGradientType.
 */
Type WithGradientType(const Type&);

// return an expression that represent differentiation of e (according to WithGradientType)
Expr FOWithGradient(const Environment& env, const Expr& e);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_AD_H_
