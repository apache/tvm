/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Softmax op constructions
 * \file nn/softmax.h
 */
#ifndef TOPI_NN_SOFTMAX_H_
#define TOPI_NN_SOFTMAX_H_

#include <algorithm>
#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Softmax activation
*
* \param x The input tensor. 2-D where softmax is performed along the second dimension
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the softmax operation
*/
inline Tensor softmax(const Tensor& x,
                      std::string name = "tensor",
                      std::string tag = "softmax_output") {
  CHECK_EQ(x->shape.size(), 2) << "Softmax requires 2-D input";

  Expr m = x->shape[0];
  Expr n = x->shape[1];

  auto k = tvm::reduce_axis(Range(0, n), "k");
  auto max_elem = tvm::compute(
    { m }, [&](Var i) {
      return tvm::max(x(i, k), Array<IterVar>{ k }); });
  k = tvm::reduce_axis(Range(0, n), "k");

  auto expsum = tvm::compute(
    { m }, [&](Var i) {
      return tvm::sum(tvm::exp(x(i, k) - max_elem(i)), { k }); });

  return tvm::compute(
    x->shape, [&](Var i, Var j) {
      return tvm::exp(x(i, j) - max_elem(i)) / expsum(i);
    });
}

/*!
* \brief Log softmax activation
*
* \param x The input tensor. 2-D where log softmax is performed along the second dimension
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the log softmax operation
*/
inline Tensor log_softmax(const Tensor& x,
                          std::string name = "tensor",
                          std::string tag = "log_softmax_output") {
  CHECK_EQ(x->shape.size(), 2) << "Log softmax requires 2-D input";

  Expr m = x->shape[0];
  Expr n = x->shape[1];

  auto k = tvm::reduce_axis(Range(0, n), "k");
  auto max_elem = tvm::compute(
    { m }, [&](Var i) {
      return tvm::max(x(i, k), Array<IterVar>{ k }); });
  k = tvm::reduce_axis(Range(0, n), "k");

  auto expsum = tvm::compute(
    { m }, [&](Var i) {
      return tvm::sum(tvm::exp(x(i, k) - max_elem(i)), { k }); });

  return tvm::compute(
    x->shape, [&](Var i, Var j) {
      return x(i, j) - max_elem(i) - tvm::log(expsum(i));
    });
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_SOFTMAX_H_
