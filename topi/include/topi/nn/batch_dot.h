/*!
 *  Copyright (c) 2019 by Contributors
 * \brief Batch dot op constructions
 * \file nn/batch_dot.h
 */
#ifndef TOPI_NN_BATCH_DOT_H_
#define TOPI_NN_BATCH_DOT_H_

#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Creates an operation that calculates data * weight^T + bias
*
* \param data Tensor with shape [batch, in_dim]
* \param weight Tensor with shape [out_dim, in_dim]
* \param bias Tensor with shape [out_dim]. Optional; to omit bias, pass Tensor()
*
* \return Tensor with shape [batch, out_dim]
*/
inline tvm::Tensor batch_dot(const tvm::Tensor& x,
                             const tvm::Tensor& y) {
  CHECK_EQ(x->shape.size(), 3) << "batch_dot requires 3-D data";
  CHECK_EQ(y->shape.size(), 3) << "batch_dot requires 3-D data";

  auto batch = x->shape[0];
  auto M = x->shape[1];
  auto K = x->shape[2];
  auto N = y->shape[1];

  auto k = tvm::reduce_axis(Range(0, K), "k");
  auto result = tvm::compute(
      { batch, M, N },
      [&](Var b, Var i, Var j) {
        return tvm::sum(x(b, i, k) * y(b, j, k), { k });
      }, "tensor", "batch_dot");

  return result;
}

}  // namespace nn
}  // namespace topi

#endif  // TOPI_NN_BATCH_DOT_H_
