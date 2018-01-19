/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Batch normalization op constructions
 * \file nn/batch_norm.h
 */
#ifndef TOPI_NN_BATCH_NORM_H_
#define TOPI_NN_BATCH_NORM_H_

#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Batch normalization inference operator with NCHW layout
*
* \param x The input tensor. 4-D with shape [batch, channel, height, width]
* \param gamma 1-D with shape [channel]
* \param beta 1-D with shape [channel]
* \param moving_mean 1-D with shape [channel]
* \param moving_var 1-D with shape [channel]
* \param eps Epsilon to prevent div by 0
* \param fix_gamma Fix gamma while training
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the batch normalization operation
*/
inline Tensor batch_norm_inference(const Tensor& x,
                                   const Tensor& gamma,
                                   const Tensor& beta,
                                   const Tensor& moving_mean,
                                   const Tensor& moving_var,
                                   float eps,
                                   bool fix_gamma,
                                   std::string name = "tensor",
                                   std::string tag = kBroadcast) {
  CHECK_EQ(x->shape.size(), 4) << "Batch norm requires 4-D input";

  Tensor out;
  if (fix_gamma) {
    out = tvm::compute(
      x->shape,
      [&](const Array<Var>& indices) {
        auto c = Array<Var>({ indices[1] });
        return (x(indices) - moving_mean(c)) / tvm::sqrt(moving_var(c) + eps) + beta(c);
      }, name, tag);
  } else {
    out = tvm::compute(
      x->shape,
      [&](const Array<Var>& indices) {
        auto c = Array<Var>({ indices[1] });
        return (x(indices) - moving_mean(c)) / tvm::sqrt(moving_var(c) + eps) * gamma(c) + beta(c);
      }, name, tag);
  }
  return out;
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_BATCH_NORM_H_
