/*!
 *  Copyright (c) 2018 by Contributors
 * \brief l2 normalization op constructions
 * \file nn/l2_normalize.h
 */
#ifndef TOPI_NN_L2_NORMALIZE_H_
#define TOPI_NN_L2_NORMALIZE_H_

#include <string>
#include <algorithm>
#include "topi/tags.h"
#include "tvm/tvm.h"
namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief L2 normalization inference operator
*
* \param data The input tensor. 4-D with shape [batch, channel, height, width]
* \param eps Epsilon to prevent div by 0
* \param axis Axes over the normalization applied
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the l2 normalization operation
*/
inline Tensor l2_normalize(const Tensor& data,
                           float eps,
                           const Array<Expr>& axis,
                           std::string name = "tensor",
                           std::string tag = "l2_normalize") {
  CHECK_EQ(data->shape.size(), 4) << "L2 normalization requires 4-D input";
  auto input_shape = data->shape;
  Tensor dot_value = topi::power(data, static_cast<float>(2.0));
  Tensor sum_value = topi::sum(dot_value, axis, true);
  Tensor expand_sum = topi::broadcast_to(sum_value, input_shape);
  return topi::divide(data,
                      topi::sqrt(tvm::compute(expand_sum->shape,
                                              [&](const Array<Var>& i){
                                                return (max(expand_sum(i), eps));
                                              }, name = name, tag = tag)));
}
}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_L2_NORMALIZE_H_
