/*!
 *  Copyright (c) 2017 by Contributors
 * \brief bias_add op constructions
 * \file nn/bias_add.h
 */
#ifndef TOPI_NN_BIAS_ADD_H
#define TOPI_NN_BIAS_ADD_H

#include <string>

#include "topi/tags.h"
#include "topi/broadcast.h"
#include "topi/transform.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Creates an operation that calculates data + bias
*
* \param data Tensor with shape [batch, in_dim]
* \param bias Tensor with shape [batch].
*
* \return Tensor with shape [batch, in_dim]
*/
inline tvm::Tensor bias_add(const tvm::Tensor& data, const tvm::Tensor& bias, int axis) {
  CHECK_EQ(data->shape.size(), 2) << "dense requires 2-D data";
  CHECK_EQ(bias->shape.size(), 1) << "dense requires 1-D bias";
  int data_ndim = data->shape.size();
  if (axis < 0) {
    axis += data_ndim;
  }
  int num_newaxis = data_ndim - axis - 1;
  return add(data, (num_newaxis ? expand_dims(bias, 1, num_newaxis) : bias));
}
}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_BIAS_ADD_H
