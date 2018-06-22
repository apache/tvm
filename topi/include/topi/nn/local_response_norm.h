/*!
 *  Copyright (c) 2018 by Contributors
 * \brief local response normalization op constructions
 * \file nn/local_response_norm.h
 */
#ifndef TOPI_NN_LOCAL_RESPONSE_NORM_H_
#define TOPI_NN_LOCAL_RESPONSE_NORM_H_

#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Local response normalization inference operator
*
* \param data The input tensor. 4-D shape NCHW or NHWC
* \param size Integer to define normalisation window size
* \param axis Input data layout channel axis
* \param alpha Float scaling factor
* \param beta Exponent value
* \param bias Offset to avoid dividing by zero
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the Local response normalization operation
*/
inline Tensor lrn(const Tensor& data,
                  int size,
                  int axis = 1,
                  float alpha = 0.0001,
                  float beta = 0.75,
                  float bias = 2,
                  std::string name = "tensor",
                  std::string tag = kBroadcast) {
  CHECK_EQ(data->shape.size(), 4) << "LRN requires 4-D input";
  CHECK_EQ(size % 2, 1) << "size should be odd number";
  CHECK(axis == 1 || axis == 3) << "axis should be 1 or 3 for NCHW and NHWC";
  auto input_shape = data->shape;
  Array<Expr> pad_before{ 0, 0, 0, 0};
  Array<Expr> pad_after{ 0, 0, 0, 0};
  pad_before.Set(axis, static_cast<Expr>(size/2));
  pad_after.Set(axis, static_cast<Expr>(size/2));
  auto pad_data = pad(data, pad_before, pad_after, 0, "pad_data");
  auto rxs = tvm::reduce_axis(Range(0, size), "rxs");
  Tensor sqr_sum;
  if (axis == 1) {
    sqr_sum = tvm::compute(input_shape,
                           [&](Var i, Var l, Var j, Var k) {
                           return tvm::sum(pad_data(i, l + rxs, j, k) *
                                           pad_data(i, l + rxs, j, k),
                                           {rxs});
                           });
  } else if (axis == 3) {
    sqr_sum = tvm::compute(input_shape,
                           [&](Var i, Var l, Var j, Var k) {
                           return tvm::sum(pad_data(i, l, j, k + rxs) *
                                           pad_data(i, l, j, k + rxs),
                                           {rxs});
                           });
  }
  auto sqrt_sum_up = tvm::compute(
      input_shape,
      [&](Var i, Var j, Var k, Var l) {
        return tvm::pow(bias +
                        (alpha * sqr_sum(i, j, k, l) / size),
                        beta);
      });
  return topi::divide(data, sqrt_sum_up);
}
}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_LOCAL_RESPONSE_NORM_H_
