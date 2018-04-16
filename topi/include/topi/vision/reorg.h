/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Reorg op constructions
 * \file vision/reorg.h
 */
#ifndef TOPI_VISION_REORG_H_
#define TOPI_VISION_REORG_H_

#include <algorithm>
#include <string>

#include "topi/detail/constant_utils.h"
#include "topi/reduction.h"
#include "topi/tags.h"
#include "topi/transform.h"
#include "tvm/tvm.h"

namespace topi {
namespace vision {
using namespace tvm;

/*!
* \brief Reorg operation
*
* \param data The input tensor. Can be any dimension
* \param stride The input integer used as stride in reorg operation
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the reorg operation
*/
inline Tensor reorg(const Tensor &data,
                    int stride = 1,
                    std::string name = "tensor",
                    std::string tag = "reorg_output") {
  auto input_shape = data->shape;

  int batch = GetConstInt(input_shape[0]);
  int c_in = GetConstInt(input_shape[1]);
  int h_in = GetConstInt(input_shape[2]);
  int w_in = GetConstInt(input_shape[3]);
  int out_c = c_in / (stride * stride);

  auto out = tvm::compute(input_shape,
                          [&](Var b, Var k, Var j, Var i) {
                          return data(b * stride * stride,
                                      (k % out_c) * stride * stride,
                                      (j*stride + (k / out_c) / stride) * stride,
                                      (i*stride + (k / out_c) % stride));
                          },
                          name,
                          tag);

  out_c = c_in * stride * stride;
  int out_h = h_in / stride;
  int out_w = w_in / stride;

  Array<Expr> out_shape = {batch, out_c, out_h, out_w};
  return reshape(out, out_shape);
}
}  // namespace vision
}  // namespace topi
#endif  // TOPI_VISION_REORG_H_
