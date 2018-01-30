/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Mapping op constructions
 * \file nn/mapping.h
 */
#ifndef TOPI_NN_MAPPING_H_
#define TOPI_NN_MAPPING_H_

#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Scale and shift with NCHW order
*
* \param x The input tensor.
* \param scale Scale tensor, 1-D of size channel
* \param shift Shift tensor, 1-D of size channel
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the scale shift operation
*/
inline Tensor scale_shift_nchw(const Tensor& x,
                               const Tensor& scale,
                               const Tensor& shift,
                               std::string name = "ScaleShift",
                               std::string tag = kBroadcast) {
  return tvm::compute(
    x->shape,
    [&](Var b, Var c, Var h, Var w) {
      return x(b, c, h, w) * scale(c) + shift(w);
    }, name, tag);
}

/*!
* \brief Scale and shift with NHWC order
*
* \param x The input tensor.
* \param scale Scale tensor, 1-D of size channel
* \param shift Shift tensor, 1-D of size channel
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the scale shift operation
*/
inline Tensor scale_shift_nhwc(const Tensor& x,
                               const Tensor& scale,
                               const Tensor& shift,
                               std::string name = "ScaleShift",
                               std::string tag = kBroadcast) {
  return tvm::compute(
    x->shape,
    [&](Var b, Var h, Var w, Var c) {
      return x(b, h, w, c) * scale(c) + shift(w);
    }, name, tag);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_MAPPING_H_
