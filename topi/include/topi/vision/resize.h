/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Resize op constructions
 * \file topi/vision/resize.h
 */
#ifndef TOPI_VISION_RESIZE_H_
#define TOPI_VISION_RESIZE_H_

#include <algorithm>
#include <string>

#include "topi/detail/constant_utils.h"
#include "topi/reduction.h"
#include "topi/tags.h"
#include "topi/scale.h"
#include "tvm/tvm.h"

namespace topi {
namespace vision {
using namespace tvm;

/*!
* \brief Resize given tensor to given shape
*
* \param inputs The input tensor array.
* Bilinear will have 2 inputs one being the weights.
* \param shape Output shape to scale to.
* \param layout input layout
* \param align_corners To preserve centers of 4 corner pixels
* \param mode Angorithm to use (NN / BILINEAR)
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize(const Array<Tensor>& inputs,
                     const Array<Expr> shape,
                     std::string layout = "NCHW",
                     bool align_corners = false,
                     std::string mode = "BILINEAR",
                     std::string name = "tensor",
                     std::string tag = kInjective) {
  return scale(inputs, shape, layout, align_corners, mode);
}

}  // namespace vision
}  // namespace topi
#endif  // TOPI_VISION_RESIZE_H_
