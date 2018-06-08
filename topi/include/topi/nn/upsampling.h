/*!
 *  Copyright (c) 2017 by Contributors
 * \file topi/nn/upsampling.h
 * \brief upsampling op constructors
 */
#ifndef TOPI_NN_UPSAMPLING_H_
#define TOPI_NN_UPSAMPLING_H_

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

#include "topi/image/resize.h"

namespace topi {
namespace nn {
using namespace tvm;
using namespace topi::image;

/*!
* \brief Upsample given tensor to given shape
*
* \param inputs The input tensor array.
* Bilinear will have 2 inputs one being the weights.
* \param shape Output shape to upsample.
* \param layout input layout
* \param align_corners To preserve centers of 4 corner pixels
* \param mode Angorithm to use (NN / BILINEAR)
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor upsampled to given shape
*/
inline Tensor upsampling(const Array<Tensor>& inputs,
                         const Array<Expr> shape,
                         std::string layout = "NCHW",
                         bool align_corners = false,
                         std::string mode = "NN",
                         std::string name = "tensor",
                         std::string tag = kInjective) {
  return resize(inputs, shape, layout, align_corners, mode);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_UPSAMPLING_H_
