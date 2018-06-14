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
* \param input The input tensor.
* \param shape Output shape to upsample.
* \param layout input layout
* \param mode Angorithm to use (NEAREST_NEIGHBOR / BILINEAR)
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor upsampled to given shape
*/
inline Tensor upsampling(const Tensor& input,
                         const Array<Expr> shape,
                         std::string layout = "NCHW",
                         std::string mode = "NEAREST_NEIGHBOR",
                         std::string name = "tensor",
                         std::string tag = kInjective) {
  return resize(input, shape, layout, false, mode);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_UPSAMPLING_H_
