/*!
 *  Copyright (c) 2017 by Contributors
 * \file topi/transform.h
 * \brief Transform op constructors
 */
#ifndef TOPI_NN_SCALE_H_
#define TOPI_NN_SCALE_H_

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

#include "topi/tags.h"
#include "topi/detail/ravel_unravel.h"
#include "topi/detail/constant_utils.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Resize given tensor to given shape using nearest neighbour for NHWC
*
* \param inputs The input tensor array.
* \param shape Output shape to scale to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor scale_nn_nhwc(const Array<Tensor>& inputs,
                            const Array<Expr> shape,
                            bool align_corners = false,
                            std::string name = "tensor",
                            std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(inputs[0]->shape[0]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);
  out_shape.push_back(inputs[0]->shape[3]);

  Expr h_scale = shape[0] / inputs[0]->shape[1];
  Expr w_scale = shape[1] / inputs[0]->shape[2];

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    Array<Expr> idx;
    idx.push_back(indices[0]);
    idx.push_back(indices[1] / h_scale);
    idx.push_back(indices[2] / w_scale);
    idx.push_back(indices[3]);

    return inputs[0](idx);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using nearest neighbour for NCHW
*
* \param inputs The input tensor array.
* \param shape Output shape to scale to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor scale_nn_nchw(const Array<Tensor>& inputs,
                            const Array<Expr> shape,
                            bool align_corners = false,
                            std::string name = "tensor",
                            std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(inputs[0]->shape[0]);
  out_shape.push_back(inputs[0]->shape[1]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);

  Expr h_scale = shape[0] / inputs[0]->shape[2];
  Expr w_scale = shape[1] / inputs[0]->shape[3];

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    Array<Expr> idx;
    idx.push_back(indices[0]);
    idx.push_back(indices[1]);
    idx.push_back(indices[2] / h_scale);
    idx.push_back(indices[3] / w_scale);

    return inputs[0](idx);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using nearest neighbour
*
* \param inputs The input tensor array.
* \param shape Output shape to scale to.
* \param layout input layout
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor scale_nn(const Array<Tensor>& inputs,
                       const Array<Expr> shape,
                       std::string layout = "NCHW",
                       bool align_corners = false,
                       std::string name = "tensor",
                       std::string tag = kInjective) {
  CHECK_EQ(align_corners, false) << "Align corners not supported for nearest neighbour";

  if (layout == "NHWC") {
    return scale_nn_nhwc(inputs, shape, align_corners);
  } else {
    return scale_nn_nchw(inputs, shape, align_corners);
  }
}

/*!
* \brief Resize given tensor to given shape using bilinear interpolation for NHWC
*
* \param inputs The input tensor array.
* \param shape Output shape to scale to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor scale_bilinear_nhwc(const Array<Tensor>& inputs,
                                  const Array<Expr> shape,
                                  bool align_corners = false,
                                  std::string name = "tensor",
                                  std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(inputs[0]->shape[0]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);
  out_shape.push_back(inputs[0]->shape[3]);

  Array<Expr> split_ind;
  split_ind.push_back(make_const(UInt(32), 2));

  Array<Tensor> weights = split(inputs[1], split_ind, 2);

  Tensor coords = cast(weights[0], Int(32));

  Expr cone = make_const(UInt(32), 1);
  Expr other_y = tvm::ir::Simplify(inputs[0]->shape[1] - cone);
  Expr other_x = tvm::ir::Simplify(inputs[0]->shape[2] - cone);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    auto y0 = coords(indices[1], indices[2], 0);
    auto x0 = coords(indices[1], indices[2], 1);

    auto x1 = tvm::select(((x0 + cone) > other_x), other_x, (x0 + cone));
    auto y1 = tvm::select(((y0 + cone) > other_y), other_y, (y0 + cone));

    auto h = weights[1](indices[1], indices[2], 0);
    auto w = weights[1](indices[1], indices[2], 1);

    auto A = inputs[0](indices[0], y0, x0, indices[3]);
    auto B = inputs[0](indices[0], y0, x1, indices[3]);
    auto C = inputs[0](indices[0], y1, x0, indices[3]);
    auto D = inputs[0](indices[0], y1, x1, indices[3]);

    return  (A*(cone-w)*(cone-h) + B*(w)*(cone-h) + C*(h)*(cone-w) + D*w*h);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using bilinear interpolation for NCHW
*
* \param inputs The input tensor array.
* \param shape Output shape to scale to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor scale_bilinear_nchw(const Array<Tensor>& inputs,
                                  const Array<Expr> shape,
                                  bool align_corners = false,
                                  std::string name = "tensor",
                                  std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(inputs[0]->shape[0]);
  out_shape.push_back(inputs[0]->shape[1]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);

  Array<Expr> split_ind;
  split_ind.push_back(make_const(UInt(32), 2));

  Array<Tensor> weights = split(inputs[1], split_ind, 2);
  Tensor coords = cast(weights[0], Int(32));

  Expr cone = make_const(UInt(32), 1);
  Expr other_y = tvm::ir::Simplify(inputs[0]->shape[2] - cone);
  Expr other_x = tvm::ir::Simplify(inputs[0]->shape[3] - cone);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    auto y0 = coords(indices[2], indices[3], 0);
    auto x0 = coords(indices[2], indices[3], 1);

    auto x1 = tvm::select(((x0 + cone) > other_x), other_x, (x0 + cone));
    auto y1 = tvm::select(((y0 + cone) > other_y), other_y, (y0 + cone));

    auto h = weights[1](indices[2], indices[3], 0);
    auto w = weights[1](indices[2], indices[3], 1);

    auto A = inputs[0](indices[0], indices[1], y0, x0);
    auto B = inputs[0](indices[0], indices[1], y0, x1);
    auto C = inputs[0](indices[0], indices[1], y1, x0);
    auto D = inputs[0](indices[0], indices[1], y1, x1);

    return  (A*(1-w)*(1-h) + B*(w)*(1-h) + C*(h)*(1-w) + D*w*h);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using bilinear interpolation
*
* \param inputs The input tensor array.
* \param shape Output shape to scale to.
* \param layout input layout
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor scale_bilinear(const Array<Tensor>& inputs,
                             const Array<Expr> shape,
                             std::string layout = "NCHW",
                             bool align_corners = false,
                             std::string name = "tensor",
                             std::string tag = kInjective) {
  Tensor ret;

  if (layout == "NHWC") {
    ret = scale_bilinear_nhwc(inputs, shape, align_corners);
  } else {
    ret = scale_bilinear_nchw(inputs, shape, align_corners);
  }

  return cast(ret, inputs[0]->dtype);
}

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
inline Tensor scale(const Array<Tensor>& inputs,
                    const Array<Expr> shape,
                    std::string layout = "NCHW",
                    bool align_corners = false,
                    std::string mode = "BILINEAR",
                    std::string name = "tensor",
                    std::string tag = kInjective) {
  if (mode == "NN") {
    return scale_nn(inputs, shape, layout, align_corners);
  } else {
    return scale_bilinear(inputs, shape, layout, align_corners);
  }
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_SCALE_H_
