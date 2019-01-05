/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Pooling op constructions
 * \file nn/pooling.h
 */
#ifndef TOPI_NN_POOLING_H_
#define TOPI_NN_POOLING_H_

#include <string>
#include <vector>

#include "tvm/tvm.h"
#include "tvm/ir_pass.h"
#include "topi/tags.h"
#include "topi/detail/constant_utils.h"
#include "topi/detail/pad_utils.h"
#include "topi/nn.h"
#include "topi/reduction.h"

namespace topi {
namespace nn {
using namespace tvm;

/*! \brief Pooling type */
enum PoolType : int {
  kAvgPool,
  kMaxPool,
};

/*!
* \brief Perform pooling on height and width dimension of data.
*
* \param x The input tensor
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
* \param height_axis index of the height dimension
* \param width_axis index of the width dimension
* \param count_include_pad Whether include padding in the calculation
*
* \return The output tensor in same layout order
*/
inline Tensor pool_impl(const Tensor& x,
                        const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size,
                        const Array<Expr>& padding_size,
                        PoolType pool_type,
                        bool ceil_mode,
                        const size_t height_axis,
                        const size_t width_axis,
                        bool count_include_pad) {
  CHECK(x->shape.size() >= 2) << "Pooling input must >= 2-D (H, W)";
  CHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 4) << "Pooling padding_size must have 4 elements";

  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto stride_height = stride_size[0];
  auto stride_width = stride_size[1];

  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto pad_top = padding_size[0];
  auto pad_left = padding_size[1];
  auto pad_bottom = padding_size[2];
  auto pad_right = padding_size[3];

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_bottom += stride_height - 1;
    pad_right += stride_width - 1;
  }

  Array<Expr> pad_before(std::vector<Expr>(x->shape.size(), 0));
  pad_before.Set(height_axis, pad_top);
  pad_before.Set(width_axis, pad_left);

  Array<Expr> pad_after(std::vector<Expr>(x->shape.size(), 0));
  pad_after.Set(height_axis, pad_bottom);
  pad_after.Set(width_axis, pad_right);

  auto out_height = tvm::ir::Simplify(
    (height - kernel_height + pad_top + pad_bottom) / stride_height + 1);
  auto out_width = tvm::ir::Simplify(
    (width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_axis, out_height);
  out_shape.Set(width_axis, out_width);

  const int64_t default_pad = 0;
  const int64_t *padding_h0 = as_const_int(pad_top, &default_pad);
  const int64_t *padding_w0 = as_const_int(pad_left, &default_pad);
  const int64_t *padding_h1 = as_const_int(pad_bottom, &default_pad);
  const int64_t *padding_w1 = as_const_int(pad_right, &default_pad);
  const bool do_pad = *padding_h0 + *padding_w0 + *padding_h1 + *padding_w1 > 0;

  if (pool_type == kMaxPool) {
    auto temp = do_pad ? pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp") : x;
    return tvm::compute(out_shape, [&](const Array<Var>& output) {
      Array<Expr> indices;
      for (const Var& var : output) indices.push_back(var);
      indices.Set(height_axis, output[height_axis] * stride_height + dheight);
      indices.Set(width_axis, output[width_axis] * stride_width + dwidth);
      return tvm::max(temp(indices), { dheight, dwidth });
    }, "tensor", "pool_max");
  } else if (pool_type == kAvgPool) {
    auto temp = do_pad ? pad(x, pad_before, pad_after, 0, "pad_temp") : x;
    auto tavg = [&](const Array<Var>& output, Expr divide_factor) {
      Array<Expr> indices;
      for (const Var& var : output) indices.push_back(var);
      indices.Set(height_axis, output[height_axis] * stride_height + dheight);
      indices.Set(width_axis, output[width_axis] * stride_width + dwidth);
      return tvm::sum(temp(indices) / divide_factor, { dheight, dwidth });
    };

    return tvm::compute(out_shape,
    [&](const Array<Var>& output) {
      if (count_include_pad) {
        return tavg(output, kernel_height * kernel_width);
      } else {
        Expr h_start = output[height_axis] * stride_height - pad_top;
        Expr w_start = output[width_axis] * stride_width - pad_left;
        Expr h_end = ir::Min::make(h_start + kernel_height, height);
        Expr w_end = ir::Min::make(w_start + kernel_width, width);
        h_start = ir::Max::make(h_start, make_const(Int(32), 0));
        w_start = ir::Max::make(w_start, make_const(Int(32), 0));
        Expr divide_factor = ir::Max::make((h_end - h_start) * (w_end - w_start),
                                           make_const(Int(32), 1));
        return tavg(output, divide_factor);
      }
    }, "tensor", "pool_avg");
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

inline Tensor pool_grad_impl(const Tensor& ograd,
                             const Tensor& x,
                             const Array<Expr>& kernel_size,
                             const Array<Expr>& stride_size,
                             const Array<Expr>& padding_size,
                             PoolType pool_type,
                             bool ceil_mode,
                             const size_t height_axis,
                             const size_t width_axis,
                             bool count_include_pad) {
  CHECK(ograd->shape.size() >= 2) << "Pooling grad output must >= 2-D (H, W)";
  CHECK(x->shape.size() >= 2) << "Pooling input must >= 2-D (H, W)";
  CHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 4) << "Pooling padding_size must have 4 elements";

  // std::vector<int> k = GetConstIntValues(kernel_size, )
  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto stride_height = stride_size[0];
  auto stride_width = stride_size[1];

  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto pad_top = padding_size[0];
  auto pad_left = padding_size[1];
  auto pad_bottom = padding_size[2];
  auto pad_right = padding_size[3];

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_bottom += stride_height - 1;
    pad_right += stride_width - 1;
  }

  Array<Expr> pad_before(std::vector<Expr>(x->shape.size(), 0));
  pad_before.Set(height_axis, pad_top);
  pad_before.Set(width_axis, pad_left);

  Array<Expr> pad_after(std::vector<Expr>(x->shape.size(), 0));
  pad_after.Set(height_axis, pad_bottom);
  pad_after.Set(width_axis, pad_right);

  auto out_height = tvm::ir::Simplify(
    (height - kernel_height + pad_top + pad_bottom) / stride_height + 1);
  auto out_width = tvm::ir::Simplify(
    (width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_axis, out_height);
  out_shape.Set(width_axis, out_width);

  const int64_t default_pad = 0;
  const int64_t *padding_h0 = as_const_int(pad_top, &default_pad);
  const int64_t *padding_w0 = as_const_int(pad_left, &default_pad);
  const int64_t *padding_h1 = as_const_int(pad_bottom, &default_pad);
  const int64_t *padding_w1 = as_const_int(pad_right, &default_pad);
  const bool do_pad = *padding_h0 + *padding_w0 + *padding_h1 + *padding_w1 > 0;

  if (pool_type == kMaxPool) {
    auto argmax = MakeArgmaxReducer();
    auto pad_x = do_pad ? pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp") : x;

    Array<Expr> ravel_shape;
    for (const auto& sh : x->shape) ravel_shape.push_back(sh);
    ravel_shape.Set(height_axis, ravel_shape[height_axis] + pad_top + pad_bottom);
    ravel_shape.Set(width_axis, ravel_shape[width_axis] + pad_left + pad_right);

    auto mp_argmax = tvm::compute(out_shape, [&](const Array<Var>& inds) {
      Array<Expr> window_inds;
      for (const Var& ind : inds) window_inds.push_back(ind);
      window_inds.Set(height_axis, inds[height_axis] * stride_height + dheight);
      window_inds.Set(width_axis, inds[width_axis] * stride_width + dwidth);
      auto idx = detail::RavelIndex(window_inds, ravel_shape);
      return argmax({ idx, pad_x(window_inds) }, { dheight, dwidth }, nullptr);
    }, "maxpool_grad_argmax", kCommReduceIdx);

    auto mp_inds = tvm::compute(out_shape, [&](const Array<Var>& inds) {
      return mp_argmax[0](inds);
    }, "maxpool_grad_inds", kCommReduceIdx);

    auto windowh = tvm::reduce_axis(
        Range(0, (kernel_height + stride_height - 1) / stride_height));
    auto windoww = tvm::reduce_axis(
        Range(0, (kernel_width + stride_width - 1) / stride_width));

    return tvm::compute(x->shape, [&](const Array<Var>& inds) {
      Array<Expr> pad_inds;
      for (const Var& ind : inds) pad_inds.push_back(ind);
      pad_inds.Set(height_axis, pad_inds[height_axis] + pad_top);
      pad_inds.Set(width_axis, pad_inds[width_axis] + pad_left);
      auto idx = detail::RavelIndex(pad_inds, ravel_shape);

      Array<Expr> out_idx;
      for (const Var& ind : inds) out_idx.push_back(ind);
      out_idx.Set(height_axis, (inds[height_axis] + pad_top) / stride_height - windowh);
      out_idx.Set(width_axis, (inds[width_axis] + pad_left) / stride_width - windoww);
      return tvm::sum(
          tvm::select(mp_inds(out_idx) == idx, ograd(out_idx), make_const(x->dtype, 0))
      , { windowh, windoww });
    }, "tensor", "pool_grad_max");
  } else if (pool_type == kAvgPool) {
    LOG(FATAL) << "implemented in python land";
    return x;
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

inline bool find_height_width(const std::string& layout,
                              int* height_axis,
                              int* width_axis) {
  *height_axis = -1, *width_axis = -1;
  int curr_idx = 0;
  for (size_t i = 0; i < layout.size(); ++i) {
    if ((layout[i] >= 'A' && layout[i] <= 'Z') ||
        (layout[i] >= 'a' && layout[i] <= 'z')) {
      if (layout[i] == 'H') {
        if (*height_axis != -1) return false;
        *height_axis = curr_idx;
      } else if (layout[i] == 'W') {
        if (*width_axis != -1) return false;
        *width_axis = curr_idx;
      } else if (layout[i] == 'h' || layout[i] == 'w') {
        // do not support split on height or width, e.g., NCHW16w
        return false;
      }
      ++curr_idx;
    }
  }
  if (*height_axis == -1 || *width_axis == -1) return false;
  return true;
}

/*!
* \brief Perform pooling on height and width dimension of data.
*        It decides the height and width dimension according to the layout string,
*        in which 'W' and 'H' means width and height respectively.
*        Width and height dimension cannot be split.
*        For example, NCHW, NCHW16c, etc. are valid for pool,
*        while NCHW16w, NCHW16h are not.
*        See \a layout for more information of the layout string convention.
* \param x The input tensor.
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
* \param layout The input layout. Pooling supports any layout as long as 'H' and 'W' appear.
*        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
*        where upper case indicates a dimension and
*        the corresponding lower case (with factor size) indicates the split dimension.
*        For example, NCHW16c can describe a 5-D tensor of
*        [batch_size, channel, height, width, channel_block].
*        (in which factor size `16` will not be used in pooling but for other operators,
*        it can be used to decide the output shape).
*        Since pooling does not care about the factor size of dimensions
*        other than `H` and `W`, one can pass `NCHWc` as well.
* \param  count_include_pad Whether include padding in the calculation when pool_type is 'avg'
*
*
* \return The output tensor in the same layout
*/
inline Tensor pool(const Tensor& x,
                   const Array<Expr>& kernel_size,
                   const Array<Expr>& stride_size,
                   const Array<Expr>& padding_size,
                   PoolType pool_type,
                   bool ceil_mode,
                   const std::string& layout = "NCHW",
                   bool count_include_pad = true) {
  int height_axis = -1, width_axis = -1;
  CHECK(find_height_width(layout, &height_axis, &width_axis))
    << "Unsupported layout " << layout;
  return pool_impl(x, kernel_size, stride_size, padding_size,
                   pool_type, ceil_mode, height_axis, width_axis,
                   count_include_pad);
}

inline Tensor pool_grad(const Tensor& ograd,
                        const Tensor& x,
                        const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size,
                        const Array<Expr>& padding_size,
                        PoolType pool_type,
                        bool ceil_mode,
                        const std::string& layout = "NCHW",
                        bool count_include_pad = true) {
  int height_axis = -1, width_axis = -1;
  CHECK(find_height_width(layout, &height_axis, &width_axis))
    << "Unsupported layout " << layout;
  return pool_grad_impl(ograd, x, kernel_size, stride_size, padding_size,
                        pool_type, ceil_mode, height_axis, width_axis,
                        count_include_pad);
}

/*!
* \brief Perform global pooling on height and width dimension of data.
*        It decides the height and width dimension according to the layout string,
*        in which 'W' and 'H' means width and height respectively.
*        Width and height dimension cannot be split.
*        For example, NCHW, NCHW16c, ... are valid for global_pool,
*        while NCHW16w, NCHW16h are not.
*        See \a layout for more information of the layout string convention.
*
* \param x The input tensor represent as layout
* \param pool_type The type of pooling operator
* \param layout The input layout. global-pooling supports any layout as long as 'H' and 'W' appear.
*        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
*        where upper case indicates a dimension and
*        the corresponding lower case (with factor size) indicates the sub-dimension.
*        For example, `NCHW16c` can describe a 5-D tensor of
*        [batch_size, channel, height, width, channel_block].
*        (in which factor size `16` will not be used in pooling but for other operators,
*        it can be used to decide the output shape).
*        Since pooling does not care about the factor size of
*        dimensions other than `H` and `W`, one can pass `NCHWc` as well.
*
* \return The output tensor in same layout with height and width dimension size of 1.
*         e.g., for NCHW, the output shape will be [batch, channel, 1, 1]
*/
inline Tensor global_pool(const Tensor& x,
                          PoolType pool_type,
                          const std::string& layout = "NCHW") {
  CHECK(x->shape.size() >= 2) << "Pooling input must >= 2-D (H, W)";

  int height_axis = -1, width_axis = -1;
  CHECK(find_height_width(layout, &height_axis, &width_axis))
    << "Unsupported layout " << layout;

  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_axis, 1);
  out_shape.Set(width_axis, 1);

  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto dheight = tvm::reduce_axis(Range(0, height));
  auto dwidth = tvm::reduce_axis(Range(0, width));

  if (pool_type == kMaxPool) {
    return tvm::compute(out_shape,
      [&](const Array<Var>& output) {
        Array<Expr> indices;
        for (const Var& var : output) indices.push_back(var);
        indices.Set(height_axis, dheight);
        indices.Set(width_axis, dwidth);
        return tvm::max(x(indices), { dheight, dwidth });  // NOLINT(*)
      }, "tensor", "global_pool_max");
  } else if (pool_type == kAvgPool) {
    auto tsum = tvm::compute(out_shape,
      [&](const Array<Var>& output) {
        Array<Expr> indices;
        for (const Var& var : output) indices.push_back(var);
        indices.Set(height_axis, dheight);
        indices.Set(width_axis, dwidth);
        return tvm::sum(x(indices), { dheight, dwidth });
      }, "tensor", "global_pool_sum");

    return tvm::compute(out_shape,
      [&](const Array<Var>& output) {
        return tsum(output) / tvm::cast(x->dtype, height * width);
      }, "tensor", kElementWise);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_POOLING_H_
