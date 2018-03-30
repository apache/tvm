/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Pooling op constructions
 * \file nn/pooling.h
 */
#ifndef TOPI_NN_POOLING_H_
#define TOPI_NN_POOLING_H_

#include <string>

#include "tvm/tvm.h"
#include "tvm/ir_pass.h"
#include "topi/tags.h"
#include "topi/detail/pad_utils.h"
#include "topi/nn.h"

namespace topi {
namespace nn {
using namespace tvm;

/*! \brief Pooling type */
enum PoolType : int {
  kAvgPool,
  kMaxPool,
};

/*!
* \brief Perform pooling on data in NCHW order
*
* \param x The input tensor in NCHW order
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
*
* \return The output tensor in NCHW order
*/

inline Tensor pool_nchw(const Tensor& x,
                        const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size,
                        const Array<Expr>& padding_size,
                        PoolType pool_type,
                        bool ceil_mode) {
  CHECK_EQ(x->shape.size(), 4) << "Pooling input must be 4-D";
  CHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 2) << "Pooling padding_size must have 2 elements";

  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto stride_height = stride_size[0];
  auto stride_width = stride_size[1];
  auto padding_height = padding_size[0];
  auto padding_width = padding_size[1];

  auto batch = x->shape[0];
  auto channel = x->shape[1];
  auto height = x->shape[2];
  auto width = x->shape[3];

  auto pad_tuple = detail::GetPadTuple(padding_height, padding_width);
  auto pad_top = pad_tuple[0];
  auto pad_left = pad_tuple[1];
  auto pad_down = pad_tuple[2];
  auto pad_right = pad_tuple[3];

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_down += stride_height - 1;
    pad_right += stride_width - 1;
  }

  Array<Expr> pad_before{ 0, 0, pad_top, pad_left };
  Array<Expr> pad_after{ 0, 0, pad_down, pad_right };

  auto out_height = tvm::ir::Simplify(
    (height - kernel_height + pad_top + pad_down) / stride_height + 1);
  auto out_width = tvm::ir::Simplify(
    (width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  if (pool_type == kMaxPool) {
    auto temp = pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp");
    return tvm::compute(
      { batch, channel, out_height, out_width },
      [&](Var n, Var c, Var h, Var w) {
        return tvm::max(temp(n, c, h * stride_height + dheight, w * stride_width + dwidth),
        { dheight, dwidth });
      }, "tensor", "pool_max");
  } else if (pool_type == kAvgPool) {
    auto temp = pad(x, pad_before, pad_after, 0, "pad_temp");

    auto tsum = tvm::compute(
      { batch, channel, out_height, out_width },
      [&](Var n, Var c, Var h, Var w) {
        return tvm::sum(temp(n, c, h * stride_height + dheight, w * stride_width + dwidth),
        { dheight, dwidth });
      }, "tensor", "pool_avg");

    return tvm::compute(
      { batch, channel, out_height, out_width },
      [&](Var n, Var c, Var h, Var w) {
        return tsum(n, c, h, w) / (kernel_height * kernel_width);
      }, "tensor", kElementWise);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

/*!
* \brief Perform pooling on data in NHWC order
*
* \param x The input tensor in NHWC order
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
*
* \return The output tensor in NCHW order
*/

inline Tensor pool_nhwc(const Tensor& x,
                        const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size,
                        const Array<Expr>& padding_size,
                        PoolType pool_type,
                        bool ceil_mode) {
  CHECK_EQ(x->shape.size(), 4) << "Pooling input must be 4-D";
  CHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 2) << "Pooling padding_size must have 2 elements";

  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto stride_height = stride_size[0];
  auto stride_width = stride_size[1];
  auto padding_height = padding_size[0];
  auto padding_width = padding_size[1];

  auto batch = x->shape[0];
  auto height = x->shape[1];
  auto width = x->shape[2];
  auto channel = x->shape[3];

  auto pad_tuple = detail::GetPadTuple(padding_height, padding_width);
  auto pad_top = pad_tuple[0];
  auto pad_left = pad_tuple[1];
  auto pad_down = pad_tuple[2];
  auto pad_right = pad_tuple[3];

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_down += stride_height - 1;
    pad_right += stride_width - 1;
  }

  Array<Expr> pad_before{ 0, pad_top, pad_left, 0};
  Array<Expr> pad_after{ 0, pad_down, pad_right, 0};

  auto out_height = tvm::ir::Simplify(
    (height - kernel_height + pad_top + pad_down) / stride_height + 1);
  auto out_width = tvm::ir::Simplify(
    (width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  if (pool_type == kMaxPool) {
    auto temp = pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp");
    return tvm::compute(
     { batch, out_height, out_width, channel },
      [&](Var n, Var h, Var w, Var c) {
        return tvm::max(temp(n, h * stride_height + dheight, w * stride_width + dwidth, c),
        { dheight, dwidth });
      }, "tensor", "pool_max");
  } else if (pool_type == kAvgPool) {
    auto temp = pad(x, pad_before, pad_after, 0, "pad_temp");

    auto tsum = tvm::compute(
     { batch, out_height, out_width, channel },
      [&](Var n, Var h, Var w, Var c) {
        return tvm::sum(temp(n, h * stride_height + dheight, w * stride_width + dwidth, c),
        { dheight, dwidth });
      }, "tensor", "pool_avg");

    return tvm::compute(
     { batch, out_height, out_width, channel },
     [&](Var n, Var h, Var w, Var c) {
       return tsum(n, h, w, c) / (kernel_height * kernel_width);
      }, "tensor", kElementWise);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

/*!
* \brief Perform pooling on data
*
* \param x The input tensor in NCHW or NHWC order
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
* \param layout The input layout
*
* \return The output tensor in NCHW order
*/

inline Tensor pool(const Tensor& x,
                   const Array<Expr>& kernel_size,
                   const Array<Expr>& stride_size,
                   const Array<Expr>& padding_size,
                   PoolType pool_type,
                   bool ceil_mode,
                   const std::string& layout = "NCHW") {
  CHECK(layout == "NCHW" || layout == "NHWC") << "Unsupported layout.";
  if (layout == "NCHW")
    return pool_nchw(x, kernel_size, stride_size, padding_size, pool_type, ceil_mode);
  else
    return pool_nhwc(x, kernel_size, stride_size, padding_size, pool_type, ceil_mode);
}

/*!
* \brief Perform global pooling on data in NCHW order
*
* \param x The input tensor in NCHW order
* \param pool_type The type of pooling operator
*
* \return The output tensor with shape [batch, channel, 1, 1]
*/
inline Tensor global_pool(const Tensor& x,
                          PoolType pool_type) {
  CHECK_EQ(x->shape.size(), 4) << "Pooling input must be 4-D";

  auto batch = x->shape[0];
  auto channel = x->shape[1];
  auto height = x->shape[2];
  auto width = x->shape[3];

  auto dheight = tvm::reduce_axis(Range(0, height));
  auto dwidth = tvm::reduce_axis(Range(0, width));

  if (pool_type == kMaxPool) {
    return tvm::compute(
      { batch, channel, 1, 1 },
      [&](Var n, Var c, Var h, Var w) {
        return tvm::max(x(n, c, dheight, dwidth), { dheight, dwidth });  // NOLINT(*)
      }, "tensor", "global_pool_max");
  } else if (pool_type == kAvgPool) {
    auto tsum = tvm::compute(
      { batch, channel, 1, 1 },
      [&](Var n, Var c, Var h, Var w) {
        return tvm::sum(x(n, c, dheight, dwidth), { dheight, dwidth });
      }, "tensor", "global_pool_sum");

    return tvm::compute(
      { batch, channel, 1, 1 },
      [&](Var n, Var c, Var h, Var w) {
        return tsum(n, c, h, w) / tvm::cast(x->dtype, height * width);
      }, "tensor", kElementWise);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_POOLING_H_
