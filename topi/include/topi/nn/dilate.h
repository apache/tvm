/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Dilate op constructions
 * \file nn/dilate.h
 */
#ifndef TOPI_NN_DILATE_H_
#define TOPI_NN_DILATE_H_

#include <string>

#include "tvm/tvm.h"
#include "tvm/ir_pass.h"
#include "topi/tags.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Create a new experssion of the intersection of all
* conditions in the arguments.
*
* \return The intersection expression
*/
Expr all(Array<Expr> args) {
  CHECK_GT(args.size(), 0) << "all requires at least one argument";

  Expr ret = args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    ret = ret && args[i];
  }
  return ret;
}

/*!
* \brief Dilate data with zeros
*
* \param x The input tensor, this can have any number of
* dimensions and any layout.
* \param strides Dilation stride for each dimension. Stride 1
* means no dilation.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return The output tensor.
*/
inline Tensor dilate(const Tensor& x,
                     Array<Expr> strides,
                     std::string name = "tensor",
                     std::string tag = kInjective) {
  auto n = x->shape.size();
  CHECK_EQ(n, strides.size())
    << "strides size (" << strides.size()
    << ") must match dimension of x (" << n << ")";

  Array<Expr> out_shape;
  for (size_t i = 0; i < n; ++i) {
    out_shape.push_back(tvm::ir::Simplify(
      (x->shape[i] - 1) * strides[i] + 1));
  }

  return tvm::compute(
    out_shape,
    [&](const Array<Var>& indices) {
      Array<Expr> not_zero;
      Array<Expr> index_tuple;
      for (size_t i = 0; i < n; ++i) {
        if (IsConstInt(strides[i]) && GetConstInt(strides[i]) == 1) {
          index_tuple.push_back(indices[i]);
        } else {
          index_tuple.push_back(indices[i] / strides[i]);
          not_zero.push_back((indices[i] % strides[i]) == 0);
        }
      }
      if (not_zero.size() > 0) {
        auto all_not_zero = all(not_zero);
        return tvm::select(all_not_zero, x(index_tuple), make_const(x->dtype, 0));
      }
      return x(index_tuple);
    }, name, tag);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_DILATE_H_
