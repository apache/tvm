/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Softmax op constructions
 * \file nn/flatten.h
 */
#ifndef TOPI_NN_FLATTEN_H_
#define TOPI_NN_FLATTEN_H_

#include <string>
#include <vector>

#include "topi/tags.h"
#include "topi/detail/constant_utils.h"
#include "tvm/tvm.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Flattens the input tensor into a 2-D tensor by collapsing higher dimensions.
* This requires the input tensor to have constant sized dimensions.
*
* \param x The input tensor.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A 2-D tensor.
*/
inline Tensor flatten(const Tensor& x,
                      std::string name = "tensor",
                      std::string tag = kInjective) {
  auto ishape = x->shape;
  int dim = 1;
  for (size_t i = 1; i < ishape.size(); ++i) {
    dim = dim * static_cast<int>(topi::detail::GetConstInt(ishape[i]));
  }

  Array<Expr> oshape({ ishape[0], dim });

  std::vector<Expr> extra_shape;
  for (size_t i = 1; i < ishape.size(); ++i) {
    extra_shape.push_back(ishape[i]);
  }
  std::reverse(extra_shape.begin(), extra_shape.end());

  return tvm::compute(
    oshape, [&](Var i, Var j) {
      Expr idx = j;
      std::vector<Expr> index;
      for (auto s : extra_shape) {
        index.push_back(idx % s);
        idx = idx / s;
      }
      index.push_back(i);
      std::reverse(index.begin(), index.end());
      return x(index);
    }, name, tag);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_FLATTEN_H_
