/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Softmax op constructions
 * \file nn/softmax.h
 */
#ifndef TOPI_NN_SOFTMAX_H_
#define TOPI_NN_SOFTMAX_H_

#include <algorithm>
#include <string>

#include "topi/reduction.h"
#include "topi/tags.h"
#include "tvm/operation.h"
#include "tvm/expr_operator.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Softmax activation
*
* \param x The input tensor. Can be any dimension
* \param axis The channel axis along which softmax is performed
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the softmax operation
*/
inline Tensor softmax(const Tensor &x,
                      int axis = -1,
                      std::string name = "tensor",
                      std::string tag = "softmax_output") {
  auto input_shape = x->shape;
  auto ndim = input_shape.size();
  if (axis < 0) {
    axis = ndim + axis;
  }
  CHECK_LT(axis, ndim) << "axis parameter should be less than input dim";

  auto k1 = tvm::reduce_axis(Range(0, input_shape[axis]), "k1");
  auto k2 = tvm::reduce_axis(Range(0, input_shape[axis]), "k2");
  auto reduced_shape = MakeReduceTargetShape({axis}, x, false, false);

  auto insert_reduce_index = [axis, ndim](const Array<Var> &indices,
                                          const IterVar &reduce_index) {
    Array<Expr> eval_range;
    int arg_counter = 0;
    for (size_t i = 0; i < ndim; ++i) {
      if (static_cast<int>(i) == axis)
        eval_range.push_back(reduce_index);
      else
        eval_range.push_back(indices[arg_counter++]);
    }
    return eval_range;
  };

  auto _compute_max = [&](const Array<Var> &indices) {
    auto eval_range = insert_reduce_index(indices, k1);
    return topi::MaxOp(x(eval_range), {k1});
  };

  auto _compute_expsum = [&](const Tensor &max_elem,
                             const Array<Var> &indices) {
    auto eval_range = insert_reduce_index(indices, k2);
    return tvm::sum(tvm::exp(x(eval_range) - max_elem(indices)), {k2});
  };

  auto _normalize = [&](const Tensor &max_elem, const Tensor &expsum,
                        const Array<Var> &indices) {
    Array<Expr> non_reduce_indices;
    for (size_t i = 0; i < ndim; ++i) {
      if (static_cast<int>(i) != axis)
        non_reduce_indices.push_back(indices[i]);
    }
    return tvm::exp(x(indices) - max_elem(non_reduce_indices)) /
           expsum(non_reduce_indices);
  };

  auto max_elem = tvm::compute(reduced_shape, _compute_max);
  auto expsum = tvm::compute(reduced_shape, [&](const Array<Var> &indices) {
      return _compute_expsum(max_elem, indices);
  });
  return tvm::compute(input_shape, [&](const Array<Var> &indices) {
      return _normalize(max_elem, expsum, indices);
  }, name, tag);
}

/*!
* \brief Log softmax activation
*
* \param x The input tensor. 2-D where log softmax is performed along the second dimension
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the log softmax operation
*/
inline Tensor log_softmax(const Tensor& x,
                          std::string name = "tensor",
                          std::string tag = "log_softmax_output") {
  CHECK_EQ(x->shape.size(), 2) << "Log softmax requires 2-D input";

  Expr m = x->shape[0];
  Expr n = x->shape[1];

  auto k = tvm::reduce_axis(Range(0, n), "k");
  auto max_elem = tvm::compute(
    { m }, [&](Var i) {
      return tvm::max(x(i, k), Array<IterVar>{ k }); });
  k = tvm::reduce_axis(Range(0, n), "k");

  auto expsum = tvm::compute(
    { m }, [&](Var i) {
      return tvm::sum(tvm::exp(x(i, k) - max_elem(i)), { k }); });

  return tvm::compute(
    x->shape, [&](Var i, Var j) {
      return x(i, j) - max_elem(i) - tvm::log(expsum(i));
    }, name, tag);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_SOFTMAX_H_
