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
 * \brief Dilate op constructions
 * \file nn/dilate.h
 */
#ifndef TOPI_NN_DILATE_H_
#define TOPI_NN_DILATE_H_

#include <string>

#include "tvm/operation.h"
#include "tvm/ir_pass.h"
#include "topi/tags.h"

namespace topi {
namespace nn {
using namespace tvm;
using namespace tvm::top;

/*!
* \brief Create a new expression of the logical and of all
* conditions in the arguments.
*
* \param args The arguments to find the logical conjunction of
*
* \return The logical conjunction expression
*/
PrimExpr all(Array<PrimExpr> args) {
  CHECK_GT(args.size(), 0) << "all requires at least one argument";

  PrimExpr ret = args[0];
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
                     Array<PrimExpr> strides,
                     std::string name = "tensor",
                     std::string tag = kInjective) {
  auto n = x->shape.size();
  CHECK_EQ(n, strides.size())
    << "strides size (" << strides.size()
    << ") must match dimension of x (" << n << ")";

  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < n; ++i) {
    out_shape.push_back(tvm::ir::Simplify(
      (x->shape[i] - 1) * cast(DataType::Int(32), strides[i] + 1)));
  }

  return tvm::top::compute(
    out_shape,
    [&](const Array<Var>& indices) {
      Array<PrimExpr> not_zero;
      Array<PrimExpr> index_tuple;
      for (size_t i = 0; i < n; ++i) {
        if (IsConstInt(strides[i]) && GetConstInt(strides[i]) == 1) {
          index_tuple.push_back(indices[i]);
        } else {
          index_tuple.push_back(indexdiv(indices[i], strides[i]));
          not_zero.push_back((indexmod(indices[i], strides[i])) == 0);
        }
      }
      if (not_zero.size() > 0) {
        auto all_not_zero = all(not_zero);
        return tvm::if_then_else(
            all_not_zero, x(index_tuple), make_const(x->dtype, 0));
      }
      return x(index_tuple);
    }, name, tag);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_DILATE_H_
