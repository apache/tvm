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
 * \brief Softmax op constructions
 * \file nn/flatten.h
 */
#ifndef TVM_TOPI_NN_FLATTEN_H_
#define TVM_TOPI_NN_FLATTEN_H_

#include <tvm/te/operation.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/tags.h>

#include <string>
#include <vector>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

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
inline Tensor flatten(const Tensor& x, std::string name = "tensor", std::string tag = kInjective) {
  auto ishape = x->shape;
  PrimExpr dim = 1;
  for (size_t i = 1; i < ishape.size(); ++i) {
    dim = dim * ishape[i];
  }

  Array<PrimExpr> oshape({ishape[0], dim});

  std::vector<PrimExpr> extra_shape;
  for (size_t i = 1; i < ishape.size(); ++i) {
    extra_shape.push_back(ishape[i]);
  }
  std::reverse(extra_shape.begin(), extra_shape.end());

  return tvm::te::compute(
      oshape,
      [&](Var i, Var j) {
        PrimExpr idx = j;
        std::vector<PrimExpr> index;
        for (auto s : extra_shape) {
          index.push_back(indexmod(idx, s));
          idx = indexdiv(idx, s);
        }
        index.push_back(i);
        std::reverse(index.begin(), index.end());
        return x(index);
      },
      name, tag);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_FLATTEN_H_
