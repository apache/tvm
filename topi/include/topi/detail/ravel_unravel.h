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
* \file ravel_unravel.h
* \brief Index ravel and unraval operations
*/
#ifndef TOPI_DETAIL_RAVEL_UNRAVEL_H_
#define TOPI_DETAIL_RAVEL_UNRAVEL_H_

#include <vector>

#include "tvm/operation.h"
#include "tvm/expr_operator.h"

namespace topi {
namespace detail {
using namespace tvm;

/*!
* \brief Flatten the indices to 1D
*
* \param indices The input coordinates
* \param shape Shape of the tensor
*
* \return The index after flattening
*/
inline Expr RavelIndex(Array<Expr> indices, Array<Expr> shape) {
  CHECK_EQ(indices.size(), shape.size()) << "indices and shape must have equal size";
  CHECK_GT(indices.size(), 0) << "indices must not be empty";
  Expr idx;
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i == 0) {
      idx = indices[i];
    } else {
      idx = idx * shape[i] + indices[i];
    }
  }
  return idx;
}

/*!
* \brief Convert flattened index to coordinate array
*
* \param idx The 1D index
* \param shape Shape of the tensor
*
* \return The coordinate corresponding to the 1D index
*/
inline Array<Expr> UnravelIndex(Expr idx, Array<Expr> shape) {
  std::vector<Expr> indices;

  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    indices.push_back(idx % shape[i]);
    idx = idx / shape[i];
  }
  std::reverse(indices.begin(), indices.end());
  return indices;
}

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_RAVEL_UNRAVEL_H_
