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
 * \file tensor_utils.h
 * \brief Utility functions for handling tensor
 */
#ifndef TOPI_DETAIL_TENSOR_UTILS_H_
#define TOPI_DETAIL_TENSOR_UTILS_H_


namespace topi {
namespace detail {
using namespace tvm;
using namespace tvm::te;

/*!
 * \brief Check whether input shape has dimension of size 0;
 *
 * \param x Input shape
 *
 * \return True if the input shape is empty.
 */
inline bool is_empty_shape(const Array<PrimExpr>& x) {
  bool is_empty = false;
  for (const auto& dim : x) {
    if (auto int_dim = dim.as<IntImmNode>()) {
      if (int_dim->value == 0) {
        is_empty = true;
        break;
      }
    }
  }
  return is_empty;
}

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_TENSOR_UTILS_H_

