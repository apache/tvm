/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  Sex The NOTICE file
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
 * KIND, either express or implied.  Sex The License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file set.h
 * \brief The functions to make Relax set operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_SET_H_
#define TVM_RELAX_OP_TENSOR_SET_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Find the unique elements in a given tensor.
 * In addition, it optionally returns
 * - the indices of the input tensor that give the unique values;
 * - the indices of the unique tensor that reconstruct the input tensor;
 * - the number of times each unique value comes up in the input tensor.
 * \param x The input tensor.
 * \param sorted Whether to sort the unique elements in ascending order before
 *        returning as output.
 * \param return_index Whether to return an additional tensor with indices for where elements in
 *        the unique tensor come from the original input.
 * \param return_inverse Whether to return an additional tensor with indices for where elements in
 *        the original input ended up in the returned unique list.
 * \param return_counts Whether to return an additional tensor with counts of each unique elements.
 * \param axis The dimension to apply unique.
 *        If not specified, the unique values of the flattened input are returned.
 * \return The unique elements of the array. The returned array will be sorted if `sorted` is True.
 *         Additional return values depend on `return_index`, `return_inverse`, and `return_counts`.
 */
Expr unique(Expr x, PrimValue sorted, PrimValue return_index, PrimValue return_inverse,
            PrimValue return_counts, Optional<PrimValue> axis);

/*!
 * \brief Returns the indices of the non-zero elements of the input tensor.
 * \param x The input tensor.
 * \return a list of 1-D tensors containing indices of non-zero elements for each dimension.
 * \note This function behaves similarly to numpy.nonzero(), but return a multi-dimensional array
 *       instead of a tuple of 1-D arrays.
 */
Expr nonzero(Expr x);
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_SET_H_
