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
 * \file search.h
 * \brief The functions to make Relax searching operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_SEARCH_H_
#define TVM_RELAX_OP_TENSOR_SEARCH_H_

#include <tvm/relax/attrs/search.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Selecting elements from either the input tensors depending on the value of the
 * condition.
 */
Expr where(Expr condition, Expr x1, Expr x2);

/*! \brief Computes the argmax of tensor elements over given axis. */
Expr argmax(Expr x, Optional<Integer> axis, bool keepdims);

/*! \brief Computes the argmin of tensor elements over given axis. */
Expr argmin(Expr x, Optional<Integer> axis, bool keepdims);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_SEARCH_H_
