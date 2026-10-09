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
 * \file tvm/tirx/op/vector.h
 * \brief Vector operations for TIRx.
 */
#ifndef TVM_TIRX_OP_VECTOR_H_
#define TVM_TIRX_OP_VECTOR_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace tirx {

/*!
 * \brief Get the high level half of the vector.
 *
 * Arguments, in order:
 * - args[0]: vec, The input vector.
 */
TVM_DLL const Op& vectorhigh_op();

/*!
 * \brief Get the low-level half of the vector.
 *
 * Arguments, in order:
 * - args[0]: vec, The input vector.
 */
TVM_DLL const Op& vectorlow_op();

/*!
 * \brief Concat two vectors.
 *
 * Arguments, in order:
 * - args[0]: vec1, The first input vector.
 * - args[1]: vec2, The second input vector.
 */
TVM_DLL const Op& vectorcombine_op();

/*!
 * \brief Calculate a predicate mask given an upper bound (limit) and a current value (base).
 *
 * Arguments, in order:
 * - args[0]: base, The base value.
 * - args[1]: limit, The limit value.
 */
TVM_DLL const Op& get_active_lane_mask_op();

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_OP_VECTOR_H_
