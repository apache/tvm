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
 * \file ternary.h
 * \brief The functions to make Relax ternary operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_TERNARY_H_
#define TVM_RELAX_OP_TENSOR_TERNARY_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Elementwise fused multiply-add operator
 * Returns elementwise result of `x1 * x2 + x3`
 * \param x1 The left hand operand of the multiplication
 * \param x2 The right hand operand of the multiplication
 * \param x3 The operand of the addition
 * \return The computed result.
 */
Expr ewise_fma(Expr x1, Expr x2, Expr x3);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_TERNARY_H_
