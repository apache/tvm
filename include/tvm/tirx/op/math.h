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
 * \file tvm/tirx/op/math.h
 * \brief Specialized mathematical expression builders.
 */
#ifndef TVM_TIRX_OP_MATH_H_
#define TVM_TIRX_OP_MATH_H_

#include <tvm/ir/prim/op.h>

namespace tvm::tirx {

/*!
 * \brief Compute log(exp(a) + exp(b)).
 *
 * \param a Left operand.
 * \param b Right operand.
 * \param span The location of this operation in the source.
 * \return The result expression.
 */
TVM_DLL PrimExpr logaddexp(PrimExpr a, PrimExpr b, Span span = Span());

TVM_DEFINE_INT_OP_CONST_VAL_OVERLOAD_SPANNED(logaddexp);

/*!
 * \brief Fast_erf_float expression from Eigen
 *
 * \param arg The input expression.
 * \param bits The number of bits in the type.
 * \return The constructed expression.
 */
TVM_DLL PrimExpr fast_erf_float_expr(PrimExpr arg, int bits);

}  // namespace tvm::tirx

#endif  // TVM_TIRX_OP_MATH_H_
