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
#ifndef TVM_RELAX_OP_UNARY_H_
#define TVM_RELAX_OP_UNARY_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {

// (TVM-TOOL) cc_op begin decl/elementwise/*
/*!
 * Elementwise abs
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call abs(relax::Expr a);
/*!
 * Elementwise acos
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call acos(relax::Expr a);
/*!
 * Elementwise acosh
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call acosh(relax::Expr a);
/*!
 * Elementwise add
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call add(relax::Expr a, relax::Expr b);
/*!
 * Elementwise asin
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call asin(relax::Expr a);
/*!
 * Elementwise asinh
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call asinh(relax::Expr a);
/*!
 * Elementwise atan
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call atan(relax::Expr a);
/*!
 * Elementwise atan2
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call atan2(relax::Expr a, relax::Expr b);
/*!
 * Elementwise atanh
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call atanh(relax::Expr a);
/*!
 * Elementwise bitwise and
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call bitwise_and(relax::Expr a, relax::Expr b);
/*!
 * Elementwise bitwise invert
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call bitwise_invert(relax::Expr a);
/*!
 * Elementwise bitwise left shift
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call bitwise_left_shift(relax::Expr a, relax::Expr b);
/*!
 * Elementwise bitwise or
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call bitwise_or(relax::Expr a, relax::Expr b);
/*!
 * Elementwise bitwise right shift
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call bitwise_right_shift(relax::Expr a, relax::Expr b);
/*!
 * Elementwise bitwise xor
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call bitwise_xor(relax::Expr a, relax::Expr b);
/*!
 * Elementwise ceil
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call ceil(relax::Expr a);
/*!
 * Elementwise cos
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call cos(relax::Expr a);
/*!
 * Elementwise cosh
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call cosh(relax::Expr a);
/*!
 * Elementwise divide
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call divide(relax::Expr a, relax::Expr b);
/*!
 * Elementwise equal
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call equal(relax::Expr a, relax::Expr b);
/*!
 * Elementwise exp
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call exp(relax::Expr a);
/*!
 * Elementwise floor
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call floor(relax::Expr a);
/*!
 * Elementwise floor divide
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call floor_divide(relax::Expr a, relax::Expr b);
/*!
 * Elementwise greater
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call greater(relax::Expr a, relax::Expr b);
/*!
 * Elementwise greater equal
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call greater_equal(relax::Expr a, relax::Expr b);
/*!
 * Elementwise isfinite
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call isfinite(relax::Expr a);
/*!
 * Elementwise isinf
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call isinf(relax::Expr a);
/*!
 * Elementwise isnan
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call isnan(relax::Expr a);
/*!
 * Elementwise less
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call less(relax::Expr a, relax::Expr b);
/*!
 * Elementwise less equal
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call less_equal(relax::Expr a, relax::Expr b);
/*!
 * Elementwise log
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call log(relax::Expr a);
/*!
 * Elementwise log10
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call log10(relax::Expr a);
/*!
 * Elementwise log1p
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call log1p(relax::Expr a);
/*!
 * Elementwise log2
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call log2(relax::Expr a);
/*!
 * Elementwise logical and
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call logical_and(relax::Expr a, relax::Expr b);
/*!
 * Elementwise logical not
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call logical_not(relax::Expr a);
/*!
 * Elementwise logical or
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call logical_or(relax::Expr a, relax::Expr b);
/*!
 * Elementwise multiply
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call multiply(relax::Expr a, relax::Expr b);
/*!
 * Elementwise negative
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call negative(relax::Expr a);
/*!
 * Elementwise not equal
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call not_equal(relax::Expr a, relax::Expr b);
/*!
 * Elementwise positive
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call positive(relax::Expr a);
/*!
 * Elementwise pow
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call pow(relax::Expr a, relax::Expr b);
/*!
 * Elementwise pow
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call power(relax::Expr a, relax::Expr b);
/*!
 * Elementwise remainder
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call remainder(relax::Expr a, relax::Expr b);
/*!
 * Elementwise round
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call round(relax::Expr a);
/*!
 * Elementwise sin
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call sin(relax::Expr a);
/*!
 * Elementwise sinh
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call sinh(relax::Expr a);
/*!
 * Elementwise sqrt
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call sqrt(relax::Expr a);
/*!
 * Elementwise square
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call square(relax::Expr a);
/*!
 * Elementwise subtract
 * \param a TODO(tvm-unity-team): add doc
 * \param b TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call subtract(relax::Expr a, relax::Expr b);
/*!
 * Elementwise tan
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call tan(relax::Expr a);
/*!
 * Elementwise tanh
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call tanh(relax::Expr a);
/*!
 * Elementwise trunc
 * \param a TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call trunc(relax::Expr a);
// (TVM-TOOL) cc_op end decl/elementwise/*

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_UNARY_H_
