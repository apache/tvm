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
 * \file unary.h
 * \brief The functions to make Relax unary arithmetic operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_UNARY_H_
#define TVM_RELAX_OP_TENSOR_UNARY_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Quick helper macro to
 * - expose a make-function interface which construct the call node.
 * - register op to the registry.
 * \param OpName The name of operator to register.
 * \param RequireFloatDtype A boolean indicating if the input is required to have float dtype.
 *  (Only for unary arith operators since all check operators don't require float dtype.)
 */
#define RELAX_REGISTER_UNARY_OP_AND_IMPL(OpName) \
  RELAX_UNARY_OP_INTERFACE(OpName, #OpName);     \
  RELAX_REGISTER_UNARY_OP(#OpName)

#define RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL(OpName, RequireFloatDtype) \
  RELAX_REGISTER_UNARY_OP_AND_IMPL(OpName).set_attr<FInferStructInfo>(    \
      "FInferStructInfo", InferStructInfoUnaryArith<RequireFloatDtype>)

#define RELAX_REGISTER_UNARY_CHECK_OP_AND_IMPL(OpName)                 \
  RELAX_REGISTER_UNARY_OP_AND_IMPL(OpName).set_attr<FInferStructInfo>( \
      "FInferStructInfo", InferStructInfoUnaryCheck)  // require_float_dtype=false for check op

/***************** Arithmetic operators *****************/

/*!
 * \brief Compute element-wise absolute value of the input data.
 * \param x The input data.
 * \return The computed result.
 */
Expr abs(Expr x);

/*! \brief Compute element-wise arc cos of the input data. */
Expr acos(Expr x);

/*! \brief Compute element-wise arc cosh of the input data. */
Expr acosh(Expr x);

/*! \brief Compute element-wise arc sin of the input data. */
Expr asin(Expr x);

/*! \brief Compute element-wise arc sinh of the input data. */
Expr asinh(Expr x);

/*! \brief Compute element-wise arc tan of the input data. */
Expr atan(Expr x);

/*! \brief Compute element-wise arc tanh of the input data. */
Expr atanh(Expr x);

/*! \brief Compute element-wise bitwise not */
Expr bitwise_not(Expr x);

/*! \brief Take ceil of input data. */
Expr ceil(Expr x);

/*! \brief Compute element-wise cos of the input data. */
Expr cos(Expr x);

/*! \brief Compute element-wise cosh of the input data. */
Expr cosh(Expr x);

/*! \brief Compute element-wise exp of data. */
Expr exp(Expr x);

/*! \brief Take floor of input data. */
Expr floor(Expr x);

/*! \brief Compute element-wise natural logarithm of data. */
Expr log(Expr x);

/*! \brief Compute element-wise logical not */
Expr logical_not(Expr x);

/*! \brief Compute element-wise negative value of data. */
Expr negative(Expr x);

/*! \brief Rounds each element of the input data to nearest integer. */
Expr round(Expr x);

/*! \brief Compute element-wise reciprocal square root of data. */
Expr rsqrt(Expr x);

/*! \brief Compute element-wise sigmoid of data. */
Expr sigmoid(Expr x);

/*! \brief Returns an indication of the sign of a number for each element of the input data. */
Expr sign(Expr x);

/*! \brief Compute element-wise sin of data. */
Expr sin(Expr x);

/*! \brief Compute element-wise sinh of data. */
Expr sinh(Expr x);

/*! \brief Compute element-wise square root of data. */
Expr sqrt(Expr x);

/*! \brief Squares each element of the input data. */
Expr square(Expr x);

/*! \brief Compute element-wise tan of data. */
Expr tan(Expr x);

/*! \brief Compute element-wise tanh of data. */
Expr tanh(Expr x);

/*! \brief Clips tensor values to a specified min and max. */
Expr clip(Expr x, Expr min, Expr max);

/***************** Check operators *****************/

/*! \brief Check if input value is finite. */
Expr isfinite(Expr x);

/*! \brief Check if input value is infinite. */
Expr isinf(Expr x);

/*! \brief Check if input value is Nan. */
Expr isnan(Expr x);

/*! \brief Apply error function to input value. */
Expr erf(Expr x);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_UNARY_H_
