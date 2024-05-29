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
 * \file tvm/arith/scalable_expression.h
 * \brief Analyze scalable expressions.
 */

#ifndef TVM_ARITH_SCALABLE_EXPRESSION_H_
#define TVM_ARITH_SCALABLE_EXPRESSION_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/target/target.h>

#include <optional>
#include <vector>

namespace tvm {
namespace arith {

/*! \brief A list of known vscale values to try for an AArch64 SVE target. */
static const std::vector<unsigned int> kAArch64VScaleValues = {1, 2, 4, 8, 16};

/*!
 * \brief Check if an expr is a call to the vscale intrinsic.
 * \param expr The expr to check
 * \return True if the expr is a call to the vscale intrinsic, false if not.
 */
bool IsVScaleCall(const PrimExpr& expr);

/*!
 * \brief Check if an expr contains a call to the vscale intrinsic.
 * \param expr The expr to check
 * \return True if the expr contains a call to the vscale intrinsic, false if not.
 */
bool ContainsVscaleCall(const PrimExpr& expr);

/*!
 * \brief Substitute a vscale intrinsic call with a known scalar value.
 * \param expr The expr to apply substitutions to.
 * \param vscale_value The scalar value to replace vscale with.
 * \return A rewritten expression with vscale values replaced with a scalar value.
 */
PrimExpr SubstituteVScaleWithKnownValue(const PrimExpr& expr, unsigned int vscale_value);

/*!
 * \brief Returns the vscale multiplier as a nullable type
 * \param lanes The scalable lanes as a PrimExpr
 * \return vscale multiplier as std::optional<int>
 */
std::optional<int> ExtractVscaleFactor(const PrimExpr& lanes);

/*!
 * \brief Check if the expression can be proven when evaluating it on all possible values
           of vscale.
 * \param analyzer An analyzer instance.
 * \param expr The expression to try to prove.
 * \param vscale_values A list of values to substitute vscale with.
 * \return Whether or not the expression can be proven with this technique.
 */
bool CanProveVscaleExpressionFromKnownValues(arith::Analyzer* analyzer, const PrimExpr& expr,
                                             const std::vector<unsigned int>& vscale_values);

/*!
 * \brief Check whether the compilation target supports SVE
 * \param target The target to check.
 * \return Whether SVE is supported
 */
bool TargetHasSVE(Target target);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_SCALABLE_EXPRESSION_H_
