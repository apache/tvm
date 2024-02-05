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

#include <tvm/ir/expr.h>

namespace tvm {
namespace arith {

/*!
 * \brief Check if an expr is a call to the vscale intrinsic.
 * \param expr The expr to check
 * \return True if the expr is a call to the vscale intrinsic, false if not.
 */
bool IsVScaleCall(const PrimExpr& expr);

/*!
 * \brief Returns the scalable lanes in a form multiplier * vscale
 * \param lanes The scalable lanes as a PrimExpr
 * \return Scalable lanes in a form multiplier * vscale
 */
PrimExpr CanonicalizeScalableLanes(const PrimExpr& lanes);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_SCALABLE_EXPRESSION_H_