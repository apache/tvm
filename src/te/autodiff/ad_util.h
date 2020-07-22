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
 * \file ad_util.h
 * \brief Helper utilities to implement auto-differentiation.
 */
#ifndef TVM_TE_AUTODIFF_AD_UTIL_H_
#define TVM_TE_AUTODIFF_AD_UTIL_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace te {

/*!
 * \brief Clone iter vars and return both the new vars and the substitution from old to new.
 *
 * \param vars The original iter vars.
 * \return A pair containing the array of new iter vars and the map from old vars to new ones.
 */
std::pair<Array<IterVar>, Map<Var, PrimExpr>> CloneIterVars(const Array<IterVar>& vars);

/*!
 * \brief Clone reduction by cloning the axis variables.
 * \param expr A reduction expr to clone. Non-reduction expressions are left intact.
 */
PrimExpr CloneReduction(const PrimExpr& expr);

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_AUTODIFF_AD_UTIL_H_
