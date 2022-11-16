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
 * \file contraint_extract.h
 *
 * \brief Centralized location for extraction of constraints from a boolean expression.
 */

#ifndef TVM_ARITH_CONSTRAINT_EXTRACT_H_
#define TVM_ARITH_CONSTRAINT_EXTRACT_H_

#include <tvm/tir/expr.h>

#include <vector>

namespace tvm {
namespace arith {

/* \brief Returns constraints that are true if the expression is true.
 *
 * Utility to break up a boolean expression into independent
 * constraints.
 *
 * Example: `i==5 && j==3` => `[i==5 && j==3, i==5, j==3]`
 * Example: `i==5 || j==3` => `[i==5 || j==3]`
 * Example: `!(i>5 || j==3)` => `[!(i==5 || j==3), i<=5, j!=3]`
 *
 * If `keep_composite_constraints` is true (default), a constraint
 * that can be decomposed will be included in the output.  If false,
 * they will be excluded.
 *
 * Example, removing composite: `!(i>5 || j==3)` => `[i<=5, j!=3]`
 *
 * Intended for use in bounds analysis or simplification within a
 * conditional, or identifying independent conditionals that may be
 * hoisted.
 *
 * \param expr The expression to be analyzers
 *
 * \param keep_composite_constraints Whether to include composite
 * constraints in the output.
 *
 * \returns A vector of independent constraints
 */
std::vector<PrimExpr> ExtractConstraints(const PrimExpr& expr,
                                         bool keep_composite_constraints = true);

/* \brief Returns components that are false if the expression is false.
 *
 * Utility to break up a boolean expression into independent
 * components.
 *
 * Example: `i==5 || j==3` => `[i==5, j==3]`
 * Example: `i==5 && j==3` => `[i==5 && j==3]`
 * Example: `!(i>5 && j==3)` => `[i<=5, j!=3]`
 *
 * Intended for use in bounds analysis or simplification within a
 * conditional, or identifying independent conditionals that may be
 * hoisted.
 *
 * \param expr The expression to be analyzers
 *
 * \returns A vector of independent constraints
 */
std::vector<PrimExpr> ExtractComponents(const PrimExpr& expr);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_CONSTRAINT_EXTRACT_H_
