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
 * \file narrow_predicate_expression.h
 * \brief Utility for extracting and interacting with buffer touch points
 */

#include <tvm/ir/expr.h>
#include <tvm/tir/var.h>

#ifndef TVM_ARITH_NARROW_PREDICATE_EXPRESSION_H_
#define TVM_ARITH_NARROW_PREDICATE_EXPRESSION_H_

namespace tvm {
namespace arith {

/* \brief Narrow a true expression to remove free parameters
 *
 * This function provides two guarantees:
 *
 * 1. If the resulting expression evaluates to True, then the original
 * expression also evaluates to True.
 *
 * 2. The resulting expression does not contain any of the free
 * parameters.
 *
 * 3. The resulting expression does not contain any BufferLoad
 *
 * \param expr The expression to be examined.
 *
 * \param ranges The variables to be removed from the expression
 *
 * \returns An expression that, if true, implies that the original
 * expression is also true.
 */
PrimExpr NarrowPredicateExpression(PrimExpr expr, Map<tir::Var, Range> free_parameters);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_NARROW_PREDICATE_EXPRESSION_H_
