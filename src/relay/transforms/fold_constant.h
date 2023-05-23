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
 * \file fold_constant.h
 * \brief Utility functions for folding constants in expressions.
 */
#ifndef TVM_RELAY_TRANSFORMS_FOLD_CONSTANT_H_
#define TVM_RELAY_TRANSFORMS_FOLD_CONSTANT_H_

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {
namespace transform {

/*!
 * \brief Apply constant folding on an expression.
 *
 * \param expr The expression to fold.
 * \param fold_qnn Whether to fold constants for QNN operations.
 * \returns The new folded expression.
 */
Expr FoldConstantExpr(const Expr& expr, bool fold_qnn = true);

/*!
 * \brief Returns \p expr with any constants expressions evaluated and let-bound constants
 * inlined. Returns \p expr unchanged if no change.
 *
 * CAUTION: The importers rely on this function returning \p expr unchanged to preserve sharing
 * from their p.o.v. Furthermore, this function can be called before conversion to ANF so
 * we must avoid all recursion.
 */
Expr FoldConstantExpr(const Expr& expr, const IRModule& mod, bool fold_qnn);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_FOLD_CONSTANT_H_
