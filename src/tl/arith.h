/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file arith.h
 * \brief Some arith tools for layout & fragment inference
 *
 */

#ifndef TVM_TL_ARITH_H_
#define TVM_TL_ARITH_H_

#include <tvm/arith/iter_affine_map.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Collect the IterSplit that is not used in expr.
 *
 *  If the expr is (x // 2) and x is in Range(4),
 *  than the result should be (x % 2)
 */
Array<arith::IterSplitExpr> DivideUnusedIterators(const Array<PrimExpr>& exprs,
                                                  const Array<IterVar> input_iters,
                                                  arith::Analyzer* analyzer);

/*!
 * \brief Conpress the iterator, remove the unused part of the iterator not present in the expr
 *
 *  Returns the compressed IterVar as well as the Updated iter sum expression.
 */
std::pair<PrimExpr, IterVar> CompressIterator(const PrimExpr& expr,
                                              const Array<IterVar> input_iters, const IterVar& iv,
                                              arith::Analyzer* analyzer);

/*!
 * \brief Convert the iter splits returned by DivideUnusedIterators into flattened expression
 *
 */
PrimExpr MakeFlattenedExpression(const Array<arith::IterSplitExpr>& splits);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_ARITH_H_
