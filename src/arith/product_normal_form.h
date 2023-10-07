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
 * \file product_normal_form.h
 * \brief Centralized location related to simplifying prod of results.
 */
#ifndef TVM_ARITH_PRODUCT_NORMAL_FORM_H_
#define TVM_ARITH_PRODUCT_NORMAL_FORM_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

/*!
 * \brief Unpack reduction by calling each leaf via fleaf
 *
 * \param value The expression value.
 * \tparam TNode the reduction node to match.
 * \tparam FLeaf The callback function at leaf.
 */
template <typename TNode, typename FLeaf>
inline void UnpackReduction(const PrimExpr& value, FLeaf fleaf) {
  if (const TNode* node = value.as<TNode>()) {
    UnpackReduction<TNode, FLeaf>(node->a, fleaf);
    UnpackReduction<TNode, FLeaf>(node->b, fleaf);
  } else {
    fleaf(value);
  }
}

/**
 * \brief Unpack chain of add sub by calling each leaf via fleaf
 * \param value The expression value.
 * \tparam FLeaf The callback function at leaf.
 */
template <typename FLeaf>
inline void UnpackSum(const PrimExpr& value, FLeaf fleaf, int sign = 1) {
  if (const tir::AddNode* node = value.as<tir::AddNode>()) {
    UnpackSum(node->a, fleaf, sign);
    UnpackSum(node->b, fleaf, sign);
  } else if (const tir::SubNode* node = value.as<tir::SubNode>()) {
    UnpackSum(node->a, fleaf, sign);
    UnpackSum(node->b, fleaf, -sign);
  } else {
    fleaf(value, sign);
  }
}

/*!
 * \brief Helper function to multiply extent and re-normalize.
 *
 * Multiply extent scale and re-normalize to form (x * y) * z
 *
 * NOTE on multiplication order: when have shape (s[0], s[1], s[2]),
 * we prefer to multiple in order of s[0] * s[1] * s[2]

 * \param lhs The lhs iterator
 * \param rhs The rhs iterator
 * \return the result.
 */
inline PrimExpr MulAndNormalize(const PrimExpr& lhs, const PrimExpr& rhs) {
  int64_t cscale = 1;
  PrimExpr res = tir::make_const(lhs.dtype(), 1);
  auto fcollect = [&](PrimExpr val) {
    if (const auto* intimm = val.as<IntImmNode>()) {
      cscale *= intimm->value;
    } else {
      res = res * val;
    }
  };
  UnpackReduction<tir::MulNode>(lhs, fcollect);
  UnpackReduction<tir::MulNode>(rhs, fcollect);
  if (cscale != 1) {
    res = res * tir::make_const(res.dtype(), cscale);
  }
  return res;
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_PRODUCT_NORMAL_FORM_H_
