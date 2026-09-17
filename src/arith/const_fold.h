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
 * \file const_fold.h
 * \brief Arithmetic aliases for shared constant folding and scoped index analysis.
 */
#ifndef TVM_ARITH_CONST_FOLD_H_
#define TVM_ARITH_CONST_FOLD_H_

#include "../ir/prim/const_fold.h"
#include "int_operator.h"

#define TVM_ARITH_CONST_PROPAGATION(BODY) TVM_PRIM_CONST_PROPAGATION(BODY)
#define TVM_INDEX_CONST_PROPAGATION(BODY) TVM_PRIM_INDEX_CONST_PROPAGATION(BODY)

namespace tvm {
namespace arith {

using prim::detail::GetFoldResultDoubleRepr;
using prim::detail::GetFoldResultInt64Repr;
using prim::detail::is_neg_inf;
using prim::detail::is_pos_inf;
using prim::detail::IsIndexType;
using prim::detail::IsIndexTypedExpr;
using prim::detail::neg_inf;
using prim::detail::pos_inf;
using prim::detail::SymbolicLimits;
using prim::detail::TryConstFold;

/*!
 * \brief Scoped opt-in flag for unsigned index analysis (allow_u32).
 *
 * Unsigned arithmetic wraps modulo 2^bits, so most index rewrite rules are
 * unsound for it in general. When the caller can guarantee values never
 * approach the type limit (e.g. compile-time layout proofs over SMEM
 * offsets), enabling this flag activates a small set of otherwise-unsafe
 * rewrite rules for unsigned operands (see the unsigned blocks in
 * rewrite_simplify.cc). It is OFF by default; thread-local and nested-safe.
 */
namespace uint_as_index {

inline thread_local int g_depth = 0;

inline bool Enabled() { return g_depth > 0; }

}  // namespace uint_as_index

/*! \brief RAII guard enabling uint_as_index mode in the current scope. */
class AllowUintAsIndexGuard {
 public:
  AllowUintAsIndexGuard() { ++uint_as_index::g_depth; }
  ~AllowUintAsIndexGuard() { --uint_as_index::g_depth; }
  AllowUintAsIndexGuard(const AllowUintAsIndexGuard&) = delete;
  AllowUintAsIndexGuard& operator=(const AllowUintAsIndexGuard&) = delete;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_CONST_FOLD_H_
