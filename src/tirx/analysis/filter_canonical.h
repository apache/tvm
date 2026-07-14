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
 * \file filter_canonical.h
 * \brief Canonical-form classifier for thread-filter predicates.
 *
 * A thread-filter predicate selects a subset of threads to enter the body of
 * an `if`. The canonical form is the subset of Bool PrimExpr shapes that the
 * compiler can statically analyze for active-thread-set narrowing:
 *
 *   pred := atom (AND atom)*        // pure n-ary conjunction (no OR/NOT)
 *   atom := scopeid_var <op> const  // op in {==, <, <=, >, >=}
 *         | Call("tirx.ptx.elect_sync")
 *
 * Consumers:
 *   1. tile_primitive_dispatch routes a bare `if cond:` to atom-based
 *      narrowing when cond is canonical; otherwise treats it as a regular
 *      data-dependent branch (no narrowing).
 *   2. The `tirx.filter(var, pred)` escape-hatch wrapper is intended for the
 *      *non-canonical* case -- callers who want thread-filter semantics on a
 *      predicate the classifier cannot decode. A canonical predicate inside
 *      `tirx.filter` is redundant: the wrapper can be dropped in favor of a
 *      bare `if`.
 */
#ifndef TVM_TIRX_ANALYSIS_FILTER_CANONICAL_H_
#define TVM_TIRX_ANALYSIS_FILTER_CANONICAL_H_

#include <tvm/tirx/expr.h>
#include <tvm/tirx/var.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief Kind of an atomic predicate in canonical form.
 *
 * All five comparison operators (==, <, <=, >, >=) are normalized into a
 * single half-open range atom `[lo, hi)`. Use `arith::ConstIntBound::kNegInf`
 * for an unbounded lower side and `kPosInf` for an unbounded upper side.
 */
enum class FilterAtomKind {
  kRange,      // scopeid_var in [lo, hi); covers ==, <, <=, >, >=
  kElectSync,  // Call("tirx.ptx.elect_sync")
};

/*!
 * \brief One atomic predicate. Variant keyed by `kind`.
 *
 * For `kRange`:
 *   - `scopeid_var`: the ScopeIdDef-declared variable on the LHS of the
 *     comparison (mirrored automatically if the input had `const <op> var`).
 *   - `lo`, `hi`: half-open bounds. `lo` may be
 *     `arith::ConstIntBound::kNegInf` for an unbounded lower side; `hi` may
 *     be `kPosInf` for an unbounded upper side.
 *   - `elect_sync_call` is unset.
 *
 * For `kElectSync`:
 *   - `elect_sync_call`: the original `Call("tirx.ptx.elect_sync")` PrimExpr,
 *     preserved verbatim so downstream consumers (e.g. selector construction
 *     in tile_primitive_dispatch) can reuse it without re-synthesizing.
 *   - `scopeid_var`, `lo`, `hi` are unset.
 */
struct FilterAtom {
  FilterAtomKind kind;
  Var scopeid_var;
  int64_t lo = 0;
  int64_t hi = 0;
  PrimExpr elect_sync_call;
};

/*!
 * \brief Canonical form: an ordered list of atomic predicates whose
 *        conjunction equals the original predicate.
 *
 * Ordering follows source flattening (left-to-right traversal of the AND
 * tree) but is not semantically significant -- consumers should treat
 * `atoms` as an unordered set of constraints.
 */
struct CanonicalForm {
  std::vector<FilterAtom> atoms;
};

/*!
 * \brief Callback: returns true iff `var` is a ScopeIdDef-declared scope id.
 *
 * The classifier consults this for every variable that appears on the LHS
 * of a comparison atom. The callback abstracts over how scope ids are
 * tracked in the caller's context:
 *   - TilePrimitiveDispatcher passes a lambda that walks its
 *     `scope_id_defs_at_level_` stack.
 *   - Tests may pass a simpler lambda over a fixed allow-list of vars.
 *
 * A var that fails this check causes the enclosing predicate to be classified
 * as non-canonical (`TryClassifyCanonical` returns `std::nullopt`).
 */
using ScopeIdPredicate = std::function<bool(const Var&)>;

/*!
 * \brief Try to classify `cond` as a canonical thread-filter predicate.
 *
 * Grammar (see file header):
 *   pred := atom (AND atom)*
 *   atom := scopeid_var <op> const  (op in {==, <, <=, >, >=})
 *         | Call("tirx.ptx.elect_sync")
 *
 * Returns:
 *   - `std::nullopt` if `cond` does not match the grammar. The caller should
 *     treat the enclosing if-statement as either a regular data-dependent
 *     branch (no narrowing) or -- if wrapped in `tirx.filter(var, cond)` --
 *     an explicit escape-hatch (binding-var-driven singleton fallback).
 *   - A `CanonicalForm` with the parsed atom list otherwise.
 *
 * Implementation notes:
 *   - Conjunction is recognized via both `tir::And` nodes and
 *     `tirx.bitwise_and` calls (matching existing FlattenConjuncts behavior
 *     in tile_primitive_dispatch.cc).
 *   - Comparison atoms with `const <op> var` are mirrored so the
 *     `scopeid_var` is on the LHS of the returned atom.
 *   - `c1 == c2` (two constants), `v1 == v2` (two vars), and any other
 *     non-grammar shape causes the whole classification to fail.
 *   - The classifier is purely syntactic: it does NOT call
 *     `arith::Analyzer::Simplify` on subexpressions. Callers that want
 *     `2 + 1` to collapse to `3` should pre-simplify their input.
 *   - This function does NOT unwrap `tirx.filter` Calls. The caller is
 *     responsible for extracting the inner predicate (`call->args[1]`)
 *     before passing it here. A `tirx.filter` Call passed in directly is
 *     classified as non-canonical.
 *
 * Thread safety: pure function; safe to call concurrently provided the
 * callback `is_scope_id` is itself thread-safe.
 */
TVM_DLL std::optional<CanonicalForm> TryClassifyCanonical(const PrimExpr& cond,
                                                          const ScopeIdPredicate& is_scope_id);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_ANALYSIS_FILTER_CANONICAL_H_
