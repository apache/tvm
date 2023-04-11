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
 * \file tvm/relax/dataflow_matcher.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAX_DATAFLOW_MATCHER_H_
#define TVM_RELAX_DATAFLOW_MATCHER_H_

#include <tvm/relax/dataflow_pattern.h>
#include <tvm/runtime/container/optional.h>

#include <memory>

namespace tvm {
namespace relax {

/**
 * \brief Determine if a pattern matches an expression.
 * \note The behavior of MatchExpr is to match a relax.Expr (`expr`) syntactically through
 * one given pattern (`pattern`).
 *
 * \param pattern The pattern to match
 * \param expr The expression to match
 * \param bindings The mapping from relax.Var to relax.Expr
 * \return true if matched
 * \return false if unmatched
 */
bool MatchExpr(DFPattern pattern, Expr expr, Optional<runtime::Map<Var, Expr>> bindings = NullOpt);

/* \brief Similar to above, but return pairs of a matching pattern and an expression.  */
Optional<Map<DFPattern, Expr>> ExtractMatchedExpr(
    DFPattern pattern, Expr expr, Optional<runtime::Map<Var, Expr>> bindings = NullOpt);

/**
 * \brief Match a sub-graph in a DataflowBlock with a graph of patterns and return the mapping.
 * \param ctx The graph-wise patterns.
 * \param dfb The function to match.
 * \return Matched patterns and corresponding bound variables
 */
TVM_DLL Optional<Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx,
                                                 const DataflowBlock& dfb);

/**
 * \brief Rewrite a function with the given pattern and the rewriter function.
 * \param ctx The pattern constraint context under which rewriting takes place.
 * \param rewriter The function to be called on a successful matching for rewriting.
    Given the map of patterns and corresponding variables (bound variables or parameters),
    it should return a map that specifies new values for matched bound variables.
 * \param f The function to rewrite
 * \return The rewritten or the input function, depending on the pattern matching result.
 */
TVM_DLL Function RewriteBindings(const PatternContext& ctx, PackedFunc rewriter, Function f);
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DATAFLOW_MATCHER_H_
