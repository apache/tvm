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
 * \note This algorithm returns the first matched sub-graph. Use `start_hint` to specify the
 * starting point of the matching so that we can distinguish multiple matches.
 *
 * \param ctx The graph-wise patterns.
 * \param dfb The function to match.
 * \param start_hint The starting point expression to match to distinguish multiple matches.
 * \param must_include_hint If start_hint is given, the return pattern must include start_hint.
 * \return tvm::runtime::Map<DFPattern, Var>
 */
TVM_DLL tvm::runtime::Map<DFPattern, Var> MatchGraph(const PatternContext& ctx,
                                                     const DataflowBlock& dfb,
                                                     Optional<Var> start_hint = NullOpt,
                                                     bool must_include_hint = false);

/**
 * \brief Match a graph-wise pattern with the current context (PatternContext::Current()).
 */
inline tvm::runtime::Map<DFPattern, Var> MatchGraphDefault(const DataflowBlock& dfb,
                                                           Optional<Var> start_hint = NullOpt,
                                                           bool must_include_hint = false) {
  return MatchGraph(PatternContext::Current(), dfb, start_hint, must_include_hint);
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DATAFLOW_MATCHER_H_
