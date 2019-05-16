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
 *  Copyright (c) 2019 by Contributors
 * \file match_exhaustion.cc
 * \brief Checking Relay match expression exhaustiveness.
 *
 * This file implements a function that checks whether a match
 * expression is exhaustive, that is, whether a given match clause
 * matches every possible case. This is important for ensuring
 * code correctness, since hitting an unmatched case results in a
 * dynamic error unless exhaustiveness is checked in advance.
 */
#include "match_exhaustion.h"
#include <tvm/relay/adt.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/pass.h>
#include <deque>
#include <stack>

namespace tvm {
namespace relay {

/*! \brief Possible pattern match results */
enum MatchResult : int {
  kMatch = 0,        // pattern matches
  kClash = 1,        // pattern conflicts
  kUnspecified = 2,  // ambiguous: candidate needs more constructors specified
};

class CandidateChecker : public PatternFunctor<MatchResult(const Pattern&, const Pattern&)> {
 public:
  explicit CandidateChecker() {}

  MatchResult Check(const Pattern& pat, const Pattern& candidate) {
    return this->VisitPattern(pat, candidate);
  }

  // for a constructor pattern, we must ensure that the candidate is
  // a ConstructorPattern, that it has the same constructor, and
  // that its fields match the subpatterns.
  MatchResult VisitPattern_(const PatternConstructorNode* op, const Pattern& cand) override {
    auto* ctor_cand = cand.as<PatternConstructorNode>();
    // attempting to match non-constructor to constructor pattern: need to specify
    if (ctor_cand == nullptr) {
      return MatchResult::kUnspecified;
    }

    // check that constructors match
    if (!op->constructor.same_as(ctor_cand->constructor)) {
      return MatchResult::kClash;
    }

    // now check that subpatterns match
    for (size_t i = 0; i < op->patterns.size(); i++) {
      MatchResult submatch = this->Check(op->patterns[i], ctor_cand->patterns[i]);
      if (submatch != MatchResult::kMatch) {
        return submatch;
      }
    }
    return MatchResult::kMatch;
  }

  // wildcard and var patterns always match
  MatchResult VisitPattern_(const PatternWildcardNode*, const Pattern&) override {
    return MatchResult::kMatch;
  }

  MatchResult VisitPattern_(const PatternVarNode*, const Pattern&) override {
    return MatchResult::kMatch;
  }
};

// Returns list of arrays corresponding to Cartesian product of input list
std::deque<Array<Pattern>> CartesianProduct(std::deque<Array<Pattern>>* fields) {
  CHECK(!fields->empty());
  Array<Pattern> field_vals = fields->back();
  fields->pop_back();
  std::deque<Array<Pattern>> ret;

  // base case: this is the last field left
  if (fields->empty()) {
    for (auto val : field_vals) {
      ret.push_back(Array<Pattern>{val});
    }
    return ret;
  }

  // if we have more fields left, get the sub-candidates by getting
  // their cartesian product and appending the elements here onto those
  std::deque<Array<Pattern>> candidates = CartesianProduct(fields);
  for (auto val : field_vals) {
    for (auto candidate : candidates) {
      // make a copy because we will mutate
      Array<Pattern> new_candidate = Array<Pattern>(candidate);
      new_candidate.push_back(val);
      ret.push_back(candidate);
    }
  }
  return ret;
}

// Expands all wildcards in the candidate pattern once, using the global type var
// to decide which constructors to insert. Returns a list of all possible expansions.
Array<Pattern> ExpandWildcards(const Pattern& cand, const GlobalTypeVar& gtv, const Module& mod) {
  auto ctor_cand = cand.as<PatternConstructorNode>();

  // for a wildcard node, create constructor nodes with wildcards
  // for all args
  if (!ctor_cand) {
    TypeData td = mod->LookupDef(gtv);
    // for each constructor add a candidate
    Array<Pattern> ret;
    for (auto constructor : td->constructors) {
      Array<Pattern> args;
      for (auto inp : constructor->inputs) {
        args.push_back(PatternWildcardNode::make());
      }
      ret.push_back(PatternConstructorNode::make(constructor, args));
    }
    return ret;
  }

  // for constructors, we will expand the wildcards in any field
  // that is an ADT
  std::deque<Array<Pattern>> values_by_field;
  for (size_t i = 0; i < ctor_cand->constructor->inputs.size(); i++) {
    auto type_call = ctor_cand->constructor->inputs[i].as<TypeCallNode>();
    // for non-ADT fields, we can only have a wildcard for the value
    if (!type_call) {
      values_by_field.push_back(Array<Pattern>{PatternWildcardNode::make()});
    }
    // otherwise, recursively expand
    auto nested_gtv = Downcast<GlobalTypeVar>(type_call->func);
    values_by_field.push_back(ExpandWildcards(ctor_cand->patterns[i], nested_gtv, mod));
  }

  // generate new candidates using a cartesian product
  auto all_subfields = CartesianProduct(&values_by_field);
  Array<Pattern> ret;
  for (auto subfields : all_subfields) {
    ret.push_back(PatternConstructorNode::make(ctor_cand->constructor, subfields));
  }
  return ret;
}

/*!
 * \brief Tests whether all match expressions in the given program
 * are exhaustive.
 * \return Returns a list of cases that are not handled by the match
 * expression.
 */
Array<Pattern> CheckMatchExhaustion(const Match& match, const Module& mod) {
  /* algorithm:
   * candidates = { Wildcard }
   * while candidates not empty {
   *   cand = candidates.pop()
   *   for clause in clauses {
   *     if clause matches candidate: next candidate
   *     if candidate is not specific enough:
   *        candidates += expand_possible_wildcards(cand)
   *        continue
   *   }
   *   failed_candidates += { cand }
   * }
   * return failed_candidates
   */
  std::stack<Pattern> candidates;
  candidates.push(PatternWildcardNode::make());
  CandidateChecker checker;

  Array<Pattern> failures;

  while (!candidates.empty()) {
    Pattern cand = candidates.top();
    candidates.pop();
    GlobalTypeVar gtv = GlobalTypeVar(nullptr);
    bool failure = true;
    for (auto clause : match->clauses) {
      // if the check succeeds, then this candidate can be eliminated
      MatchResult check = checker.Check(clause->lhs, cand);
      if (check == MatchResult::kClash) {
        continue;
      }

      failure = false;

      // match was not specific enough: need to expand wildcards
      if (check == MatchResult::kUnspecified) {
        // only a constructor pattern can fail to match so this will give
        // us a global type var to use
        auto ctor_pat = Downcast<PatternConstructor>(clause->lhs);
        gtv = ctor_pat->constructor->belong_to;
        auto new_candidates = ExpandWildcards(cand, gtv, mod);
        for (auto candidate : new_candidates) {
          candidates.push(candidate);
        }
      }
      break;
    }

    if (failure) {
      failures.push_back(cand);
    }
  }

  return failures;
}

TVM_REGISTER_API("relay._ir_pass.check_match_exhaustion")
.set_body_typed<Array<Pattern>(const Match&, const Module&)>
([]
 (const Match& match, const Module& mod_ref) {
    return CheckMatchExhaustion(match, mod_ref);
  });
}  // namespace relay
}  // namespace tvm
