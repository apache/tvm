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
 * \file match_exhaustion.cc
 * \brief Checking Relay match expression exhaustiveness.
 *
 * This file implements a function that checks whether a match
 * expression is exhaustive, that is, whether a given match clause
 * matches every possible case. This is important for ensuring
 * code correctness, since hitting an unmatched case results in a
 * dynamic error unless exhaustiveness is checked in advance.
 */
#include <tvm/relay/adt.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>

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
    ICHECK_EQ(op->patterns.size(), ctor_cand->patterns.size());
    bool unspecified = false;
    for (size_t i = 0; i < op->patterns.size(); i++) {
      MatchResult submatch = this->Check(op->patterns[i], ctor_cand->patterns[i]);
      // if we have a clash anywhere, then we can return clash
      if (submatch == MatchResult::kClash) {
        return MatchResult::kClash;
      }
      if (submatch == MatchResult::kUnspecified) {
        unspecified = true;
      }
    }
    // only return unspecified if we have ruled out a clash
    if (unspecified) {
      return MatchResult::kUnspecified;
    }
    return MatchResult::kMatch;
  }

  MatchResult VisitPattern_(const PatternTupleNode* op, const Pattern& cand) override {
    auto* tuple_cand = cand.as<PatternTupleNode>();
    // attempting to match non-tuple to constructor pattern: need to specify
    if (tuple_cand == nullptr) {
      return MatchResult::kUnspecified;
    }

    // now check that subpatterns match
    ICHECK_EQ(op->patterns.size(), tuple_cand->patterns.size());
    bool unspecified = false;
    for (size_t i = 0; i < op->patterns.size(); i++) {
      MatchResult submatch = this->Check(op->patterns[i], tuple_cand->patterns[i]);
      // if we have a clash anywhere, then we can return clash
      if (submatch == MatchResult::kClash) {
        return MatchResult::kClash;
      }
      if (submatch == MatchResult::kUnspecified) {
        unspecified = true;
      }
    }
    // only return unspecified if we have ruled out a clash
    if (unspecified) {
      return MatchResult::kUnspecified;
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

// Returns list of arrays corresponding to Cartesian product of input list.
// Note: CartesianProduct({}) = {{}}
Array<Array<Pattern>> CartesianProduct(Array<Array<Pattern>> fields) {
  // the only combination of 0 fields is 0 fields
  if (fields.size() == 0) {
    return {{}};
  }

  Array<Pattern> field_vals = fields[fields.size() - 1];
  Array<Array<Pattern>> ret;

  // base case: this is the last field left
  if (fields.size() == 1) {
    for (auto val : field_vals) {
      ret.push_back(Array<Pattern>{val});
    }
    return ret;
  }

  // if we have more fields left, get the sub-candidates by getting
  // their cartesian product and appending the elements here onto those
  Array<Array<Pattern>> remaining_fields;
  for (size_t i = 0; i < fields.size() - 1; i++) {
    remaining_fields.push_back(fields[i]);
  }
  Array<Array<Pattern>> candidates = CartesianProduct(remaining_fields);
  for (auto val : field_vals) {
    for (auto candidate : candidates) {
      candidate.push_back(val);
      ret.push_back(candidate);
    }
  }
  return ret;
}

Array<Pattern> ExpandWildcardsConstructor(const PatternConstructor& clause_ctor,
                                          const Pattern& cand, const IRModule& mod);

Array<Pattern> ExpandWildcardsTuple(const PatternTuple& clause_tuple, const Pattern& cand,
                                    const IRModule& mod);

// Expands all wildcards in the candidate pattern once
// Returns a list of all possible expansions.
Array<Pattern> ExpandWildcards(const Pattern& clause_pat, const Pattern& cand,
                               const IRModule& mod) {
  if (auto clause_ctor = clause_pat.as<PatternConstructor>()) {
    return ExpandWildcardsConstructor(clause_ctor.value(), cand, mod);
  } else if (auto clause_tup = clause_pat.as<PatternTuple>()) {
    return ExpandWildcardsTuple(clause_tup.value(), cand, mod);
  } else {
    return {cand};
  }
}

// Expands all wildcards in the candidate pattern once.
// Use the pattern to decide which constructors to insert.
// Returns a list of all possible expansions.
Array<Pattern> ExpandWildcardsConstructor(const PatternConstructor& clause_ctor,
                                          const Pattern& cand, const IRModule& mod) {
  auto gtv = Downcast<GlobalTypeVar>(clause_ctor->constructor->belong_to);

  // for a wildcard node, create constructor nodes with wildcards for all args.
  if (cand.as<PatternWildcardNode>()) {
    TypeData td = mod->LookupTypeDef(gtv);
    // for each constructor add a candidate.
    Array<Pattern> ret;
    for (auto constructor : td->constructors) {
      Array<Pattern> args;
      for (auto inp : constructor->inputs) {
        args.push_back(PatternWildcard());
      }
      ret.push_back(PatternConstructor(constructor, args));
    }
    return ret;
  }

  auto ctor_cand = Downcast<PatternConstructor>(cand);

  // expand all fields' wildcards
  Array<Array<Pattern>> values_by_field;
  for (size_t i = 0; i < ctor_cand->constructor->inputs.size(); i++) {
    values_by_field.push_back(
        ExpandWildcards(clause_ctor->patterns[i], ctor_cand->patterns[i], mod));
  }

  // generate new candidates using a cartesian product.
  auto all_subfields = CartesianProduct(values_by_field);
  Array<Pattern> ret;
  for (auto subfields : all_subfields) {
    ret.push_back(PatternConstructor(ctor_cand->constructor, subfields));
  }
  return ret;
}

// Expands all wildcards in the candidate pattern once.
// Returns a list of all possible expansions.
Array<Pattern> ExpandWildcardsTuple(const PatternTuple& clause_tuple, const Pattern& cand,
                                    const IRModule& mod) {
  // for a wildcard node, create tuple with wildcards for all args.
  if (cand.as<PatternWildcardNode>()) {
    Array<Pattern> args;
    for (auto inp : clause_tuple->patterns) {
      args.push_back(PatternWildcard());
    }
    return {PatternTuple(args)};
  }

  auto tuple_cand = Downcast<PatternTuple>(cand);

  // expand all members' patterns
  Array<Array<Pattern>> values_by_field;
  for (size_t i = 0; i < tuple_cand->patterns.size(); i++) {
    values_by_field.push_back(
        ExpandWildcards(clause_tuple->patterns[i], tuple_cand->patterns[i], mod));
  }

  // generate new candidates using a cartesian product
  auto all_subfields = CartesianProduct(values_by_field);
  Array<Pattern> ret;
  for (auto subfields : all_subfields) {
    ret.push_back(PatternTuple(subfields));
  }
  return ret;
}

/*!
 * \brief Finds cases that the match expression does not catch, if any.
 * \return Returns a list of cases that are not handled by the match
 * expression.
 */
Array<Pattern> UnmatchedCases(const Match& match, const IRModule& mod) {
  /* algorithm:
   * candidates = { Wildcard }
   * while candidates not empty {
   *   cand = candidates.pop()
   *   for clause in clauses {
   *     if clause fails: next clause
   *     if clause matches candidate: next candidate
   *     if candidate is not specific enough:
   *        candidates += expand_possible_wildcards(cand)
   *        next candidate
   *   }
   *   failed_candidates += { cand }
   * }
   * return failed_candidates
   */
  std::stack<Pattern> candidates;
  candidates.push(PatternWildcard());
  CandidateChecker checker;

  Array<Pattern> failures;

  while (!candidates.empty()) {
    Pattern cand = candidates.top();
    candidates.pop();

    bool failure = true;
    for (auto clause : match->clauses) {
      // if the check fails, we move on to the next
      MatchResult check = checker.Check(clause->lhs, cand);
      if (check == MatchResult::kClash) {
        continue;
      }

      // either success or we need to generate more candidates;
      // either way, we're done with this candidate
      failure = false;
      if (check == MatchResult::kUnspecified) {
        auto new_candidates = ExpandWildcards(clause->lhs, cand, mod);
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

// expose for testing only
TVM_REGISTER_GLOBAL("relay.analysis.unmatched_cases")
    .set_body_typed([](const Match& match, const Optional<IRModule>& mod_ref) {
      IRModule call_mod = mod_ref.defined() ? mod_ref.value() : IRModule({}, {});
      return UnmatchedCases(match, call_mod);
    });

}  // namespace relay
}  // namespace tvm
