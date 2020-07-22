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
 * \file src/relay/ir/pattern_functor.cc
 * \brief Implementations of visitors and mutators for ADT patterns.
 */

#include <tvm/relay/pattern_functor.h>

namespace tvm {
namespace relay {

Pattern PatternMutator::Mutate(const Pattern& pat) { return (*this)(pat); }

Pattern PatternMutator::VisitPattern_(const PatternWildcardNode* op) { return GetRef<Pattern>(op); }

Pattern PatternMutator::VisitPattern_(const PatternVarNode* op) {
  return PatternVar(VisitVar(op->var));
}

Pattern PatternMutator::VisitPattern_(const PatternConstructorNode* op) {
  std::vector<Pattern> pat;
  for (const auto& p : op->patterns) {
    pat.push_back(VisitPattern(p));
  }
  return PatternConstructor(VisitConstructor(op->constructor), pat);
}

Pattern PatternMutator::VisitPattern_(const PatternTupleNode* op) {
  std::vector<Pattern> pat;
  for (const auto& p : op->patterns) {
    pat.push_back(VisitPattern(p));
  }
  return PatternTuple(pat);
}

Type PatternMutator::VisitType(const Type& t) { return t; }

Var PatternMutator::VisitVar(const Var& v) {
  if (var_map_.count(v) == 0) {
    var_map_.insert(std::pair<Var, Var>(v, Var(v->name_hint(), VisitType(v->type_annotation))));
  }
  return var_map_.at(v);
}

Constructor PatternMutator::VisitConstructor(const Constructor& v) { return v; }

void PatternVisitor::VisitPattern_(const PatternWildcardNode* op) {}

void PatternVisitor::VisitPattern_(const PatternVarNode* op) { VisitVar(op->var); }

void PatternVisitor::VisitPattern_(const PatternConstructorNode* op) {
  VisitConstructor(op->constructor);
  for (const auto& p : op->patterns) {
    VisitPattern(p);
  }
}

void PatternVisitor::VisitPattern_(const PatternTupleNode* op) {
  for (const auto& p : op->patterns) {
    VisitPattern(p);
  }
}

void PatternVisitor::VisitType(const Type& t) {}

void PatternVisitor::VisitVar(const Var& v) { VisitType(v->type_annotation); }

void PatternVisitor::VisitConstructor(const Constructor& c) {
  for (const auto& inp : c->inputs) {
    VisitType(inp);
  }
}

}  // namespace relay
}  // namespace tvm
