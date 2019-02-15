/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/pattern_functor.cc
 * \brief Implementations of visitors and mutators for ADT patterns.
 */

#include <tvm/relay/pattern_functor.h>

namespace tvm {
namespace relay {

Pattern PatternMutator::Mutate(const Pattern& pat) {
  return (*this)(pat);
}

Pattern PatternMutator::VisitPattern_(const PatternWildcardNode* op) {
  return GetRef<Pattern>(op);
}

Pattern PatternMutator::VisitPattern_(const PatternVarNode* op) {
  return PatternVarNode::make(VisitVar(op->var));
}

Pattern PatternMutator::VisitPattern_(const PatternConstructorNode* op) {
  std::vector<Pattern> pat;
  for (const auto& p : op->patterns) {
    pat.push_back(VisitPattern(p));
  }
  return PatternConstructorNode::make(VisitConstructor(op->constructor), pat);
}

Type PatternMutator::VisitType(const Type& t) {
  return t;
}

Var PatternMutator::VisitVar(const Var& v) {
  if (var_map_.count(v) == 0) {
    var_map_.insert(std::pair<Var, Var>(v,
                                        VarNode::make(v->name_hint(),
                                                      VisitType(v->type_annotation))));
  }
  return var_map_.at(v);
}

Constructor PatternMutator::VisitConstructor(const Constructor& v) {
  return v;
}

void PatternVisitor::VisitPattern_(const PatternWildcardNode* op) { }

void PatternVisitor::VisitPattern_(const PatternVarNode* op) {
  VisitVar(op->var);
}

void PatternVisitor::VisitPattern_(const PatternConstructorNode* op) {
  VisitConstructor(op->constructor);
  for (const auto& p : op->patterns) {
    VisitPattern(p);
  }
}

void PatternVisitor::VisitType(const Type& t) { }

void PatternVisitor::VisitVar(const Var& v) {
  VisitType(v->type_annotation);
}

void PatternVisitor::VisitConstructor(const Constructor& c) {
  for (const auto& inp : c->inputs) {
    VisitType(inp);
  }
}

}  // namespace relay
}  // namespace tvm
