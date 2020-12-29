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
 * \file src/ir/adt.cc
 * \brief AST nodes for Relay algebraic data types (ADTs).
 */
#include <tvm/relay/adt.h>
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

PatternWildcard::PatternWildcard() {
  ObjectPtr<PatternWildcardNode> n = make_object<PatternWildcardNode>();
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternWildcardNode);

TVM_REGISTER_GLOBAL("relay.ir.PatternWildcard").set_body_typed([]() { return PatternWildcard(); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PatternWildcardNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "PatternWildcardNode()";
    });

PatternVar::PatternVar(tvm::relay::Var var) {
  ObjectPtr<PatternVarNode> n = make_object<PatternVarNode>();
  n->var = std::move(var);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternVarNode);

TVM_REGISTER_GLOBAL("relay.ir.PatternVar").set_body_typed([](tvm::relay::Var var) {
  return PatternVar(var);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PatternVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PatternVarNode*>(ref.get());
      p->stream << "PatternVarNode(" << node->var << ")";
    });

PatternConstructor::PatternConstructor(Constructor constructor, tvm::Array<Pattern> patterns) {
  ObjectPtr<PatternConstructorNode> n = make_object<PatternConstructorNode>();
  n->constructor = std::move(constructor);
  n->patterns = std::move(patterns);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternConstructorNode);

TVM_REGISTER_GLOBAL("relay.ir.PatternConstructor")
    .set_body_typed([](Constructor constructor, tvm::Array<Pattern> patterns) {
      return PatternConstructor(constructor, patterns);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PatternConstructorNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PatternConstructorNode*>(ref.get());
      p->stream << "PatternConstructorNode(" << node->constructor << ", " << node->patterns << ")";
    });

PatternTuple::PatternTuple(tvm::Array<Pattern> patterns) {
  ObjectPtr<PatternTupleNode> n = make_object<PatternTupleNode>();
  n->patterns = std::move(patterns);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternTupleNode);

TVM_REGISTER_GLOBAL("relay.ir.PatternTuple").set_body_typed([](tvm::Array<Pattern> patterns) {
  return PatternTuple(patterns);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PatternTupleNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PatternTupleNode*>(ref.get());
      p->stream << "PatternTupleNode(" << node->patterns << ")";
    });

Clause::Clause(Pattern lhs, Expr rhs) {
  ObjectPtr<ClauseNode> n = make_object<ClauseNode>();
  n->lhs = std::move(lhs);
  n->rhs = std::move(rhs);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ClauseNode);

TVM_REGISTER_GLOBAL("relay.ir.Clause").set_body_typed([](Pattern lhs, Expr rhs) {
  return Clause(lhs, rhs);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ClauseNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ClauseNode*>(ref.get());
      p->stream << "ClauseNode(" << node->lhs << ", " << node->rhs << ")";
    });

Match::Match(Expr data, tvm::Array<Clause> clauses, bool complete, Span span) {
  ObjectPtr<MatchNode> n = make_object<MatchNode>();
  n->data = std::move(data);
  n->clauses = std::move(clauses);
  n->complete = complete;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(MatchNode);

TVM_REGISTER_GLOBAL("relay.ir.Match")
    .set_body_typed([](Expr data, tvm::Array<Clause> clauses, bool complete) {
      return Match(data, clauses, complete);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MatchNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const MatchNode*>(ref.get());
      p->stream << "MatchNode(" << node->data << ", " << node->clauses << ", " << node->complete
                << ")";
    });

}  // namespace relay
}  // namespace tvm
