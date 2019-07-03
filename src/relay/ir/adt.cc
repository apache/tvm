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
 * \file src/tvm/ir/adt.cc
 * \brief AST nodes for Relay algebraic data types (ADTs).
 */
#include <tvm/relay/type.h>
#include <tvm/relay/adt.h>

namespace tvm {
namespace relay {

PatternWildcard PatternWildcardNode::make() {
  NodePtr<PatternWildcardNode> n = make_node<PatternWildcardNode>();
  return PatternWildcard(n);
}

TVM_REGISTER_NODE_TYPE(PatternWildcardNode);

TVM_REGISTER_API("relay._make.PatternWildcard")
.set_body_typed(PatternWildcardNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PatternWildcardNode>([](const PatternWildcardNode* node,
                                      tvm::IRPrinter* p) {
  p->stream << "PatternWildcardNode()";
});

PatternVar PatternVarNode::make(tvm::relay::Var var) {
  NodePtr<PatternVarNode> n = make_node<PatternVarNode>();
  n->var = std::move(var);
  return PatternVar(n);
}

TVM_REGISTER_NODE_TYPE(PatternVarNode);

TVM_REGISTER_API("relay._make.PatternVar")
.set_body_typed(PatternVarNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PatternVarNode>([](const PatternVarNode* node,
                                 tvm::IRPrinter* p) {
  p->stream << "PatternVarNode(" << node->var << ")";
});

PatternConstructor PatternConstructorNode::make(Constructor constructor,
                                                tvm::Array<Pattern> patterns) {
  NodePtr<PatternConstructorNode> n = make_node<PatternConstructorNode>();
  n->constructor = std::move(constructor);
  n->patterns = std::move(patterns);
  return PatternConstructor(n);
}

TVM_REGISTER_NODE_TYPE(PatternConstructorNode);

TVM_REGISTER_API("relay._make.PatternConstructor")
.set_body_typed(PatternConstructorNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<PatternConstructorNode>([](const PatternConstructorNode* node,
                                         tvm::IRPrinter* p) {
  p->stream << "PatternConstructorNode(" << node->constructor
            << ", " << node->patterns << ")";
});

Constructor ConstructorNode::make(std::string name_hint,
                                  tvm::Array<Type> inputs,
                                  GlobalTypeVar belong_to) {
  NodePtr<ConstructorNode> n = make_node<ConstructorNode>();
  n->name_hint = std::move(name_hint);
  n->inputs = std::move(inputs);
  n->belong_to = std::move(belong_to);
  return Constructor(n);
}

TVM_REGISTER_NODE_TYPE(ConstructorNode);

TVM_REGISTER_API("relay._make.Constructor")
.set_body_typed(ConstructorNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ConstructorNode>([](const ConstructorNode* node,
                                  tvm::IRPrinter* p) {
  p->stream << "ConstructorNode(" << node->name_hint << ", "
            << node->inputs << ", " << node->belong_to << ")";
});

TypeData TypeDataNode::make(GlobalTypeVar header,
                            tvm::Array<TypeVar> type_vars,
                            tvm::Array<Constructor> constructors) {
  NodePtr<TypeDataNode> n = make_node<TypeDataNode>();
  n->header = std::move(header);
  n->type_vars = std::move(type_vars);
  n->constructors = std::move(constructors);
  return TypeData(n);
}

TVM_REGISTER_NODE_TYPE(TypeDataNode);

TVM_REGISTER_API("relay._make.TypeData")
.set_body_typed(TypeDataNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TypeDataNode>([](const TypeDataNode* node,
                               tvm::IRPrinter* p) {
  p->stream << "TypeDataNode(" << node->header << ", " << node->type_vars << ", "
            << node->constructors << ")";
});

Clause ClauseNode::make(Pattern lhs, Expr rhs) {
  NodePtr<ClauseNode> n = make_node<ClauseNode>();
  n->lhs = std::move(lhs);
  n->rhs = std::move(rhs);
  return Clause(n);
}

TVM_REGISTER_NODE_TYPE(ClauseNode);

TVM_REGISTER_API("relay._make.Clause")
.set_body_typed(ClauseNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ClauseNode>([](const ClauseNode* node,
                             tvm::IRPrinter* p) {
  p->stream << "ClauseNode(" << node->lhs << ", "
            << node->rhs << ")";
  });

Match MatchNode::make(Expr data, tvm::Array<Clause> clauses) {
  NodePtr<MatchNode> n = make_node<MatchNode>();
  n->data = std::move(data);
  n->clauses = std::move(clauses);
  return Match(n);
}

TVM_REGISTER_NODE_TYPE(MatchNode);

TVM_REGISTER_API("relay._make.Match")
.set_body_typed(MatchNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<MatchNode>([](const MatchNode* node,
                            tvm::IRPrinter* p) {
  p->stream << "MatchNode(" << node->data << ", "
            << node->clauses << ")";
});

}  // namespace relay
}  // namespace tvm
