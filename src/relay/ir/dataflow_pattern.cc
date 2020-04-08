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
 * \file src/tvm/relay/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relay.
 */
#include <tvm/relay/dataflow_pattern.h>

namespace tvm {
namespace relay {

ExprPattern::ExprPattern(Expr expr) {
  ObjectPtr<ExprPatternNode> n = make_object<ExprPatternNode>();
  n->expr = std::move(expr);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExprPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.ExprPattern")
.set_body_typed([](Expr e) {
    return ExprPattern(e);
  });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ExprPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const ExprPatternNode*>(ref.get());
    p->Print(node->expr);
  });


VarPattern VarPatternNode::make(std::string name_hint, Type type_annotation) {
  ObjectPtr<VarPatternNode> n = make_object<VarPatternNode>();
  n->name = std::move(name_hint);
  n->type_annotation = std::move(type_annotation);
  return VarPattern(n);
}

TVM_REGISTER_NODE_TYPE(VarPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.VarPattern")
.set_body_typed(static_cast<VarPattern (*)(std::string, Type)>(VarPatternNode::make));

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<VarPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const VarPatternNode*>(ref.get());
    p->stream << "VarPattern(" << node->name_hint();
    if (node->type_annotation.defined()) {
      p->stream << ", ty=";
      p->Print(node->type_annotation);
    }
    p->stream << ")";
  });

CallPattern CallPatternNode::make(DFPattern op, Array<DFPattern> args, Attrs attrs,
                                  Array<Type> type_args) {
  ObjectPtr<CallPatternNode> n = make_object<CallPatternNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  return CallPattern(n);
}

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.CallPattern")
.set_body_typed(CallPatternNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<CallPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const CallPatternNode*>(ref.get());
  p->stream << "CallPatternNode(" << node->op << ", " << node->args << ", " << node->attrs
            << ", " << node->type_args << ")";
});

TuplePattern TuplePatternNode::make(tvm::Array<DFPattern> fields) {
  ObjectPtr<TuplePatternNode> n = make_object<TuplePatternNode>();
  n->fields = std::move(fields);
  return TuplePattern(n);
}

TVM_REGISTER_NODE_TYPE(TuplePatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.TuplePattern")
.set_body_typed(TuplePatternNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TuplePatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const TuplePatternNode*>(ref.get());
    p->stream << "TuplePattern(" << node->fields << ")";
  });

TupleGetItemPattern TupleGetItemPatternNode::make(DFPattern tuple, int index) {
  ObjectPtr<TupleGetItemPatternNode> n = make_object<TupleGetItemPatternNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  return TupleGetItemPattern(n);
}

TVM_REGISTER_NODE_TYPE(TupleGetItemPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.TupleGetItemPattern")
.set_body_typed(TupleGetItemPatternNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TupleGetItemPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const TupleGetItemPatternNode*>(ref.get());
  p->stream << "TupleGetItemPatternNode(" << node->tuple << ", " << node->index << ")";
});

AltPattern AltPatternNode::make(DFPattern left, DFPattern right) {
  ObjectPtr<AltPatternNode> n = make_object<AltPatternNode>();
  n->left = std::move(left);
  n->right = std::move(right);
  return AltPattern(n);
}

TVM_REGISTER_NODE_TYPE(AltPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.AltPattern")
.set_body_typed(AltPatternNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<AltPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const AltPatternNode*>(ref.get());
    p->stream << "AltPattern(" << node->left << " | " << node->right << ")";
  });

TVM_REGISTER_NODE_TYPE(WildcardPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.WildcardPattern")
.set_body_typed([]() {
    auto w = WildcardPattern(make_object<WildcardPatternNode>());
    return w;
  });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<WildcardPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
    p->stream << "*";
  });

TypePattern TypePatternNode::make(DFPattern pattern, Type type) {
  ObjectPtr<TypePatternNode> n = make_object<TypePatternNode>();
  n->pattern = std::move(pattern);
  n->type = std::move(type);
  return TypePattern(n);
}

TVM_REGISTER_NODE_TYPE(TypePatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.TypePattern")
.set_body_typed(TypePatternNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TypePatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const TypePatternNode*>(ref.get());
    p->stream << "TypePattern(" << node->pattern << " has type " << node->type << ")";
  });

AttrPattern AttrPatternNode::make(DFPattern pattern, Attrs attrs) {
  ObjectPtr<AttrPatternNode> n = make_object<AttrPatternNode>();
  n->pattern = std::move(pattern);
  n->attrs = std::move(attrs);
  return AttrPattern(n);
}

TVM_REGISTER_NODE_TYPE(AttrPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.AttrPattern")
.set_body_typed(AttrPatternNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<AttrPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const AttrPatternNode*>(ref.get());
    p->stream << "AttrPattern(" << node->pattern << " has attributes " << node->attrs << ")";
  });

DominatorPattern DominatorPatternNode::make(DFPattern parent, DFPattern path, DFPattern child) {
  ObjectPtr<DominatorPatternNode> n = make_object<DominatorPatternNode>();
  n->parent = std::move(parent);
  n->path = std::move(path);
  n->child = std::move(child);
  return DominatorPattern(n);
}

TVM_REGISTER_NODE_TYPE(DominatorPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.DominatorPattern").set_body_typed(DominatorPatternNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<DominatorPatternNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const DominatorPatternNode*>(ref.get());
      p->stream << "DominatorPattern(" << node->parent << ", " << node->path << ", " << node->child
                << ")";
    });

}  // namespace relay
}  // namespace tvm
