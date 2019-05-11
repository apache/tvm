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
 *  Copyright (c) 2018 by Contributors
 * \file base.cc
 * \brief The core base types for Relay.
 */
#include <tvm/api_registry.h>
#include <tvm/relay/base.h>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

NodePtr<SourceNameNode> GetSourceNameNode(const std::string& name) {
  // always return pointer as the reference can change as map re-allocate.
  // or use another level of indirection by creating a unique_ptr
  static std::unordered_map<std::string, NodePtr<SourceNameNode> > source_map;

  auto sn = source_map.find(name);
  if (sn == source_map.end()) {
    NodePtr<SourceNameNode> n = make_node<SourceNameNode>();
    source_map[name] = n;
    n->name = std::move(name);
    return n;
  } else {
    return sn->second;
  }
}

SourceName SourceName::Get(const std::string& name) {
  return SourceName(GetSourceNameNode(name));
}

TVM_REGISTER_API("relay._make.SourceName")
.set_body_typed(SourceName::Get);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SourceNameNode>([](const SourceNameNode* node, tvm::IRPrinter* p) {
    p->stream << "SourceName(" << node->name << ", " << node << ")";
  });

TVM_REGISTER_NODE_TYPE(SourceNameNode)
.set_creator(GetSourceNameNode)
.set_global_key([](const Node* n) {
    return static_cast<const SourceNameNode*>(n)->name;
  });

Span SpanNode::make(SourceName source, int lineno, int col_offset) {
  auto n = make_node<SpanNode>();
  n->source = std::move(source);
  n->lineno = lineno;
  n->col_offset = col_offset;
  return Span(n);
}

TVM_REGISTER_NODE_TYPE(SpanNode);

TVM_REGISTER_API("relay._make.Span")
.set_body_typed(SpanNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SpanNode>([](const SpanNode* node, tvm::IRPrinter* p) {
    p->stream << "SpanNode(" << node->source << ", " << node->lineno << ", "
              << node->col_offset << ")";
  });

TVM_REGISTER_NODE_TYPE(IdNode);

TVM_REGISTER_API("relay._base.set_span")
.set_body_typed<void(NodeRef, Span)>([](NodeRef node_ref, Span sp) {
    auto rn = node_ref.as_derived<RelayNode>();
    CHECK(rn);
    rn->span = sp;
});

}  // namespace relay
}  // namespace tvm
