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
 * \file span.cc
 * \brief The span data structure.
 */
#include <tvm/ir/span.h>
#include <tvm/runtime/registry.h>

namespace tvm {

ObjectPtr<Object> GetSourceNameNode(const std::string& name) {
  // always return pointer as the reference can change as map re-allocate.
  // or use another level of indirection by creating a unique_ptr
  static std::unordered_map<std::string, ObjectPtr<SourceNameNode> > source_map;

  auto sn = source_map.find(name);
  if (sn == source_map.end()) {
    ObjectPtr<SourceNameNode> n = make_object<SourceNameNode>();
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

TVM_REGISTER_GLOBAL("ir.SourceName")
.set_body_typed(SourceName::Get);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SourceNameNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const SourceNameNode*>(ref.get());
    p->stream << "SourceName(" << node->name << ", " << node << ")";
  });

TVM_REGISTER_NODE_TYPE(SourceNameNode)
.set_creator(GetSourceNameNode)
.set_repr_bytes([](const Object* n) {
    return static_cast<const SourceNameNode*>(n)->name;
  });

Span SpanNode::make(SourceName source, int lineno, int col_offset) {
  auto n = make_object<SpanNode>();
  n->source = std::move(source);
  n->lineno = lineno;
  n->col_offset = col_offset;
  return Span(n);
}

TVM_REGISTER_NODE_TYPE(SpanNode);

TVM_REGISTER_GLOBAL("ir.Span")
.set_body_typed(SpanNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<SpanNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const SpanNode*>(ref.get());
    p->stream << "Span(" << node->source << ", " << node->lineno << ", "
              << node->col_offset << ")";
  });
}  // namespace tvm
