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
 * \file src/relay/ir/expr.cc
 * \brief The expression AST nodes of Relay.
 */
#include <tvm/node/repr_printer.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/registry.h>
#include <tvm/meta/expr.h>

namespace tvm {
namespace meta {

using tvm::ReprPrinter;
using namespace tvm::runtime;

VarDef::VarDef(String name, MetaIR type_info) {
  ObjectPtr<VarDefNode> n = make_object<VarDefNode>();
  n->name = std::move(name);
  n->type_info = std::move(type_info);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(VarDefNode);

TVM_REGISTER_GLOBAL("meta.VarDef").set_body_typed(
  [](String name, MetaIR type_info) {
    return VarDef(name, type_info);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<VarDefNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const VarDefNode*>(ref.get());
  CHECK(node);
  p->stream << "VarDef("
    << "name=" << node->name
    << "type_info=" << node->type_info
    << ")";
});

ObjectDef::ObjectDef(String name, String ref_name, String nmspace, MetaIR base, Array<VarDef> variables) {
  ObjectPtr<ObjectDefNode> n = make_object<ObjectDefNode>();
  n->name = std::move(name);
  n->ref_name = std::move(ref_name);
  n->nmspace = std::move(nmspace);
  n->base = std::move(base);
  n->variables = std::move(variables);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ObjectDefNode);

TVM_REGISTER_GLOBAL("meta.ObjectDef").set_body_typed(
  [](String name, String ref_name, String nmspace, MetaIR base, Array<VarDef> variables) {
    return ObjectDef(name, ref_name, nmspace, base, variables);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ObjectDefNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* n = static_cast<const ObjectDefNode*>(ref.get());
  CHECK(n);
  p->stream << "ObjectDef("
    << "name=" << n->name
    << ", ref_name=" << n->ref_name
    << ", nmspace=" << n->nmspace
    << ", base=";
  auto* pbase = static_cast<const ObjectDefNode*>(n->base.get());
  if (pbase == nullptr) {
    p->stream << "nullptr";
  } else {
    p->stream << pbase->name;
  }
  p->stream << ", variables=[";
  if (!n->variables.empty()) {
    for (auto v : n->variables) {
      p->stream << v->name;
    }
  }
  p->stream << "])";
});

}  // namespace meta
}  // namespace tvm
