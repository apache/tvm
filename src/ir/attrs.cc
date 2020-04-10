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
 * \file attrs.cc
 */
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include "attr_functor.h"

namespace tvm {

void DictAttrsNode::VisitAttrs(AttrVisitor* v)  {
  v->Visit("__dict__", &dict);
}

void DictAttrsNode::VisitNonDefaultAttrs(AttrVisitor* v) {
  v->Visit("__dict__", &dict);
}

void DictAttrsNode::InitByPackedArgs(
    const runtime::TVMArgs& args, bool allow_unknown) {
  for (int i = 0; i < args.size(); i += 2) {
    std::string key = args[i];
    runtime::TVMArgValue val = args[i + 1];
    if (val.IsObjectRef<ObjectRef>()) {
      dict.Set(key, val.operator ObjectRef());
    } else if (val.type_code() == kTVMStr) {
      dict.Set(key, val.operator String());
    } else {
      dict.Set(key, val.operator PrimExpr());
    }
  }
}

Array<AttrFieldInfo> DictAttrsNode::ListFieldInfo() const {
  return {};
}

DictAttrs::DictAttrs(Map<std::string, ObjectRef> dict) {
  ObjectPtr<DictAttrsNode> n = make_object<DictAttrsNode>();
  n->dict = std::move(dict);
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<DictAttrsNode>([](const ObjectRef& node, ReprPrinter* p) {
  auto* op = static_cast<const DictAttrsNode*>(node.get());
  p->stream << op->dict;
});

TVM_REGISTER_NODE_TYPE(DictAttrsNode);

TVM_REGISTER_NODE_TYPE(AttrFieldInfoNode);

TVM_REGISTER_GLOBAL("ir.DictAttrsGetDict")
.set_body_typed([](DictAttrs attrs) {
  return attrs->dict;
});

TVM_REGISTER_GLOBAL("ir.AttrsListFieldInfo")
.set_body_typed([](Attrs attrs) {
  return attrs->ListFieldInfo();
});

}  // namespace tvm
