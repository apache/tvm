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
#include <tvm/runtime/container/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>

namespace tvm {
namespace script {
namespace printer {

Doc IRDocsifierNode::AsDocImpl(const TracedObject<ObjectRef>& obj) const {
  return IRDocsifier::vtable()(dispatch_tokens.back(), obj, GetRef<IRDocsifier>(this));
}

IRDocsifier::IRDocsifier(Map<String, String> ir_prefix) {
  auto n = make_object<IRDocsifierNode>();
  n->ir_prefix = std::move(ir_prefix);
  n->dispatch_tokens.push_back(kDefaultDispatchToken);
  data_ = std::move(n);
}

IRDocsifier::FType& IRDocsifier::vtable() {
  static IRDocsifier::FType inst;
  return inst;
}

RootNodeContainer::RootNodeContainer(ObjectRef root_node) {
  auto n = make_object<RootNodeContainerNode>();
  n->root_node = std::move(root_node);
  data_ = std::move(n);
}

// Add a default dispatch for the RootNodeContainer to throw error.
// To add implementation for a new IR, RootNodeContainer needs to be
// registered under the dispatch token of that IR, like:
// \code
// TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
//     .set_dispatch("relax", [](TracedObject<RootNodeContainer> obj, IRDocsifier p) {
//       const ObjectRef& root_node = obj.Get()->root_node;
//       \\ More specialized logic for your IR.
//       return p->AsDoc<Doc>(MakeTraced(root_node));
//     });
// \endcode
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch([](TracedObject<RootNodeContainer> obj, IRDocsifier p) -> Doc {
      String top_dispatch_token = p->dispatch_tokens.back();
      ICHECK_NE(top_dispatch_token, "");
      ICHECK(false) << "Printing IR " << top_dispatch_token << " is not implemented.";
      throw;
    });

TVM_REGISTER_NODE_TYPE(IRDocsifierNode);
TVM_REGISTER_GLOBAL("script.printer.IRDocsifier").set_body_typed([](Map<String, String> ir_prefix) {
  return IRDocsifier(ir_prefix);
});
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierAsDoc")
    .set_body_typed([](IRDocsifier p, ObjectRef obj, ObjectPath obj_path) {
      return p->AsDoc<Doc>(MakeTraced(obj, obj_path));
    });

TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPushDispatchToken")
    .set_body_typed([](IRDocsifier p, String token) { p->dispatch_tokens.push_back(token); });
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPopDispatchToken").set_body_typed([](IRDocsifier p) {
  p->dispatch_tokens.pop_back();
});

TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPushFrame")
    .set_body_typed([](IRDocsifier p, Frame frame) { p->frames.push_back(frame); });
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierPopFrame").set_body_typed([](IRDocsifier p) {
  p->frames.pop_back();
});

TVM_REGISTER_GLOBAL("script.printer.IRDocsifierSetDispatch")
    .set_body_typed([](String token, uint64_t type_index, runtime::PackedFunc f) {
      IRDocsifier::vtable().set_dispatch(token, type_index, std::move(f));
    });
TVM_REGISTER_GLOBAL("script.printer.IRDocsifierRemoveDispatch")
    .set_body_typed([](String token, uint64_t type_index) {
      IRDocsifier::vtable().remove_dispatch(token, type_index);
    });

TVM_REGISTER_NODE_TYPE(RootNodeContainerNode);
TVM_REGISTER_GLOBAL("script.printer.RootNodeContainer").set_body_typed([](ObjectRef root_node) {
  return RootNodeContainer(root_node);
});

}  // namespace printer
}  // namespace script
}  // namespace tvm
