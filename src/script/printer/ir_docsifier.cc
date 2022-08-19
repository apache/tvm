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
}  // namespace printer
}  // namespace script
}  // namespace tvm
