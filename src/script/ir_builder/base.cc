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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/script/ir_builder/base.h>

namespace tvm {
namespace script {
namespace ir_builder {

TVM_FFI_STATIC_INIT_BLOCK() {
  IRBuilderFrameNode::RegisterReflection();
  IRBuilderNode::RegisterReflection();
}

void IRBuilderFrameNode::EnterWithScope() {
  IRBuilder::Current()->frames.push_back(ffi::GetRef<IRBuilderFrame>(this));
}

void IRBuilderFrameNode::ExitWithScope() {
  for (auto it = callbacks.rbegin(); it != callbacks.rend(); ++it) {
    (*it)();
  }
  this->callbacks.clear();
  IRBuilder::Current()->frames.pop_back();
}

void IRBuilderFrameNode::AddCallback(ffi::TypedFunction<void()> callback) {
  if (IRBuilder::Current()->frames.empty()) {
    LOG(FATAL) << "ValueError: No frames in Builder to add callback";
  }
  IRBuilder::Current()->frames.back()->callbacks.push_back(callback);
}

IRBuilder::IRBuilder() {
  ObjectPtr<IRBuilderNode> n = ffi::make_object<IRBuilderNode>();
  n->frames.clear();
  n->result = std::nullopt;
  data_ = n;
}

std::vector<IRBuilder>* ThreadLocalBuilderStack() {
  thread_local std::vector<IRBuilder> stack;
  return &stack;
}

void IRBuilder::EnterWithScope() {
  IRBuilderNode* n = this->get();
  CHECK(n->frames.empty()) << "ValueError: There are frame(s) left in the builder: "
                           << n->frames.size()
                           << ". Please use a fresh new builder every time building IRs";
  n->result = std::nullopt;
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  stack->push_back(*this);
}

void IRBuilder::ExitWithScope() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  ICHECK(!stack->empty());
  stack->pop_back();
}

IRBuilder IRBuilder::Current() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  CHECK(!stack->empty()) << "ValueError: No builder in current scope";
  return stack->back();
}

bool IRBuilder::IsInScope() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  return !stack->empty();
}

namespace details {

Namer::FType& Namer::vtable() {
  static FType inst;
  return inst;
}

void Namer::Name(ObjectRef node, ffi::String name) {
  static const FType& f = vtable();
  CHECK(node.defined()) << "ValueError: Cannot name nullptr with: " << name;
  CHECK(f.can_dispatch(node)) << "ValueError: Do not know how to name type \""
                              << node->GetTypeKey();
  f(node, name);
}

}  // namespace details

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("script.ir_builder.IRBuilderFrameEnter", &IRBuilderFrameNode::EnterWithScope)
      .def_method("script.ir_builder.IRBuilderFrameExit", &IRBuilderFrameNode::ExitWithScope)
      .def_method("script.ir_builder.IRBuilderFrameAddCallback", &IRBuilderFrameNode::AddCallback)
      .def("script.ir_builder.IRBuilder", []() { return IRBuilder(); })
      .def_method("script.ir_builder.IRBuilderEnter", &IRBuilder::EnterWithScope)
      .def_method("script.ir_builder.IRBuilderExit", &IRBuilder::ExitWithScope)
      .def("script.ir_builder.IRBuilderCurrent", IRBuilder::Current)
      .def("script.ir_builder.IRBuilderIsInScope", IRBuilder::IsInScope)
      .def_method("script.ir_builder.IRBuilderGet", &IRBuilderNode::Get<ObjectRef>)
      .def("script.ir_builder.IRBuilderName", IRBuilder::Name<ObjectRef>);
}

}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
