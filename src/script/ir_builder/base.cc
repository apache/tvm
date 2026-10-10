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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/script/ir_builder/base.h>

#include <algorithm>
#include <utility>

namespace tvm {
namespace script {
namespace ir_builder {

namespace {

bool PositionLessEqual(int lhs_line, int lhs_column, int rhs_line, int rhs_column) {
  return lhs_line < rhs_line || (lhs_line == rhs_line && lhs_column <= rhs_column);
}

bool Contains(const Location& outer, const Location& inner) {
  const auto* outer_source = outer.as<SourceLocNode>();
  const auto* inner_source = inner.as<SourceLocNode>();
  if (!outer_source || !inner_source ||
      !outer_source->source_name.same_as(inner_source->source_name)) {
    return false;
  }
  return PositionLessEqual(outer_source->start_line, outer_source->start_column,
                           inner_source->start_line, inner_source->start_column) &&
         PositionLessEqual(inner_source->end_line, inner_source->end_column, outer_source->end_line,
                           outer_source->end_column);
}

bool SameLocation(const Location& lhs, const Location& rhs) {
  return Contains(lhs, rhs) && Contains(rhs, lhs);
}

void AppendNormalizedLoc(const Location& loc, std::vector<Location>* normalized) {
  if (loc.as<UnknownLocNode>()) {
    return;
  }
  if (const auto* call_site = loc.as<CallSiteLocNode>()) {
    // Stored node/frame context can repeat the active caller prefix.  Merge
    // overlapping chains before appending their distinct definition locations.
    std::vector<Location> nested;
    AppendNormalizedLoc(call_site->caller, &nested);
    AppendNormalizedLoc(call_site->callee, &nested);
    size_t common_prefix = 0;
    while (common_prefix < normalized->size() && common_prefix < nested.size() &&
           SameLocation((*normalized)[common_prefix], nested[common_prefix])) {
      ++common_prefix;
    }
    size_t overlap = std::min(normalized->size(), nested.size());
    for (; overlap > 0; --overlap) {
      bool matches = true;
      for (size_t i = 0; i < overlap; ++i) {
        if (!SameLocation((*normalized)[normalized->size() - overlap + i], nested[i])) {
          matches = false;
          break;
        }
      }
      if (matches) {
        break;
      }
    }
    for (size_t i = std::max(overlap, common_prefix); i < nested.size(); ++i) {
      AppendNormalizedLoc(nested[i], normalized);
    }
    return;
  }
  if (!normalized->empty() && Contains(normalized->back(), loc)) {
    normalized->back() = loc;
  } else if (normalized->empty() || !Contains(loc, normalized->back())) {
    normalized->push_back(loc);
  }
}

Location NormalizedLoc(const std::vector<Location>& normalized) {
  if (normalized.empty()) {
    return Location();
  }
  Location loc = normalized[0];
  for (size_t i = 1; i < normalized.size(); ++i) {
    loc = CallSiteLoc(normalized[i], loc);
  }
  return loc;
}

Location ComposeLoc(const Location& active, const Location& existing) {
  std::vector<Location> normalized;
  AppendNormalizedLoc(active, &normalized);
  // A node constructed under a single caller can acquire its explicit local
  // location later. Treat that existing caller as a shared prefix, just as
  // AppendNormalizedLoc does for an existing CallSiteLoc.
  if (!normalized.empty() && SameLocation(normalized.front(), existing)) {
    return NormalizedLoc(normalized);
  }
  AppendNormalizedLoc(existing, &normalized);
  return NormalizedLoc(normalized);
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  IRBuilderFrameNode::RegisterReflection();
  IRBuilderNode::RegisterReflection();
}

IRBuilderFrameNode::IRBuilderFrameNode() {
  if (IRBuilder::IsInScope()) loc = IRBuilder::Current()->GetCurrentLoc();
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
    TVM_FFI_THROW(InternalError) << "ValueError: No frames in Builder to add callback";
  }
  IRBuilder::Current()->frames.back()->callbacks.push_back(callback);
}

IRBuilder::IRBuilder() {
  ffi::ObjectPtr<IRBuilderNode> n = ffi::make_object<IRBuilderNode>();
  n->frames.clear();
  n->result = std::nullopt;
  n->locs.clear();
  data_ = n;
}

void IRBuilderNode::PushLoc(Location loc) { locs.push_back(std::move(loc)); }

void IRBuilderNode::PopLoc() {
  TVM_FFI_CHECK(!locs.empty(), ValueError)
      << "ValueError: No source location exists in the builder scope";
  locs.pop_back();
}

Location IRBuilderNode::GetCurrentLoc(Location location) const {
  std::vector<Location> normalized;
  normalized.reserve(locs.size());
  for (const Location& loc : locs) {
    AppendNormalizedLoc(loc, &normalized);
  }
  AppendNormalizedLoc(location, &normalized);
  return NormalizedLoc(normalized);
}

ffi::ObjectRef IRBuilderNode::SetCurrentLoc(ffi::ObjectRef obj) const {
  return SetLoc(std::move(obj), Location());
}

ffi::ObjectRef IRBuilderNode::SetLoc(ffi::ObjectRef obj, Location loc) const {
  loc = ComposeLoc(GetCurrentLoc(), loc);
  if (!loc.as<UnknownLocNode>() && obj.defined()) {
    if (Location* target = details::LocationAccessor::vtable()(obj)) {
      *target = ComposeLoc(loc, *target);
    }
  }
  return obj;
}

std::vector<IRBuilder>* ThreadLocalBuilderStack() {
  thread_local std::vector<IRBuilder> stack;
  return &stack;
}

void IRBuilder::EnterWithScope() {
  IRBuilderNode* n = this->get();
  TVM_FFI_CHECK(n->frames.empty(), ValueError)
      << "ValueError: There are frame(s) left in the builder: " << n->frames.size()
      << ". Please use a fresh new builder every time building IRs";
  TVM_FFI_CHECK(n->locs.empty(), ValueError)
      << "ValueError: There are source loc(s) left in the builder: " << n->locs.size()
      << ". Please use a fresh new builder every time building IRs";
  n->result = std::nullopt;
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  stack->push_back(*this);
}

void IRBuilder::ExitWithScope() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  TVM_FFI_ICHECK(!stack->empty());
  stack->pop_back();
}

IRBuilder IRBuilder::Current() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  TVM_FFI_CHECK(!stack->empty(), ValueError) << "ValueError: No builder in current scope";
  return stack->back();
}

bool IRBuilder::IsInScope() {
  std::vector<IRBuilder>* stack = ThreadLocalBuilderStack();
  return !stack->empty();
}

namespace details {

LocationAccessor::FType& LocationAccessor::vtable() {
  static FType inst;
  return inst;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  LocationAccessor::vtable()
      .SetDispatch<ffi::Object>([](const ffi::ObjectRef&) -> Location* { return nullptr; })
      .SetDispatch<ExprNode>(
          [](const ffi::ObjectRef& obj) -> Location* { return &obj.as<ExprNode>()->loc; })
      .SetDispatch<IRBuilderFrameNode>([](const ffi::ObjectRef& obj) -> Location* {
        return &obj.as<IRBuilderFrameNode>()->loc;
      });
}

Namer::FType& Namer::vtable() {
  static FType inst;
  return inst;
}

void Namer::Name(ffi::ObjectRef node, ffi::String name) {
  static const FType& f = vtable();
  TVM_FFI_CHECK(node.defined(), ValueError) << "ValueError: Cannot name nullptr with: " << name;
  TVM_FFI_CHECK(f.CanDispatch(node), ValueError)
      << "ValueError: Do not know how to name type \"" << node->GetTypeKey() << "\"";
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
      .def_method("script.ir_builder.IRBuilderGet", &IRBuilderNode::Get<ffi::ObjectRef>)
      .def_method("script.ir_builder.IRBuilderPushLoc", &IRBuilderNode::PushLoc)
      .def_method("script.ir_builder.IRBuilderPopLoc", &IRBuilderNode::PopLoc)
      .def_method("script.ir_builder.IRBuilderSetCurrentLoc", &IRBuilderNode::SetCurrentLoc)
      .def_method("script.ir_builder.IRBuilderSetLoc", &IRBuilderNode::SetLoc)
      .def("script.ir_builder.IRBuilderName", IRBuilder::Name<ffi::ObjectRef>);
}

}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
