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

/*
 * Axis definitions, attributes, fusers, and registrations.
 */
#include <tvm/ffi/reflection/enum_def.h>

#include "utils.h"

namespace tvm {
namespace tirx {
namespace refl = tvm::ffi::reflection;

/**************** Axis ****************/
ffi::ObjectPtr<ffi::Object> CreateAxis(const std::string& name) {
  auto axis = Axis::Get(name);
  return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<ffi::Object>(axis);
}

const ffi::EnumState& AxisEnumState() {
  static refl::TypeAttrColumn state_column(refl::type_attr::kEnumState);
  static ffi::EnumState state =
      ffi::Any(state_column[AxisNode::_GetOrAllocRuntimeTypeIndex()]).as_or_throw<ffi::EnumState>();
  return state;
}

template <typename T>
ffi::Optional<T> AxisAttr(const AxisNode* axis, const char* key) {
  const ffi::EnumState& state = AxisEnumState();
  auto column = state->attrs.Get(ffi::String(key));
  if (!column) return std::nullopt;
  auto value = column.value().Get(ffi::GetRef<Axis>(axis));
  if (!value) return std::nullopt;
  return value.value().cast<T>();
}

bool AxisNode::IsThreadAxis() const {
  auto thread = AxisAttr<bool>(this, "thread");
  TVM_FFI_ICHECK(thread.has_value()) << "Axis '" << _str_index << "' has no thread classification";
  return thread.value();
}

bool AxisNode::IsMemoryAxis() const {
  auto thread = AxisAttr<bool>(this, "thread");
  TVM_FFI_ICHECK(thread.has_value()) << "Axis '" << _str_index << "' has no thread classification";
  return !thread.value();
}

ffi::Optional<ffi::String> AxisNode::GetBackend() const {
  return AxisAttr<ffi::String>(this, "backend");
}

ffi::Optional<ExecScope> AxisNode::GetScope() const { return AxisAttr<ExecScope>(this, "scope"); }

ffi::Optional<ExecScope> AxisNode::GetSubscope() const {
  return AxisAttr<ExecScope>(this, "subscope");
}

ffi::Optional<FAxisFuser> AxisNode::GetFuser() const { return AxisAttr<FAxisFuser>(this, "fuser"); }

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::ObjectDef<AxisNode>(refl::init(false));
  refl::TypeAttrDef<AxisNode>()
      .def(tvm::ffi::reflection::type_attr::kDataToJson,
           [](const AxisNode* node) -> ffi::String { return node->_str_index; })
      .def(tvm::ffi::reflection::type_attr::kDataFromJson,
           [](const ffi::String& name) -> Axis { return Axis::Get(name); });

  refl::GlobalDef()
      .def("tirx.AxisIsThreadAxis", [](Axis axis) { return axis->IsThreadAxis(); })
      .def("tirx.AxisIsMemoryAxis", [](Axis axis) { return axis->IsMemoryAxis(); })
      .def("tirx.AxisGetBackend", [](Axis axis) { return axis->GetBackend(); })
      .def("tirx.AxisGetScope", [](Axis axis) { return axis->GetScope(); })
      .def("tirx.AxisGetSubscope", [](Axis axis) { return axis->GetSubscope(); })
      .def("tirx.AxisGet", [](ffi::String name) -> Axis { return Axis::Get(name); });
}

// Axis
Axis Axis::Get(const ffi::String& name) {
  const ffi::EnumState& state = AxisEnumState();
  auto existing = state->indexes.Get(ffi::Any(name));
  if (!existing) {
    refl::EnumDef<AxisNode>(name.c_str());
    existing = state->indexes.Get(ffi::Any(name));
  }
  return existing.value().as_or_throw<Axis>();
}

TVM_FFI_STATIC_INIT_BLOCK() { refl::EnumDef<AxisNode>("m").set_attr("thread", false); }
}  // namespace tirx
}  // namespace tvm
