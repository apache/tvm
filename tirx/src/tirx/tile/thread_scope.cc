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
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/tile/thread_scope.h>

#include <string>

namespace tvm::tirx {
namespace refl = ffi::reflection;
namespace {
template <typename T>
T ScopeAttr(const ThreadScopeObj* scope, refl::AttrKey<T> key) {
  auto state = refl::TypeAttrColumn(
                   ffi::type_attr::kEnumState)[ThreadScopeObj::_GetOrAllocRuntimeTypeIndex()]
                   .cast<ffi::EnumState>();
  auto column = state->attrs.Get(key.name);
  auto value = key.ValueFrom(
      column ? column->Get(ffi::GetRef<ThreadScope>(scope)).value_or(ffi::Any()) : ffi::Any());
  TVM_FFI_CHECK(value.has_value(), ValueError)
      << "Thread scope " << scope->name() << " has no " << key.name << " attribute";
  return value.value();
}
}  // namespace

ffi::String ThreadScopeObj::backend() const { return ScopeAttr(this, thread_scope_attr::kBackend); }
ffi::String ThreadScopeObj::local_name() const {
  const std::string prefix = std::string(backend()) + ".";
  const std::string name(_str_index);
  TVM_FFI_CHECK(name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0,
                ValueError)
      << "Thread scope " << name << " must be named as backend.scope";
  return name.substr(prefix.size());
}
int64_t ThreadScopeObj::rank() const { return ScopeAttr(this, thread_scope_attr::kRank); }

ThreadScope ThreadScope::Get(const ffi::String& name) {
  auto scope = ffi::EnumObj::_GetByStrIndex<ThreadScopeObj>(name).as_or_throw<ThreadScope>();
  scope->local_name();
  scope->rank();
  return scope;
}

bool ScopeHigher(const ThreadScope& a, const ThreadScope& b) {
  TVM_FFI_CHECK(a->backend() == b->backend(), ValueError)
      << "Cannot compare thread scopes from different backends";
  return a->rank() < b->rank();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::ObjectDef<ThreadScopeObj>(refl::init(false));
  refl::TypeAttrDef<ThreadScopeObj>()
      .def(ffi::type_attr::kDataToJson.name,
           [](const ThreadScopeObj* scope) -> ffi::String { return scope->name(); })
      .def(ffi::type_attr::kDataFromJson.name, ThreadScope::Get);
  refl::GlobalDef()
      .def("tirx.ThreadScopeGet", ThreadScope::Get)
      .def("tirx.ThreadScopeGetBackend", [](ThreadScope scope) { return scope->backend(); })
      .def("tirx.ThreadScopeGetLocalName", [](ThreadScope scope) { return scope->local_name(); })
      .def("tirx.ThreadScopeGetRank", [](ThreadScope scope) { return scope->rank(); })
      .def("tirx.ScopeHigher", ScopeHigher);
}
}  // namespace tvm::tirx
