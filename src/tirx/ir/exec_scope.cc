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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/exec_scope.h>

#include <map>
#include <string>
#include <utility>

namespace tvm::tirx {
namespace {
using ScopeKey = std::pair<std::string, std::string>;
std::map<ScopeKey, int>& Scopes() {
  static std::map<ScopeKey, int> scopes;
  return scopes;
}
int Lookup(const ffi::String& backend, const ffi::String& name) {
  auto it = Scopes().find({std::string(backend), std::string(name)});
  TVM_FFI_CHECK(it != Scopes().end(), ValueError)
      << "Unknown execution scope: " << backend << "." << name;
  return it->second;
}
}  // namespace

void RegisterExecScope(const ffi::String& backend, const ffi::String& name, int rank) {
  Scopes().insert_or_assign({std::string(backend), std::string(name)}, rank);
}
int ExecScopeNode::rank() const { return Lookup(backend, scope_name); }
ExecScope::ExecScope(ffi::String backend, ffi::String name) {
  Lookup(backend, name);
  auto n = ffi::make_object<ExecScopeNode>();
  n->backend = std::move(backend);
  n->scope_name = std::move(name);
  data_ = std::move(n);
}
bool ScopeHigher(const ExecScope& a, const ExecScope& b) {
  TVM_FFI_CHECK(a->backend == b->backend, ValueError)
      << "Cannot compare execution scopes from different backends";
  return a->rank() < b->rank();
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ExecScopeNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.ExecScope",
           [](ffi::String backend, ffi::String name) {
             return ExecScope(std::move(backend), std::move(name));
           })
      .def("tirx.RegisterExecScope", RegisterExecScope);
}
}  // namespace tvm::tirx
