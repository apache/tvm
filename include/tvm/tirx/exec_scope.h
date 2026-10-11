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
/*! \brief Backend-defined execution scopes shared by tensor layouts and tile calls. */
#ifndef TVM_TIRX_EXEC_SCOPE_H_
#define TVM_TIRX_EXEC_SCOPE_H_
#include <tvm/ffi/object.h>
#include <tvm/ir/expr.h>

namespace tvm::tirx {

/*! \brief Register a backend's execution hierarchy. Smaller rank means wider scope. */
TVM_DLL void RegisterExecScope(const ffi::String& backend, const ffi::String& name, int rank,
                               bool executable = true);

class ExecScopeNode : public ffi::Object {
 public:
  ffi::String backend;
  ffi::String scope_name;
  ffi::String name() const { return scope_name; }
  int rank() const;
  bool executable() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExecScopeNode>()
        .def_ro("backend", &ExecScopeNode::backend)
        .def_ro("name", &ExecScopeNode::scope_name);
  }
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.ExecScope", ExecScopeNode, ffi::Object);
};

class ExecScope : public ffi::ObjectRef {
 public:
  TVM_DLL ExecScope(ffi::String backend, ffi::String name);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ExecScope, ffi::ObjectRef, ExecScopeNode);
};

/*! \brief Compare scopes within the same backend's hierarchy. */
TVM_DLL bool ScopeHigher(const ExecScope& a, const ExecScope& b);

}  // namespace tvm::tirx
#endif  // TVM_TIRX_EXEC_SCOPE_H_
