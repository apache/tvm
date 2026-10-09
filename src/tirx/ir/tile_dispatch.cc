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
 * \file tirx/ir/tile_dispatch.cc
 * \brief Context for dispatching and lowering TIRx tile operations.
 */
#include <tvm/tirx/tile_dispatch.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() { DispatchContextNode::RegisterReflection(); }

template <typename Key, typename Value>
Value getOrSetDefault(ffi::Map<ffi::String, ffi::ObjectRef>& m, const Key& key,
                      const Value& defaultValue) {
  // try_emplace inserts the defaultValue only if key does not exist.
  auto it = m.find(key);
  if (it == m.end()) {
    m.Set(key, defaultValue);
    return defaultValue;
  }
  return (*it).second.template as_or_throw<Value>();
}

void DispatchContextNode::AddAllocBuffer(TensorVar buffer) {
  auto buffers = getOrSetDefault(callbacks, callback::kPrivateAlloc, ffi::Array<TensorVar>());
  buffers.push_back(buffer);
  callbacks.Set(callback::kPrivateAlloc, buffers);
}

void DispatchContextNode::AddInitStmt(Stmt stmt, bool host) {
  auto tag = host ? callback::kHostInitStmt : callback::kDeviceInitStmt;
  auto stmts = getOrSetDefault(callbacks, tag, ffi::Array<Stmt>());
  stmts.push_back(stmt);
  callbacks.Set(tag, stmts);
}

void DispatchContextNode::AddPostBufferDefStmt(TensorVar buffer, Stmt stmt) {
  auto mapping = getOrSetDefault(callbacks, callback::kPostBufferDefStmt,
                                 ffi::Map<TensorVar, ffi::Array<Stmt>>());
  auto it = mapping.find(buffer);
  ffi::Array<Stmt> stmts;
  if (it != mapping.end()) {
    stmts = (*it).second;
  }
  stmts.push_back(stmt);
  mapping.Set(buffer, stmts);
  callbacks.Set(callback::kPostBufferDefStmt, mapping);
}

void DispatchContextNode::SharedStateSet(ffi::String key, ffi::ObjectRef value) {
  shared_state.Set(key, value);
}

ffi::Optional<ffi::ObjectRef> DispatchContextNode::SharedStateGet(ffi::String key) {
  auto it = shared_state.find(key);
  if (it != shared_state.end()) {
    return (*it).second;
  }
  return ffi::Optional<ffi::ObjectRef>();
}

DispatchContext::DispatchContext(Target target, ExecScope exec_scope,
                                 ffi::Map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params,
                                 ffi::Map<Var, Range> var_range_map, bool alloc_only,
                                 ffi::Map<ffi::String, ffi::ObjectRef> callbacks,
                                 ffi::Map<ffi::String, ffi::ObjectRef> shared_state,
                                 ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter,
                                 ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra,
                                 ffi::String scope_kind) {
  auto n = ffi::make_object<DispatchContextNode>();
  n->target = std::move(target);
  n->exec_scope = std::move(exec_scope);
  for (const auto& [tag, binding] : launch_params) {
    PrimType var_ty = binding.get<0>().ty();
    PrimType extent_ty = binding.get<1>().ty();
    TVM_FFI_CHECK(extent_ty.code() == DLDataTypeCode::kDLInt && extent_ty == var_ty, TypeError)
        << "Launch parameter " << tag << " requires a signed integer extent matching its variable";
  }
  n->launch_params = std::move(launch_params);
  n->var_range_map = std::move(var_range_map);
  n->alloc_only = alloc_only;
  n->callbacks = std::move(callbacks);
  n->shared_state = std::move(shared_state);
  n->inter = std::move(inter);
  n->intra = std::move(intra);
  n->scope_kind = std::move(scope_kind);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.DispatchContext",
           [](Target target, ExecScope exec_scope,
              ffi::Map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params,
              ffi::Map<Var, Range> var_range_map, bool alloc_only,
              ffi::Map<ffi::String, ffi::ObjectRef> callbacks,
              ffi::Map<ffi::String, ffi::ObjectRef> shared_state,
              ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter,
              ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra, ffi::String scope_kind) {
             return DispatchContext(target, exec_scope, launch_params, var_range_map, alloc_only,
                                    callbacks, shared_state, inter, intra, scope_kind);
           })
      .def_method("tirx.DispatchContextAddAllocBuffer", &DispatchContextNode::AddAllocBuffer)
      .def_method("tirx.DispatchContextAddInitStmt", &DispatchContextNode::AddInitStmt)
      .def_method("tirx.DispatchContextAddPostBufferDefStmt",
                  &DispatchContextNode::AddPostBufferDefStmt)
      .def_method("tirx.DispatchContextSharedStateSet", &DispatchContextNode::SharedStateSet)
      .def_method("tirx.DispatchContextSharedStateGet", &DispatchContextNode::SharedStateGet);
}

}  // namespace tirx
}  // namespace tvm
