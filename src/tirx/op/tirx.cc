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
 * \file tir/op/tirx.cc
 * TIRX built-in operators.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tirx_op.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() { DispatchContextNode::RegisterReflection(); }

/********************* Utils **********************/

#define TIRX_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                              \
    static const Op& op = Op::Get("tirx." #OpName); \
    return op;                                      \
  }                                                 \
  TVM_REGISTER_OP("tirx." #OpName)                  \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String(#OpName), /*plevel=*/9)

#define TIRX_DEFINE_OP(OpName) TIRX_DEFINE_BUILTIN_FUNC(OpName).set_attr<bool>("TIsTIRxOp", true)

/********************* Context utils **********************/
template <typename Key, typename Value>
Value getOrSetDefault(ffi::Map<ffi::String, ffi::ObjectRef>& m, const Key& key,
                      const Value& defaultValue) {
  // try_emplace inserts the defaultValue only if key does not exist.
  auto it = m.find(key);
  if (it == m.end()) {
    m.Set(key, defaultValue);
    return defaultValue;
  }
  return Downcast<Value>((*it).second);
}

/********************* DispatchContext **********************/

void DispatchContextNode::AddAllocBuffer(Buffer buffer) {
  auto buffers = getOrSetDefault(callbacks, callback::kPrivateAlloc, ffi::Array<Buffer>());
  buffers.push_back(buffer);
  callbacks.Set(callback::kPrivateAlloc, buffers);
}

void DispatchContextNode::AddInitStmt(Stmt stmt, bool host) {
  auto tag = host ? callback::kHostInitStmt : callback::kDeviceInitStmt;
  auto stmts = getOrSetDefault(callbacks, tag, ffi::Array<Stmt>());
  stmts.push_back(stmt);
  callbacks.Set(tag, stmts);
}

void DispatchContextNode::AddPostBufferDefStmt(Buffer buffer, Stmt stmt) {
  auto mapping = getOrSetDefault(callbacks, callback::kPostBufferDefStmt,
                                 ffi::Map<Buffer, ffi::Array<Stmt>>());
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
                                 ffi::Map<ffi::String, IterVar> launch_params,
                                 ffi::Map<Var, Range> var_range_map, bool alloc_only,
                                 ffi::Map<ffi::String, ffi::ObjectRef> callbacks,
                                 ffi::Map<ffi::String, ffi::ObjectRef> shared_state,
                                 ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter,
                                 ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra,
                                 ffi::String scope_kind) {
  auto n = ffi::make_object<DispatchContextNode>();
  n->target = std::move(target);
  n->exec_scope = std::move(exec_scope);
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
           [](Target target, ExecScope exec_scope, ffi::Map<ffi::String, IterVar> launch_params,
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

/********************* Dispatch Ops **********************/
#define TIRX_DEFINE_DISPATCH_OP(OpName) TIRX_DEFINE_OP(OpName).set_attr<bool>("TIsDispatchOp", true)

TIRX_DEFINE_DISPATCH_OP(zero);
TIRX_DEFINE_DISPATCH_OP(sqrt);
TIRX_DEFINE_DISPATCH_OP(exp);
TIRX_DEFINE_DISPATCH_OP(exp2);
TIRX_DEFINE_DISPATCH_OP(add);
TIRX_DEFINE_DISPATCH_OP(sub);
TIRX_DEFINE_DISPATCH_OP(mul);
TIRX_DEFINE_DISPATCH_OP(fdiv);
TIRX_DEFINE_DISPATCH_OP(minimum);
TIRX_DEFINE_DISPATCH_OP(maximum);
TIRX_DEFINE_DISPATCH_OP(copy);
TIRX_DEFINE_DISPATCH_OP(fill);
TIRX_DEFINE_DISPATCH_OP(gemm);
TIRX_DEFINE_DISPATCH_OP(reciprocal);
TIRX_DEFINE_DISPATCH_OP(sum);
TIRX_DEFINE_DISPATCH_OP(max);
TIRX_DEFINE_DISPATCH_OP(min);
TIRX_DEFINE_DISPATCH_OP(memset);
TIRX_DEFINE_DISPATCH_OP(reduce_negate);
TIRX_DEFINE_DISPATCH_OP(binary_reduce);
TIRX_DEFINE_DISPATCH_OP(unary_reduce);
TIRX_DEFINE_DISPATCH_OP(binary_chain);
TIRX_DEFINE_DISPATCH_OP(select);
TIRX_DEFINE_DISPATCH_OP(cast);
TIRX_DEFINE_DISPATCH_OP(fma);
TIRX_DEFINE_DISPATCH_OP(silu);

/********************* Compose Ops **********************/
#define TIRX_DEFINE_COMPOSE_OP(OpName) TIRX_DEFINE_OP(OpName).set_attr<bool>("TIsComposeOp", true)

TIRX_DEFINE_COMPOSE_OP(compose_op);

/********************* Async Ops **********************/
#define TIRX_DEFINE_ASYNC_OP(OpName) TIRX_DEFINE_OP(OpName).set_attr<bool>("TIsAsyncOp", true)

TIRX_DEFINE_ASYNC_OP(copy_async);
TIRX_DEFINE_ASYNC_OP(gemm_async);

/********************* Misc Ops **********************/
TIRX_DEFINE_OP(tvm_kernel_replace_point);

}  // namespace tirx
}  // namespace tvm
