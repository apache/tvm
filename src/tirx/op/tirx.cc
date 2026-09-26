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
#include <tvm/tirx/tile_primitive.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() { DispatchContextNode::RegisterReflection(); }

/********************* Utils **********************/

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
  return (*it).second.template as_or_throw<Value>();
}

/********************* DispatchContext **********************/

void DispatchContextNode::AddAllocBuffer(BufferVar buffer) {
  auto buffers = getOrSetDefault(callbacks, callback::kPrivateAlloc, ffi::Array<BufferVar>());
  buffers.push_back(buffer);
  callbacks.Set(callback::kPrivateAlloc, buffers);
}

void DispatchContextNode::AddInitStmt(Stmt stmt, bool host) {
  auto tag = host ? callback::kHostInitStmt : callback::kDeviceInitStmt;
  auto stmts = getOrSetDefault(callbacks, tag, ffi::Array<Stmt>());
  stmts.push_back(stmt);
  callbacks.Set(tag, stmts);
}

void DispatchContextNode::AddPostBufferDefStmt(BufferVar buffer, Stmt stmt) {
  auto mapping = getOrSetDefault(callbacks, callback::kPostBufferDefStmt,
                                 ffi::Map<BufferVar, ffi::Array<Stmt>>());
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

/********************* Tile Ops **********************/
Op zero() {
  static const Op op = Op::Get("tirx.tile.zero");
  return op;
}

Op sqrt() {
  static const Op op = Op::Get("tirx.tile.sqrt");
  return op;
}

Op exp() {
  static const Op op = Op::Get("tirx.tile.exp");
  return op;
}

Op exp2() {
  static const Op op = Op::Get("tirx.tile.exp2");
  return op;
}

Op log2() {
  static const Op op = Op::Get("tirx.tile.log2");
  return op;
}

Op add() {
  static const Op op = Op::Get("tirx.tile.add");
  return op;
}

Op sub() {
  static const Op op = Op::Get("tirx.tile.sub");
  return op;
}

Op mul() {
  static const Op op = Op::Get("tirx.tile.mul");
  return op;
}

Op fdiv() {
  static const Op op = Op::Get("tirx.tile.fdiv");
  return op;
}

Op minimum() {
  static const Op op = Op::Get("tirx.tile.minimum");
  return op;
}

Op maximum() {
  static const Op op = Op::Get("tirx.tile.maximum");
  return op;
}

Op copy() {
  static const Op op = Op::Get("tirx.tile.copy");
  return op;
}

Op fill() {
  static const Op op = Op::Get("tirx.tile.fill");
  return op;
}

Op gemm() {
  static const Op op = Op::Get("tirx.tile.gemm");
  return op;
}

Op reciprocal() {
  static const Op op = Op::Get("tirx.tile.reciprocal");
  return op;
}

Op sum() {
  static const Op op = Op::Get("tirx.tile.sum");
  return op;
}

Op max() {
  static const Op op = Op::Get("tirx.tile.max");
  return op;
}

Op min() {
  static const Op op = Op::Get("tirx.tile.min");
  return op;
}

Op memset() {
  static const Op op = Op::Get("tirx.tile.memset");
  return op;
}

Op reduce_negate() {
  static const Op op = Op::Get("tirx.tile.reduce_negate");
  return op;
}

Op binary_reduce() {
  static const Op op = Op::Get("tirx.tile.binary_reduce");
  return op;
}

Op unary_reduce() {
  static const Op op = Op::Get("tirx.tile.unary_reduce");
  return op;
}

Op binary_chain() {
  static const Op op = Op::Get("tirx.tile.binary_chain");
  return op;
}

Op select() {
  static const Op op = Op::Get("tirx.tile.select");
  return op;
}

Op cast() {
  static const Op op = Op::Get("tirx.tile.cast");
  return op;
}

Op fma() {
  static const Op op = Op::Get("tirx.tile.fma");
  return op;
}

Op silu() {
  static const Op op = Op::Get("tirx.tile.silu");
  return op;
}

Op permute_layout() {
  static const Op op = Op::Get("tirx.tile.permute_layout");
  return op;
}

Op copy_async() {
  static const Op op = Op::Get("tirx.tile.copy_async");
  return op;
}

Op gemm_async() {
  static const Op op = Op::Get("tirx.tile.gemm_async");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.zero")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("zero"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sqrt")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("sqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("exp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("exp2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.log2")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("log2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.add")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("add"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sub")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("sub"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.mul")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("mul"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fdiv")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fdiv"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.minimum")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("minimum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.maximum")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("maximum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.copy")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("copy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fill")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.gemm")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("gemm"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.reciprocal")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("reciprocal"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sum")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("sum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.max")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("max"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.min")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("min"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.memset")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("memset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.reduce_negate")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("reduce_negate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.binary_reduce")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("binary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.unary_reduce")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("unary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.binary_chain")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("binary_chain"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.select")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("select"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.cast")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cast"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fma")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.silu")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("silu"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.permute_layout")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("permute_layout"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.copy_async")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("copy_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.gemm_async")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("gemm_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

}  // namespace tirx
}  // namespace tvm
