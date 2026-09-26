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
#define TVM_DEFINE_CACHED_OP_GETTER(Name, RegisteredName) \
  const Op& Name() {                                      \
    static const Op op = Op::Get(RegisteredName);         \
    return op;                                            \
  }

TVM_DEFINE_CACHED_OP_GETTER(zero, "tirx.tile.zero")
TVM_DEFINE_CACHED_OP_GETTER(sqrt, "tirx.tile.sqrt")
TVM_DEFINE_CACHED_OP_GETTER(exp, "tirx.tile.exp")
TVM_DEFINE_CACHED_OP_GETTER(exp2, "tirx.tile.exp2")
TVM_DEFINE_CACHED_OP_GETTER(log2, "tirx.tile.log2")
TVM_DEFINE_CACHED_OP_GETTER(add, "tirx.tile.add")
TVM_DEFINE_CACHED_OP_GETTER(sub, "tirx.tile.sub")
TVM_DEFINE_CACHED_OP_GETTER(mul, "tirx.tile.mul")
TVM_DEFINE_CACHED_OP_GETTER(fdiv, "tirx.tile.fdiv")
TVM_DEFINE_CACHED_OP_GETTER(minimum, "tirx.tile.minimum")
TVM_DEFINE_CACHED_OP_GETTER(maximum, "tirx.tile.maximum")
TVM_DEFINE_CACHED_OP_GETTER(copy, "tirx.tile.copy")
TVM_DEFINE_CACHED_OP_GETTER(fill, "tirx.tile.fill")
TVM_DEFINE_CACHED_OP_GETTER(gemm, "tirx.tile.gemm")
TVM_DEFINE_CACHED_OP_GETTER(reciprocal, "tirx.tile.reciprocal")
TVM_DEFINE_CACHED_OP_GETTER(sum, "tirx.tile.sum")
TVM_DEFINE_CACHED_OP_GETTER(max, "tirx.tile.max")
TVM_DEFINE_CACHED_OP_GETTER(min, "tirx.tile.min")
TVM_DEFINE_CACHED_OP_GETTER(memset, "tirx.tile.memset")
TVM_DEFINE_CACHED_OP_GETTER(reduce_negate, "tirx.tile.reduce_negate")
TVM_DEFINE_CACHED_OP_GETTER(binary_reduce, "tirx.tile.binary_reduce")
TVM_DEFINE_CACHED_OP_GETTER(unary_reduce, "tirx.tile.unary_reduce")
TVM_DEFINE_CACHED_OP_GETTER(binary_chain, "tirx.tile.binary_chain")
TVM_DEFINE_CACHED_OP_GETTER(select, "tirx.tile.select")
TVM_DEFINE_CACHED_OP_GETTER(cast, "tirx.tile.cast")
TVM_DEFINE_CACHED_OP_GETTER(fma, "tirx.tile.fma")
TVM_DEFINE_CACHED_OP_GETTER(silu, "tirx.tile.silu")
TVM_DEFINE_CACHED_OP_GETTER(permute_layout, "tirx.tile.permute_layout")
TVM_DEFINE_CACHED_OP_GETTER(copy_async, "tirx.tile.copy_async")
TVM_DEFINE_CACHED_OP_GETTER(gemm_async, "tirx.tile.gemm_async")

#undef TVM_DEFINE_CACHED_OP_GETTER

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.zero")
      .arg("dst", "")
      .arg("src", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("zero"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sqrt")
      .arg("dst", "")
      .arg("src", "")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("sqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp")
      .arg("dst", "")
      .arg("src", "")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("exp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp2")
      .arg("dst", "")
      .arg("src", "")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("exp2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.log2")
      .arg("dst", "")
      .arg("src", "")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("log2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.add")
      .arg("dst", "")
      .arg("src1", "")
      .arg("src2", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("add"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sub")
      .arg("dst", "")
      .arg("src1", "")
      .arg("src2", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("sub"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.mul")
      .arg("dst", "")
      .arg("src1", "")
      .arg("src2", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("mul"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fdiv")
      .arg("dst", "")
      .arg("src1", "")
      .arg("src2", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fdiv"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.minimum")
      .arg("dst", "")
      .arg("src1", "")
      .arg("src2", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("minimum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.maximum")
      .arg("dst", "")
      .arg("src1", "")
      .arg("src2", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("maximum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.copy")
      .arg("dst", "")
      .arg("src", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("copy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fill")
      .arg("dst", "")
      .arg("value", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.gemm")
      .arg("D", "")
      .arg("A", "")
      .arg("B", "")
      .arg("C", "")
      .arg("transpose_A", "")
      .arg("transpose_B", "")
      .arg("alpha", "")
      .arg("beta", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("gemm"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.reciprocal")
      .arg("dst", "")
      .arg("src", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("reciprocal"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sum")
      .arg("dst", "")
      .arg("src", "")
      .arg("axes", "")
      .arg("accum", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("sum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.max")
      .arg("dst", "")
      .arg("src", "")
      .arg("axes", "")
      .arg("accum", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("max"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.min")
      .arg("dst", "")
      .arg("src", "")
      .arg("axes", "")
      .arg("accum", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("min"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.memset")
      .arg("dst", "")
      .arg("value", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("memset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.reduce_negate")
      .arg("output", "")
      .arg("input", "")
      .arg("reduce_axes", "")
      .arg("accum", "")
      .arg("reduce_op", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("reduce_negate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.binary_reduce")
      .arg("binary_output", "")
      .arg("reduce_output", "")
      .arg("binary_input1", "")
      .arg("binary_input2", "")
      .arg("binary_op", "")
      .arg("reduce_op", "")
      .arg("reduce_axes", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("binary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.unary_reduce")
      .arg("unary_output", "")
      .arg("reduce_output", "")
      .arg("unary_input", "")
      .arg("unary_op", "")
      .arg("reduce_op", "")
      .arg("bias", "")
      .arg("scale", "")
      .arg("reduce_axes", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("unary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.binary_chain")
      .arg("output", "")
      .arg("data", "")
      .arg("operand0", "")
      .arg("operand1", "")
      .arg("op0", "")
      .arg("op1", "")
      .arg("reverse1", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("binary_chain"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.select")
      .arg("dst", "")
      .arg("true_value", "")
      .arg("false_value", "")
      .arg("pred", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("select"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.cast")
      .arg("dst", "")
      .arg("src", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cast"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fma")
      .arg("dst", "")
      .arg("src", "")
      .arg("scale", "")
      .arg("bias", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.silu")
      .arg("dst", "")
      .arg("src", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("silu"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.permute_layout")
      .arg("dst", "")
      .arg("src", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("permute_layout"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.copy_async")
      .arg("dst", "")
      .arg("src", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("copy_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.gemm_async")
      .arg("C", "")
      .arg("A", "")
      .arg("B", "")
      .allow_extra_args()
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("gemm_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

}  // namespace tirx
}  // namespace tvm
