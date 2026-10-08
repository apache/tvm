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
#include <tvm/tirx/tile_op.h>

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

/********************* Tile Ops **********************/
namespace tile {
#define TVM_DEFINE_CACHED_OP_GETTER(Name, RegisteredName) \
  const Op& Name() {                                      \
    static const Op op = Op::Get(RegisteredName);         \
    return op;                                            \
  }

TVM_DEFINE_CACHED_OP_GETTER(zero_op, "tirx.tile.zero")
TVM_DEFINE_CACHED_OP_GETTER(sqrt_op, "tirx.tile.sqrt")
TVM_DEFINE_CACHED_OP_GETTER(sqrt_with_scale_bias_op, "tirx.tile.sqrt_with_scale_bias")
TVM_DEFINE_CACHED_OP_GETTER(exp_op, "tirx.tile.exp")
TVM_DEFINE_CACHED_OP_GETTER(exp_with_scale_bias_op, "tirx.tile.exp_with_scale_bias")
TVM_DEFINE_CACHED_OP_GETTER(exp2_op, "tirx.tile.exp2")
TVM_DEFINE_CACHED_OP_GETTER(exp2_with_scale_bias_op, "tirx.tile.exp2_with_scale_bias")
TVM_DEFINE_CACHED_OP_GETTER(log2_op, "tirx.tile.log2")
TVM_DEFINE_CACHED_OP_GETTER(log2_with_scale_bias_op, "tirx.tile.log2_with_scale_bias")
TVM_DEFINE_CACHED_OP_GETTER(add_op, "tirx.tile.add")
TVM_DEFINE_CACHED_OP_GETTER(sub_op, "tirx.tile.sub")
TVM_DEFINE_CACHED_OP_GETTER(mul_op, "tirx.tile.mul")
TVM_DEFINE_CACHED_OP_GETTER(fdiv_op, "tirx.tile.fdiv")
TVM_DEFINE_CACHED_OP_GETTER(minimum_op, "tirx.tile.minimum")
TVM_DEFINE_CACHED_OP_GETTER(maximum_op, "tirx.tile.maximum")
TVM_DEFINE_CACHED_OP_GETTER(copy_op, "tirx.tile.copy")
TVM_DEFINE_CACHED_OP_GETTER(fill_op, "tirx.tile.fill")
TVM_DEFINE_CACHED_OP_GETTER(gemm_op, "tirx.tile.gemm")
TVM_DEFINE_CACHED_OP_GETTER(reciprocal_op, "tirx.tile.reciprocal")
TVM_DEFINE_CACHED_OP_GETTER(sum_op, "tirx.tile.sum")
TVM_DEFINE_CACHED_OP_GETTER(max_op, "tirx.tile.max")
TVM_DEFINE_CACHED_OP_GETTER(min_op, "tirx.tile.min")
TVM_DEFINE_CACHED_OP_GETTER(memset_op, "tirx.tile.memset")
TVM_DEFINE_CACHED_OP_GETTER(reduce_negate_op, "tirx.tile.reduce_negate")
TVM_DEFINE_CACHED_OP_GETTER(binary_reduce_op, "tirx.tile.binary_reduce")
TVM_DEFINE_CACHED_OP_GETTER(unary_reduce_op, "tirx.tile.unary_reduce")
TVM_DEFINE_CACHED_OP_GETTER(unary_reduce_with_scale_bias_op,
                            "tirx.tile.unary_reduce_with_scale_bias")
TVM_DEFINE_CACHED_OP_GETTER(binary_chain_op, "tirx.tile.binary_chain")
TVM_DEFINE_CACHED_OP_GETTER(select_op, "tirx.tile.select")
TVM_DEFINE_CACHED_OP_GETTER(cast_op, "tirx.tile.cast")
TVM_DEFINE_CACHED_OP_GETTER(fma_op, "tirx.tile.fma")
TVM_DEFINE_CACHED_OP_GETTER(silu_op, "tirx.tile.silu")
TVM_DEFINE_CACHED_OP_GETTER(permute_layout_op, "tirx.tile.permute_layout")
TVM_DEFINE_CACHED_OP_GETTER(copy_async_op, "tirx.tile.copy_async")
TVM_DEFINE_CACHED_OP_GETTER(gemm_async_op, "tirx.tile.gemm_async")

#undef TVM_DEFINE_CACHED_OP_GETTER

}  // namespace tile

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tile.zero")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.zero"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sqrt")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.sqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sqrt_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.sqrt_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.exp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.exp_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp2")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.exp2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.exp2_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.exp2_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.log2")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.log2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.log2_with_scale_bias")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.log2_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.add")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.add"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sub")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.sub"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.mul")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.mul"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fdiv")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.fdiv"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.minimum")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.minimum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.maximum")
      .signature(sig::arg("dst", "The destination."), sig::arg("src1", "The first source."),
                 sig::arg("src2", "The second source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.maximum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.copy")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.copy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fill")
      .signature(sig::arg("dst", "The destination."), sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.gemm")
      .signature(
          sig::arg("D", "The D tile."), sig::arg("A", "The A tile."), sig::arg("B", "The B tile."),
          sig::arg("C", "The C tile."), sig::arg("transpose_A", "Whether to transpose A."),
          sig::arg("transpose_B", "Whether to transpose B."),
          sig::arg("alpha", "The alpha scale factor."), sig::arg("beta", "The beta scale factor."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.gemm"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.reciprocal")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.reciprocal"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.sum")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("axes", "The axes."), sig::arg("accum", "The accumulator."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.sum"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.max")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("axes", "The axes."), sig::arg("accum", "The accumulator."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.max"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.min")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("axes", "The axes."), sig::arg("accum", "The accumulator."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.min"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.memset")
      .signature(sig::arg("dst", "The destination."), sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.memset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.reduce_negate")
      .signature(sig::arg("output", "The output."), sig::arg("input", "The input."),
                 sig::arg("reduce_axes", "The reduction axes."),
                 sig::arg("accum", "The accumulator."),
                 sig::arg("reduce_op", "The reduction operation."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.reduce_negate"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.binary_reduce")
      .signature(sig::arg("binary_output", "The binary operation output."),
                 sig::arg("reduce_output", "The reduction output."),
                 sig::arg("binary_input1", "The first binary input."),
                 sig::arg("binary_input2", "The second binary input."),
                 sig::arg("binary_op", "The binary operation."),
                 sig::arg("reduce_op", "The reduction operation."),
                 sig::arg("reduce_axes", "The reduction axes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.binary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.unary_reduce")
      .signature(sig::arg("unary_output", "The unary operation output."),
                 sig::arg("reduce_output", "The reduction output."),
                 sig::arg("unary_input", "The unary operation input."),
                 sig::arg("unary_op", "The unary operation."),
                 sig::arg("reduce_op", "The reduction operation."),
                 sig::arg("reduce_axes", "The reduction axes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.unary_reduce"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.unary_reduce_with_scale_bias")
      .signature(sig::arg("unary_output", "The unary operation output."),
                 sig::arg("reduce_output", "The reduction output."),
                 sig::arg("unary_input", "The unary operation input."),
                 sig::arg("unary_op", "The unary operation."),
                 sig::arg("reduce_op", "The reduction operation."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."),
                 sig::arg("reduce_axes", "The reduction axes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.tile.unary_reduce_with_scale_bias"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.binary_chain")
      .signature(sig::arg("output", "The output."), sig::arg("data", "The input data."),
                 sig::arg("operand0", "The first operand."),
                 sig::arg("operand1", "The second operand."),
                 sig::arg("op0", "The first operation."), sig::arg("op1", "The second operation."),
                 sig::arg("reverse1", "Whether to reverse the second operation."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.binary_chain"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.select")
      .signature(sig::arg("dst", "The destination."),
                 sig::arg("true_value", "The value when the condition is true."),
                 sig::arg("false_value", "The value when the condition is false."),
                 sig::arg("pred", "The predicate."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.select"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.cast")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.cast"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.fma")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."),
                 sig::arg("scale", "The scale factor."), sig::arg("bias", "The bias."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.fma"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.silu")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.silu"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.permute_layout")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.permute_layout"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.copy_async")
      .signature(sig::arg("dst", "The destination."), sig::arg("src", "The source."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.copy_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));

  OpDef("tirx.tile.gemm_async")
      .signature(sig::arg("C", "The C tile."), sig::arg("A", "The A tile."),
                 sig::arg("B", "The B tile."), sig::var_args("args"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tile.gemm_async"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("tile_primitive"));
}

}  // namespace tirx
}  // namespace tvm
