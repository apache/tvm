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
 * \file tirx/op/gpu.cc
 * \brief TIRx gpu operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

template <size_t N>
static Type InferTypeReturnArgType(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

const Op& gpu_thread_return_op() {
  static const Op op = Op::Get("tirx.gpu_thread_return");
  return op;
}

PrimExpr gpu_thread_return(ffi::Optional<Location> loc) {
  return Call(PrimType::Void(), tirx::gpu_thread_return_op(), {}, {}, {}, loc)
      .as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_thread_return")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_thread_return"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kControlJump));
}

const Op& gpu_thread_filter_op() {
  static const Op op = Op::Get("tirx.gpu_thread_filter");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_thread_filter")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Bool())
      .signature(sig::arg<PrimVar>("var", "The thread-axis variable."),
                 sig::arg<PrimExpr>("pred", "The predicate."))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_thread_filter"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& gpu_active_thread_selector_op() {
  static const Op op = Op::Get("tirx.gpu_active_thread_selector");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_active_thread_selector")
      .signature(sig::arg<PrimVar>("var", "The thread-axis variable."),
                 sig::arg<PrimExpr>("pred", "The predicate."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_active_thread_selector"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_thread_invariant_op() {
  static const Op op = Op::Get("tirx.gpu_thread_invariant");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_thread_invariant")
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .signature(sig::arg<PrimExpr>("cond", "The condition."))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_thread_invariant"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& gpu_storage_sync_op() {
  static const Op op = Op::Get("tirx.gpu_storage_sync");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_storage_sync")
      .signature(sig::arg<StringImm>("storage_scope", "The storage scope."))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_storage_sync"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_warp_shuffle_op() {
  static const Op op = Op::Get("tirx.gpu_warp_shuffle");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_warp_shuffle")
      .signature(sig::arg<IntExpr>("mask", "The mask."),
                 sig::arg<PrimExpr>("value", "The value to use."),
                 sig::arg<IntExpr>("warp_id", "The warp identifier."),
                 sig::arg<IntExpr>("width", "The width."),
                 sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_warp_shuffle"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_warp_shuffle_up_op() {
  static const Op op = Op::Get("tirx.gpu_warp_shuffle_up");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_warp_shuffle_up")
      .signature(
          sig::arg<IntExpr>("mask", "The mask."), sig::arg<PrimExpr>("value", "The value to use."),
          sig::arg<IntExpr>("offset", "The offset."), sig::arg<IntExpr>("width", "The width."),
          sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_warp_shuffle_up"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_warp_shuffle_down_op() {
  static const Op op = Op::Get("tirx.gpu_warp_shuffle_down");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_warp_shuffle_down")
      .signature(
          sig::arg<IntExpr>("mask", "The mask."), sig::arg<PrimExpr>("value", "The value to use."),
          sig::arg<IntExpr>("offset", "The offset."), sig::arg<IntExpr>("width", "The width."),
          sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_warp_shuffle_down"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_warp_shuffle_xor_op() {
  static const Op op = Op::Get("tirx.gpu_warp_shuffle_xor");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_warp_shuffle_xor")
      .signature(sig::arg<IntExpr>("mask", "The mask."),
                 sig::arg<PrimExpr>("value", "The value to use."),
                 sig::arg<IntExpr>("lane_mask", "The lane mask."),
                 sig::arg<IntExpr>("width", "The width."),
                 sig::arg<IntExpr>("warp_size", "The number of threads per warp."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<1>>())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_warp_shuffle_xor"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_warp_activemask_op() {
  static const Op op = Op::Get("tirx.gpu_warp_activemask");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_warp_activemask")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_warp_activemask"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_thread_allreduce_op() {
  static const Op op = Op::Get("tirx.gpu_thread_allreduce");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_thread_allreduce")
      .signature(sig::arg<LambdaExpr>("combine", "The typed combining lambda."),
                 sig::arg<Expr>("identity", "The identity values."),
                 sig::arg<Expr>("values", "The reduction values."),
                 sig::arg<PrimExpr>("predicate", "Whether this thread contributes."),
                 sig::arg<Expr>("destinations", "The destination tensor loads."),
                 sig::arg<Expr>("thread_axes", "The reduction thread axes."))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_thread_allreduce"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& gpu_dp4a_op() {
  static const Op op = Op::Get("tirx.gpu_dp4a");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_dp4a")
      .signature(sig::arg<PrimExpr>("vec1", "The first input vector."),
                 sig::arg<PrimExpr>("vec2", "The second input vector."),
                 sig::arg<PrimExpr>("acc", "The accumulator."))
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_dp4a"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_load_matrix_sync")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(
          sig::arg("fragment", "The matrix fragment."), sig::arg<IntExpr>("m", "The M dimension."),
          sig::arg<IntExpr>("n", "The N dimension."), sig::arg<IntExpr>("k", "The K dimension."),
          sig::arg<IntExpr>("index", "The index."), sig::arg("buffer_ptr", "The buffer pointer."),
          sig::arg<IntExpr>("stride", "The stride."), sig::arg("layout", "The layout."))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_load_matrix_sync"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kReadState));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_mma_sync")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(sig::arg("fragment_d", "The D fragment."),
                 sig::arg<IntExpr>("index_d", "The D fragment index."),
                 sig::arg("fragment_a", "The A fragment."),
                 sig::arg<IntExpr>("index_a", "The A fragment index."),
                 sig::arg("fragment_b", "The B fragment."),
                 sig::arg<IntExpr>("index_b", "The B fragment index."),
                 sig::arg("fragment_c", "The C fragment."),
                 sig::arg<IntExpr>("index_c", "The C fragment index."))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_mma_sync"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_fill_fragment")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(
          sig::arg("fragment", "The matrix fragment."), sig::arg<IntExpr>("m", "The M dimension."),
          sig::arg<IntExpr>("n", "The N dimension."), sig::arg<IntExpr>("k", "The K dimension."),
          sig::arg<IntExpr>("index", "The index."), sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_fill_fragment"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.gpu_store_matrix_sync")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(
          sig::arg("fragment", "The matrix fragment."), sig::arg<IntExpr>("m", "The M dimension."),
          sig::arg<IntExpr>("n", "The N dimension."), sig::arg<IntExpr>("k", "The K dimension."),
          sig::arg<IntExpr>("index", "The index."), sig::arg("buffer_ptr", "The buffer pointer."),
          sig::arg<IntExpr>("stride", "The stride."), sig::arg("layout", "The layout."))
      .set_attr<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName,
                                    ffi::String("tirx.gpu_store_matrix_sync"))
      .set_attr<TIRxOpCategory>(tvm::tirx::op_attr::kOpCategory, ffi::String("builtin"))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kOpaque));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tirx.gpu_thread_return", gpu_thread_return);
}

}  // namespace tirx
}  // namespace tvm
