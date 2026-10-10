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
 * \file backend/cuda/op/target_builtin.cc
 *
 *  builtin intrinsic operators specific to CUDA target.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/type.h>

#include <string>

namespace tvm {

namespace tirx {
namespace builtin {

namespace {
void RegisterDeviceIntrinsics();

Type InferTypeMovSreg(const CallNode* call) {
  TVM_FFI_CHECK(!call->args.empty(), TypeError) << "mov_sreg expects a bit width";
  const auto* bits = call->args[0].as<IntImmNode>();
  TVM_FFI_CHECK(bits, TypeError) << "mov_sreg expects a constant bit width";
  return PrimType::Int(static_cast<int64_t>(bits->value));
}

Type InferTypeDeclTmem(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError) << "decl_tmem expects one address operand";
  TVM_FFI_CHECK_EQ(call->ty_args.size(), 1U, ValueError)
      << "decl_tmem expects exactly one TensorType argument";
  auto type = call->ty_args[0].as_or_throw<TensorType>();
  TVM_FFI_CHECK_EQ(type->storage_scope, "tmem", ValueError)
      << "decl_tmem requires a tmem tensor type";
  auto addr_type = call->args[0]->ty.as_or_throw<PrimType>();
  TVM_FFI_CHECK(addr_type.IsScalar() &&
                    addr_type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
                    addr_type.bits() == 32,
                TypeError)
      << "decl_tmem address must be a scalar 32-bit integer";
  TVM_FFI_CHECK(prim::IsZero(type->elem_offset) && type->strides.empty(), ValueError)
      << "decl_tmem addresses use physical columns; express tensor offsets through the layout";
  return type;
}

void ValidateDeclTmem(const CallNode* call) {
  Type inferred = InferTypeDeclTmem(call);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, inferred), TypeError)
      << "decl_tmem result must match its TensorType argument";
  TVM_FFI_CHECK(!call->attrs.defined(), ValueError) << "decl_tmem does not accept attributes";
}

Type InferTypeCudaAxis(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError);
  const auto* axis = call->args[0].as<StringImmNode>();
  TVM_FFI_CHECK(axis && (axis->value == "x" || axis->value == "y" || axis->value == "z"),
                ValueError)
      << "CUDA index axis must be x, y, or z";
  return PrimType::Int(32);
}

template <size_t N>
static Type InferTypeReturnArgType(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

}  // namespace

void RegisterCudaTargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  OpDef("tirx.cuda.decl_tmem")
      .signature(sig::arg<IntExpr>("addr", "Encoded TMEM base address in physical columns."),
                 sig::ty_arg<TensorType>("tensor_type"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeDeclTmem>())
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateDeclTmem>(),
                     true)
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.decl_tmem"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.cuda.bmma_sync")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .signature(sig::arg("fragment_d", "The D fragment."),
                 sig::arg<IntExpr>("index_d", "The D fragment index."),
                 sig::arg("fragment_a", "The A fragment."),
                 sig::arg<IntExpr>("index_a", "The A fragment index."),
                 sig::arg("fragment_b", "The B fragment."),
                 sig::arg<IntExpr>("index_b", "The B fragment index."),
                 sig::arg("fragment_c", "The C fragment."),
                 sig::arg<IntExpr>("index_c", "The C fragment index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.bmma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  // Raw legacy cp.async form emitted by InjectPTXAsyncCopy (and round-tripped by
  // the T.s_tir.cp_async_raw.legacy 6-arg surface). It carries the element dtype in Call.dtype
  // and prints it with ty=; user-issued copies go through T.ptx instead.
  OpDef("tirx.s_tir.cp_async_raw")
      .signature(sig::arg("dst_ptr", "The destination pointer."),
                 sig::arg<IntExpr>("dst_offset", "The destination offset."),
                 sig::arg("src_ptr", "The source pointer."),
                 sig::arg<IntExpr>("src_offset", "The source offset."),
                 sig::arg<IntExpr>("cp_size", "The copy size."), sig::var_args("args"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.s_tir.cp_async_raw"));

  OpDef("tirx.cuda.mma_store")
      .signature(sig::arg<IntExpr>("m", "The M dimension."),
                 sig::arg<IntExpr>("n", "The N dimension."),
                 sig::arg("dst_ptr", "The destination pointer."),
                 sig::arg("src_ptr", "The source pointer."),
                 sig::arg<IntExpr>("src_offset", "The source offset."),
                 sig::arg<IntExpr>("dst_stride", "The destination stride."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.mma_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.mma_fill")
      .signature(sig::arg<IntExpr>("local_size", "The local allocation size."),
                 sig::arg("local_ptr", "The local pointer."),
                 sig::arg<IntExpr>("offset", "The offset."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.mma_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.timer_init")
      .signature(sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("num_groups", "The number of groups."),
                 sig::arg<IntExpr>("group_id", "The group identifier."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.timer_init"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.timer_start")
      .signature(sig::arg("event_type", "The event type."),
                 sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.timer_start"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.timer_end")
      .signature(sig::arg("event_type", "The event type."),
                 sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.timer_end"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.cuda.timer_finalize")
      .signature(sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cuda.timer_finalize"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  RegisterDeviceIntrinsics();
}

namespace {

TVM_FFI_NO_INLINE void RegisterDeviceIntrinsicAttrs(OpDef& def, const char* op_namespace,
                                                    CallEffectKind effect_kind) {
  std::string printer_name = def.op()->name;
  if (std::string(op_namespace) == "nvshmem" &&
      ((printer_name.size() >= 6 &&
        printer_name.compare(printer_name.size() - 6, 6, "_block") == 0) ||
       (printer_name.size() >= 5 &&
        printer_name.compare(printer_name.size() - 5, 5, "_warp") == 0))) {
    printer_name[printer_name.rfind('_')] = '.';
  }
  def.set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String(op_namespace))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(effect_kind));
  if (std::string(op_namespace) != "cuda") {
    def.set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String(printer_name));
  }
}

template <typename... Specs>
OpDef& RegisterDeviceIntrinsic(OpDef&& def, const char* op_namespace, CallEffectKind effect_kind,
                               const Specs&... specs) {
  def.signature(specs...);
  RegisterDeviceIntrinsicAttrs(def, op_namespace, effect_kind);
  return def;
}

void RegisterDeviceIntrinsics() {
  for (const char* name : {"block_idx", "thread_idx", "cluster_id", "cluster_cta_id", "grid_dim",
                           "block_dim", "cluster_dim"}) {
    RegisterDeviceIntrinsic(OpDef(std::string("tirx.cuda.") + name), "cuda", CallEffectKind::kPure,
                            sig::arg<StringImm>("axis"))
        .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeCudaAxis>());
  }
  for (const char* name : {"linear_thread_id", "warp_id", "lane_id", "warpgroup_id",
                           "warp_in_warpgroup", "thread_in_warpgroup", "cta_pair_id"}) {
    RegisterDeviceIntrinsic(OpDef(std::string("tirx.cuda.") + name), "cuda", CallEffectKind::kPure)
        .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32));
  }
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.any_sync"), "cuda", CallEffectKind::kPure,
                          sig::arg<IntExpr>("mask"), sig::arg("pred"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.atomic_add"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("res_addr"), sig::arg("value"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<1>>());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.atomic_cas"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("ptr"), sig::arg("old_val"), sig::arg("new_val"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<1>>());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.wait_until"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("ptr"), sig::arg("condition"),
                          sig::arg("scope"), sig::arg("space"), sig::arg("ptx_type"),
                          sig::arg<IntExpr>("backoff_ns"));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.ballot_sync"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("mask"), sig::arg("pred"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.bfloat1622float2"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("packed"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(64));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.bfloat162float"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("src"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Float(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.clock64"), "cuda", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(64));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cluster_sync"), "cuda", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cta_reduce"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("value"), sig::arg("op"), sig::arg<IntExpr>("num_warps"),
                          sig::arg("scratch"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cta_sync"), "cuda", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cvta_generic_to_shared"), "cuda",
                          CallEffectKind::kOpaque, sig::arg("ptr"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.elect_sync"), "cuda", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fadd2_rn"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("a"), sig::arg("b"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(64));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fdividef"), "cuda", CallEffectKind::kPure, sig::arg("x"),
                          sig::arg("y"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Float(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.ffs_u32"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("value"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float22bfloat162_rn"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("v0"), sig::arg("v1"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float22bfloat162_rn_from_float2"), "cuda",
                          CallEffectKind::kOpaque, sig::arg("packed"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float22half2"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float2_x"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("packed"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Float(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float2_y"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("packed"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Float(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float8tohalf8"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("src_addr"), sig::arg("dst_addr"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float_as_uint"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("x"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fmul2_rn"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("a"), sig::arg("b"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(64));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fp8x4_e4m3_from_float4"), "cuda",
                          CallEffectKind::kOpaque, sig::arg("x"), sig::arg("y"), sig::arg("z"),
                          sig::arg("w"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.func_call"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("func_name"), sig::var_args("args"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.get_tmem_addr"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("addr"), sig::arg<IntExpr>("row_offset"),
                          sig::arg<IntExpr>("col_offset"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.grid_sync"), "cuda", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.half2float"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("src"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Float(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.half8tofloat8"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("src_addr"), sig::arg("dst_addr"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.hmax2"), "cuda", CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.hmin2"), "cuda", CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.make_float2"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("x"), sig::arg("y"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(64));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.mbarrier_wait"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("bar"), sig::arg<IntExpr>("phase"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.mbarrier_wait_acquire_cluster"), "cuda",
                          CallEffectKind::kOpaque, sig::arg("bar"), sig::arg<IntExpr>("phase"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.mov_sreg"), "cuda", CallEffectKind::kPure,
                          sig::arg<IntExpr>("bits"), sig::arg("reg_name"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeMovSreg>());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.nano_sleep"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("time"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.printf"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("fmt"), sig::var_args("args"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.reduce_add_sync_u32"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("mask"), sig::arg<IntExpr>("value"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.reduce_min_sync_u32"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("mask"), sig::arg<IntExpr>("value"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.runtime_instr_desc"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("desc"), sig::arg<IntExpr>("sf_id"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.smem_addr_from_uint64"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("cluster_addr"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::UInt(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.syncthreads_and"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("cond"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(64));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.syncthreads_or"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("cond"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(64));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.thread_fence"), "cuda", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.thread_rank"), "cuda", CallEffectKind::kPure)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.trap_when_assert_failed"), "cuda",
                          CallEffectKind::kOpaque, sig::arg("cond"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.uint_as_float"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("bits"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Float(32));
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.warp_reduce"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("value"), sig::arg("op"), sig::arg<IntExpr>("width"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.warp_sync"), "cuda", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.warpgroup_sync"), "cuda", CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("bar_no"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.wgmma_encode_matrix_descriptor"), "cuda",
                          CallEffectKind::kOpaque, sig::arg("desc"), sig::arg("addr"),
                          sig::arg<IntExpr>("ldo"), sig::arg<IntExpr>("sdo"),
                          sig::arg<IntExpr>("swizzle"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.wgmma_noop_barrier"), "cuda", CallEffectKind::kOpaque,
                          sig::arg("reg"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.barrier_all"), "nvshmem", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.fence"), "nvshmem", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.getmem_nbi"), "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.getmem_nbi_block"), "nvshmem",
                          CallEffectKind::kOpaque, sig::arg("dst"), sig::arg("src"),
                          sig::arg<IntExpr>("nelems"), sig::arg<IntExpr>("pe"));
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.getmem_nbi_warp"), "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.my_pe"), "nvshmem", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32));
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.n_pes"), "nvshmem", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32));
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_nbi"), "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_nbi_block"), "nvshmem",
                          CallEffectKind::kOpaque, sig::arg("dst"), sig::arg("src"),
                          sig::arg<IntExpr>("nelems"), sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_nbi_warp"), "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_signal_nbi"), "nvshmem",
                          CallEffectKind::kOpaque, sig::arg("dst"), sig::arg("src"),
                          sig::arg<IntExpr>("nelems"), sig::arg("sig_addr"),
                          sig::arg<IntExpr>("signal"), sig::arg("sig_op"), sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_signal_nbi_block"), "nvshmem",
                          CallEffectKind::kOpaque, sig::arg("dst"), sig::arg("src"),
                          sig::arg<IntExpr>("nelems"), sig::arg("sig_addr"),
                          sig::arg<IntExpr>("signal"), sig::arg("sig_op"), sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_signal_nbi_warp"), "nvshmem",
                          CallEffectKind::kOpaque, sig::arg("dst"), sig::arg("src"),
                          sig::arg<IntExpr>("nelems"), sig::arg("sig_addr"),
                          sig::arg<IntExpr>("signal"), sig::arg("sig_op"), sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.quiet"), "nvshmem", CallEffectKind::kOpaque)
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.signal_op"), "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("sig_addr"), sig::arg<IntExpr>("signal"), sig::arg("sig_op"),
                          sig::arg<IntExpr>("pe"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.wait_until"), "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("ivar"), sig::arg("cmp"), sig::arg<IntExpr>("cmp_value"),
                          sig::arg("type"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void());
  RegisterDeviceIntrinsic(OpDef("tirx.ptx_legacy.ldmatrix"), "ptx_legacy", CallEffectKind::kOpaque,
                          sig::arg("trans"), sig::arg<IntExpr>("num"), sig::arg("dtype"),
                          sig::arg("local_ptr"), sig::arg<IntExpr>("local_offset"),
                          sig::arg("smem_ptr"), sig::arg<IntExpr>("smem_offset"));
  RegisterDeviceIntrinsic(
      OpDef("tirx.ptx_legacy.mma"), "ptx_legacy", CallEffectKind::kOpaque, sig::arg("shape"),
      sig::arg("a_layout"), sig::arg("b_layout"), sig::arg("a_dtype"), sig::arg("b_dtype"),
      sig::arg("c_dtype"), sig::arg("a_ptr"), sig::arg<IntExpr>("a_offset"), sig::arg("b_ptr"),
      sig::arg<IntExpr>("b_offset"), sig::arg("acc_ptr"), sig::arg<IntExpr>("c_offset"),
      sig::arg("saturate"), sig::var_args("args"));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
