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
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <initializer_list>
#include <string>

namespace tvm {
namespace tirx {
namespace builtin {

namespace {
void RegisterDeviceIntrinsicAliases();
}

void RegisterCudaTargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  OpDef("tirx.tvm_load_matrix_sync")
      .arg("fragment", "")
      .arg("m", "")
      .arg("n", "")
      .arg("k", "")
      .arg("index", "")
      .arg("buffer_ptr", "")
      .arg("stride", "")
      .arg("layout", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_load_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.tvm_mma_sync")
      .arg("fragment_d", "")
      .arg("index_d", "")
      .arg("fragment_a", "")
      .arg("index_a", "")
      .arg("fragment_b", "")
      .arg("index_b", "")
      .arg("fragment_c", "")
      .arg("index_c", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_mma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_bmma_sync")
      .arg("fragment_d", "")
      .arg("index_d", "")
      .arg("fragment_a", "")
      .arg("index_a", "")
      .arg("fragment_b", "")
      .arg("index_b", "")
      .arg("fragment_c", "")
      .arg("index_c", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_bmma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_fill_fragment")
      .arg("fragment", "")
      .arg("m", "")
      .arg("n", "")
      .arg("k", "")
      .arg("index", "")
      .arg("value", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_fill_fragment"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_store_matrix_sync")
      .arg("fragment", "")
      .arg("m", "")
      .arg("n", "")
      .arg("k", "")
      .arg("index", "")
      .arg("buffer_ptr", "")
      .arg("stride", "")
      .arg("layout", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_store_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  // Siblings of mma_store / mma_fill that accept
  // (ptr_var, offset) pairs. Codegen emits `ptr + offset` C-pointer
  // arithmetic and lower_warp_memory rewrites the offset's group component
  // to its thread-local index. Used by the s_tir tensor_intrin tensorize
  // path so per-thread fragment offsets stay element-accurate.
  OpDef("tirx.mma_store_legacy")
      .arg("m", "")
      .arg("n", "")
      .arg("dst_ptr", "")
      .arg("src_ptr", "")
      .arg("src_offset", "")
      .arg("dst_stride", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.mma_fill_legacy")
      .arg("local_size", "")
      .arg("local_ptr", "")
      .arg("offset", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.s_tir.ldg32")
      .arg("reg", "")
      .arg("guard", "")
      .arg("addr", "")
      .arg("local_addr", "")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.ldg32"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"));

  // Raw legacy cp.async form emitted by InjectPTXAsyncCopy (and round-tripped by
  // the T.s_tir.cp_async_raw.legacy 6-arg surface). It carries the element dtype in Call.dtype
  // and prints it dtype-first; user-issued copies go through T.ptx instead.
  OpDef("tirx.s_tir.cp_async_raw")
      .arg("dst_ptr", "")
      .arg("dst_offset", "")
      .arg("src_ptr", "")
      .arg("src_offset", "")
      .arg("cp_size", "")
      .allow_extra_args()
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.cp_async_raw"))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_store")
      .arg("m", "")
      .arg("n", "")
      .arg("dst_ptr", "")
      .arg("src_ptr", "")
      .arg("src_offset", "")
      .arg("dst_stride", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_fill")
      .arg("local_size", "")
      .arg("local_ptr", "")
      .arg("offset", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.timer_init_cuda")
      .arg("profiler_buffer", "")
      .arg("profiler_tag", "")
      .arg("profiler_write_offset", "")
      .arg("num_groups", "")
      .arg("group_id", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_init"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_start_cuda")
      .arg("event_type", "")
      .arg("profiler_buffer", "")
      .arg("profiler_tag", "")
      .arg("profiler_write_offset", "")
      .arg("profiler_write_stride", "")
      .arg("leader_cond", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_start"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_end_cuda")
      .arg("event_type", "")
      .arg("profiler_buffer", "")
      .arg("profiler_tag", "")
      .arg("profiler_write_offset", "")
      .arg("profiler_write_stride", "")
      .arg("leader_cond", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_end"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_finalize_cuda")
      .arg("profiler_buffer", "")
      .arg("profiler_tag", "")
      .arg("profiler_write_offset", "")
      .arg("profiler_write_stride", "")
      .arg("leader_cond", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_finalize"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  RegisterDeviceIntrinsicAliases();
}

namespace {

struct DeviceIntrinsicRegistration {
  const char* name;
  const char* namespace_name;
  CallEffectKind effect_kind;
  std::initializer_list<const char*> args;
  bool allow_extra_args;
};

void RegisterDeviceIntrinsic(const DeviceIntrinsicRegistration& reg) {
  std::string name(reg.name);
  std::string namespace_name(reg.namespace_name);
  std::string prefix = namespace_name + "_";
  std::string suffix = name;
  if (suffix.rfind(prefix, 0) == 0) {
    suffix = suffix.substr(prefix.size());
  }

  std::string canonical_op_name = "tirx." + namespace_name + "." + suffix;
  ffi::String namespace_attr(namespace_name);
  // Match the nested construction namespaces at the canonical registration site.
  if (namespace_name == "cuda" &&
      (suffix.rfind("tcgen05_", 0) == 0 || suffix.rfind("wgmma_", 0) == 0)) {
    suffix[suffix.find('_')] = '.';
  } else if (namespace_name == "nvshmem" &&
             ((suffix.size() >= 6 && suffix.compare(suffix.size() - 6, 6, "_block") == 0) ||
              (suffix.size() >= 5 && suffix.compare(suffix.size() - 5, 5, "_warp") == 0))) {
    suffix[suffix.rfind('_')] = '.';
  }
  ffi::String printer_name(namespace_name + "." + suffix);
  int64_t effect = static_cast<int64_t>(reg.effect_kind);

  OpDef def(canonical_op_name);
  for (const char* name : reg.args) {
    def.arg(name, "");
  }
  if (reg.allow_extra_args) {
    def.allow_extra_args();
  }
  def.set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", namespace_attr)
      .set_attr<TCallEffectKind>("TCallEffectKind", effect)
      .set_attr<TScriptPrinterName>("TScriptPrinterName", printer_name);
}

#define TIRX_DEVICE_INTRIN_ALIAS(OpName, Namespace, EffectKind, ...) \
  {#OpName, #Namespace, CallEffectKind::EffectKind, __VA_ARGS__}

const DeviceIntrinsicRegistration kDeviceIntrinsics[] = {
    TIRX_DEVICE_INTRIN_ALIAS(cuda_any_sync, cuda, kPure, {"mask", "pred"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_atomic_add, cuda, kOpaque, {"res_addr", "value"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_atomic_cas, cuda, kOpaque, {"ptr", "old_val", "new_val"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(
        cuda_wait_until, cuda, kOpaque,
        {"dst", "ptr", "condition", "scope", "space", "ptx_type", "backoff_ns"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_ballot_sync, cuda, kOpaque, {"mask", "pred"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_bfloat1622float2, cuda, kOpaque, {"packed"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_bfloat162float, cuda, kOpaque, {"src"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_clock64, cuda, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cluster_sync, cuda, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cta_reduce, cuda, kOpaque,
                             {"value", "op", "num_warps", "scratch"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cta_sync, cuda, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cvta_generic_to_shared, cuda, kOpaque, {"ptr"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_elect_sync, cuda, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fadd2_rn, cuda, kOpaque, {"a", "b"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fdividef, cuda, kPure, {"x", "y"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_ffs_u32, cuda, kOpaque, {"value"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float22bfloat162_rn, cuda, kOpaque, {"v0", "v1"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float22bfloat162_rn_from_float2, cuda, kOpaque, {"packed"},
                             false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float22half2, cuda, kOpaque, {"dst", "src"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float2_x, cuda, kOpaque, {"packed"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float2_y, cuda, kOpaque, {"packed"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float8tohalf8, cuda, kOpaque, {"src_addr", "dst_addr"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float_as_uint, cuda, kOpaque, {"x"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fmul2_rn, cuda, kOpaque, {"a", "b"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fp8x4_e4m3_from_float4, cuda, kOpaque, {"x", "y", "z", "w"},
                             false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_func_call, cuda, kOpaque, {"func_name"}, true),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_get_tmem_addr, cuda, kOpaque,
                             {"addr", "row_offset", "col_offset"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_grid_sync, cuda, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_half2float, cuda, kOpaque, {"src"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_half8tofloat8, cuda, kOpaque, {"src_addr", "dst_addr"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_hmax2, cuda, kOpaque, {"a", "b"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_hmin2, cuda, kOpaque, {"a", "b"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_ldg, cuda, kOpaque, {}, true),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_make_float2, cuda, kOpaque, {"x", "y"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_mbarrier_wait, cuda, kOpaque, {"bar", "phase"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_mbarrier_wait_acquire_cluster, cuda, kOpaque, {"bar", "phase"},
                             false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_mov_sreg, cuda, kPure, {"bits", "reg_name"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_nano_sleep, cuda, kOpaque, {"time"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_printf, cuda, kOpaque, {"fmt"}, true),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_reduce_add_sync_u32, cuda, kOpaque, {"mask", "value"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_reduce_min_sync_u32, cuda, kOpaque, {"mask", "value"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_runtime_instr_desc, cuda, kOpaque, {"desc", "sf_id"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_sm100_2sm_leader_smem_addr, cuda, kOpaque, {"ptr"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_smem_addr_from_uint64, cuda, kOpaque, {"cluster_addr"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_syncthreads_and, cuda, kOpaque, {"cond"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_syncthreads_or, cuda, kOpaque, {"cond"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_tcgen05_encode_instr_descriptor, cuda, kOpaque,
                             {"desc", "d_dtype", "a_dtype", "b_dtype", "M", "N", "K", "trans_a",
                              "trans_b", "n_cta_groups", "neg_a", "neg_b", "sat_d", "is_sparse"},
                             false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_tcgen05_encode_instr_descriptor_block_scaled, cuda, kOpaque,
                             {"desc", "d_dtype", "a_dtype", "b_dtype", "sfa_dtype", "sfb_dtype",
                              "sfa_tmem_addr", "sfb_tmem_addr", "M", "N", "K", "trans_a", "trans_b",
                              "n_cta_groups", "neg_a", "neg_b", "is_sparse"},
                             false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_tcgen05_encode_matrix_descriptor, cuda, kOpaque,
                             {"desc", "addr", "ldo", "sdo", "swizzle"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_thread_fence, cuda, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_thread_rank, cuda, kPure, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_trap_when_assert_failed, cuda, kOpaque, {"cond"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_uint_as_float, cuda, kOpaque, {"bits"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warp_reduce, cuda, kOpaque, {"value", "op", "width"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warp_sync, cuda, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warpgroup_sync, cuda, kOpaque, {"bar_no"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_wgmma_encode_matrix_descriptor, cuda, kOpaque,
                             {"desc", "addr", "ldo", "sdo", "swizzle"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_wgmma_noop_barrier, cuda, kOpaque, {"reg"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_barrier_all, nvshmem, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_fence, nvshmem, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_getmem_nbi, nvshmem, kOpaque, {"dst", "src", "nelems", "pe"},
                             false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_getmem_nbi_block, nvshmem, kOpaque,
                             {"dst", "src", "nelems", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_getmem_nbi_warp, nvshmem, kOpaque,
                             {"dst", "src", "nelems", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_my_pe, nvshmem, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_n_pes, nvshmem, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_nbi, nvshmem, kOpaque, {"dst", "src", "nelems", "pe"},
                             false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_nbi_block, nvshmem, kOpaque,
                             {"dst", "src", "nelems", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_nbi_warp, nvshmem, kOpaque,
                             {"dst", "src", "nelems", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_signal_nbi, nvshmem, kOpaque,
                             {"dst", "src", "nelems", "sig_addr", "signal", "sig_op", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_signal_nbi_block, nvshmem, kOpaque,
                             {"dst", "src", "nelems", "sig_addr", "signal", "sig_op", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_signal_nbi_warp, nvshmem, kOpaque,
                             {"dst", "src", "nelems", "sig_addr", "signal", "sig_op", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_quiet, nvshmem, kOpaque, {}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_signal_op, nvshmem, kOpaque,
                             {"sig_addr", "signal", "sig_op", "pe"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_wait_until, nvshmem, kOpaque,
                             {"ivar", "cmp", "cmp_value", "type"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(
        ptx_legacy_ldmatrix,
        ptx_legacy, kOpaque,
        {"trans", "num", "dtype", "local_ptr", "local_offset", "smem_ptr", "smem_offset"}, false),
    TIRX_DEVICE_INTRIN_ALIAS(
        ptx_legacy_mma, ptx_legacy, kOpaque,
        {"shape", "a_layout", "b_layout", "a_dtype", "b_dtype", "c_dtype", "a_ptr", "a_offset",
         "b_ptr", "b_offset", "acc_ptr", "c_offset", "saturate"},
        true),
};

#undef TIRX_DEVICE_INTRIN_ALIAS

void RegisterDeviceIntrinsicAliases() {
  for (const auto& reg : kDeviceIntrinsics) {
    RegisterDeviceIntrinsic(reg);
  }
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
