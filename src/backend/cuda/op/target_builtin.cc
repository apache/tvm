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
      .arg<Expr>("fragment", "")
      .arg<Expr>("m", "")
      .arg<Expr>("n", "")
      .arg<Expr>("k", "")
      .arg<Expr>("index", "")
      .arg<Expr>("buffer_ptr", "")
      .arg<Expr>("stride", "")
      .arg<Expr>("layout", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_load_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.tvm_mma_sync")
      .arg<Expr>("fragment_d", "")
      .arg<Expr>("index_d", "")
      .arg<Expr>("fragment_a", "")
      .arg<Expr>("index_a", "")
      .arg<Expr>("fragment_b", "")
      .arg<Expr>("index_b", "")
      .arg<Expr>("fragment_c", "")
      .arg<Expr>("index_c", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_mma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_bmma_sync")
      .arg<Expr>("fragment_d", "")
      .arg<Expr>("index_d", "")
      .arg<Expr>("fragment_a", "")
      .arg<Expr>("index_a", "")
      .arg<Expr>("fragment_b", "")
      .arg<Expr>("index_b", "")
      .arg<Expr>("fragment_c", "")
      .arg<Expr>("index_c", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_bmma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_fill_fragment")
      .arg<Expr>("fragment", "")
      .arg<Expr>("m", "")
      .arg<Expr>("n", "")
      .arg<Expr>("k", "")
      .arg<Expr>("index", "")
      .arg<Expr>("value", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_fill_fragment"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_store_matrix_sync")
      .arg<Expr>("fragment", "")
      .arg<Expr>("m", "")
      .arg<Expr>("n", "")
      .arg<Expr>("k", "")
      .arg<Expr>("index", "")
      .arg<Expr>("buffer_ptr", "")
      .arg<Expr>("stride", "")
      .arg<Expr>("layout", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_store_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  // Siblings of mma_store / mma_fill that accept
  // (ptr_var, offset) pairs. Codegen emits `ptr + offset` C-pointer
  // arithmetic and lower_warp_memory rewrites the offset's group component
  // to its thread-local index. Used by the s_tir tensor_intrin tensorize
  // path so per-thread fragment offsets stay element-accurate.
  OpDef("tirx.mma_store_legacy")
      .arg<Expr>("m", "")
      .arg<Expr>("n", "")
      .arg<Expr>("dst_ptr", "")
      .arg<Expr>("src_ptr", "")
      .arg<Expr>("src_offset", "")
      .arg<Expr>("dst_stride", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.mma_fill_legacy")
      .arg<Expr>("local_size", "")
      .arg<Expr>("local_ptr", "")
      .arg<Expr>("offset", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.s_tir.ldg32")
      .arg<Expr>("reg", "")
      .arg<Expr>("guard", "")
      .arg<Expr>("addr", "")
      .arg<Expr>("local_addr", "")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.ldg32"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"));

  // Raw legacy cp.async form emitted by InjectPTXAsyncCopy (and round-tripped by
  // the T.s_tir.cp_async_raw.legacy 6-arg surface). It carries the element dtype in Call.dtype
  // and prints it dtype-first; user-issued copies go through T.ptx instead.
  OpDef("tirx.s_tir.cp_async_raw")
      .arg<Expr>("dst_ptr", "")
      .arg<Expr>("dst_offset", "")
      .arg<Expr>("src_ptr", "")
      .arg<Expr>("src_offset", "")
      .arg<Expr>("cp_size", "")
      .allow_extra_args()
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.cp_async_raw"))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_store")
      .arg<Expr>("m", "")
      .arg<Expr>("n", "")
      .arg<Expr>("dst_ptr", "")
      .arg<Expr>("src_ptr", "")
      .arg<Expr>("src_offset", "")
      .arg<Expr>("dst_stride", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_fill")
      .arg<Expr>("local_size", "")
      .arg<Expr>("local_ptr", "")
      .arg<Expr>("offset", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.timer_init_cuda")
      .arg<Expr>("profiler_buffer", "")
      .arg<Expr>("profiler_tag", "")
      .arg<Expr>("profiler_write_offset", "")
      .arg<Expr>("num_groups", "")
      .arg<Expr>("group_id", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_init"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_start_cuda")
      .arg<Expr>("event_type", "")
      .arg<Expr>("profiler_buffer", "")
      .arg<Expr>("profiler_tag", "")
      .arg<Expr>("profiler_write_offset", "")
      .arg<Expr>("profiler_write_stride", "")
      .arg<Expr>("leader_cond", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_start"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_end_cuda")
      .arg<Expr>("event_type", "")
      .arg<Expr>("profiler_buffer", "")
      .arg<Expr>("profiler_tag", "")
      .arg<Expr>("profiler_write_offset", "")
      .arg<Expr>("profiler_write_stride", "")
      .arg<Expr>("leader_cond", "")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_end"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_finalize_cuda")
      .arg<Expr>("profiler_buffer", "")
      .arg<Expr>("profiler_tag", "")
      .arg<Expr>("profiler_write_offset", "")
      .arg<Expr>("profiler_write_stride", "")
      .arg<Expr>("leader_cond", "")
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
    def.arg<Expr>(name, "");
  }
  if (reg.allow_extra_args) {
    def.allow_extra_args();
  }
  def.set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", namespace_attr)
      .set_attr<TCallEffectKind>("TCallEffectKind", effect)
      .set_attr<TScriptPrinterName>("TScriptPrinterName", printer_name);
}

const DeviceIntrinsicRegistration kDeviceIntrinsics[] = {
    {"cuda_any_sync", "cuda", CallEffectKind::kPure, {"mask", "pred"}, false},
    {"cuda_atomic_add", "cuda", CallEffectKind::kOpaque, {"res_addr", "value"}, false},
    {"cuda_atomic_cas", "cuda", CallEffectKind::kOpaque, {"ptr", "old_val", "new_val"}, false},
    {"cuda_wait_until",
     "cuda",
     CallEffectKind::kOpaque,
     {"dst", "ptr", "condition", "scope", "space", "ptx_type", "backoff_ns"},
     false},
    {"cuda_ballot_sync", "cuda", CallEffectKind::kOpaque, {"mask", "pred"}, false},
    {"cuda_bfloat1622float2", "cuda", CallEffectKind::kOpaque, {"packed"}, false},
    {"cuda_bfloat162float", "cuda", CallEffectKind::kOpaque, {"src"}, false},
    {"cuda_clock64", "cuda", CallEffectKind::kOpaque, {}, false},
    {"cuda_cluster_sync", "cuda", CallEffectKind::kOpaque, {}, false},
    {"cuda_cta_reduce",
     "cuda",
     CallEffectKind::kOpaque,
     {"value", "op", "num_warps", "scratch"},
     false},
    {"cuda_cta_sync", "cuda", CallEffectKind::kOpaque, {}, false},
    {"cuda_cvta_generic_to_shared", "cuda", CallEffectKind::kOpaque, {"ptr"}, false},
    {"cuda_elect_sync", "cuda", CallEffectKind::kOpaque, {}, false},
    {"cuda_fadd2_rn", "cuda", CallEffectKind::kOpaque, {"a", "b"}, false},
    {"cuda_fdividef", "cuda", CallEffectKind::kPure, {"x", "y"}, false},
    {"cuda_ffs_u32", "cuda", CallEffectKind::kOpaque, {"value"}, false},
    {"cuda_float22bfloat162_rn", "cuda", CallEffectKind::kOpaque, {"v0", "v1"}, false},
    {"cuda_float22bfloat162_rn_from_float2", "cuda", CallEffectKind::kOpaque, {"packed"}, false},
    {"cuda_float22half2", "cuda", CallEffectKind::kOpaque, {"dst", "src"}, false},
    {"cuda_float2_x", "cuda", CallEffectKind::kOpaque, {"packed"}, false},
    {"cuda_float2_y", "cuda", CallEffectKind::kOpaque, {"packed"}, false},
    {"cuda_float8tohalf8", "cuda", CallEffectKind::kOpaque, {"src_addr", "dst_addr"}, false},
    {"cuda_float_as_uint", "cuda", CallEffectKind::kOpaque, {"x"}, false},
    {"cuda_fmul2_rn", "cuda", CallEffectKind::kOpaque, {"a", "b"}, false},
    {"cuda_fp8x4_e4m3_from_float4", "cuda", CallEffectKind::kOpaque, {"x", "y", "z", "w"}, false},
    {"cuda_func_call", "cuda", CallEffectKind::kOpaque, {"func_name"}, true},
    {"cuda_get_tmem_addr",
     "cuda",
     CallEffectKind::kOpaque,
     {"addr", "row_offset", "col_offset"},
     false},
    {"cuda_grid_sync", "cuda", CallEffectKind::kOpaque, {}, false},
    {"cuda_half2float", "cuda", CallEffectKind::kOpaque, {"src"}, false},
    {"cuda_half8tofloat8", "cuda", CallEffectKind::kOpaque, {"src_addr", "dst_addr"}, false},
    {"cuda_hmax2", "cuda", CallEffectKind::kOpaque, {"a", "b"}, false},
    {"cuda_hmin2", "cuda", CallEffectKind::kOpaque, {"a", "b"}, false},
    {"cuda_ldg", "cuda", CallEffectKind::kOpaque, {}, true},
    {"cuda_make_float2", "cuda", CallEffectKind::kOpaque, {"x", "y"}, false},
    {"cuda_mbarrier_wait", "cuda", CallEffectKind::kOpaque, {"bar", "phase"}, false},
    {"cuda_mbarrier_wait_acquire_cluster",
     "cuda",
     CallEffectKind::kOpaque,
     {"bar", "phase"},
     false},
    {"cuda_mov_sreg", "cuda", CallEffectKind::kPure, {"bits", "reg_name"}, false},
    {"cuda_nano_sleep", "cuda", CallEffectKind::kOpaque, {"time"}, false},
    {"cuda_printf", "cuda", CallEffectKind::kOpaque, {"fmt"}, true},
    {"cuda_reduce_add_sync_u32", "cuda", CallEffectKind::kOpaque, {"mask", "value"}, false},
    {"cuda_reduce_min_sync_u32", "cuda", CallEffectKind::kOpaque, {"mask", "value"}, false},
    {"cuda_runtime_instr_desc", "cuda", CallEffectKind::kOpaque, {"desc", "sf_id"}, false},
    {"cuda_sm100_2sm_leader_smem_addr", "cuda", CallEffectKind::kOpaque, {"ptr"}, false},
    {"cuda_smem_addr_from_uint64", "cuda", CallEffectKind::kOpaque, {"cluster_addr"}, false},
    {"cuda_syncthreads_and", "cuda", CallEffectKind::kOpaque, {"cond"}, false},
    {"cuda_syncthreads_or", "cuda", CallEffectKind::kOpaque, {"cond"}, false},
    {"cuda_tcgen05_encode_instr_descriptor",
     "cuda",
     CallEffectKind::kOpaque,
     {"desc", "d_dtype", "a_dtype", "b_dtype", "M", "N", "K", "trans_a", "trans_b", "n_cta_groups",
      "neg_a", "neg_b", "sat_d", "is_sparse"},
     false},
    {"cuda_tcgen05_encode_instr_descriptor_block_scaled",
     "cuda",
     CallEffectKind::kOpaque,
     {"desc", "d_dtype", "a_dtype", "b_dtype", "sfa_dtype", "sfb_dtype", "sfa_tmem_addr",
      "sfb_tmem_addr", "M", "N", "K", "trans_a", "trans_b", "n_cta_groups", "neg_a", "neg_b",
      "is_sparse"},
     false},
    {"cuda_tcgen05_encode_matrix_descriptor",
     "cuda",
     CallEffectKind::kOpaque,
     {"desc", "addr", "ldo", "sdo", "swizzle"},
     false},
    {"cuda_thread_fence", "cuda", CallEffectKind::kOpaque, {}, false},
    {"cuda_thread_rank", "cuda", CallEffectKind::kPure, {}, false},
    {"cuda_trap_when_assert_failed", "cuda", CallEffectKind::kOpaque, {"cond"}, false},
    {"cuda_uint_as_float", "cuda", CallEffectKind::kOpaque, {"bits"}, false},
    {"cuda_warp_reduce", "cuda", CallEffectKind::kOpaque, {"value", "op", "width"}, false},
    {"cuda_warp_sync", "cuda", CallEffectKind::kOpaque, {}, false},
    {"cuda_warpgroup_sync", "cuda", CallEffectKind::kOpaque, {"bar_no"}, false},
    {"cuda_wgmma_encode_matrix_descriptor",
     "cuda",
     CallEffectKind::kOpaque,
     {"desc", "addr", "ldo", "sdo", "swizzle"},
     false},
    {"cuda_wgmma_noop_barrier", "cuda", CallEffectKind::kOpaque, {"reg"}, false},
    {"nvshmem_barrier_all", "nvshmem", CallEffectKind::kOpaque, {}, false},
    {"nvshmem_fence", "nvshmem", CallEffectKind::kOpaque, {}, false},
    {"nvshmem_getmem_nbi",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "pe"},
     false},
    {"nvshmem_getmem_nbi_block",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "pe"},
     false},
    {"nvshmem_getmem_nbi_warp",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "pe"},
     false},
    {"nvshmem_my_pe", "nvshmem", CallEffectKind::kOpaque, {}, false},
    {"nvshmem_n_pes", "nvshmem", CallEffectKind::kOpaque, {}, false},
    {"nvshmem_putmem_nbi",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "pe"},
     false},
    {"nvshmem_putmem_nbi_block",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "pe"},
     false},
    {"nvshmem_putmem_nbi_warp",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "pe"},
     false},
    {"nvshmem_putmem_signal_nbi",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "sig_addr", "signal", "sig_op", "pe"},
     false},
    {"nvshmem_putmem_signal_nbi_block",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "sig_addr", "signal", "sig_op", "pe"},
     false},
    {"nvshmem_putmem_signal_nbi_warp",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"dst", "src", "nelems", "sig_addr", "signal", "sig_op", "pe"},
     false},
    {"nvshmem_quiet", "nvshmem", CallEffectKind::kOpaque, {}, false},
    {"nvshmem_signal_op",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"sig_addr", "signal", "sig_op", "pe"},
     false},
    {"nvshmem_wait_until",
     "nvshmem",
     CallEffectKind::kOpaque,
     {"ivar", "cmp", "cmp_value", "type"},
     false},
    {"ptx_legacy_ldmatrix",
     "ptx_legacy",
     CallEffectKind::kOpaque,
     {"trans", "num", "dtype", "local_ptr", "local_offset", "smem_ptr", "smem_offset"},
     false},
    {"ptx_legacy_mma",
     "ptx_legacy",
     CallEffectKind::kOpaque,
     {"shape", "a_layout", "b_layout", "a_dtype", "b_dtype", "c_dtype", "a_ptr", "a_offset",
      "b_ptr", "b_offset", "acc_ptr", "c_offset", "saturate"},
     true},
};

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
