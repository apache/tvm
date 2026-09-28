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
#include <utility>

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
      .signature(sig::arg("fragment", "The matrix fragment."), sig::arg("m", "The M dimension."),
                 sig::arg("n", "The N dimension."), sig::arg("k", "The K dimension."),
                 sig::arg("index", "The index."), sig::arg("buffer_ptr", "The buffer pointer."),
                 sig::arg("stride", "The stride."), sig::arg("layout", "The layout."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_load_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.tvm_mma_sync")
      .signature(
          sig::arg("fragment_d", "The D fragment."), sig::arg("index_d", "The D fragment index."),
          sig::arg("fragment_a", "The A fragment."), sig::arg("index_a", "The A fragment index."),
          sig::arg("fragment_b", "The B fragment."), sig::arg("index_b", "The B fragment index."),
          sig::arg("fragment_c", "The C fragment."), sig::arg("index_c", "The C fragment index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_mma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_bmma_sync")
      .signature(
          sig::arg("fragment_d", "The D fragment."), sig::arg("index_d", "The D fragment index."),
          sig::arg("fragment_a", "The A fragment."), sig::arg("index_a", "The A fragment index."),
          sig::arg("fragment_b", "The B fragment."), sig::arg("index_b", "The B fragment index."),
          sig::arg("fragment_c", "The C fragment."), sig::arg("index_c", "The C fragment index."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_bmma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_fill_fragment")
      .signature(sig::arg("fragment", "The matrix fragment."), sig::arg("m", "The M dimension."),
                 sig::arg("n", "The N dimension."), sig::arg("k", "The K dimension."),
                 sig::arg("index", "The index."), sig::arg("value", "The value to use."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_fill_fragment"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_store_matrix_sync")
      .signature(sig::arg("fragment", "The matrix fragment."), sig::arg("m", "The M dimension."),
                 sig::arg("n", "The N dimension."), sig::arg("k", "The K dimension."),
                 sig::arg("index", "The index."), sig::arg("buffer_ptr", "The buffer pointer."),
                 sig::arg("stride", "The stride."), sig::arg("layout", "The layout."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_store_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  // Siblings of mma_store / mma_fill that accept
  // (ptr_var, offset) pairs. Codegen emits `ptr + offset` C-pointer
  // arithmetic and lower_warp_memory rewrites the offset's group component
  // to its thread-local index. Used by the s_tir tensor_intrin tensorize
  // path so per-thread fragment offsets stay element-accurate.
  OpDef("tirx.mma_store_legacy")
      .signature(sig::arg("m", "The M dimension."), sig::arg("n", "The N dimension."),
                 sig::arg("dst_ptr", "The destination pointer."),
                 sig::arg("src_ptr", "The source pointer."),
                 sig::arg("src_offset", "The source offset."),
                 sig::arg("dst_stride", "The destination stride."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.mma_fill_legacy")
      .signature(sig::arg("local_size", "The local allocation size."),
                 sig::arg("local_ptr", "The local pointer."), sig::arg("offset", "The offset."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.s_tir.ldg32")
      .signature(sig::arg("reg", "The register."), sig::arg("guard", "The guard predicate."),
                 sig::arg("addr", "The address."), sig::arg("local_addr", "The local address."))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.ldg32"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"));

  // Raw legacy cp.async form emitted by InjectPTXAsyncCopy (and round-tripped by
  // the T.s_tir.cp_async_raw.legacy 6-arg surface). It carries the element dtype in Call.dtype
  // and prints it dtype-first; user-issued copies go through T.ptx instead.
  OpDef("tirx.s_tir.cp_async_raw")
      .signature(sig::arg("dst_ptr", "The destination pointer."),
                 sig::arg("dst_offset", "The destination offset."),
                 sig::arg("src_ptr", "The source pointer."),
                 sig::arg("src_offset", "The source offset."),
                 sig::arg("cp_size", "The copy size."), sig::var_args("args"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.cp_async_raw"))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_store")
      .signature(sig::arg("m", "The M dimension."), sig::arg("n", "The N dimension."),
                 sig::arg("dst_ptr", "The destination pointer."),
                 sig::arg("src_ptr", "The source pointer."),
                 sig::arg("src_offset", "The source offset."),
                 sig::arg("dst_stride", "The destination stride."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_fill")
      .signature(sig::arg("local_size", "The local allocation size."),
                 sig::arg("local_ptr", "The local pointer."), sig::arg("offset", "The offset."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.timer_init_cuda")
      .signature(sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg("profiler_write_offset", "The profiler write offset."),
                 sig::arg("num_groups", "The number of groups."),
                 sig::arg("group_id", "The group identifier."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_init"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_start_cuda")
      .signature(sig::arg("event_type", "The event type."),
                 sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg("profiler_write_offset", "The profiler write offset."),
                 sig::arg("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_start"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_end_cuda")
      .signature(sig::arg("event_type", "The event type."),
                 sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg("profiler_write_offset", "The profiler write offset."),
                 sig::arg("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_end"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_finalize_cuda")
      .signature(sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg("profiler_write_offset", "The profiler write offset."),
                 sig::arg("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_finalize"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  RegisterDeviceIntrinsicAliases();
}

namespace {

struct DeviceIntrinsicNames {
  std::string canonical;
  std::string printer;
};

TVM_FFI_NO_INLINE DeviceIntrinsicNames MakeDeviceIntrinsicNames(const char* op_name,
                                                                const char* op_namespace) {
  std::string name(op_name);
  std::string namespace_name(op_namespace);
  std::string prefix = namespace_name + "_";
  std::string suffix = name;
  if (suffix.rfind(prefix, 0) == 0) {
    suffix = suffix.substr(prefix.size());
  }

  std::string canonical = "tirx." + namespace_name + "." + suffix;
  // Match the nested construction namespaces at the canonical registration site.
  if (namespace_name == "cuda" &&
      (suffix.rfind("tcgen05_", 0) == 0 || suffix.rfind("wgmma_", 0) == 0)) {
    suffix[suffix.find('_')] = '.';
  } else if (namespace_name == "nvshmem" &&
             ((suffix.size() >= 6 && suffix.compare(suffix.size() - 6, 6, "_block") == 0) ||
              (suffix.size() >= 5 && suffix.compare(suffix.size() - 5, 5, "_warp") == 0))) {
    suffix[suffix.rfind('_')] = '.';
  }
  return {std::move(canonical), namespace_name + "." + suffix};
}

TVM_FFI_NO_INLINE void RegisterDeviceIntrinsicAttrs(OpDef& def, const char* op_namespace,
                                                    CallEffectKind effect_kind,
                                                    const std::string& printer_name) {
  def.set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String(op_namespace))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(effect_kind))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String(printer_name));
}

template <typename... Specs>
void RegisterDeviceIntrinsic(const char* op_name, const char* op_namespace,
                             CallEffectKind effect_kind, const Specs&... specs) {
  DeviceIntrinsicNames names = MakeDeviceIntrinsicNames(op_name, op_namespace);
  OpDef def(names.canonical);
  def.signature(specs...);
  RegisterDeviceIntrinsicAttrs(def, op_namespace, effect_kind, names.printer);
}

void RegisterDeviceIntrinsicAliases() {
  RegisterDeviceIntrinsic("cuda_any_sync", "cuda", CallEffectKind::kPure, sig::arg("mask"),
                          sig::arg("pred"));
  RegisterDeviceIntrinsic("cuda_atomic_add", "cuda", CallEffectKind::kOpaque, sig::arg("res_addr"),
                          sig::arg("value"));
  RegisterDeviceIntrinsic("cuda_atomic_cas", "cuda", CallEffectKind::kOpaque, sig::arg("ptr"),
                          sig::arg("old_val"), sig::arg("new_val"));
  RegisterDeviceIntrinsic("cuda_wait_until", "cuda", CallEffectKind::kOpaque, sig::arg("dst"),
                          sig::arg("ptr"), sig::arg("condition"), sig::arg("scope"),
                          sig::arg("space"), sig::arg("ptx_type"), sig::arg("backoff_ns"));
  RegisterDeviceIntrinsic("cuda_ballot_sync", "cuda", CallEffectKind::kOpaque, sig::arg("mask"),
                          sig::arg("pred"));
  RegisterDeviceIntrinsic("cuda_bfloat1622float2", "cuda", CallEffectKind::kOpaque,
                          sig::arg("packed"));
  RegisterDeviceIntrinsic("cuda_bfloat162float", "cuda", CallEffectKind::kOpaque, sig::arg("src"));
  RegisterDeviceIntrinsic("cuda_clock64", "cuda", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("cuda_cluster_sync", "cuda", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("cuda_cta_reduce", "cuda", CallEffectKind::kOpaque, sig::arg("value"),
                          sig::arg("op"), sig::arg("num_warps"), sig::arg("scratch"));
  RegisterDeviceIntrinsic("cuda_cta_sync", "cuda", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("cuda_cvta_generic_to_shared", "cuda", CallEffectKind::kOpaque,
                          sig::arg("ptr"));
  RegisterDeviceIntrinsic("cuda_elect_sync", "cuda", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("cuda_fadd2_rn", "cuda", CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"));
  RegisterDeviceIntrinsic("cuda_fdividef", "cuda", CallEffectKind::kPure, sig::arg("x"),
                          sig::arg("y"));
  RegisterDeviceIntrinsic("cuda_ffs_u32", "cuda", CallEffectKind::kOpaque, sig::arg("value"));
  RegisterDeviceIntrinsic("cuda_float22bfloat162_rn", "cuda", CallEffectKind::kOpaque,
                          sig::arg("v0"), sig::arg("v1"));
  RegisterDeviceIntrinsic("cuda_float22bfloat162_rn_from_float2", "cuda", CallEffectKind::kOpaque,
                          sig::arg("packed"));
  RegisterDeviceIntrinsic("cuda_float22half2", "cuda", CallEffectKind::kOpaque, sig::arg("dst"),
                          sig::arg("src"));
  RegisterDeviceIntrinsic("cuda_float2_x", "cuda", CallEffectKind::kOpaque, sig::arg("packed"));
  RegisterDeviceIntrinsic("cuda_float2_y", "cuda", CallEffectKind::kOpaque, sig::arg("packed"));
  RegisterDeviceIntrinsic("cuda_float8tohalf8", "cuda", CallEffectKind::kOpaque,
                          sig::arg("src_addr"), sig::arg("dst_addr"));
  RegisterDeviceIntrinsic("cuda_float_as_uint", "cuda", CallEffectKind::kOpaque, sig::arg("x"));
  RegisterDeviceIntrinsic("cuda_fmul2_rn", "cuda", CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"));
  RegisterDeviceIntrinsic("cuda_fp8x4_e4m3_from_float4", "cuda", CallEffectKind::kOpaque,
                          sig::arg("x"), sig::arg("y"), sig::arg("z"), sig::arg("w"));
  RegisterDeviceIntrinsic("cuda_func_call", "cuda", CallEffectKind::kOpaque, sig::arg("func_name"),
                          sig::var_args("args"));
  RegisterDeviceIntrinsic("cuda_get_tmem_addr", "cuda", CallEffectKind::kOpaque, sig::arg("addr"),
                          sig::arg("row_offset"), sig::arg("col_offset"));
  RegisterDeviceIntrinsic("cuda_grid_sync", "cuda", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("cuda_half2float", "cuda", CallEffectKind::kOpaque, sig::arg("src"));
  RegisterDeviceIntrinsic("cuda_half8tofloat8", "cuda", CallEffectKind::kOpaque,
                          sig::arg("src_addr"), sig::arg("dst_addr"));
  RegisterDeviceIntrinsic("cuda_hmax2", "cuda", CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"));
  RegisterDeviceIntrinsic("cuda_hmin2", "cuda", CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"));
  RegisterDeviceIntrinsic("cuda_ldg", "cuda", CallEffectKind::kOpaque, sig::var_args("args"));
  RegisterDeviceIntrinsic("cuda_make_float2", "cuda", CallEffectKind::kOpaque, sig::arg("x"),
                          sig::arg("y"));
  RegisterDeviceIntrinsic("cuda_mbarrier_wait", "cuda", CallEffectKind::kOpaque, sig::arg("bar"),
                          sig::arg("phase"));
  RegisterDeviceIntrinsic("cuda_mbarrier_wait_acquire_cluster", "cuda", CallEffectKind::kOpaque,
                          sig::arg("bar"), sig::arg("phase"));
  RegisterDeviceIntrinsic("cuda_mov_sreg", "cuda", CallEffectKind::kPure, sig::arg("bits"),
                          sig::arg("reg_name"));
  RegisterDeviceIntrinsic("cuda_nano_sleep", "cuda", CallEffectKind::kOpaque, sig::arg("time"));
  RegisterDeviceIntrinsic("cuda_printf", "cuda", CallEffectKind::kOpaque, sig::arg("fmt"),
                          sig::var_args("args"));
  RegisterDeviceIntrinsic("cuda_reduce_add_sync_u32", "cuda", CallEffectKind::kOpaque,
                          sig::arg("mask"), sig::arg("value"));
  RegisterDeviceIntrinsic("cuda_reduce_min_sync_u32", "cuda", CallEffectKind::kOpaque,
                          sig::arg("mask"), sig::arg("value"));
  RegisterDeviceIntrinsic("cuda_runtime_instr_desc", "cuda", CallEffectKind::kOpaque,
                          sig::arg("desc"), sig::arg("sf_id"));
  RegisterDeviceIntrinsic("cuda_sm100_2sm_leader_smem_addr", "cuda", CallEffectKind::kOpaque,
                          sig::arg("ptr"));
  RegisterDeviceIntrinsic("cuda_smem_addr_from_uint64", "cuda", CallEffectKind::kOpaque,
                          sig::arg("cluster_addr"));
  RegisterDeviceIntrinsic("cuda_syncthreads_and", "cuda", CallEffectKind::kOpaque,
                          sig::arg("cond"));
  RegisterDeviceIntrinsic("cuda_syncthreads_or", "cuda", CallEffectKind::kOpaque, sig::arg("cond"));
  RegisterDeviceIntrinsic(
      "cuda_tcgen05_encode_instr_descriptor", "cuda", CallEffectKind::kOpaque, sig::arg("desc"),
      sig::arg("d_dtype"), sig::arg("a_dtype"), sig::arg("b_dtype"), sig::arg("M"), sig::arg("N"),
      sig::arg("K"), sig::arg("trans_a"), sig::arg("trans_b"), sig::arg("n_cta_groups"),
      sig::arg("neg_a"), sig::arg("neg_b"), sig::arg("sat_d"), sig::arg("is_sparse"));
  RegisterDeviceIntrinsic("cuda_tcgen05_encode_instr_descriptor_block_scaled", "cuda",
                          CallEffectKind::kOpaque, sig::arg("desc"), sig::arg("d_dtype"),
                          sig::arg("a_dtype"), sig::arg("b_dtype"), sig::arg("sfa_dtype"),
                          sig::arg("sfb_dtype"), sig::arg("sfa_tmem_addr"),
                          sig::arg("sfb_tmem_addr"), sig::arg("M"), sig::arg("N"), sig::arg("K"),
                          sig::arg("trans_a"), sig::arg("trans_b"), sig::arg("n_cta_groups"),
                          sig::arg("neg_a"), sig::arg("neg_b"), sig::arg("is_sparse"));
  RegisterDeviceIntrinsic("cuda_tcgen05_encode_matrix_descriptor", "cuda", CallEffectKind::kOpaque,
                          sig::arg("desc"), sig::arg("addr"), sig::arg("ldo"), sig::arg("sdo"),
                          sig::arg("swizzle"));
  RegisterDeviceIntrinsic("cuda_thread_fence", "cuda", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("cuda_thread_rank", "cuda", CallEffectKind::kPure);
  RegisterDeviceIntrinsic("cuda_trap_when_assert_failed", "cuda", CallEffectKind::kOpaque,
                          sig::arg("cond"));
  RegisterDeviceIntrinsic("cuda_uint_as_float", "cuda", CallEffectKind::kOpaque, sig::arg("bits"));
  RegisterDeviceIntrinsic("cuda_warp_reduce", "cuda", CallEffectKind::kOpaque, sig::arg("value"),
                          sig::arg("op"), sig::arg("width"));
  RegisterDeviceIntrinsic("cuda_warp_sync", "cuda", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("cuda_warpgroup_sync", "cuda", CallEffectKind::kOpaque,
                          sig::arg("bar_no"));
  RegisterDeviceIntrinsic("cuda_wgmma_encode_matrix_descriptor", "cuda", CallEffectKind::kOpaque,
                          sig::arg("desc"), sig::arg("addr"), sig::arg("ldo"), sig::arg("sdo"),
                          sig::arg("swizzle"));
  RegisterDeviceIntrinsic("cuda_wgmma_noop_barrier", "cuda", CallEffectKind::kOpaque,
                          sig::arg("reg"));
  RegisterDeviceIntrinsic("nvshmem_barrier_all", "nvshmem", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("nvshmem_fence", "nvshmem", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("nvshmem_getmem_nbi", "nvshmem", CallEffectKind::kOpaque, sig::arg("dst"),
                          sig::arg("src"), sig::arg("nelems"), sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_getmem_nbi_block", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("nelems"), sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_getmem_nbi_warp", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("nelems"), sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_my_pe", "nvshmem", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("nvshmem_n_pes", "nvshmem", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("nvshmem_putmem_nbi", "nvshmem", CallEffectKind::kOpaque, sig::arg("dst"),
                          sig::arg("src"), sig::arg("nelems"), sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_putmem_nbi_block", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("nelems"), sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_putmem_nbi_warp", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("nelems"), sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_putmem_signal_nbi", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("nelems"),
                          sig::arg("sig_addr"), sig::arg("signal"), sig::arg("sig_op"),
                          sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_putmem_signal_nbi_block", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("nelems"),
                          sig::arg("sig_addr"), sig::arg("signal"), sig::arg("sig_op"),
                          sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_putmem_signal_nbi_warp", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg("nelems"),
                          sig::arg("sig_addr"), sig::arg("signal"), sig::arg("sig_op"),
                          sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_quiet", "nvshmem", CallEffectKind::kOpaque);
  RegisterDeviceIntrinsic("nvshmem_signal_op", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("sig_addr"), sig::arg("signal"), sig::arg("sig_op"),
                          sig::arg("pe"));
  RegisterDeviceIntrinsic("nvshmem_wait_until", "nvshmem", CallEffectKind::kOpaque,
                          sig::arg("ivar"), sig::arg("cmp"), sig::arg("cmp_value"),
                          sig::arg("type"));
  RegisterDeviceIntrinsic("ptx_legacy_ldmatrix", "ptx_legacy", CallEffectKind::kOpaque,
                          sig::arg("trans"), sig::arg("num"), sig::arg("dtype"),
                          sig::arg("local_ptr"), sig::arg("local_offset"), sig::arg("smem_ptr"),
                          sig::arg("smem_offset"));
  RegisterDeviceIntrinsic("ptx_legacy_mma", "ptx_legacy", CallEffectKind::kOpaque,
                          sig::arg("shape"), sig::arg("a_layout"), sig::arg("b_layout"),
                          sig::arg("a_dtype"), sig::arg("b_dtype"), sig::arg("c_dtype"),
                          sig::arg("a_ptr"), sig::arg("a_offset"), sig::arg("b_ptr"),
                          sig::arg("b_offset"), sig::arg("acc_ptr"), sig::arg("c_offset"),
                          sig::arg("saturate"), sig::var_args("args"));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
