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
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_load_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));

  OpDef("tirx.tvm_mma_sync")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_mma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_bmma_sync")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_bmma_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_fill_fragment")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_fill_fragment"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.tvm_store_matrix_sync")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tvm_store_matrix_sync"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  // Siblings of mma_store / mma_fill that accept
  // (ptr_var, offset) pairs. Codegen emits `ptr + offset` C-pointer
  // arithmetic and lower_warp_memory rewrites the offset's group component
  // to its thread-local index. Used by the s_tir tensor_intrin tensorize
  // path so per-thread fragment offsets stay element-accurate.
  OpDef("tirx.mma_store_legacy")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.mma_fill_legacy")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill_legacy"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.s_tir.ldg32")
      .set_num_inputs(4)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.ldg32"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"));

  // Raw legacy cp.async form emitted by InjectPTXAsyncCopy (and round-tripped by
  // the T.s_tir.cp_async_raw.legacy 6-arg surface). It carries the element dtype in Call.dtype
  // and prints it dtype-first; user-issued copies go through T.ptx instead.
  OpDef("tirx.s_tir.cp_async_raw")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("s_tir"))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("s_tir.cp_async_raw"))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_store")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.mma_fill")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.mma_fill"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

  OpDef("tirx.timer_init_cuda")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_init"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_start_cuda")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_start"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_end_cuda")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.timer_end"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.timer_finalize_cuda")
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

  auto register_one = [&](const std::string& op_name) {
    OpDef(op_name)
        .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
        .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", namespace_attr)
        .set_attr<TCallEffectKind>("TCallEffectKind", effect)
        .set_attr<TScriptPrinterName>("TScriptPrinterName", printer_name);
  };

  register_one(canonical_op_name);
}

#define TIRX_DEVICE_INTRIN_ALIAS(OpName, Namespace, EffectKind) \
  {#OpName, #Namespace, CallEffectKind::EffectKind}

const DeviceIntrinsicRegistration kDeviceIntrinsics[] = {
    TIRX_DEVICE_INTRIN_ALIAS(cuda_any_sync, cuda, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_atomic_add, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_atomic_cas, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_wait_until, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_ballot_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_bfloat1622float2, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_bfloat162float, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_clock64, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cluster_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cta_reduce, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cta_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cvta_generic_to_shared, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_elect_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fadd2_rn, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fdividef, cuda, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_ffs_u32, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float22bfloat162_rn, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float22bfloat162_rn_from_float2, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float22half2, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float2_x, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float2_y, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float8tohalf8, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_float_as_uint, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fmul2_rn, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fp8x4_e4m3_from_float4, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_func_call, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_get_tmem_addr, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_grid_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_half2float, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_half8tofloat8, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_hmax2, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_hmin2, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_ldg, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_make_float2, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_mbarrier_wait, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_mbarrier_wait_acquire_cluster, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_mov_sreg, cuda, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_nano_sleep, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_printf, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_reduce_add_sync_u32, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_reduce_min_sync_u32, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_runtime_instr_desc, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_sm100_2sm_leader_smem_addr, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_smem_addr_from_uint64, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_syncthreads_and, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_syncthreads_or, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_tcgen05_encode_instr_descriptor, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_tcgen05_encode_instr_descriptor_block_scaled, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_tcgen05_encode_matrix_descriptor, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_thread_fence, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_thread_rank, cuda, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_trap_when_assert_failed, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_uint_as_float, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warp_reduce, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warp_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warpgroup_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_wgmma_encode_matrix_descriptor, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_wgmma_noop_barrier, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_barrier_all, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_fence, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_getmem_nbi, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_getmem_nbi_block, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_getmem_nbi_warp, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_my_pe, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_n_pes, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_nbi, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_nbi_block, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_nbi_warp, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_signal_nbi, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_signal_nbi_block, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_putmem_signal_nbi_warp, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_quiet, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_signal_op, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(nvshmem_wait_until, nvshmem, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_legacy_ldmatrix, ptx_legacy, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_legacy_mma, ptx_legacy, kOpaque),
};

void RegisterDeviceIntrinsicAliases() {
  for (const auto& reg : kDeviceIntrinsics) {
    RegisterDeviceIntrinsic(reg);
  }
}

#undef TIRX_DEVICE_INTRIN_ALIAS

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
