
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
 * \file tir/op/target_builtin/cuda.cc
 *
 *  builtin intrinsic operators specific to CUDA target.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <string>

namespace tvm {
namespace tirx {
namespace builtin {

#define TIRX_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                              \
    static const Op& op = Op::Get("tirx." #OpName); \
    return op;                                      \
  }                                                 \
  TVM_TIRX_REGISTER_OP(#OpName)

TIRX_DEFINE_BUILTIN_FUNC(tvm_load_matrix_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kReadState));

TIRX_DEFINE_BUILTIN_FUNC(tvm_mma_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(tvm_bmma_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(tvm_fill_fragment)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(tvm_store_matrix_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_mma)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

// Siblings of ptx_mma / ptx_ldmatrix / mma_store / mma_fill that accept
// (ptr_var, offset) pairs. Codegen emits `ptr + offset` C-pointer
// arithmetic and lower_warp_memory rewrites the offset's group component
// to its thread-local index. Used by the s_tir tensor_intrin tensorize
// path so per-thread fragment offsets stay element-accurate.
TIRX_DEFINE_BUILTIN_FUNC(ptx_mma_legacy)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(ptx_ldmatrix_legacy)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(mma_store_legacy)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(mma_fill_legacy)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

const Op& ptx_ldg32() {
  static const Op& op = Op::Get("tirx.ptx.ldg32");
  return op;
}

TVM_REGISTER_OP("tirx.ptx.ldg32")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ptx.ldg32"), 20)
    .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"), 10)
    .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("ptx"), 10);

TIRX_DEFINE_BUILTIN_FUNC(ptx_mma_sp)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(ptx_ldmatrix)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_shared_to_cluster)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_commit_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_wait_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_mbarrier_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_fence).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_fence_proxy_async)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_mbarrier_init)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_mbarrier_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_mbarrier_arrive_expect_tx)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_mbarrier_try_wait)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_bar_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_bar_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_tensor_global_to_cluster)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_tensor_shared_to_global)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_tensor_global_to_cluster_prefetch)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_tensor_shared_to_global_reduce)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_commit_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_cp_async_bulk_wait_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_barrier_cluster_arrive)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_barrier_cluster_wait)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_elect_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_fence_mbarrier_init)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

const Op& ptx_fetch_register() {
  static const Op& op = Op::Get("tirx.ptx.fetch_register");
  return op;
}

TVM_REGISTER_OP("tirx.ptx.fetch_register")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
    .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("ptx"))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ptx.fetch_register"));

TVM_REGISTER_OP("tirx.ptx_fetch_register")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
    .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("ptx"))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("ptx.fetch_register"));

// griddepcontrol — programmatic dependent launch synchronization (sm_90+).
// Both are memory barriers; mark kOpaque to prevent CSE/reordering.
TIRX_DEFINE_BUILTIN_FUNC(ptx_griddepcontrol_wait)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_griddepcontrol_launch_dependents)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(mma_store)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(mma_fill)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

TIRX_DEFINE_BUILTIN_FUNC(ptx_wgmma_encode_matrix_descriptor)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_wgmma_noop_barrier)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_wgmma_mma_async_ss)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_wgmma_mma_async_rs)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_wgmma_fence)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_wgmma_commit_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_wgmma_wait_group)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_stmatrix)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_setmaxnreg)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_ld_global_acquire)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_alloc)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_dealloc)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_relinquish_alloc_permit)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_fence_before_thread_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_fence_after_thread_sync)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_ld)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_st)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_wait_ld)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_wait_st)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_encode_matrix_descriptor)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_encode_instr_descriptor)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_encode_instr_descriptor_block_scaled)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_mma)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_mma_block_scale)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_mma_sp)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_mma_sp_block_scale)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_commit)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_cp)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_tcgen05_shift)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(ptx_map_shared_rank)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(cuda_func_call)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_my_pe)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_n_pes)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_getmem_nbi)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_putmem_nbi)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_getmem_nbi_warp)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_putmem_nbi_warp)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_getmem_nbi_block)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_putmem_nbi_block)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_signal_op)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_wait_until)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_quiet)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_putmem_signal_nbi)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_putmem_signal_nbi_warp)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_putmem_signal_nbi_block)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_fence)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TIRX_DEFINE_BUILTIN_FUNC(nvshmem_barrier_all)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

namespace {

struct DeviceIntrinsicRegistration {
  const char* flat_name;
  const char* namespace_name;
  CallEffectKind effect_kind;
};

void RegisterDeviceIntrinsic(const DeviceIntrinsicRegistration& reg) {
  std::string flat_name(reg.flat_name);
  std::string namespace_name(reg.namespace_name);
  std::string prefix = namespace_name + "_";
  std::string suffix = flat_name;
  if (suffix.rfind(prefix, 0) == 0) {
    suffix = suffix.substr(prefix.size());
  }

  std::string flat_op_name = "tirx." + flat_name;
  std::string canonical_op_name = "tirx." + namespace_name + "." + suffix;
  ffi::String namespace_attr(namespace_name);
  ffi::String printer_name(namespace_name + "." + suffix);
  int64_t effect = static_cast<int64_t>(reg.effect_kind);

  auto register_one = [&](const std::string& op_name) {
    OpRegEntry::RegisterOrGet(op_name)
        .set_name()
        .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"),
                                  /*plevel=*/15)
        .set_attr<TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", namespace_attr,
                                             /*plevel=*/15)
        .set_attr<TCallEffectKind>("TCallEffectKind", effect, /*plevel=*/15)
        .set_attr<TScriptPrinterName>("TScriptPrinterName", printer_name, /*plevel=*/15);
  };

  register_one(flat_op_name);
  register_one(canonical_op_name);
}

#define TIRX_DEVICE_INTRIN_ALIAS(OpName, Namespace, EffectKind) \
  {#OpName, #Namespace, CallEffectKind::EffectKind}

const DeviceIntrinsicRegistration kDeviceIntrinsics[] = {
    TIRX_DEVICE_INTRIN_ALIAS(cuda_atomic_add, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_atomic_cas, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_ballot_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_bfloat1622float2, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_bfloat162float, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_clock64, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cluster_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_copy_bytes, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cta_reduce, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cta_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_cvta_generic_to_shared, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_fadd2_rn, cuda, kOpaque),
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
    TIRX_DEVICE_INTRIN_ALIAS(cuda_nano_sleep, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_printf, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_reduce_add_sync_u32, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_reduce_min_sync_u32, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_runtime_instr_desc, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_smem_addr_from_uint64, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_sm100_tma_2sm_mbarrier_addr, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_syncthreads_and, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_syncthreads_or, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_thread_fence, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_thread_rank, cuda, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_trap_when_assert_failed, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_uint_as_float, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warp_reduce, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warp_sync, cuda, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(cuda_warpgroup_sync, cuda, kOpaque),
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
    TIRX_DEVICE_INTRIN_ALIAS(ptx_add_f32, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_add_f32x2, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_add_f64, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_add_rn_f32_bf16, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_any_sync, ptx, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_atom_scalar, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_bar_arrive, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_bar_sync, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_barrier_cluster_arrive, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_barrier_cluster_wait, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_commit_group, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_g2s_cluster, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_g2s_cta, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_s2g, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_s2s_cluster, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_shared_to_cluster, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_tensor_global_to_cluster, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_tensor_global_to_cluster_prefetch, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_tensor_shared_to_global, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_tensor_shared_to_global_reduce, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_bulk_wait_group, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_commit_group, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_mbarrier_arrive, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_cp_async_wait_group, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_elect_sync, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_exp2, ptx, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fence, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fence_mbarrier_init, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fence_proxy_async, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fetch_register, ptx, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fma_f32, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fma_f32x2, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fma_f64, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_fns_b32, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_griddepcontrol_launch_dependents, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_griddepcontrol_wait, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_ld, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_ld_acquire, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_ld_global_acquire, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_ld_volatile, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_ldmatrix, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_ldmatrix_legacy, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mapa, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_map_shared_rank, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_max_f32, ptx, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mbarrier_arrive, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mbarrier_arrive_expect_tx, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mbarrier_init, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mbarrier_test_wait_parity, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mbarrier_try_wait, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mbarrier_try_wait_once, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mma, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mma_legacy, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mma_sp, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mul_f32, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mul_f32x2, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_mul_f64, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_prefetch_tensormap, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_rcp, ptx, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_red_scalar, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_reduce3_max_f32, ptx, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_reduce3_min_f32, ptx, kPure),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_setmaxnreg, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_st, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_st_bulk, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_stmatrix, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_sub_f32, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_sub_f32x2, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_sub_f64, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_alloc, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_commit, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_cp, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_dealloc, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_encode_instr_descriptor, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_encode_instr_descriptor_block_scaled, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_encode_matrix_descriptor, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_fence_after_thread_sync, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_fence_before_thread_sync, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_ld, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_mma, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_mma_block_scale, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_mma_sp, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_mma_sp_block_scale, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_relinquish_alloc_permit, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_shift, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_st, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_wait_ld, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_tcgen05_wait_st, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_wgmma_commit_group, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_wgmma_encode_matrix_descriptor, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_wgmma_fence, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_wgmma_mma_async_rs, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_wgmma_mma_async_ss, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_wgmma_noop_barrier, ptx, kOpaque),
    TIRX_DEVICE_INTRIN_ALIAS(ptx_wgmma_wait_group, ptx, kOpaque),
};

const bool kDeviceIntrinsicAliasesRegistered = []() {
  for (const auto& reg : kDeviceIntrinsics) {
    RegisterDeviceIntrinsic(reg);
  }
  return true;
}();

#undef TIRX_DEVICE_INTRIN_ALIAS

}  // namespace

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
