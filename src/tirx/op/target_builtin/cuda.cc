
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

TIRX_DEFINE_BUILTIN_FUNC(ptx_ldg32).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

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

TIRX_DEFINE_BUILTIN_FUNC(ptx_fetch_register)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

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

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
