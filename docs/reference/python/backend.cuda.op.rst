..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

tvm_tirx.backend.cuda.op
========================

.. automodule:: tvm_tirx.backend.cuda.op
   :no-members:

.. currentmodule:: tvm_tirx.backend.cuda.op

Classes
-------

.. autosummary::
   :toctree: generated/

   TCGen05InstrDescriptorAttrs
   TCGen05InstrDescriptorBlockScaledAttrs
   TensorMapEncodeTiledAttr

Functions
---------

.. autosummary::
   :toctree: generated/

   iket_mark
   iket_range_start
   iket_range_end
   iket_range_push
   iket_official_event
   func_call
   warp_reduce
   warp_sum
   warp_max
   warp_min
   cta_reduce
   cta_sum
   cta_max
   cta_min
   wait_until
   ptx_cp_async_legacy
   mov_sreg
   ptx_legacy_mma
   bmma_sync
   mma_store
   mma_fill
   ptx_legacy_ldmatrix
   tcgen05_encode_instr_descriptor
   tcgen05_encode_instr_descriptor_block_scaled
   timer_init
   timer_start
   timer_end
   timer_finalize
   atomic_add
   sm100_2sm_leader_smem_addr
   atomic_cas
   nvshmem_my_pe
   nvshmem_n_pes
   nvshmem_getmem_nbi
   nvshmem_putmem_nbi
   nvshmem_getmem_nbi_warp
   nvshmem_putmem_nbi_warp
   nvshmem_getmem_nbi_block
   nvshmem_putmem_nbi_block
   nvshmem_signal_op
   nvshmem_wait_until
   nvshmem_quiet
   nvshmem_putmem_signal_nbi
   nvshmem_putmem_signal_nbi_warp
   nvshmem_putmem_signal_nbi_block
   nvshmem_fence
   nvshmem_barrier_all
   tensormap_encode_tiled
   any_sync
   ballot_sync
   bfloat1622float2
   bfloat162float
   block_dim
   block_idx
   clock64
   cluster_cta_id
   cluster_dim
   cluster_id
   cluster_sync
   cta_pair_id
   cta_sync
   cvta_generic_to_shared
   decl_tmem
   elect_sync
   fadd2_rn
   fdividef
   ffs_u32
   float22bfloat162_rn
   float22bfloat162_rn_from_float2
   float22half2
   float2_x
   float2_y
   float8tohalf8
   float_as_uint
   fmul2_rn
   fp8x4_e4m3_from_float4
   get_tmem_addr
   grid_dim
   grid_sync
   half2float
   half8tofloat8
   hmax2
   hmin2
   iket_range_pop
   iket_sentinel_token
   lane_id
   linear_thread_id
   make_float2
   mbarrier_wait
   mbarrier_wait_acquire_cluster
   nano_sleep
   printf
   reduce_add_sync_u32
   reduce_min_sync_u32
   runtime_instr_desc
   smem_addr_from_uint64
   syncthreads_and
   syncthreads_or
   tcgen05_encode_matrix_descriptor
   thread_fence
   thread_idx
   thread_in_warpgroup
   thread_rank
   trap_when_assert_failed
   uint_as_float
   warp_id
   warp_in_warpgroup
   warp_sync
   warpgroup_id
   warpgroup_sync
   wgmma_encode_matrix_descriptor
   wgmma_noop_barrier
