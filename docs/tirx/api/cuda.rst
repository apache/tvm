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

CUDA Authoring and Support APIs
===============================

CUDA helpers
------------

The CUDA backend installs the ``Tx.cuda`` namespace when it is loaded::

   from tvm.script import tirx as Tx

   Tx.cuda.cta_sync()
   leader = Tx.cuda.elect_sync()

These helpers cover operations that need multiple instructions, C/C++
expressions, descriptor packing, compiler annotations, or other behavior that
is not represented as one table-driven PTX instruction.  For a single supported
PTX instruction, use :doc:`ptx`.

The current ``Tx.cuda`` surface includes:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Category
     - Helpers
   * - Synchronization and participation
     - ``any_sync``, ``elect_sync``, ``warp_sync``, ``warpgroup_sync``,
       ``cta_sync``, ``grid_sync``, ``cluster_sync``, ``syncthreads_and``,
       ``syncthreads_or``, ``thread_rank``, ``ballot_sync``, ``__shfl_sync``,
       ``__shfl_up_sync``, ``__shfl_down_sync``, ``__shfl_xor_sync``, and
       ``__activemask``
   * - Barriers and memory ordering
     - ``mbarrier_wait``, ``mbarrier_wait_acquire_cluster``,
       ``thread_fence``, ``atomic_add``, and ``atomic_cas``
   * - Reductions
     - ``warp_reduce``, ``warp_sum``, ``warp_max``, ``warp_min``,
       ``cta_reduce``, ``cta_sum``, ``cta_max``, ``cta_min``,
       ``reduce_add_sync_u32``, and ``reduce_min_sync_u32``
   * - Descriptors and addresses
     - ``wgmma.noop_barrier``, ``wgmma.encode_matrix_descriptor``,
       ``tcgen05.encode_matrix_descriptor``, ``tcgen05.encode_instr_descriptor``,
       ``tcgen05.encode_instr_descriptor_block_scaled``, ``runtime_instr_desc``,
       ``get_tmem_addr``, ``cvta_generic_to_shared``,
       ``smem_addr_from_uint64``, ``sm100_2sm_leader_smem_addr``, and ``mov_sreg``
   * - Loads, calls, and diagnostics
     - ``ldg``, ``func_call``, ``printf``, ``trap_when_assert_failed``,
       ``nano_sleep``, ``clock64``, and ``ffs_u32``
   * - Numeric conversion and packed math
     - ``half2float``, ``bfloat162float``, ``float22half2``,
       ``half8tofloat8``, ``float8tohalf8``, ``uint_as_float``,
       ``float_as_uint``, ``make_float2``, ``float2_x``, ``float2_y``,
       ``fmul2_rn``, ``fadd2_rn``, ``float22bfloat162_rn``,
       ``float22bfloat162_rn_from_float2``, ``bfloat1622float2``, ``hmin2``,
       ``hmax2``, ``fp8x4_e4m3_from_float4``, and ``fdividef``
   * - Instrumentation and compatibility
     - ``iket.mark``, ``iket.range_start``, ``iket.range_end``,
       ``iket.range_push``, ``iket.range_pop``, ``iket.sentinel_token``,
       ``iket.official_event``, ``timer_init``, ``timer_start``, ``timer_end``,
       ``timer_finalize``, ``mma_store``, ``mma_fill``, ``mma_store_legacy``,
       and ``mma_fill_legacy``

NVSHMEM namespace
-----------------

The same backend installs ``Tx.nvshmem`` for ``my_pe``, ``n_pes``,
``signal_op``, ``wait_until``, ``quiet``, ``fence``, and ``barrier_all``.
Its ``getmem_nbi``, ``putmem_nbi``, and ``putmem_signal_nbi`` operations also
provide ``.warp`` and ``.block`` variants.

CUDA language utilities
-----------------------

``tvm.backend.cuda.lang`` provides reusable objects for hand-written CUDA TIRx
kernels, including schedulers, pipelines, roles, barriers, descriptors, and
memory pools.

.. autoclass:: tvm.backend.cuda.lang.BaseTileScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.ClusterPersistentScheduler2D
   :members:

.. autoclass:: tvm.backend.cuda.lang.FlashAttentionLPTScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.FlashAttentionLinearScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.GroupMajor3D
   :members:

.. autoclass:: tvm.backend.cuda.lang.IndexedTripleTileScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.MBarrier
   :members:

.. autoclass:: tvm.backend.cuda.lang.Pipeline
   :members:

.. autoclass:: tvm.backend.cuda.lang.PipelineState
   :members:

.. autoclass:: tvm.backend.cuda.lang.RankAwareGroupMajorTileScheduler
   :members:

.. autoclass:: tvm.backend.cuda.lang.SMEMPool
   :members:

.. autoclass:: tvm.backend.cuda.lang.SmemDescriptor
   :members:

.. autoclass:: tvm.backend.cuda.lang.TCGen05Bar
   :members:

.. autoclass:: tvm.backend.cuda.lang.TMABar
   :members:

.. autoclass:: tvm.backend.cuda.lang.TMEMPool
   :members:

.. autoclass:: tvm.backend.cuda.lang.WarpRole
   :members:

.. autoclass:: tvm.backend.cuda.lang.WarpgroupRole
   :members:

IKET profiling
--------------

.. automodule:: tvm.backend.cuda.iket
   :members:
   :no-index:

CUDA-specific transforms
------------------------

.. automodule:: tvm.backend.cuda.transforms
   :members:
   :no-index:
