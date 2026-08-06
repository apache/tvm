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

copy_async → dsmem
==================

The ``dsmem`` variant lowers a ``copy_async`` whose **source and destination are
both shared** memory but in **different CTAs of a cluster** (distributed shared
memory). One elected thread on the source CTA maps the destination CTA's shared
address into its own address space (PTX ``mapa``) and issues a bulk copy
(``cp.async.bulk.shared::cluster``); the hardware decrements the *destination* CTA's
mbarrier when the bytes land. Source:
``python/tvm/backend/cuda/tile_primitive/copy_async/dsmem.py``.

What it accepts
---------------

Three predicates: a valid copy, a single-thread scope, and a shared → shared pair:

.. code-block:: python

    # register_dispatch(..., priority=10, when=[
    predicate("validate_copy_op", ...),
    predicate("single_thread",    lambda op, sctx: (single_thread(op, sctx), "expected single thread")),
    predicate("is_shared_to_shared", lambda op, sctx: (_is_shared_to_shared(op), "not shared-to-shared")),
    # ])

    def _is_shared_to_shared(op_call):
        src_scope = op_call.src.buffer.scope()
        dst_scope = op_call.dst.buffer.scope()
        return src_scope.startswith("shared") and dst_scope.startswith("shared")

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / priority
     - ``cuda``; priority ``10``
   * - scope
     - **single thread** issues the copy (the source CTA elects one thread)
   * - memory pair
     - both ``shared*`` (``_is_shared_to_shared``); the copy targets a *remote* CTA
       via ``remote_cta_id``
   * - chunk size
     - the contiguous chunk must be **≥ 16 bytes and a multiple of 16**
       (``cp.async.bulk`` requirement) — else the dispatch declines (``fail``)
   * - environment
     - a **cluster launch** (so a remote CTA's shared memory exists), plus a caller
       mbarrier on the destination CTA

Demonstration program
----------------------

A 2-CTA cluster: CTA 0 stages a ``128×64`` ``float16`` tile global → its shared,
then bulk-copies it into **CTA 1's** shared via ``dsmem``; CTA 1 waits on the
mbarrier and writes the result out (from ``test_dsmem.py``):

.. code-block:: python

    from tvm.tirx.lang.pipeline import MBarrier

    shape, dtype, CLUSTER_N = (128, 64), "float16", 2
    src_layout = dst_layout = TileLayout(S[128, 64])
    copy_bytes = 128 * 64 * 2
    r = (slice(0, 128), slice(0, 64))

    @T.prim_func
    def dsmem_copy(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, shape, dtype); B = T.match_buffer(B_ptr, shape, dtype)
        T.device_entry()
        cbx = T.cta_id_in_cluster([CLUSTER_N]); T.cta_id([CLUSTER_N]); tid = T.thread_id([1])
        pool = T.SMEMPool()
        src_smem = T.decl_buffer(list(shape), dtype, pool.alloc([8192], dtype, align=128).data,
                                 elem_offset=0, scope="shared.dyn", layout=src_layout)
        dst_smem = T.decl_buffer(list(shape), dtype, pool.alloc([8192], dtype, align=128).data,
                                 elem_offset=0, scope="shared.dyn", layout=dst_layout)
        mbar = MBarrier(pool, 1); pool.commit()
        mbar.init(1); T.ptxd.fence.mbarrier_init.release.cluster(); T.cuda.cluster_sync()
        if tid == 0:
            if cbx == 0:                                      # source CTA
                Tx.copy(src_smem[r], A[r])                    # global -> local shared
                T.ptxd.fence.proxy.async_.shared__cta()
                Tx.copy_async(dst_smem[r], src_smem[r], dispatch="dsmem",
                              mbar=mbar.ptr_to([0]), remote_cta_id=T.int32(1))   # -> CTA 1
            else:                                             # destination CTA
                T.ptxd.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(copy_bytes))
                mbar.wait(0, 0)
                Tx.copy(B[r], dst_smem[r])                    # remote shared -> global
        T.cuda.cluster_sync()

Algorithm
---------

**1. Find the contiguous chunk.** The dispatch slices and groups both layouts to the
copy region, walks inward to the longest matching contiguous stride-1 shard chain,
and multiplies those extents into ``chunk_elements``; ``chunk_bytes`` must be ≥ 16
and a multiple of 16 (a ``cp.async.bulk`` constraint), else it declines:

.. code-block:: python

    chunk_bytes = chunk_elements * dtype_bytes
    if chunk_bytes < 16 or chunk_bytes % 16 != 0:
        fail(...)

**2. Map the remote address.** ``T.ptxd.mapa.u64`` translates a local shared
pointer into the destination CTA's window — applied to both the destination
buffer pointer and the mbarrier (``mapa`` writes into a declared register, so
the mapped addresses live in a small local scratch buffer):

.. code-block:: python

    mapped = T.alloc_local([2], "uint64")
    T.ptxd.mapa.u64(mapped[0], mbar, T.uint32(remote_cta_id))                    # remote_mbar
    T.ptxd.mapa.u64(mapped[1], dst_buf.ptr_to(dst_st), T.uint32(remote_cta_id))  # cluster_dst

**3. Issue one bulk copy per chunk.** Fully contiguous → a single instruction; a
strided region loops over the outer (non-contiguous) extents, re-deriving the
chunk's offsets each step:

.. code-block:: python

    if not outer_extents:                                 # one contiguous chunk
        T.ptxd["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            T.cast(mapped[1], "uint32"), src_buf.ptr_to(src_st),
            T.cast(chunk_bytes, "uint32"), T.cast(mapped[0], "uint32"))
    else:
        for loop_vars in T.grid(*outer_extents):          # one chunk per outer coord
            ...  # re-decl src/dst views at the per-chunk offset
            T.ptxd["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
                T.cast(mapped[1], "uint32"), src_ptr,
                T.cast(chunk_bytes, "uint32"), T.cast(mapped[0], "uint32"))

The ``complete_tx::bytes`` form makes the hardware decrement ``remote_mbar`` by
``chunk_bytes`` on completion; the dispatch emits no wait — the caller arms the
mbarrier (``arrive.expect_tx``) and waits.

Generated TIRx IR
-----------------

The fully contiguous ``128×64`` fp16 tile (``16384`` bytes) is a **single chunk**:

.. code-block:: python

    T.ptxd["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
        T.cast(mapped[1], "uint32"), src_ptr[0], T.uint32(16384), T.cast(mapped[0], "uint32"))

Generated CUDA
--------------

.. code-block:: c++

    // map local shared addresses into CTA 1's window (mapa)
    remote_mbar = tvm_builtin_ptx_mapa_u64(&mbar,     /*rank=*/1);   // asm: mapa.u64
    cluster_dst = tvm_builtin_ptx_mapa_u64(&dst_smem, /*rank=*/1);
    // bulk-copy 16384 bytes local shared -> CTA 1 shared, signalling its mbarrier
    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes ..."

One thread on CTA 0 launches the whole 16 KB transfer; CTA 1's mbarrier fires when
it lands.

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - input
     - effect
   * - layout contiguity
     - fully contiguous (matching row-major both sides) → **one** ``cp.async.bulk``;
       a stride gap or mismatched outer stride → a loop of **N** chunks (one per
       outer coord)
   * - dtype / chunk size
     - sets ``chunk_bytes`` (must stay ≥ 16 and a multiple of 16); smaller
       contiguous runs mean smaller, more numerous chunks
   * - ``remote_cta_id``
     - the ``mapa`` rank — which cluster CTA receives the data
   * - incompatible layouts
     - e.g. row-major source vs column-major destination → no matching contiguous
       chain → the dispatch declines (``fail``)
