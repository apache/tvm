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

copy → fallback
===============

The ``fallback`` variant is the **priority-0 last resort**: a scalar, single-thread
copy that runs only when ``ldstmatrix`` and both ``vec_auto`` paths
(:doc:`gmem_smem` and :doc:`reg`) have declined.  Explicit ``vec_*`` variants
are considered only when requested through ``dispatch=``. The fallback works for
valid copies among directly addressable global, shared, and local memory at
thread, warp, warpgroup, or CTA scope. It is intentionally slow, so it emits a
``UserWarning`` when chosen. Source:
``python/tvm/backend/cuda/tile_primitive/copy/fallback.py``.

What it accepts
---------------

The registered predicate checks only ``_is_valid_copy`` (layouts present, equal
dtype, and equal non-unit extents); it has no separate execution-scope or
memory-pair predicate.  The emitter has an unguarded thread path and first-thread
paths for warp, warpgroup, and CTA scope.  Because its body uses ordinary buffer
loads and stores, use it with directly addressable global, shared, or local
buffers.  There is no divisibility restriction. It is registered at
``priority=0`` so it is the last candidate the dispatcher tries:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - priority
     - ``0`` — only reached after the applicable priority-10 variants decline
   * - target / scope
     - ``cuda``; the emitter implements ``thread`` / ``warp`` / ``warpgroup`` /
       ``cta``. The registration itself has no scope-kind filter
   * - memory pair
     - no registration-time filter; the scalar body is usable for directly
       addressable global / shared / local buffers. It cannot ordinarily load
       from or store to tensor memory
   * - dtype / shape
     - ``_is_valid_copy`` only — equal dtype and equal non-unit extents

A typical reason the fast variants all decline: the region's element count does not
divide the thread count, so the ``[outer, threads, vec]`` split has no solution
(``vec_auto``'s global/shared path declines), and neither side is a supported
register-layout pair (its register path and ``ldstmatrix`` decline). Example: a
warp (32 threads) copying a ``4×6 = 24``-element tile (``24 ∤ 32``).

Demonstration program
----------------------

A warp copies a ``4×6`` ``float32`` tile global → shared and back. ``24`` is not
divisible by ``32``, so this falls through to ``fallback`` (from
``test_fallback.py``):

.. code-block:: python

    shape, dtype = (4, 6), "float32"            # 24 elements, 32 threads -> 24 ∤ 32
    s_layout = TileLayout(S[shape])
    full = (slice(0, 4), slice(0, 6))

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, shape, dtype)
        B = Tx.match_buffer(B_ptr, shape, dtype)
        Tx.device_entry(); Tx.cta_id([1]); Tx.lane_id([32]); Tx.thread_id([32])
        A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
        Tx.tile.warp.copy(A_smem[full], A[full])    # fallback
        Tx.cuda.cta_sync()
        Tx.tile.warp.copy(B[full], A_smem[full])    # fallback

Algorithm
---------

**1. Elect one thread.** For a multi-thread scope the copy is done entirely by the
**first thread of the scope** — its ``laneid`` base, plus the warp offset within a
warpgroup / cta (radix-32 composition of the per-axis offsets); every other thread
skips it. For ``thread`` scope there is only one thread, so no guard is emitted:

.. code-block:: python

    if scope_kind == "thread":
        def impl():
            _copy_body(dst, src)            # the single thread copies everything
    else:
        first_tid = int(sctx.intra["laneid"][1])              # first thread of the scope:
        if scope_kind == "warpgroup":
            first_tid += 32 * int(sctx.intra["wid_in_wg"][1]) #   + warp offset within the wg
        elif scope_kind == "cta":
            first_tid += 32 * int(sctx.intra["warpid"][1])    #   + warp offset within the cta
        def impl():
            tid = _axis_decl(tid_axis_name, sctx)
            if tid == first_tid:            # only the scope's first thread copies
                _copy_body(dst, src)

**2. Scalar nested loop over the region.** ``_copy_body`` iterates the non-unit
extents with ``Tx.grid`` and copies one element per step — no vectorization, no
partition:

.. code-block:: python

    with Tx.grid(*copy_extents) as lvs:        # copy_extents = non-unit dst extents
        dst[_dst_coord(lvs)] = src[_src_coord(lvs)]

Generated TIRx IR
-----------------

``LowerTIRx`` produces the guarded scalar grid (global → shared shown):

.. code-block:: python

    if tid == 0:                              # first_tid (lane 0 for this warp scope)
        for v_3, v_4 in Tx.grid(4, 6):
            A_smem[v_3, v_4] = A[v_3, v_4]

Generated CUDA
--------------

.. code-block:: c++

    __shared__ alignas(64) float A_smem_ptr[24];
    if (((int)threadIdx.x) == 0) {            // lane 0 does the whole copy
      for (int v_3 = 0; v_3 < 4; ++v_3)
        for (int v_4 = 0; v_4 < 6; ++v_4)
          A_smem_ptr[(v_3 * 6) + v_4] = A_ptr[(v_3 * 6) + v_4];
    }

Lane 0 copies all 24 elements one at a time; the other 31 lanes do nothing. (At
lowering the dispatch also prints ``UserWarning: copy/fallback (scalar
single-thread) picked … all faster variants rejected``.)

How inputs change the algorithm
-------------------------------

The **scope** decides who runs the loop:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - scope
     - who copies
   * - ``thread``
     - the single thread, no ``if`` guard
   * - ``warp`` / ``warpgroup`` / ``cta``
     - only ``first_tid`` (the scope's first thread = lane base + warp offset); all
       other threads skip

The **shape** sets the ``Tx.grid`` bounds (the non-unit extents); the loop body is
always one scalar element copy, regardless of dtype. The algorithm remains scalar
for every dtype; element width and the target memory spaces can still affect the
cost of each load and store. This variant exists for correctness, not speed.
