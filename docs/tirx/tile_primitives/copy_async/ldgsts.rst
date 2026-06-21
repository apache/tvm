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

copy_async → ldgsts
===================

The ``ldgsts`` variant lowers ``copy_async`` for a **global → shared** transfer to
the PTX ``cp.async`` (LDGSTS) instruction: each thread issues an *asynchronous*
vectorized copy that the hardware completes in the background, so the warp can keep
computing while the load is in flight. It reuses the exact ``[outer, threads, vec]``
partition of the synchronous :doc:`../copy/gmem_smem` variant; the differences are
all in *what* is emitted and *when* it completes. Source:
``python/tvm/backend/cuda/operator/tile_primitive/copy_async/ldgsts.py``.

What it accepts
---------------

.. code-block:: python

    _LDGSTS_PAIRS = [("global", "shared*")]      # cp.async is unidirectional
    _LDGSTS_VEC_BITS = (128, 64, 32)             # cp_size ∈ {16, 8, 4} bytes

    def _is_ldgsts(op_call, sctx):
        if not sctx.is_target("cuda"):
            return False, "non-cuda target"
        if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
            return False, f"unsupported exec_scope {sctx.scope_kind}"
        for check in (
            lambda: _all_threads_active(sctx),
            lambda: _is_valid_copy(op_call, sctx),
            lambda: _scope_allowed(op_call, sctx, allowed_pairs=_LDGSTS_PAIRS),
            lambda: _divides_thread_cnt_ldgsts(op_call, sctx),
        ):
            ok, msg = check()
            if not ok:
                return False, msg
        return True, None

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope
     - ``cuda``; ``thread`` / ``warp`` / ``warpgroup`` / ``cta``, all active
   * - direction
     - **global → shared only** (``_LDGSTS_PAIRS``) — ``cp.async`` is unidirectional
   * - dtype / shape
     - ``_is_valid_copy``; region element count divides the thread count
   * - vector width
     - ``cp.async`` only accepts cp_size ∈ {4, 8, 16} bytes, so the candidate set is
       restricted to ``_LDGSTS_VEC_BITS`` = ``{128, 64, 32}`` bits
   * - priority
     - ``20`` — selected for ``copy_async`` global→shared (vs the bulk/TMA variants)

Demonstration program
----------------------

A CTA (128 threads) asynchronously loads a ``128×32`` ``float16`` tile global →
shared, then commits and waits before reading it back (from ``test_ldgsts.py``):

.. code-block:: python

    shape, dtype = (128, 32), "float16"
    s_layout = TileLayout(S[shape]); full = (slice(0, 128), slice(0, 32))

    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, shape, dtype)
        B = T.match_buffer(B_ptr, shape, dtype)
        T.device_entry(); T.cta_id([1]); T.warp_id([4]); T.lane_id([32]); tid = T.thread_id([128])
        A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
        Tx.cta.copy_async(A_smem[full], A[full], dispatch="ldgsts")   # async global -> shared
        T.ptx.cp_async.commit_group()                                # caller commits ...
        T.ptx.cp_async.wait_group()                                  # ... and waits
        T.cuda.cta_sync()
        Tx.cta.copy(B[full], A_smem[full])

Algorithm
---------

**1. Same partition as** :doc:`../copy/gmem_smem`. ``align_layouts_gs`` builds the
``[outer, threads, vec]`` split with the global side driving the canonical order —
but the vector candidates are clamped to ``{128, 64, 32}`` bits so the byte size is
a legal ``cp.async`` cp_size. For ``128×32 = 4096`` ``float16`` over 128 threads the
widest legal width is ``vec = 8`` (``8 × 2 B = 16 B``), giving ``outer = 4``.

**2. Emit** ``cp.async`` **instead of a synchronous copy**, and do **not** sync:

.. code-block:: python

    for f in range(total_outer):
        s_lin = s_p.apply(f, tid, v0, shape=apply_shape)["m"]
        g_lin = g_p.apply(f, tid, v0, shape=apply_shape)["m"]
        s_ptr = _ptr_off(s_buf.ptr_to(s_zero), _s_off(f, s_lin))
        g_ptr = _ptr_off(g_buf.ptr_to(g_zero), g_lin)
        T.evaluate(T.ptx.cp_async(s_ptr, g_ptr, cp_size))   # async; cp_size = vec_bits // 8
    # NO cta_sync — commit_group / wait_group / cta_sync are the caller's job

Completion is the caller's responsibility (``cp_async.commit_group()`` then
``cp_async.wait_group()``); the dispatch only issues the in-flight loads.

Generated TIRx IR
-----------------

.. code-block:: python

    for f in range(4):                                       # outer = 4
        s_ptr = pointer_offset(A_smem, ...)
        g_ptr = pointer_offset(A_1, ...)
        T.ptx.cp_async(s_ptr, g_ptr, 16, T.uint64(0), 0, -1, -1, "")   # cp_size = 16 B

Generated CUDA
--------------

.. code-block:: c++

    // cp.async.cg copies 16 bytes shared <- global, asynchronously
    tvm_builtin_ptx_cp_async_cg_16(s_ptr, g_ptr, /*cache_policy=*/0);   // x4 (outer = 4)
    // ...
    tvm_builtin_ptx_cp_async_commit_group();    // asm: cp.async.commit_group;
    tvm_builtin_ptx_cp_async_wait_group_0();    // asm: cp.async.wait_group 0;

where the helper is ``asm volatile("cp.async.cg.shared.global [%0], [%1], 16;")``.
Each thread issues 4 asynchronous 16-byte copies; nothing blocks until the caller's
``wait_group``.

How inputs change the algorithm
-------------------------------

The dtype/alignment set ``vec`` (hence ``cp_size`` and ``outer``), but unlike the
synchronous variant the width is capped at 16 B (cp.async maximum):

.. list-table::
   :header-rows: 1
   :widths: 24 18 20 38

   * - case
     - ``vec``
     - ``cp_size``
     - ``outer = 4096 / (128 · vec)``
   * - ``float16``, aligned
     - 8
     - 16 B
     - 4
   * - ``float32``, aligned
     - 4
     - 16 B
     - 8
   * - 8-B-aligned only
     - (clamped)
     - 8 B
     - (doubles)
   * - 4-B-aligned only
     - (clamped)
     - 4 B
     - (×4)

If the region can't satisfy even a 4-byte (32-bit) cp_size, ``align_layouts_gs``
finds no candidate and the variant declines. The **direction is fixed**: a
shared → global ``copy_async`` is never ``ldgsts`` (hardware has no store form).
