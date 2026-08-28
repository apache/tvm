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

copy → gmem_smem
================

The ``gmem_smem`` variant lowers a synchronous ``copy`` between **global and
shared** memory (either direction) when **neither side is a register**. Because
neither operand carries a thread partition, the dispatch *synthesizes* one from the
execution scope: it splits the region into ``[outer, threads, vec]`` and emits a
serial loop of vectorized loads/stores. Source:
``python/tvm/backend/cuda/tile_primitive/copy/vec_auto_gmem_smem.py``.

What it accepts
---------------

The predicate ``_is_gmem_smem`` gates the variant:

.. code-block:: python

    def _is_gmem_smem(op_call, sctx):
        if not sctx.is_target("cuda"):
            return False, "non-cuda target"
        if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
            return False, f"unsupported exec_scope {sctx.scope_kind}"
        for check in (
            lambda: _all_threads_active(sctx),                              # full scope, no narrowing
            lambda: _is_valid_copy(op_call, sctx),                          # layouts, equal dtype/extents
            lambda: _scope_allowed(op_call, sctx, allowed_pairs=_GMEM_SMEM_PAIRS),
            lambda: _divides_thread_cnt(op_call, sctx),
        ):
            ok, msg = check()
            if not ok:
                return False, msg
        return True, None

So the accepted input is:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target
     - ``cuda``
   * - scope
     - ``thread`` / ``warp`` / ``warpgroup`` / ``cta``, and **all threads active**
       (``_all_threads_active`` — ``laneid`` spans 32, etc., none narrowed by an
       enclosing ``if``)
   * - memory pair
     - ``(global, shared*)`` or ``(shared*, global)`` — ``_GMEM_SMEM_PAIRS``;
       neither side is ``local``
   * - dtype / shape
     - both operands have a layout, equal dtype, equal non-unit extents
       (``_is_valid_copy`` → ``validate_copy_op``)
   * - divisibility
     - the region's element count is divisible by the thread count
       (``_divides_thread_cnt``) — otherwise the ``[outer, threads, vec]`` split has
       no integer solution and the variant declines

Demonstration program
----------------------

A warp (32 threads) copies a ``32×32`` ``float32`` tile global → shared and back
(the round trip from ``test_gmem_smem.py``):

.. code-block:: python

    from tvm.script import tirx as Tx
    from tvm.tirx.layout import S, TileLayout

    shape, dtype = (32, 32), "float32"
    s_layout = TileLayout(S[shape])
    fs = (slice(0, 32), slice(0, 32))

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, shape, dtype)
        B = Tx.match_buffer(B_ptr, shape, dtype)
        Tx.device_entry()
        Tx.cta_id([1]); Tx.lane_id([32]); Tx.thread_id([32])
        A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
        Tx.tile.warp.copy(A_smem[fs], A[fs])   # global -> shared  (this dispatch)
        Tx.cuda.cta_sync()
        Tx.tile.warp.copy(B[fs], A_smem[fs])   # shared -> global  (this dispatch)

Algorithm
---------

**1. Synthesize the partition.** With 32 threads and ``32×32 = 1024`` elements,
the dispatch builds a 3-D iteration ``[outer, threads, vec]`` via
``align_layouts_gs``: it slices both layouts to the region, makes the **global**
side drive the canonical (stride-descending) order, then carves a contiguous
``vec`` tail and a ``threads`` chunk off it and re-groups the shared side to match.

**2. Choose the vector width, widest first.** It tries element counts for
``{128, 64, 32, 16, 8}`` bits and accepts the widest where (a) the contiguous tail
divides it and (b) every **non-vec iter stride** (on both sides, the thread iter
included) and both base offsets is a multiple of it, so the per-thread, per-round
vector pointer is naturally aligned. (Only the innermost ``vec`` iter is excluded
from that check.) For ``float32`` that is ``vec = 4`` (``4 × 4 B = 16 B = 128 bit``),
giving ``outer = 1024 / (32 × 4) = 8``.

**3. Emit a serial loop** (``vec_auto_gmem_smem.py``) — deliberately a Python ``for`` (so
ptxas unrolls it), *not* ``Tx.unroll``:

.. code-block:: python

    for f in range(total_outer):
        g_lin = g_p.apply(f, tid, v0, shape=apply_shape)["m"]
        s_off = s_apply_layout.apply(f, tid, v0, shape=apply_shape)["m"]
        s_ptr = _ptr_off(s_buf.ptr_to(s_zero), s_off)
        g_ptr = _ptr_off(g_buf.ptr_to(g_zero), g_lin)
        if g_is_src:
            Tx.ptx[ld_g](*[tmp[i] for i in range(lanes)], g_ptr)
            Tx.ptx[st_s](s_ptr, *[tmp[i] for i in range(lanes)])
        else:
            Tx.ptx[ld_s](*[tmp[i] for i in range(lanes)], s_ptr)
            Tx.ptx[st_g](g_ptr, *[tmp[i] for i in range(lanes)])

Each ``(f, tid, 0)`` coordinate is flattened by ``layout.apply`` against
``[outer, threads, vec]``, so the emit never needs to know how the partition split
the iters.  ``ld_g``, ``st_s``, ``ld_s``, and ``st_g`` are registered direct-PTX
forms selected for the memory direction and vector width.  A 128-bit transfer uses
four ``uint32`` registers and the ``v4.u32`` forms.

Generated TIRx IR
-----------------

Running ``LowerTIRx`` on the program above turns each ``Tx.tile.warp.copy`` into the
synthesized loop (global → shared shown, trimmed):

.. code-block:: python

    tid: Tx.let = threadIdx_x % 32
    A_smem = Tx.alloc_shared((1024,))
    tmp = Tx.alloc_local((4,), "uint32")
    for f in range(8):                              # outer = 8
        s_lin = f * 128 + tid * 4                   # 32 threads × vec 4 = 128 / round
        g_lin = f * 128 + tid * 4
        s_ptr = pointer_offset(A_smem, s_lin)
        g_ptr = pointer_offset(A_1, g_lin)          # A_1 = A.view(1024)
        Tx.ptx.ld.global_.v4.u32(tmp[0], tmp[1], tmp[2], tmp[3], g_ptr)
        Tx.ptx.st.shared.v4.u32(s_ptr, tmp[0], tmp[1], tmp[2], tmp[3])

Generated PTX instructions
--------------------------

The CUDA code generator emits one vector load and one vector store per round::

    ld.global.v4.u32 {r0, r1, r2, r3}, [g_ptr];
    st.shared.v4.u32 [s_ptr], {r0, r1, r2, r3};

The shared-to-global direction uses ``ld.shared.v4.u32`` followed by
``st.global.v4.u32``.

Thread ``tid`` handles elements ``[f·128 + tid·4 .. +4)`` each round; across 8
rounds and 32 lanes that covers all 1024 elements, each as one 128-bit transfer.

How inputs change the algorithm
-------------------------------

The element **dtype** sets the vector width (widest 128-bit transfer that stays
aligned), which sets the round count. For the same ``32×32`` tile and 32 threads:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - dtype
     - ``vec``
     - transfer width
     - ``outer = 1024 / (32 · vec)``
   * - ``float32``
     - 4
     - 16 B (``v4.u32``)
     - 8
   * - ``float16``
     - 8
     - 16 B (``v4.u32``)
     - 4
   * - ``uint8``
     - 16
     - 16 B (``v4.u32``)
     - 2

The **scope** sets which axis names the thread id (``warp`` → ``laneid``,
``cta`` → ``tx``, …) and the thread count, hence the partition. A **swizzled**
shared layout caps ``vec`` to one swizzle chunk and routes ``s_off`` through the
swizzle (a recognized swizzle becomes a few register adds per round; otherwise
``swizzle.apply`` per round).
