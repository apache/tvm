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
``python/tvm/backend/cuda/operator/tile_primitive/copy/gmem_smem.py``.

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

    from tvm.script import tirx as T
    from tvm.script.tirx import tile as Tx
    from tvm.tirx.layout import S, TileLayout

    shape, dtype = (32, 32), "float32"
    s_layout = TileLayout(S[shape])
    fs = (slice(0, 32), slice(0, 32))

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, shape, dtype)
        B = T.match_buffer(B_ptr, shape, dtype)
        T.device_entry()
        T.cta_id([1]); T.lane_id([32]); T.thread_id([32])
        A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
        Tx.warp.copy(A_smem[fs], A[fs])   # global -> shared  (this dispatch)
        T.cuda.cta_sync()
        Tx.warp.copy(B[fs], A_smem[fs])   # shared -> global  (this dispatch)

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

**3. Emit a serial loop** (`gmem_smem.py`) — deliberately a Python ``for`` (so
ptxas unrolls it), *not* ``T.unroll``:

.. code-block:: python

    for f in range(total_outer):
        s_lin = s_p.apply(f, tid, v0, shape=apply_shape)["m"]   # shared element offset
        g_lin = g_p.apply(f, tid, v0, shape=apply_shape)["m"]   # global element offset
        s_off = _s_off(f, s_lin)                                # apply swizzle if any
        s_ptr = _ptr_off(s_buf.ptr_to(s_zero), s_off)
        g_ptr = _ptr_off(g_buf.ptr_to(g_zero), g_lin)
        if g_is_src:
            copy_op(s_ptr, g_ptr)     # global -> shared
        else:
            copy_op(g_ptr, s_ptr)     # shared -> global

Each ``(f, tid, 0)`` coordinate is flattened by ``layout.apply`` against
``[outer, threads, vec]``, so the emit never needs to know how the partition split
the iters; ``copy_op`` is ``T.cuda.copy_{vec_bits}b`` (here ``copy_128b``).

Generated TIRx IR
-----------------

Running ``LowerTIRx`` on the program above turns each ``Tx.warp.copy`` into the
synthesized loop (global → shared shown, trimmed):

.. code-block:: python

    tid: T.let = threadIdx_x % 32
    A_smem = T.alloc_shared((1024,))
    for f in range(8):                              # outer = 8
        s_lin = f * 128 + tid * 4                   # 32 threads × vec 4 = 128 / round
        g_lin = f * 128 + tid * 4
        s_ptr = pointer_offset(A_smem, s_lin)
        g_ptr = pointer_offset(A_1, g_lin)          # A_1 = A.view(1024)
        T.cuda.copy_bytes(s_ptr, g_ptr, 16)         # 16 B = vec 4 × 4 B

Generated CUDA
--------------

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(32)
    kernel_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tid = ((int)threadIdx.x);
      __shared__ alignas(64) float A_smem_ptr[1024];
      for (int f = 0; f < 8; ++f) {
        int   s_off = (f * 128) + (tid * 4);
        void* s_ptr = tvm_builtin_pointer_offset(&A_smem_ptr[0], s_off);
        void* g_ptr = tvm_builtin_pointer_offset(&A_ptr[0],      s_off);
        tvm_builtin_copy_128b(s_ptr, g_ptr);        // 128-bit vector load+store
      }
      // ... __syncthreads(); then the shared -> global loop into B_ptr ...
    }

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
     - ``copy_bytes``
     - ``outer = 1024 / (32 · vec)``
   * - ``float32``
     - 4
     - 16 (``copy_128b``)
     - 8
   * - ``float16``
     - 8
     - 16 (``copy_128b``)
     - 4
   * - ``uint8``
     - 16
     - 16 (``copy_128b``)
     - 2

The **scope** sets which axis names the thread id (``warp`` → ``laneid``,
``cta`` → ``tx``, …) and the thread count, hence the partition. A **swizzled**
shared layout caps ``vec`` to one swizzle chunk and routes ``s_off`` through the
swizzle (a recognized swizzle becomes a few register adds per round; otherwise
``swizzle.apply`` per round).
