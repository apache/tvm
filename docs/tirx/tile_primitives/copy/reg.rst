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

copy → reg
==========

The ``reg`` variant lowers a synchronous ``copy`` where **exactly one side is a
register** (``local``) buffer and the other is ``shared*`` or ``global``. Unlike
:doc:`gmem_smem`, the partition is **not synthesized** — it is *induced* by the
register operand's layout: that layout's thread-axis iters already say which thread
owns which logical coordinate, so the dispatch drops those axes, leaves each thread
its private bundle of elements, and copies them in a vectorized serial loop. Source:
``python/tvm/backend/cuda/operator/tile_primitive/copy/reg.py``.

What it accepts
---------------

.. code-block:: python

    def _is_reg_copy(op_call, sctx):
        if not sctx.is_target("cuda"):
            return False, "non-cuda target"
        if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
            return False, f"unsupported exec_scope {sctx.scope_kind}"
        for check in (
            lambda: _all_threads_active(sctx),
            lambda: _is_valid_copy(op_call, sctx),
            lambda: _scope_allowed(op_call, sctx, allowed_pairs=_REG_PAIRS),
            lambda: _r_side_layout_valid(op_call, sctx),   # the register operand
            lambda: _s_side_slice_ok(op_call),             # the other operand
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
     - ``cuda``; ``thread`` / ``warp`` / ``warpgroup`` / ``cta`` with all threads
       active
   * - memory pair
     - ``_REG_PAIRS`` = ``(local, shared*)`` / ``(shared*, local)`` /
       ``(local, global)`` / ``(global, local)`` — exactly one side is ``local``
   * - register layout
     - ``_r_side_layout_valid``: the ``local`` operand is a non-swizzle
       ``TileLayout`` whose thread-axis iters have **stride 1**, a register-level
       subscope no wider than the exec scope, and a **zero sliced thread offset**
       (the region doesn't split a thread axis)
   * - other side
     - ``_s_side_slice_ok``: the ``shared*`` / ``global`` operand slices cleanly to
       its region

Demonstration program
----------------------

A warp round-trips a ``32×8`` ``float32`` tile shared → register → shared, with the
register layout ``S[(32,8):(1@laneid, 1)]`` — **lane ``i`` owns row ``i``** (8
contiguous elements). From ``test_reg.py``:

.. code-block:: python

    from tvm.tirx.layout import S, TileLayout, laneid

    shape, dtype = (32, 8), "float32"
    r_layout = TileLayout(S[shape : (1 @ laneid, 1)])   # lane i -> row i, 8 regs
    s_layout = TileLayout(S[shape])
    fs = (slice(0, 32), slice(0, 8))

    @T.prim_func
    def kernel(B_ptr: T.handle):
        B = T.match_buffer(B_ptr, shape, dtype)
        T.device_entry(); T.cta_id([1]); T.lane_id([32]); tid = T.thread_id([32])
        A_smem = T.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
        for kk in range(8): A_smem[tid, kk] = T.cast(tid * 100 + kk + 1, dtype)
        T.cuda.cta_sync()
        R = T.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
        Tx.warp.copy(R[fs], A_smem[fs])    # shared -> register  (this dispatch)
        # ... clear A_smem, cta_sync ...
        Tx.warp.copy(A_smem[fs], R[fs])    # register -> shared  (this dispatch)
        # ... cta_sync; B[tid, kk] = A_smem[tid, kk] ...

Algorithm
---------

**1. Inherit the partition from R.** The register layout's thread axis (``laneid``)
states that lane ``i`` owns row ``i``; the dispatch aligns the other (shared) side
to that order, then **drops the thread iters** — what remains is each thread's
private memory bundle: here ``8`` contiguous elements per lane.

**2. Linearize and choose the vector width.** The per-thread elements are
flattened into ``(extent, stride)`` atoms; the vector width is chosen widest-first
(``128 → … → 8`` bits) so the contiguous tail divides it and the outer atom
strides + base offsets are aligned. Crucially the **thread-axis strides are
excluded** from this alignment check (they live in *partition-coordinate* space —
which thread owns which element — and never appear in a single thread's physical
address). For ``8`` contiguous ``float32`` that is ``vec = 4``, so ``outer = 2``.

**3. Per-thread base offset + serial loop.** The shared-side base offset is built
from thread-axis placeholders (substituted with the real ``T.lane_id()`` etc.),
and the register side is a flat per-thread ``local`` buffer. The emit is a serial
loop (not ``T.unroll`` — same flooding rationale as :doc:`gmem_smem`):

.. code-block:: python

    r_local = r_buf.local(*per_thread_r_shape)   # flat per-thread registers
    for f in range(total_outer):
        ds, dr = _outer_const_offsets(outer, f)               # shared / reg deltas
        s_ptr = _ptr_off(s_buf.ptr_to(s_zero_indices), _s_iter_off(f, ds, s_off))
        r_ptr = _ptr_off(r_local.ptr_to([0]), r_off_base + dr)
        if r_is_src:
            copy_op(s_ptr, r_ptr)     # register -> shared/global
        else:
            copy_op(r_ptr, s_ptr)     # shared/global -> register

Generated TIRx IR
-----------------

``LowerTIRx`` turns the shared → register copy into a per-thread loop over the
8-element register bundle (trimmed):

.. code-block:: python

    r_local = T.decl_buffer((8,), data=R.data, scope="local")   # 8 regs / lane
    for f in range(2):                                           # outer = 8 / vec 4
        s_ptr = pointer_offset(A_smem, ...)                      # this lane's row
        r_ptr = pointer_offset(r_local, dr)
        T.cuda.copy_bytes(r_ptr, s_ptr, 16)                      # 16 B = vec 4 × 4 B

(The register → shared copy is the mirror: ``copy_bytes(s_ptr, r_ptr, 16)``.)

Generated CUDA
--------------

.. code-block:: c++

    alignas(64) float r_local_ptr[8];          // 8 registers, private to the lane
    for (int f = 0; f < 2; ++f) {
      void* r_ptr = tvm_builtin_pointer_offset(&r_local_ptr[0], dr);
      void* s_ptr = tvm_builtin_pointer_offset(&A_smem_ptr[0], /* lane row + f*4 */);
      tvm_builtin_copy_128b(r_ptr, s_ptr);     // shared -> register
    }
    // ... register -> shared mirror writes A_smem back ...

Each lane copies its own 8 elements as 2 × 128-bit transfers; no cross-lane
addressing appears because the thread partition was resolved away at lowering.

How inputs change the algorithm
-------------------------------

The register layout's **per-thread element count** (the non-thread extents — here
``k``) and the **dtype** set the register count, vector width, and round count:

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 22 20

   * - dtype
     - ``k``
     - regs / lane
     - ``vec``
     - ``outer = k / vec``
   * - ``float32``
     - 8
     - 8
     - 4
     - 2
   * - ``float32``
     - 16
     - 16
     - 4
     - 4
   * - ``float16``
     - 8
     - 8
     - 8
     - 1
   * - ``float16``
     - 16
     - 16
     - 8
     - 2

The copy is always a 128-bit transfer (``copy_bytes = 16``) when the contiguous
tail allows. The **scope** sets the thread axis (``warp`` → ``laneid``, ``cta`` →
``tx``, …) the register layout must use; a different R layout (e.g. a strided or
multi-row ownership) changes which elements each lane holds and therefore the atom
list and ``outer``.
