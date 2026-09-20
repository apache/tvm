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

copy → vec_auto register path
==============================

The register implementation path inside the registered ``vec_auto`` variant
lowers a synchronous ``copy`` where **exactly one side is a register** (``local``)
buffer and the other is ``shared*`` or ``global``.  It is not a separate
``reg`` dispatch name; automatic selection or ``dispatch="vec_auto"`` reaches
this path.  Unlike :doc:`gmem_smem`, the partition is **not synthesized** — it is
*induced* by the register operand's layout: that layout's thread-axis iters
already say which thread owns which logical coordinate, so the dispatch drops
those axes, leaves each thread its private bundle of elements, and copies them
in a vectorized serial loop. Source:
``python/tvm/backend/cuda/tile_primitive/copy/vec_auto_reg.py``.

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
       ``TileLayout`` whose thread axes have a register-level subscope no wider
       than the exec scope and a **zero sliced thread offset** (the region does
       not start partway through a thread partition)
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

    @Tx.prim_func
    def kernel(B_ptr: Tx.handle):
        B = Tx.match_buffer(B_ptr, shape, dtype)
        Tx.device_entry(); Tx.cta_id([1]); Tx.lane_id([32]); tid = Tx.thread_id([32])
        A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
        for kk in range(8): A_smem[tid, kk] = Tx.cast(tid * 100 + kk + 1, dtype)
        Tx.cuda.cta_sync()
        R = Tx.alloc_buffer(shape, dtype, scope="local", layout=r_layout)
        Tx.tile.warp.copy(R[fs], A_smem[fs])    # shared -> register  (this dispatch)
        # ... clear A_smem, cta_sync ...
        Tx.tile.warp.copy(A_smem[fs], R[fs])    # register -> shared  (this dispatch)
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
from thread-axis placeholders (substituted with the real ``Tx.lane_id()`` etc.),
and the register side is a flat per-thread ``local`` buffer. The emit is a serial
loop (not ``Tx.unroll`` — same flooding rationale as :doc:`gmem_smem`):

.. code-block:: python

    r_local = r_buf.local()                      # raw per-thread physical span
    r_words = r_local.view(reg_dtype)            # PTX register-container view
    for f in range(total_outer):
        ds, dr = _outer_const_offsets(outer, f)               # shared / reg deltas
        s_ptr = _ptr_off(s_buf.ptr_to(s_zero_indices), _s_iter_off(f, ds, s_off))
        r_w = (r_off_base + dr) * words_per_elem // elems_per_word
        if r_is_src:
            Tx.ptx[st_chain](s_ptr, *[r_words[r_w + i] for i in range(lanes)])
        else:
            Tx.ptx[ld_chain](*[r_words[r_w + i] for i in range(lanes)], s_ptr)

``dr`` and ``r_off_base`` are physical storage offsets, so the register alias
must use the raw-span ``local()`` view.  This remains correct when storage
iterators are permuted or leave gaps; every physical slot through
``layout.storage().span()`` is directly addressable.

Generated TIRx IR
-----------------

``LowerTIRx`` turns the shared → register copy into a per-thread loop over the
8-element local bundle (trimmed):

.. code-block:: python

    r_local = Tx.decl_buffer((8,), data=R.data, scope="local")   # 8 fp32 elements / lane
    r_words = r_local.view("uint32")
    for f in range(2):                                           # outer = 8 / vec 4
        s_ptr = pointer_offset(A_smem, ...)                      # this lane's row
        r_w = f * 4
        Tx.ptx.ld.shared.v4.u32(
            r_words[r_w], r_words[r_w + 1], r_words[r_w + 2], r_words[r_w + 3], s_ptr
        )

(The register-to-shared copy uses ``Tx.ptx.st.shared.v4.u32`` with the pointer
first, followed by the four register-container operands.)

Generated PTX instruction
-------------------------

The CUDA code generator emits the vector load directly::

    ld.shared.v4.u32 {r0, r1, r2, r3}, [s_ptr];

The reverse direction emits ``st.shared.v4.u32 [s_ptr], {r0, r1, r2, r3};``.

Each lane copies its own 8 elements as 2 × 128-bit transfers; no cross-lane
addressing appears because the thread partition was resolved away at lowering.

How inputs change the algorithm
-------------------------------

The register layout's **per-thread element count** (the non-thread extents — here
``k``) and the **dtype** set the local element count, PTX container count, vector
width, and round count:

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 22 20

   * - dtype
     - ``k``
     - elements / lane
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

The copy uses a 128-bit ``v4.u32`` load or store when the contiguous tail allows.
The **scope** sets the thread axis (``warp`` → ``laneid``, ``cta`` → ``tx``, …) the
register layout must use; a different R layout (e.g. a strided or multi-row
ownership) changes which elements each lane holds and therefore the atom list and
``outer``.
