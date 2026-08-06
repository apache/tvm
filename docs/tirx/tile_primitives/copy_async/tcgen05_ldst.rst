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

copy_async → tcgen05_ldst
=========================

The ``tcgen05_ldst`` variant lowers a ``copy_async`` between **tensor memory and
registers** (Blackwell ``tcgen05.ld`` / ``tcgen05.st``). It is warpgroup-collective:
the four warps cooperatively move a tensor-memory tile to/from their per-thread
registers. One registration handles both directions — ``tmem → local`` lowers to
``tcgen05.ld``, ``local → tmem`` to ``tcgen05.st`` — and the dispatch picks the
widest instruction shape the register layout matches. As with the other async
variants, completion (``tcgen05.wait.ld`` / ``wait.st``) is the caller's. Source:
``python/tvm/backend/cuda/tile_primitive/copy_async/tcgen05_ldst.py``.

What it accepts
---------------

A single registration (``variant="tmem<->local"``); direction is inferred at
lowering:

.. code-block:: python

    @register_dispatch("copy_async", "cuda", variant="tmem<->local", priority=10, when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate("storage_scope", _scope_allowed,
                  allowed_pairs=[("tmem", "local"), ("local", "tmem")]),
    ])
    # direction inferred in copy_tmem_local_impl:
    #   src tmem + dst local -> "tmem2local" (ld);  else "local2tmem" (st)

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / priority
     - ``cuda`` (Blackwell, sm_100+); priority ``10``
   * - scope
     - **warpgroup** (``exec_scope_ok(expected_scopes=["warpgroup"])``) — the four
       warps act together
   * - memory pair
     - ``(tmem, local)`` or ``(local, tmem)`` — exactly one side is tensor memory
   * - register layout
     - matched against a ``tcgen05_atom_layout`` (``.16x64b`` / ``.16x128b`` /
       ``.16x256b``) for the fast path; otherwise the ``.32x32b`` fallback.
       Layout B uses the special ``.32x32b`` logical ``(64, N)`` image
   * - tmem datapath
     - classified ``D`` (M=128 identity), ``F`` (M=64 scattered), or ``B``
       (per-CTA M=64, ``cta_group=2`` column split); an F layout also selects
       the lower or upper 16-lane sub-slab of each warp partition

Demonstration program
----------------------

A warpgroup round-trips a ``128×8`` ``float16`` tile registers → tmem → registers
(the GPU smoke test ``test_copy_tmem2reg_async``; ``WIDTH = 8`` for ``width_32b=4``,
fp16):

.. code-block:: python

    from tvm.tirx.layout import S, TCol, TileLayout, TLane
    from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg

    local_view = TileLayout(S[(128, WIDTH) : (1 @ axis_tid_in_wg, 1)])

    @T.prim_func
    def copy_async_test(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128, WIDTH), "float16"); B = T.match_buffer(B_ptr, (128, WIDTH), "float16")
        T.device_entry()
        warp_id = T.warp_id([4]); wg_id = T.warpgroup_id([1]); tid = T.thread_id([128])
        tmem_addr = T.alloc_shared([1], "uint32")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=32, cta_group=1)
            T.tvm_storage_sync("shared")
            tmem = T.decl_buffer((128, WIDTH), "float16", scope="tmem", allocated_addr=tmem_addr[0],
                                 layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))
            A_reg = T.alloc_local((WIDTH,), "float16"); B_reg = T.alloc_local((WIDTH,), "float16")
            A_local = A_reg.view(128, WIDTH, layout=local_view)
            B_local = B_reg.view(128, WIDTH, layout=local_view)
            # ... load A into A_reg, zero B_reg, cta_sync ...
            Tx.wg.copy_async(tmem[:, :], A_local[:, :]); T.ptx.tcgen05.wait.st()   # store (local -> tmem)
            T.cuda.cta_sync()
            Tx.wg.copy_async(B_local[:, :], tmem[:, :]); T.ptx.tcgen05.wait.ld()   # load  (tmem -> local)
            # ... write B_reg out; tcgen05.dealloc ...

Algorithm
---------

**1. Infer direction.** ``tmem → local`` is a load (``tcgen05.ld``); ``local → tmem``
is a store (``tcgen05.st``).

**2. Pick the instruction shape.** The dispatch matches the register layout against
``tcgen05_atom_layout`` for ``.16x64b`` / ``.16x128b`` / ``.16x256b``
(``_match_tcgen05_atom_layout``); the matched shape sets the column factor (2/4/8
fp32 columns) and the ``num`` count. If nothing matches it falls back to
``.32x32b`` and probes ``num ∈ {1, 2, 4, 8, …}`` against the column width.

**3. Issue per datapath slab.** For an M=128 Layout D ``.16x*b`` copy the
fragment spans two 16-row slabs, so the warps issue the atom twice
(``row = 0`` and ``row = 16``). An M=64 Layout F copy issues once, at
``row = 0`` for ``sub_slab=0`` or ``row = 16`` for ``sub_slab=1``. The
``.32x32b`` path covers M=128 in a single issue at ``row = 0``:

.. code-block:: python

    op = T.ptx.tcgen05.ld if load else T.ptx.tcgen05.st
    classified = _check_tmem_layout_for_atom(tmem_buf, "16x*b", frag_rows)
    datapath, sub_slab = classified if classified is not None else (None, 0)
    for slab in range(n_slabs):                 # 1 for M=64; 2 for M=128
        op(tmem_buf.allocated_addr[0],
           *[local_32b[reg_base + i] for i in range(regs_eff)],
           shape=shape, num=num_eff,
           row=(sub_slab + slab) * 16, col=col_off_32b)

Layout B is routed before the ordinary atom matching. Its logical
``(64, N)`` fragment is emitted as one physical ``.32x32b.x{N/2}``
instruction over all 128 lanes.

The dispatch emits **no** wait — the caller issues ``tcgen05.wait.ld()`` /
``wait.st()`` (as in the demo).

Selecting the upper F sub-slab
-------------------------------

``sub_slab`` is part of the tensor-memory layout rather than an option on
``copy_async``. This keeps physical TMEM occupation explicit and lets two
64-row buffers alias the lower and upper halves of one 128-row allocation:

.. code-block:: python

    from tvm.tirx.layout import tmem_datapath_layout

    lower = T.decl_buffer(
        (64, cols),
        "float32",
        scope="tmem",
        allocated_addr=tmem_addr[0],
        layout=tmem_datapath_layout("F", 64, cols, sub_slab=0),
    )
    upper = T.decl_buffer(
        (64, cols),
        "float32",
        scope="tmem",
        allocated_addr=tmem_addr[0],
        layout=tmem_datapath_layout("F", 64, cols, sub_slab=1),
    )

    Tx.wg.copy_async(lower_frag, lower)
    T.ptx.tcgen05.wait.ld()
    Tx.wg.copy_async(upper_frag, upper)
    T.ptx.tcgen05.wait.ld()

The lower view emits ``row=0`` and the upper view emits ``row=16`` for
``.16x64b``, ``.16x128b``, and ``.16x256b`` atoms. Layout D has 128 rows and
already spans both sub-slabs, so it only accepts ``sub_slab=0``.

Layout B readback
-----------------

Layout B is produced by an M=64-per-CTA ``tcgen05.mma`` with
``cta_group=2``. Its logical ``(64, N)`` columns are split into two
``N/2`` halves: the low half uses physical TMEM lanes 0–63, and the high
half uses lanes 64–127. It therefore occupies all 128 lanes and ``N/2``
``TCol`` values.

Use the public allocation and fragment APIs together:

.. code-block:: python

    accumulator = tmem_pool.alloc((64, N), "float32", datapath="B")
    frag = T.alloc_tcgen05_ldst_frag("32x32b", (64, N), "float32")

    Tx.wg.copy_async(frag[:, :], accumulator[:, :])
    T.ptx.tcgen05.wait.ld()

    # The inverse direction emits tcgen05.st with the same physical image.
    Tx.wg.copy_async(accumulator[:, :], frag[:, :])
    T.ptx.tcgen05.wait.st()

This is a single ``tcgen05.{ld,st}.32x32b.x{N/2}`` issue. ``N`` must be
even, ``N/2`` must be a valid PTX ``num``, and the fragment is fp32-only.
The first implementation intentionally requires the full logical
``(64, N)`` region: a partial logical-column slice is not contiguous after
the two-way lane split. A conventional ``.16x*b`` fragment is rejected with
an actionable error.

Generated TIRx IR
-----------------

For the ``128×8`` fp16 tile the layout takes the ``.32x32b`` path with ``num = 4``
(4 registers per thread), one issue each way:

.. code-block:: python

    T.ptx.tcgen05.st(tmem_addr[0], 0, 0, "32x32b", 4, False, local_32b[0], local_32b[1],
                     local_32b[2], local_32b[3])     # local -> tmem
    T.ptx.tcgen05.ld(tmem_addr[0], 0, 0, "32x32b", 4, False, local_32b_1[0], local_32b_1[1],
                     local_32b_1[2], local_32b_1[3]) # tmem -> local

Generated CUDA
--------------

.. code-block:: c++

    "tcgen05.st.sync.aligned.32x32b.x4.b32 ..."   // 4 registers -> tmem
    "tcgen05.ld.sync.aligned.32x32b.x4.b32 ..."   // tmem -> 4 registers

Verified end-to-end on ``sm_100a`` (the round trip reproduces the input exactly).

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - register layout
     - matches a ``.16x64b`` / ``.16x128b`` / ``.16x256b`` atom → that shape; no
       match → the ``.32x32b`` fallback (this demo)
   * - column width / dtype
     - sets ``num`` (the ``.xN`` count) and the registers per thread
       (``elem_per_32b = 32 / dtype_bits``)
   * - direction
     - ``tmem → local`` → ``tcgen05.ld``; ``local → tmem`` → ``tcgen05.st`` (same
       shape/num logic)
   * - datapath D vs F vs B
     - ``D`` (M=128) covers all 128 rows; an M=128 ``.16x*b`` copy issues two slabs
       (``row = 0`` / ``row = 16``). ``F`` (M=64) scatters rows to lanes and
       its layout selects one issue at ``row = 0`` or ``row = 16``. ``B``
       (per-CTA M=64, ``cta_group=2``) splits N across two 64-lane halves and
       emits one logical-64-row ``.32x32b`` image
