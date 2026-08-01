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

copy_async → tcgen05_cp
=======================

The ``tcgen05_cp`` variant lowers a ``copy_async`` from **shared memory to tensor
memory** (Blackwell ``tmem``) through a generic planner covering every
``tcgen05.cp`` shape. A shared **matrix descriptor** names the source tile; all
descriptor fields (ldo/sdo/swizzle) and the cp issue sequence are derived from
the two buffer layouts. The dispatch issues only the copy; the caller signals
completion with ``tcgen05.commit``. Source:
``python/tvm/backend/cuda/operator/tile_primitive/copy_async/tcgen05_cp.py``.

Shape selection
---------------

- ``shape=`` config: forces that PTX shape (with ``multicast=`` where the shape
  has more than one legal qualifier).
- No ``shape`` config: the planner tries each candidate **widest atom first**
  and takes the first whose plan validates against the layouts:
  ``128x256b`` → ``4x256b`` → ``128x128b`` → ``64x128b.warpx2::02_13`` →
  ``64x128b.warpx2::01_23`` → ``32x128b.warpx4``. All candidates but the
  256b/128b pair are mutually exclusive by the tmem (lane, replica) pattern, so
  a bare warpx4 copy still resolves to ``32x128b.warpx4``.

Each shape pins a tmem row→lane mapping and replica (multicast) pattern
(verified bit-exactly on B200 by ``test_tcgen05_cp.py``):

.. list-table::
   :header-rows: 1
   :widths: 18 26 30 26

   * - shape
     - multicast
     - t lane pattern
     - t replica
   * - ``128x256b``
     - (none)
     - ``(128, 1@TLane)``
     - —
   * - ``4x256b``
     - (none)
     - ``(4, 32@TLane)``
     - —
   * - ``128x128b``
     - (none)
     - ``(128, 1@TLane)``
     - —
   * - ``64x128b``
     - ``warpx2::02_13``
     - ``(64, 1@TLane)``
     - ``(2, 64@TLane)``
   * - ``64x128b``
     - ``warpx2::01_23``
     - ``(2,32):(64,1)@TLane``
     - ``(2, 32@TLane)``
   * - ``32x128b``
     - ``warpx4``
     - ``(32, 1@TLane)``
     - ``(4, 32@TLane)``

What it accepts
---------------

Two predicates — the memory-scope envelope and a single-thread exec scope; all
shape/layout validation happens in the planner with readable errors:

.. code-block:: python

    # register_dispatch(..., variant="smem->tmem", priority=10, when=[
    predicate("validate_smem_tmem_copy", _validate_smem_tmem_copy),
    predicate("exec_scope", _single_thread_exec),       # exec_scope == "thread"
    # ])

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / priority
     - ``cuda`` (Blackwell, sm_100+); priority ``10``
   * - scope
     - **single thread** issues the copy
   * - memory pair
     - source ``shared*`` → destination ``tmem`` (with ``allocated_addr`` set by
       a prior ``tcgen05.alloc``); both buffers carry layouts, dtypes match
   * - tmem layout
     - must slice to one shape's (lane, replica) pattern from the table above
   * - smem layout
     - rows in 8-row descriptor core-matrix groups; the atom row width derives
       the swizzle mode (K-byte ∈ {16, 32, 64, 128} → sw 0..3) and must match
       the buffer's swizzle (if any)

Demonstration program
----------------------

A warpgroup allocates 16 tmem columns, fills a ``32×16`` ``uint8`` shared tile,
and copies it into tmem — no shape config, the planner infers
``32x128b.warpx4`` from the layouts (from ``test_tcgen05_cp.py``; readback /
dealloc tail elided):

.. code-block:: python

    from tvm.tirx.layout import R, S, TCol, TileLayout, TLane

    A_smem = T.alloc_buffer([32, 16], "uint8", scope="shared",
                            layout=TileLayout(S[(32, 16) : (16, 1)]), align=1024)
    tmem_addr = T.alloc_shared([1], "uint32")
    cp_mbar   = T.alloc_shared([1], "uint64")
    if warp_id == 0:
        T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=16, cta_group=1)
    # ... mbarrier.init, fence, cta_sync, fill A_smem from global ...
    tmem = T.decl_buffer([32, 16], "uint8", scope="tmem", allocated_addr=tmem_addr[0],
                         layout=TileLayout(S[(32, 16) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]))
    if tid_in_wg == 0:
        Tx.copy_async(tmem[0:32, 0:16], A_smem[0:32, 0:16], cta_group=1)   # smem -> tmem
        T.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=1)             # caller signals
    T.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
    # ... readback via tcgen05.ld, then tcgen05.dealloc ...

Algorithm
---------

**1. Resolve the shape** — explicit ``shape=`` or layout inference (above).

**2. Validate the plan.** Slice both layouts to the region, permute to
TLane-first / TCol-stride-descending, isolate broadcast iters, and split each
side into ``(lane, middle, col)`` segments against the atom: ``lane`` = one
instruction's rows (must match the shape's lane pattern), ``col`` = one
instruction's columns, ``middle`` = the loop dims — the outer columns and, for
lane-tiled atoms (e.g. 16 ``4x256b`` cps filling each warp's first 16 rows, the
M=64 Layout-F scatter), the extra row iters. The smem side derives the
descriptor fields: 8-row group strides → sdo + swizzle mode, 16B-unit stride →
ldo (256b atoms).

**3. Encode the matrix descriptor once.** A 64-bit shared descriptor is encoded
at the smem buffer base right after its allocation, cached per
``(smem_buf, ldo, sdo, swizzle)``; each cp patches only the 14-bit address
field.

**4. Issue the copies** — the middle dims flatten into one unrolled loop; each
step bumps the descriptor's 16-byte shared offset and the tmem address (column
bits, plus the lane half-word for lane-tiled atoms):

.. code-block:: python

    for flat in T.unroll(total):
        t_off, s_off = T.meta_var(compute_offsets(flat))
        T.ptx.tcgen05.cp(t_addr[0] + t_addr_off + t_off,
                         smem_desc_add_16B_offset(desc_buf[0], init_off_16B + s_off),
                         shape=shape, cta_group=cta_group, multicast=multicast)

The dispatch emits **no** ``tcgen05.commit`` / ``wait`` — the caller commits
against an mbarrier (as in the demo).

Generated CUDA
--------------

.. code-block:: c++

    // one warpx4 copy: shared (named by the matrix descriptor) -> tensor memory
    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"   // [%0]=tmem addr, %1=descriptor

(Compiled for ``sm_100a``. End-to-end correctness — including the tmem readback
— is covered by ``test_tcgen05_cp.py``.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - tmem layout
     - selects the shape under inference via its (lane, replica) pattern; a
       ``R[4:32@TLane]`` replica resolves to ``32x128b.warpx4``
   * - dtype
     - sets ``elem_per_atom = atom_bits / dtype_bits`` and, with the smem row
       strides, the descriptor swizzle mode
   * - region size
     - middle dims: one atom → a single ``tcgen05.cp``; wider regions → an
       unrolled loop over columns; more rows than one atom → lane-tiled cps
       stepping the tmem lane half-word
   * - shared swizzle layout
     - changes the encoded ``swizzle`` mode (must match the derived atom K-byte)
   * - cta_group
     - forwarded to the instruction; ``cta_group::2`` is pair-collective — one
       even-CTA issue makes each CTA copy from its own smem into its own tmem
       (B200-pinned by the cta_group=2 round-trip tests)
