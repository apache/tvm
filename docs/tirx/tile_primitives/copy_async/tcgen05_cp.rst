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
memory** (Blackwell ``tmem``). One elected thread issues
``tcgen05.cp.32x128b.warpx4``: a shared **matrix descriptor** names the source tile,
and the ``warpx4`` multicast routes 32 lanes × 128 bits into the tensor-memory lanes
owned by all four warps. The dispatch issues only the copy; the caller signals
completion with ``tcgen05.commit``. Source:
``python/tvm/backend/cuda/operator/tile_primitive/copy_async/tcgen05_cp.py``.

What it accepts
---------------

Two predicates — a valid shared→tmem copy and a single-thread scope:

.. code-block:: python

    # register_dispatch(..., variant="smem->tmem", priority=10, when=[
    predicate("validate_smem_tmem_copy", _is_valid_smem_tmem_copy),
    predicate("exec_scope", _single_thread_exec),       # exec_scope == "thread"
    # ])

    def _is_valid_smem_tmem_copy(op, sctx):
        if not (src.scope().startswith("shared") and dst.scope() == "tmem"): ...
        if not (src.layout and dst.layout): ...
        if dst.allocated_addr is None: ...
        rep = dst.layout.replica                         # the warpx4 router
        if not (len(rep) == 1 and int(rep[0].extent) == 4
                and int(rep[0].stride) == 32 and "TLane" in str(rep[0].axis)):
            return False, f"requires R[4:32@TLane] on tmem, got {list(rep)}"
        return True, None

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
     - source ``shared*`` → destination ``tmem`` (with ``allocated_addr`` set by a
       prior ``tcgen05.alloc``)
   * - tmem layout
     - the replica **must be exactly** ``R[4:32@TLane]`` — the warpx4 router that
       fans the copy across all four warps' tensor-memory lanes
   * - dtype
     - sets ``elem_per_128b = 128 / dtype_bits`` (uint8 → 16) and the descriptor
       swizzle mode

Demonstration program
----------------------

A warpgroup allocates 16 tmem columns, fills a ``32×16`` ``uint8`` shared tile, and
copies it into tmem with ``tcgen05_cp`` (from ``test_smem_tmem.py``; the readback /
dealloc tail is elided):

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

(The ``copy_async`` is auto-dispatched to the ``smem->tmem`` variant — the source is
shared and the destination is the ``R[4:32@TLane]`` tmem buffer.)

Algorithm
---------

**1. Verify the warpx4 router and re-order.** After slicing both layouts to the
region, the dispatch confirms the tmem replica is ``R[4:32@TLane]``, permutes to
TLane-first / TCol-stride-descending, isolates the broadcast, and groups the
remaining iters into ``(32, middle, elem_per_128b)`` — the 32×128-bit atom plus a
list of *middle* tiles to loop over.

**2. Encode the matrix descriptor once.** A 64-bit shared descriptor (leading-dim
offset ``ldo``, stride-dim offset ``sdo``, swizzle mode) is encoded right after the
shared buffer is allocated, cached per ``(smem_buf, ldo, sdo, swizzle)``:

.. code-block:: python

    desc_buf = decl_buffer((1,), "uint64", scope="local")
    T.ptx.tcgen05.encode_matrix_descriptor(desc_buf.data, s_buf.ptr_to([0, 0]), ldo, sdo, swizzle)

**3. Issue the copy** — one ``tcgen05.cp`` for a single atom, or an unrolled loop
that bumps the tmem column offset and the descriptor's 16-byte shared offset per
middle tile:

.. code-block:: python

    if total == 1:
        T.ptx.tcgen05.cp(t_addr[0] + t_col0,
                         smem_desc_add_16B_offset(desc_buf[0], init_off_16B),
                         shape="32x128b", cta_group=cta_group, multicast="warpx4")
    else:
        for flat in T.unroll(total):
            t_off, s_off = T.meta_var(compute_offsets(flat))
            T.ptx.tcgen05.cp(t_addr[0] + t_col0 + t_off,
                             smem_desc_add_16B_offset(desc_buf[0], init_off_16B + s_off),
                             shape="32x128b", cta_group=cta_group, multicast="warpx4")

The dispatch emits **no** ``tcgen05.commit`` / ``wait`` — the caller commits against
an mbarrier (as in the demo).

Generated TIRx IR
-----------------

The ``32×16`` uint8 tile is a single atom (ldo=16, sdo=8, swizzle=0):

.. code-block:: python

    T.ptx.tcgen05.encode_matrix_descriptor(cp_desc.data, T.address_of(A_smem[0]), 16, 8, 0)
    T.ptx.tcgen05.cp(tmem_addr[0],
                     smem_desc_add_16B_offset(cp_desc[0], 0),
                     shape="32x128b", cta_group=1, multicast="warpx4")

Generated CUDA
--------------

.. code-block:: c++

    // one warpx4 copy: shared (named by the matrix descriptor) -> tensor memory
    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"   // [%0]=tmem addr, %1=descriptor

(Compiled for ``sm_100a``. End-to-end correctness — including the tmem readback —
is covered by ``test_smem_tmem.py``.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - dtype
     - sets ``elem_per_128b = 128 / dtype_bits`` (uint8 → 16, uint32 → 4) and the
       descriptor swizzle mode (atom K-byte ∈ {16, 32, 64, 128} → swizzle 0/1/2/3)
   * - number of tiles
     - ``total`` middle tiles: ``1`` → a single ``tcgen05.cp``; ``> 1`` → an
       unrolled loop, each step bumping the tmem column and the descriptor's 16-B
       shared offset
   * - shared swizzle layout
     - changes the encoded ``swizzle`` mode (must match the shared buffer's swizzle)
   * - tmem layout (D vs F) / cta_group
     - the permutation order sets per-tile column steps; ``cta_group`` selects the
       multicast routing (``cta_group::1`` vs ``::2``)
   * - atom shape
     - fixed at ``32x128b`` ``warpx4`` — a different atom would need a new variant
