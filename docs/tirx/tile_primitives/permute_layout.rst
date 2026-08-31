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

permute_layout
==============

``permute_layout`` rearranges a warp's data from a source ``TileLayout`` to a
destination one — typically an in-place transpose. The single CUDA variant
(``warp_xor_swizzle``) stages each lane's elements through a ``local`` buffer and
writes them back under the destination layout. It chooses a per-lane **XOR
swizzle** using the shared-memory bank model; for shared operands, the selected
iteration order makes both read and write phases bank-conflict-free. A
``warp_sync`` separates the two phases so the op is safe even when source and
destination alias. Source:
``python/tvm/backend/cuda/tile_primitive/permute_layout/warp_xor_swizzle.py``.

What it accepts
---------------

The implementation first runs its ``_why_reject`` validator:

.. code-block:: python

    if sctx.scope_kind != "warp":                  return "scope is not 'warp'"
    if "threadIdx.y" in launch or "threadIdx.z" in launch: return "multi-dim threadIdx"
    if src_buf.dtype != dst_buf.dtype:             return "dtype mismatch"
    if src_ext_i != dst_ext_i:                     return "extent mismatch"
    if dtype_bytes not in (1, 2, 4, 8, 16):        return "unsupported dtype byte width"
    if not isinstance(src_buf.layout, TileLayout): return "src not a plain TileLayout"
    if not isinstance(dst_buf.layout, TileLayout): return "dst not a plain TileLayout"
    # + layouts must slice, regroup, and define bijections on the slice
    # + V % 32 == 0 and P = V/32 is a power of two in [1, 32]
    # + _choose_xor_k must find a valid k (else fail)

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda``; **warp** scope only; priority ``20``
   * - operands
     - equal dtype, equal (compile-time) extents; both plain ``TileLayout`` (no
       swizzle wrapper); dtype byte width ∈ {1, 2, 4, 8, 16}. The dispatcher has
       no storage-scope predicate; shared 32/64-bit operands use direct PTX
       ``ld.shared`` / ``st.shared``, while other accepted cases use ordinary
       buffer loads and stores
   * - launch / volume
     - one-dimensional ``threadIdx``; slice volume ``V`` is divisible by 32 and
       ``P = V/32`` is a power of two in ``[1, 32]``
   * - layout mapping
     - after slicing and regrouping, source and destination describe the same
       iteration extents and each is a bijection on the slice
   * - bank-freedom
     - ``_choose_xor_k`` must find an XOR-bit count ``k ∈ [0, log2(P)]`` that makes
       **both** phases bank-conflict-free, else the dispatch declines (``fail``)

Demonstration program
----------------------

A warp transposes the inner ``4×32`` block of a scale-factor tile — source layout
strides ``(…, 32, 1)``, destination ``(…, 1, 4)`` — for two pipeline stages (the
canonical SF-transpose, from ``test_permute_layout.py``):

.. code-block:: python

    pipe, blk, dtype = 2, 128, "float32"; high = 1
    shape = (pipe, high, 4, 32)
    pre  = TileLayout(S[shape : (blk, 128, 32, 1)])   # source
    post = TileLayout(S[shape : (blk, 128, 1, 4)])    # destination (4↔32 transposed)

    @Tx.prim_func
    def f(A: Tx.handle, B: Tx.handle):
        A_buf = Tx.match_buffer(A, shape, dtype, layout=pre)
        B_buf = Tx.match_buffer(B, shape, dtype, layout=post)
        Tx.device_entry(); Tx.cta_id([1]); Tx.thread_id([32])
        for s in Tx.serial(0, pipe):
            Tx.tile.warp.permute_layout(B_buf[s, 0:1, 0:4, 0:32], A_buf[s, 0:1, 0:4, 0:32])

Algorithm
---------

**1. Align the two layouts.** Both layouts are sliced to the region and
canonicalized; if their shards differ in structure (a linear layout collapses to 1-D
under canon, a transposed one keeps its multi-dim shape) the source is regrouped to
the destination's shape. From the destination shard come the iteration ``extent``
and the per-side strides ``src_str`` / ``dst_str``. ``P`` = elements per lane =
``prod(extent) / 32`` (here ``4``).

**2. Choose the XOR swizzle.** ``_choose_xor_k`` simulates the shared-memory bank
pattern at shard granularity for ``k = 0, 1, … log2(P)`` and picks the smallest
``k`` whose ``shift`` / ``mask`` make *both* phases conflict-free (here ``shift = 3``,
``mask = 3``).

**3. Emit two local-staged phases.** Each lane reads its ``P`` elements through
the source layout into a local temporary (the swizzle permutes which slot holds
which iteration), a ``warp_sync`` follows, then the elements are written back through the
destination layout:

.. code-block:: python

    regs = Tx.alloc_buffer((P,), dtype, scope="local")
    for r in Tx.unroll(0, P):                                   # read via src layout
        j   = r ^ ((lane_id >> shift) & mask)
        idx = decompose(lane_id + j * 32, extent)
        regs[r] = src_buf[project(idx, src_st)]
    Tx.cuda.warp_sync()
    for r in Tx.unroll(0, P):                                   # write via dst layout
        j   = r ^ ((lane_id >> shift) & mask)
        idx = decompose(lane_id + j * 32, extent)
        dst_buf[project(idx, dst_st)] = regs[r]
    Tx.cuda.warp_sync()

Generated TIRx IR
-----------------

.. code-block:: python

    regs[r] = A_buf[s*128 + (r ^ ((tx >> 3) & 3)) % 4 * 32 + tx]   # phase 1 (src order)
    Tx.cuda.warp_sync()
    B_buf[s*128 + tx * 4 + (r ^ ((tx >> 3) & 3)) % 4] = regs[r]    # phase 2 (dst order)
    Tx.cuda.warp_sync()

Generated CUDA
--------------

.. code-block:: c++

    alignas(64) float regs_ptr[4];
    regs_ptr[0] = A_buf_ptr[(s*128) + (((0 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    regs_ptr[1] = A_buf_ptr[(s*128) + (((1 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    regs_ptr[2] = A_buf_ptr[(s*128) + (((2 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    regs_ptr[3] = A_buf_ptr[(s*128) + (((3 ^ ((threadIdx.x >> 3) & 3)) & 3) * 32) + threadIdx.x];
    __syncwarp();
    // ... 4 transposed writes into B_buf_ptr, then __syncwarp();

Each lane owns column ``threadIdx.x`` and stages its 4 rows through ``regs``; the
``(threadIdx.x >> 3)`` XOR rotates the register order per lane-group of 8 so the
write phase hits distinct banks. Verified on ``sm_100a`` — the ``4×32`` block is
transposed for every pipeline stage.

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - layout strides (the permutation)
     - define ``extent`` / ``src_str`` / ``dst_str`` and hence ``P`` and the
       per-element index math (the transpose pattern)
   * - dtype byte width
     - feeds the bank simulation in ``_choose_xor_k``; no dtype width alone
       guarantees a match, and any layout whose two phases cannot both be made
       bank-free causes the dispatch to fail
   * - chosen ``k``
     - sets ``shift`` / ``mask`` of the XOR swizzle (``k = 0`` ⇒ no swizzle)
   * - ``P`` (= elements/lane)
     - the number of staged local elements and unrolled iterations per phase
