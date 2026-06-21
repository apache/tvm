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
(``warp_xor_swizzle``) stages each lane's elements through registers and writes them
back under the destination layout, with a per-lane **XOR swizzle** on the iteration
index chosen so that *both* the read and the write phase are shared-memory
bank-conflict-free. A ``warp_sync`` separates the two phases so the op is safe even
when source and destination alias. Source:
``python/tvm/backend/cuda/operator/tile_primitive/permute_layout/warp_xor_swizzle.py``.

What it accepts
---------------

The predicate (``_why_reject``) gates the variant:

.. code-block:: python

    if sctx.scope_kind != "warp":                  return "scope is not 'warp'"
    if src_buf.dtype != dst_buf.dtype:             return "dtype mismatch"
    if src_ext_i != dst_ext_i:                     return "extent mismatch"
    if dtype_bytes not in (1, 2, 4, 8, 16):        return "unsupported dtype byte width"
    if not isinstance(src_buf.layout, TileLayout): return "src not a plain TileLayout"
    if not isinstance(dst_buf.layout, TileLayout): return "dst not a plain TileLayout"
    # + layouts must slice/canonicalize; _choose_xor_k must find a valid k (else fail)

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda``; **warp** scope only; priority ``20``
   * - operands
     - equal dtype, equal (compile-time) extents; both plain ``TileLayout`` (no
       swizzle wrapper); dtype byte width ∈ {1, 2, 4, 8, 16}
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

    @T.prim_func
    def f(A: T.handle, B: T.handle):
        A_buf = T.match_buffer(A, shape, dtype, layout=pre)
        B_buf = T.match_buffer(B, shape, dtype, layout=post)
        T.device_entry(); T.cta_id([1]); T.thread_id([32])
        for s in T.serial(0, pipe):
            Tx.warp.permute_layout(B_buf[s, 0:1, 0:4, 0:32], A_buf[s, 0:1, 0:4, 0:32])

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

**3. Emit two register-staged phases.** Each lane reads its ``P`` elements through
the source layout into registers (the swizzle permutes which register holds which
iteration), a ``warp_sync`` follows, then the registers are written back through the
destination layout:

.. code-block:: python

    regs = T.alloc_buffer((P,), dtype, scope="local")
    for r in T.unroll(0, P):                                   # read via src layout
        j   = r ^ ((lane_id >> shift) & mask)
        idx = decompose(lane_id + j * 32, extent)
        regs[r] = src_buf[project(idx, src_st)]
    T.cuda.warp_sync()
    for r in T.unroll(0, P):                                   # write via dst layout
        j   = r ^ ((lane_id >> shift) & mask)
        idx = decompose(lane_id + j * 32, extent)
        dst_buf[project(idx, dst_st)] = regs[r]
    T.cuda.warp_sync()

Generated TIRx IR
-----------------

.. code-block:: python

    regs[r] = A_buf[s*128 + (r ^ ((tx >> 3) & 3)) % 4 * 32 + tx]   # phase 1 (src order)
    T.cuda.warp_sync()
    B_buf[s*128 + tx * 4 + (r ^ ((tx >> 3) & 3)) % 4] = regs[r]    # phase 2 (dst order)
    T.cuda.warp_sync()

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
     - feeds the bank simulation in ``_choose_xor_k``; 4-byte dtypes always admit a
       valid ``k`` (one element per bank), while stride-1 **sub-4-byte** reads can
       pack several lanes into one bank and make the dispatch ``fail``
   * - chosen ``k``
     - sets ``shift`` / ``mask`` of the XOR swizzle (``k = 0`` ⇒ no swizzle)
   * - ``P`` (= elements/lane)
     - the number of staged registers and unrolled iterations per phase
