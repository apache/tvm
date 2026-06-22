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

gemm
====

``gemm`` computes ``D = alpha·A@B + beta·C`` at **warp** scope as a fully-unrolled
nest of warp-collective ``mma.sync.aligned.m16n8k{16,8}`` instructions. A and B
fragments and the C/D accumulators **all live in registers** — the caller stages A
and B into register fragments first (typically via :doc:`copy/ldstmatrix`). The
dispatch tiles M/N/K into ``m16n8k`` atoms and emits one ``mma`` per output tile,
accumulating over K in place. Source:
``python/tvm/backend/cuda/operator/tile_primitive/gemm/mma_m16n8k_.py``. (For the
Blackwell async tensor-core path see :doc:`gemm_async`.)

What it accepts
---------------

.. code-block:: python

    # register_dispatch("gemm", "cuda", priority=10, when=[
    predicate("full_active_lanes", _full_active_lanes),   # whole warp, un-narrowed
    predicate("no_replica", _no_replica),                 # no broadcast axes on D/A/B/C
    # ])
    # in the impl:
    for buf, name in ((D, "D"), (A, "A"), (B, "B"), (C, "C")):
        if buf.scope() != "local":
            fail(f"gemm mma requires {name} in register (local) scope, got {buf.scope()}")

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda``; **warp** (``mma.sync`` is warp-collective — all 32 lanes active,
       ``_full_active_lanes``); priority ``10``
   * - operand scope
     - **A, B, C, D all in registers** (``local``); a shared operand makes the
       dispatch ``fail`` (stage with ldmatrix first)
   * - no replica
     - none of D/A/B/C may carry a broadcast/replica axis (``_no_replica``)
   * - shape
     - ``M % 16 == 0``, ``N % 8 == 0``, ``K % 8`` (k8) or ``% 16`` (k16) — each
       dim must tile into the m16n8k fragment frame
   * - dtype
     - inputs ``float16`` / ``bfloat16``; accumulator ``float32``
   * - alpha / beta
     - ``alpha == 1.0``; ``beta ∈ {0.0, 1.0}`` (0 → ``D = A@B``; 1 → ``D = A@B + C``)

Demonstration program
----------------------

A single warp computes ``D[16,8] = A[16,16] @ B[16,8]`` in ``float16`` (f32
accumulate) — one ``m16n8k16`` atom (from ``test_gemm_mma_m16n8k_.py``):

.. code-block:: python

    from tvm.tirx.layout import S, TileLayout, laneid

    D_FRAG    = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
    A_FRAG_K8 = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])
    B_FRAG_K8 = TileLayout(S[(4, 2, 8) : (1 @ laneid, 1, 4 @ laneid)])
    A_FRAG = A_FRAG_K8.tile_to([16, 16], [16, 8]); B_FRAG = B_FRAG_K8.tile_to([16, 8], [8, 8])

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, D_ptr: T.handle):
        A_g = T.match_buffer(A_ptr, (16, 16), "float16"); B_g = T.match_buffer(B_ptr, (16, 8), "float16")
        D_g = T.match_buffer(D_ptr, (16, 8), "float32")
        T.device_entry(); T.cta_id([1]); T.warp_id([1]); lane = T.lane_id([32])
        A_f = T.alloc_buffer((16, 16), "float16", scope="local", layout=A_FRAG)
        B_f = T.alloc_buffer((16, 8),  "float16", scope="local", layout=B_FRAG)
        D_f = T.alloc_buffer((16, 8),  "float32", scope="local", layout=D_FRAG)
        A_reg = A_f.local(8)                              # stage A into the lane's 8 regs
        for s in T.unroll(8):
            kp, kHi, rM = s % 2, (s // 2) % 2, s // 4
            A_reg[s] = A_g[lane // 4 + 8 * rM, 2 * (lane % 4) + kp + 8 * kHi]
        B_reg = B_f.local(4)                              # stage B into the lane's 4 regs
        for s in T.unroll(4):
            kp, kHi = s % 2, s // 2
            B_reg[s] = B_g[2 * (lane % 4) + kp + 8 * kHi, lane // 4]
        Tx.warp.gemm(D_f, A_f, B_f, D_f, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0)
        D_reg = D_f.local(4)                              # write the 4 result regs out
        for s in T.unroll(4):
            rN, rM = s % 2, s // 2
            D_g[lane // 4 + 8 * rM, 2 * (lane % 4) + rN] = D_reg[s]

Algorithm
---------

**1. Tile and fragment-group.** The dispatch slices each operand's layout to its
region and, for each candidate instruction (``m16n8k16`` then ``m16n8k8``), tries to
group the operand sub-layouts (``D_M, D_N, A_M, A_K, B_K, B_N, C_*``) into the fixed
m16n8k frame, anchoring A/C on D's M, B/C on D's N, and B on A's K. The first
instruction that fits, with matching warp-tiling, wins.

**2. Derive register layouts.** Each operand gets a per-lane register view: D/C as
``[Mo, No, rM, rN]`` (4 f32), A as ``[Mo, Ko, rM, kHi, k_pack]``, B as
``[Ko, No, kHi, k_pack]`` — the exact register order ``mma.sync`` expects.

**3. Emit the unrolled nest** — initialize D (from C if ``beta==1``, else 0), then
accumulate over K in place, one ``mma`` per (m, n) tile:

.. code-block:: python

    for m in T.unroll(M_tiles):
        for n in T.unroll(N_tiles):
            for rM, rN in ...: d_local[m, n, rM, rN] = c_local[...] if use_c else T.float32(0)
            for k in T.unroll(K_tiles):
                d_ptrs = [d_local.ptr_to([m, n, rM, rN]) for rM in range(2) for rN in range(2)]  # 4 f32
                a_ptrs = [a_local.ptr_to([m, k, rM, kHi, 0]) for kHi in range(n_kHi) for rM in range(2)]
                b_ptrs = [b_local.ptr_to([k, n, kHi, 0]) for kHi in range(n_kHi)]
                T.ptx.mma(shape_str, "row", "col", "float32", a_type, b_type, "float32",
                          d_ptrs, a_ptrs, b_ptrs, d_ptrs)        # d = a·b + d

Generated TIRx IR
-----------------

The single 16×8×16 tile lowers to one ``mma`` (4 D regs, 4 A regs, 2 B regs):

.. code-block:: python

    T.ptx.mma("m16n8k16", "row", "col", "float32", "float16", "float16", "float32",
              4, 4, 2, 4, False, T.address_of(d_local[0]), ...)

Generated CUDA
--------------

.. code-block:: c++

    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"

The accumulator ``{%0..%3}`` is both the C input and the D output (in-place
accumulate); ``{%4..%7}`` are A's four ``b32`` registers, ``{%8, %9}`` B's two.
Verified on ``sm_100a`` (``D == A@B`` within fp16 tolerance).

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - input dtype
     - ``float16`` → ``…f32.f16.f16.f32``; ``bfloat16`` → ``…f32.bf16.bf16.f32``
       (register counts unchanged — 2 elems per ``b32``)
   * - K instruction
     - ``k16`` → A 4 ``b32`` / B 2 ``b32``; ``k8`` → A 2 / B 1
       (``mma.…m16n8k8.…``)
   * - M / N / K extents
     - set the ``M_tiles`` / ``N_tiles`` / ``K_tiles`` unrolled loop counts (one
       ``mma`` per (m, n), K accumulated in place)
   * - beta
     - ``0`` → D zero-initialized; ``1`` → D initialized from C (the ``mma`` itself
       is identical)
   * - operand scope
     - A/B **must** be register fragments; a shared operand makes the dispatch
       ``fail`` (stage via :doc:`copy/ldstmatrix` first)
