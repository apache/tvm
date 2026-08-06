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

gemm_async
==========

``gemm_async`` lowers a matrix multiply to the **Blackwell asynchronous
tensor-core** instruction ``tcgen05.mma``. The A and B operands live in **shared
memory** (named by 64-bit *matrix descriptors*), the accumulator lives in **tensor
memory**, and one elected thread launches the MMA, which runs asynchronously; the
caller signals completion with ``tcgen05.commit`` against an mbarrier. It also
supports **block-scaled** low precision (fp8 / fp4 with per-block scale factors
``SFA`` / ``SFB`` in tensor memory). Source:
``python/tvm/backend/cuda/tile_primitive/gemm_async/tcgen05.py``. (For the
synchronous warp-register path see :doc:`gemm`.)

What it accepts
---------------

A single predicate — single-thread or warp scope:

.. code-block:: python

    # register_dispatch("gemm_async", "cuda", priority=10, when=[
    predicate("single_thread_or_warp",
              lambda op, sctx: (single_thread(op, sctx) or sctx.is_warp,
                                f"unsupported exec_scope {sctx.exec_scope}"))
    # ])

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda`` (Blackwell, sm_100+); **single thread or warp**; priority ``10``
   * - operands
     - A, B in **shared** (B always; A shared, or tmem for the TMEM-A path); the
       accumulator **C/D in tmem** (``float32``)
   * - dtype
     - regular: A/B ``float16`` / ``bfloat16``; block-scaled: A/B
       ``float8_e4m3fn`` / ``float4_e2m1fn`` with ``SFA`` / ``SFB`` scale factors in
       tmem; accumulator always ``float32``
   * - shape
     - per-CTA ``M ∈ {64, 128}``; ``N`` divisible by 8 (cta_group=1) or 16
       (cta_group=2); ``K`` divisible by ``MMA_K`` = 16 (f16/bf16) / 32 (fp8) /
       64 (fp4). With cta_group=2, the CTA pair covers twice the per-CTA M
   * - cta_group
     - ``1`` (one CTA) or ``2`` (two CTAs split the operand)
   * - descriptor mode
     - optional ``smem_desc`` controls shared matrix-descriptor construction:
       ``"hoist"`` (default), ``"local_hoist"``, ``"encode"``, or
       ``"recompute"``
   * - layout forms
     - swizzled shared layouts, no-swizzle packed shared layouts, regular tmem
       accumulators, and FlashMLA-style packed ``N/2`` tmem accumulator layouts

Demonstration program
----------------------

A warpgroup multiplies a ``128×64`` × ``64×128`` ``float16`` tile (f32 accumulate)
into a tmem accumulator, after TMA-loading A/B into shared (from
``test_gemm_async.py``; setup/readback abbreviated):

.. code-block:: python

    from tvm.tirx.layout import S, TCol, TLane, TileLayout, tid_in_wg as axis_tid_in_wg
    from tvm.tirx.cuda.tile_primitive.tma_utils import mma_shared_layout

    A_smem = T.alloc_buffer((3,128,64), "float16", scope="shared", layout=mma_shared_layout("float16", 3, (3,128,64)))
    B_smem = T.alloc_buffer((3,128,64), "float16", scope="shared", layout=mma_shared_layout("float16", 3, (3,128,64)))
    tmem_addr = T.alloc_shared([1], "uint32"); mma_mbar = T.alloc_shared([1], "uint64")
    # ... mbarrier.init, cta_sync ...
    if warp_id == 0:
        T.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
            T.address_of(tmem_addr), T.uint32(512))
    T.cuda.cta_sync()
    tmem = T.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
                         layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]))
    # ... TMA-load A_smem, B_smem from global, wait ...
    if tid_in_wg == 0:
        Tx.gemm_async(tmem[0:128, 256:384], A_smem[1:2, :, :], B_smem[2:3, :, :], dispatch="tcgen05")
        # caller signals completion
        T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            mma_mbar.ptr_to([0]))
    T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
    # ... tcgen05.fence.after_thread_sync(); read tmem back via tcgen05.ld; dealloc ...

Algorithm
---------

**1. Encode or synthesize shared matrix descriptors.** Each shared operand is named
by a 64-bit descriptor (leading-dim offset ``ldo``, stride-dim offset ``sdo``,
swizzle mode).  ``smem_desc`` selects where that descriptor comes from:

* ``"hoist"`` (default): encode one uniform descriptor per operand after shared
  allocation and add each per-MMA 16-byte offset.
* ``"local_hoist"``: encode at this ``gemm_async`` call site, under the caller's
  control flow, then add offsets.  This is for call sites where only the elected
  issue thread should construct the descriptor.
* ``"encode"``: encode the exact shared pointer for each MMA issue.
* ``"recompute"``: synthesize the descriptor value inline per MMA without a local
  descriptor cell.

.. code-block:: python

    T.cuda.tcgen05.encode_matrix_descriptor(descA.data, A_smem.ptr_to([0]), ldo, sdo, swizzle)
    T.cuda.tcgen05.encode_matrix_descriptor(descB.data, B_smem.ptr_to([0]), ldo, sdo, swizzle)

**2. Choose the MMA tile.** ``M_mma × N_mma`` are chosen to tile ``M``/``N`` (with
``MMA_K`` set by dtype: 16 f16/bf16, 32 fp8, 64 fp4); a compile-time *instruction
descriptor* packs the shape and dtypes.

**3. Issue the async MMA** in an unrolled ``(mi, ni, ki)`` nest, accumulating into
the tmem accumulator (``enable_input_d`` turns accumulation on for ``ki > 0``):

.. code-block:: python

    mma_chain = f"tcgen05.mma.cta_group::{cta_group}.kind::{kind}"
    T.ptx[mma_chain](
        T.cast(T.cuda.get_tmem_addr(tmem_addr, mi * M_mma, tmem_col), "uint32"),  # C in tmem
        smem_desc_add_16B_offset(descA, a_off), descB_val, descI,    # A / B descriptors
        *zero_masks,                                                 # disabled output lanes
        ki != 0,                                                     # accumulate over K
    )

For **block-scaled** fp8/fp4 the chain gains ``.block_scale.scale_vec::<n>X``
with two extra tmem addresses — ``SFA`` / ``SFB`` — and the scale-factor dtypes; the
instruction descriptor is encoded at runtime. As with the other async ops, the
dispatch emits **no** completion — the caller's ``tcgen05.commit`` + mbarrier wait
close it.

For row-0 schedules, the lowering folds ``T.cuda.get_tmem_addr(base, 0, col)`` to
``base + col``.  This keeps the generated issue loop close to hand-written
FlashMLA kernels while preserving the helper call for nonzero row offsets.  When
``weight_stationary=True`` is passed, the flag is forwarded to the PTX wrapper so
the wrapper can select the matching tcgen05 MMA ABI.

Accumulator datapaths and readback
----------------------------------

The accumulator layout must match the MMA's row placement:

* Layout D is the M=128 identity placement.
* Layout F is the single-CTA M=64 scattered placement.
* Layout B is the per-CTA M=64 placement for ``cta_group=2``. Its logical
  N columns split across physical lane halves 0–63 and 64–127, so it
  occupies all 128 lanes and ``N/2`` tensor-memory columns.

Allocate and read a Layout B result as follows:

.. code-block:: python

    accumulator = tmem_pool.alloc((64, N), "float32", datapath="B")
    Tx.gemm_async(
        accumulator[:, :],
        A_smem[:, :],
        B_smem[:, :],
        dispatch="tcgen05",
        cta_group=2,
    )

    frag = T.alloc_tcgen05_ldst_frag("32x32b", (64, N), "float32")
    Tx.wg.copy_async(frag[:, :], accumulator[:, :])
    T.ptx.tcgen05.wait__ld.sync.aligned()

The fragment is a logical ``(64, N)`` view of one physical
``.32x32b`` transfer over all 128 lanes. The gemm write-side layout and
the allocation/readback layout are produced by the same
``tmem_datapath_layout("B", 64, N)`` factory.

Generated TIRx IR
-----------------

For the ``128×64 × 64×128`` fp16 tile (swizzle mode 3):

.. code-block:: python

    T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descA[0]), T.address_of(A_smem[0]), 64, 64, 3)
    T.cuda.tcgen05.encode_matrix_descriptor(T.address_of(descB[0]), T.address_of(B_smem[0]), 64, 64, 3)
    T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
        T.cast(T.cuda.get_tmem_addr(tmem_addr[0], mi * 128, 256 + ni * 128), "uint32"), ...)

Generated CUDA
--------------

.. code-block:: c++

    // async tensor-core MMA: A,B (shared, via descriptors) -> C (tmem)
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, ...;"
    //  [%0] = C tmem address;  %1 = A descriptor;  %2 = B descriptor;  %3 = instr descriptor

``kind::f16`` selects the fp16/bf16 datapath. Verified on ``sm_100a`` (the tmem
result, read back, equals ``A@B`` within fp16 tolerance).

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - dtype
     - ``float16``/``bfloat16`` → ``kind::f16``, ``MMA_K = 16``; ``fp8`` →
       ``MMA_K = 32``; ``fp4`` → ``MMA_K = 64`` and the **block-scaled** path
   * - block scaling (SFA/SFB)
     - present → ``tcgen05.mma.block_scale`` with SFA/SFB tmem scale-factor
       addresses and a runtime-encoded instruction descriptor
   * - cta_group
     - ``1`` → one CTA, ``M ∈ {64, 128}``; ``2`` → two CTAs split the operand,
       each with per-CTA ``M ∈ {64, 128}`` and half the B rows. The
       per-CTA M=64 output uses Layout B
   * - M / N / K extents
     - set the ``(mi, ni, ki)`` unrolled loop counts; K iterations accumulate into
       the same tmem accumulator
   * - shared swizzle
     - sets the ``swizzle`` mode + ``ldo``/``sdo`` in the matrix descriptors;
       no-swizzle packed layouts are accepted when the selected tile has a
       hardware-compatible 16-byte packed stride
   * - ``smem_desc``
     - selects hoisted, call-site-hoisted, per-MMA encoded, or recomputed shared
       descriptor construction.  The choice changes code shape only; the MMA
       operands still describe the same selected shared tiles.
   * - packed tmem accumulator
     - layouts of the form ``TileLayout(S[(M, 2, N//2) : (1@TLane, 64@TLane,
       1@TCol)])`` are treated as packed ``N/2`` physical columns, matching
       FlashMLA-style low/high accumulator placement.
   * - ``weight_stationary``
     - forwarded to the low-level tcgen05 wrapper for kernels that require that
       issue ABI; it is a lowering/configuration mode, not a separate PTX
       instruction mnemonic.
