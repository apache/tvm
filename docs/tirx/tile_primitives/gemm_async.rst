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
tensor-core** instruction ``tcgen05.mma``. B, and normally A, live in **shared
memory** and are named by 64-bit *matrix descriptors*; A can instead use the
tensor-memory operand path. The accumulator lives in **tensor memory**, and one
thread launches the MMA, which runs asynchronously. A
single-thread call site has already selected that issuer; a warp-scoped call uses
``elect_sync`` internally. The caller signals completion with ``tcgen05.commit``
against an mbarrier. It also
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
     - ``cuda`` target with ``tcgen05`` support (this implementation is tested
       with ``sm_100a``); **single thread or warp**; priority ``10``
   * - operands
     - B is in **shared**; A is in **shared** or, for the TMEM-A path, **tmem**;
       the accumulator C/D is in **tmem** (``float32``)
   * - dtype
     - without ``is_AB_tf32``, dense A/B use the same dtype: ``float16``,
       ``bfloat16``, ``float8_e4m3fn``, or ``float8_e5m2``. Typical TF32
       authoring uses ``float32`` buffers with ``is_AB_tf32=True``; the flag
       replaces both semantic A/B dtypes with ``tf32`` before validation, so
       the current implementation does not separately validate the underlying
       storage dtypes or their equality in that mode.
       Block-scaled A/B: ``float8_e4m3fn`` with tmem ``SFA`` / ``SFB`` of
       ``float8_e8m0fnu``, or ``float4_e2m1fn`` with matching
       ``float8_e8m0fnu`` or ``float8_e4m3fn`` scale factors.
       The accumulator is always ``float32``
   * - shape
     - the per-CTA M region is tiled by 64- or 128-row instruction blocks;
       ``N`` is divisible by 8 (cta_group=1) or 16 (cta_group=2), and ``K`` is
       divisible by ``MMA_K`` = 8 (tf32) / 16 (f16/bf16) / 32 (fp8) / 64
       (fp4). With cta_group=2, the CTA pair covers twice the per-CTA M
   * - cta_group
     - ``1`` (one CTA) or ``2`` (two CTAs split the operand)
   * - descriptor mode
     - optional ``smem_desc`` controls shared matrix-descriptor construction:
       ``"hoist"`` (default), ``"local_hoist"``, ``"encode"``, or
       ``"recompute"``. ``"encode"`` is currently dense-only. Other strings
       are not rejected and take the descriptor-add path used by hoisting
   * - explicit instruction tile
     - ``mma_m`` and ``mma_n`` must be supplied together. ``mma_m`` must equal
       ``M * cta_group``; ``mma_n`` must divide N exactly; the resulting
       physical tile is also checked by the tcgen05 hardware-shape validator
   * - instruction descriptor
     - dense MMA always encodes its descriptor in the dispatcher and rejects
       ``descI``.  Block-scaled MMA may accept a pre-encoded uint32 ``descI``
   * - layout forms
     - swizzled shared layouts, no-swizzle packed shared layouts, regular tmem
       accumulators, and FlashMLA-style packed ``N/2`` tmem accumulator layouts

Demonstration program
----------------------

One selected thread in a warpgroup multiplies a ``128×64`` × ``64×128``
``float16`` tile (f32 accumulate) into a tmem accumulator, after TMA-loading A/B
into shared (from
``test_gemm_async.py``; setup/readback abbreviated):

.. code-block:: python

    from tvm.tirx.layout import S, TCol, TLane, TileLayout, tid_in_wg as axis_tid_in_wg
    from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout

    A_smem = Tx.alloc_buffer((3,128,64), "float16", scope="shared", layout=mma_shared_layout("float16", 3, (3,128,64)))
    B_smem = Tx.alloc_buffer((3,128,64), "float16", scope="shared", layout=mma_shared_layout("float16", 3, (3,128,64)))
    tmem_addr = Tx.alloc_shared([1], "uint32"); mma_mbar = Tx.alloc_shared([1], "uint64")
    # ... mbarrier.init, cta_sync ...
    if warp_id == 0:
        Tx.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
            Tx.address_of(tmem_addr), Tx.uint32(512))
    Tx.cuda.cta_sync()
    tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
                         layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]))
    # ... TMA-load A_smem, B_smem from global, wait ...
    if tid_in_wg == 0:
        Tx.tile.gemm_async(tmem[0:128, 256:384], A_smem[1:2, :, :], B_smem[2:3, :, :], dispatch="tcgen05")
        # caller signals completion
        Tx.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            mma_mbar.ptr_to([0]))
    Tx.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
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

    Tx.cuda.tcgen05.encode_matrix_descriptor(descA.data, A_smem.ptr_to([0]), ldo, sdo, swizzle)
    Tx.cuda.tcgen05.encode_matrix_descriptor(descB.data, B_smem.ptr_to([0]), ldo, sdo, swizzle)

**2. Choose the MMA tile.** ``M_mma × N_mma`` are chosen to tile ``M``/``N``
(with ``MMA_K`` set by dtype: 8 tf32, 16 f16/bf16, 32 fp8, 64 fp4); a
compile-time *instruction descriptor* packs the shape and dtypes.

**3. Issue the async MMA** in an unrolled ``(mi, ni, ki)`` nest. The
``enable_input_d`` input is ``accum or ki != 0``: with ``accum=False``, the first
K issue overwrites the destination; with ``accum=True``, it also accumulates the
destination's existing value.

.. code-block:: python

    mma_chain = f"tcgen05.mma.cta_group::{cta_group}.kind::{kind}"
    Tx.ptx[mma_chain](
        Tx.cast(Tx.cuda.get_tmem_addr(tmem_addr, mi * M_mma, tmem_col), "uint32"),  # C in tmem
        smem_desc_add_16B_offset(descA, a_off), descB_val, descI,    # A / B descriptors
        *zero_masks,                                                 # disabled output lanes
        accum or ki != 0,                                            # use existing C/D input
    )

For **block-scaled** fp8/fp4 the chain gains ``.block_scale.scale_vec::<n>X``
with two extra tmem addresses — ``SFA`` / ``SFB`` — and the scale-factor dtypes;
the instruction descriptor is encoded at runtime unless the caller supplies
``descI``. As with the other async ops, the dispatch emits **no** completion —
the caller's ``tcgen05.commit`` + mbarrier wait close it.

For row-0 schedules, the lowering folds ``Tx.cuda.get_tmem_addr(base, 0, col)`` to
``base + col``.  This keeps the generated issue loop close to hand-written
FlashMLA kernels while preserving the helper call for nonzero row offsets.
On the dense path, ``weight_stationary=True`` with ``cta_group=1`` selects the
``tcgen05.mma.ws`` ABI.  The PTX table has no ``.ws.cta_group::2`` form.  The
dispatcher also infers this mode from the packed M=64 Layout-E accumulator and
rejects layout/flag combinations that would place tensor-memory rows
incorrectly. Block-scaled MMA uses its own instruction chain and does not append
``.ws``.

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

    accumulator = tmem_pool.alloc_tcgen05_mma_D(
        (64, N), "float32", M=128, cta_group=2)
    Tx.tile.gemm_async(
        accumulator[:, :],
        A_smem[:, :],
        B_smem[:, :],
        dispatch="tcgen05",
        cta_group=2,
    )

    frag = Tx.alloc_tcgen05_ldst_frag("32x32b", (64, N), "float32")
    Tx.tile.wg.copy_async(frag[:, :], accumulator[:, :])
    Tx.ptx.tcgen05.wait__ld.sync.aligned()

The fragment is a logical ``(64, N)`` view of one physical
``.32x32b`` transfer over all 128 lanes. The gemm write-side layout and
the allocation/readback layout are produced by the same
``tmem_datapath_layout("B", 64, N)`` factory.

Generated TIRx IR
-----------------

For the ``128×64 × 64×128`` fp16 tile (swizzle mode 3):

.. code-block:: python

    Tx.cuda.tcgen05.encode_matrix_descriptor(Tx.address_of(descA[0]), Tx.address_of(A_smem[0]), 64, 64, 3)
    Tx.cuda.tcgen05.encode_matrix_descriptor(Tx.address_of(descB[0]), Tx.address_of(B_smem[0]), 64, 64, 3)
    Tx.ptx["tcgen05.mma.cta_group::1.kind::f16"](
        Tx.cast(Tx.cuda.get_tmem_addr(tmem_addr[0], mi * 128, 256 + ni * 128), "uint32"), ...)

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
     - ``tf32`` → ``MMA_K = 8``; ``float16``/``bfloat16`` → ``kind::f16`` and
       ``MMA_K = 16``; ``fp8`` → ``MMA_K = 32``; ``fp4`` → ``MMA_K = 64`` and
       the **block-scaled** path
   * - block scaling (SFA/SFB)
     - present → ``tcgen05.mma.block_scale`` with SFA/SFB tmem scale-factor
       addresses and a runtime-encoded or caller-supplied instruction descriptor
   * - cta_group
     - ``1`` → one CTA; ``2`` → two CTAs split the operand and each sees half
       of B's logical N extent.  The selected physical instruction has
       cluster-wide M of 64/128 for one CTA or 128/256 for two CTAs; larger
       operand regions are tiled by those instruction blocks.  The per-CTA
       M=64 output uses Layout B
   * - M / N / K extents
     - set the ``(mi, ni, ki)`` unrolled loop counts; K iterations accumulate into
       the same tmem accumulator.  ``accum=True`` preserves and accumulates the
       value already in that accumulator on the first K iteration
   * - shared swizzle
     - sets the ``swizzle`` mode + ``ldo``/``sdo`` in the matrix descriptors;
       no-swizzle packed layouts are accepted when the selected tile has a
       hardware-compatible 16-byte packed stride
   * - ``smem_desc``
     - selects hoisted, call-site-hoisted, per-MMA encoded, or recomputed shared
       descriptor construction.  Per-MMA ``"encode"`` is dense-only.  The
       choice changes code shape only; the MMA operands still describe the same
       selected shared tiles.
   * - packed tmem accumulator
     - layouts of the form ``TileLayout(S[(M, 2, N//2) : (1@TLane, 64@TLane,
       1@TCol)])`` are treated as packed ``N/2`` physical columns, matching
       FlashMLA-style low/high accumulator placement.
   * - ``weight_stationary``
     - on the dense path with ``cta_group=1``, selects the ``tcgen05.mma.ws``
       form when explicitly true; it is also inferred for the packed M=64
       Layout-E accumulator. The accumulator and A-operand layouts must match
       that datapath. The block-scaled path does not append ``.ws``, and the PTX
       table does not register a ``.ws.cta_group::2`` form.
