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
``python/tvm/backend/cuda/operator/tile_primitive/gemm_async/tcgen05.py``. (For the
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
     - ``M ∈ {64, 128}`` (×2 for cta_group=2); ``N`` divisible by 8 (cta_group=1) or
       16 (cta_group=2); ``K`` divisible by ``MMA_K`` = 16 (f16/bf16) / 32 (fp8) /
       64 (fp4)
   * - cta_group
     - ``1`` (one CTA) or ``2`` (two CTAs split the operand)

Demonstration program
----------------------

A warpgroup multiplies a ``128×64`` × ``64×128`` ``float16`` tile (f32 accumulate)
into a tmem accumulator, after TMA-loading A/B into shared (from
``test_gemm_async.py``; setup/readback abbreviated):

.. code-block:: python

    from tvm.tirx.layout import S, TCol, TLane, TileLayout, tid_in_wg as axis_tid_in_wg
    from tvm.tirx.cuda.operator.tile_primitive.tma_utils import mma_shared_layout

    A_smem = T.alloc_buffer((3,128,64), "float16", scope="shared", layout=mma_shared_layout("float16", 3, (3,128,64)))
    B_smem = T.alloc_buffer((3,128,64), "float16", scope="shared", layout=mma_shared_layout("float16", 3, (3,128,64)))
    tmem_addr = T.alloc_shared([1], "uint32"); mma_mbar = T.alloc_shared([1], "uint64")
    # ... mbarrier.init, cta_sync ...
    if warp_id == 0:
        T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    T.cuda.cta_sync()
    tmem = T.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
                         layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]))
    # ... TMA-load A_smem, B_smem from global, wait ...
    if tid_in_wg == 0:
        Tx.gemm_async(tmem[0:128, 256:384], A_smem[1:2, :, :], B_smem[2:3, :, :], dispatch="tcgen05")
        T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)   # caller signals completion
    T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
    # ... tcgen05.fence.after_thread_sync(); read tmem back via tcgen05.ld; dealloc ...

Algorithm
---------

**1. Encode shared matrix descriptors.** Each shared operand gets a 64-bit
descriptor (leading-dim offset ``ldo``, stride-dim offset ``sdo``, swizzle mode)
naming its tile to the tensor core:

.. code-block:: python

    T.ptx.tcgen05.encode_matrix_descriptor(descA.data, A_smem.ptr_to([0]), ldo, sdo, swizzle)
    T.ptx.tcgen05.encode_matrix_descriptor(descB.data, B_smem.ptr_to([0]), ldo, sdo, swizzle)

**2. Choose the MMA tile.** ``M_mma × N_mma`` are chosen to tile ``M``/``N`` (with
``MMA_K`` set by dtype: 16 f16/bf16, 32 fp8, 64 fp4); a compile-time *instruction
descriptor* packs the shape and dtypes.

**3. Issue the async MMA** in an unrolled ``(mi, ni, ki)`` nest, accumulating into
the tmem accumulator (``enable_input_d`` turns accumulation on for ``ki > 0``):

.. code-block:: python

    T.ptx.tcgen05.mma(
        "float32", A_type, B_type,
        T.cuda.get_tmem_addr(tmem_addr, mi * M_mma, tmem_col),       # C in tmem
        smem_desc_add_16B_offset(descA, a_off), descB_val, descI,    # A / B descriptors
        use_a_tmem=a_is_tmem, cta_group=cta_group,
        enable_input_d=(ki != 0),                                    # accumulate over K
    )

For **block-scaled** fp8/fp4 the emit becomes ``T.ptx.tcgen05.mma.block_scale(...)``
with two extra tmem addresses — ``SFA`` / ``SFB`` — and the scale-factor dtypes; the
instruction descriptor is encoded at runtime. As with the other async ops, the
dispatch emits **no** completion — the caller's ``tcgen05.commit`` + mbarrier wait
close it.

Generated TIRx IR
-----------------

For the ``128×64 × 64×128`` fp16 tile (swizzle mode 3):

.. code-block:: python

    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descA[0]), T.address_of(A_smem[0]), 64, 64, 3)
    T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(descB[0]), T.address_of(B_smem[0]), 64, 64, 3)
    T.ptx.tcgen05.mma("float32", "float16", "float16",
                      T.cuda.get_tmem_addr(tmem_addr[0], mi * 128, 256 + ni * 128), ...)

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
       ``M ∈ {128, 256}`` and half the per-CTA N
   * - M / N / K extents
     - set the ``(mi, ni, ki)`` unrolled loop counts; K iterations accumulate into
       the same tmem accumulator
   * - shared swizzle
     - sets the ``swizzle`` mode + ``ldo``/``sdo`` in the matrix descriptors
