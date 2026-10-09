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

Tensor Instructions
===================

Tensor instructions expose a backend instruction over tensors with layouts::

    from tvm.script import tirx as T

    T.cuda.tile.ld(registers, global_tile, scope="warpgroup")
    T.cuda.tile.st(shared_tile, registers, scope="warpgroup")
    T.cuda.tile.tcgen05.mma(accumulator, shared_a, shared_b)

They construct ordinary, opaque, void ``tvm.ir.Call`` expressions. A statement
in TVMScript becomes one ``Evaluate(Call(...))``. ``T.ptx`` and ``T.nki``
continue to expose raw instructions over scalar registers and addresses.

Each tensor instruction has a fixed operand list and typed static attributes.
Tensors, regions, pointers, predicates, coordinates, runtime cache policies,
accumulation controls, and optional workspace tensors are **Call arguments**.
They participate in normal free-variable analysis, substitution, serialization,
and traversal. Static qualifiers such as instruction shape, cache hint, and
execution scope live in reflected attribute types.

``scope="thread"`` is the default. Use ``"warp"``, ``"warpgroup"``, ``"cta"``,
or ``"cluster"`` only where the selected instruction supports that scope.
Layouts and the active thread set must satisfy the instruction's contract.
There is no generic ``copy``/``gemm`` operation or priority dispatch, and no
``config``, ``workspace``, or ``dispatch`` keyword bag.

CUDA instructions
-----------------

* ``ld`` and ``st`` transfer between local registers and global/shared memory.
  Their register layout determines the participating threads. ``vec_bits``
  selects an explicit 16-, 32-, 64-, 128-, or 256-bit thread transfer.
  Fixed-width global loads accept ``cache``, ``l1_evict``, ``l2_evict``, and
  ``prefetch_size`` qualifiers.
* ``mov`` copies registers or fills a destination with a scalar. ``ldmatrix``
  and ``stmatrix`` explicitly request matrix memory instructions.
* ``cp_async(dst, src, predicate=-1)`` requests global-to-shared asynchronous
  copying. ``direct=True`` requests one thread-level instruction.
  ``fill_mode="zero"`` uses the instruction's source-size operand for zero
  filling; it must not become a predicate that skips the instruction.
* ``cp_async_bulk(dst, src, mbar, remote_cta_id)`` requests DSMEM copying.
  ``dst`` and ``mbar`` refer to the executing CTA's local shared addresses;
  lowering maps both addresses to the remote CTA before issuing the copy.
* ``cp_async_bulk_tensor_load(dst, src, mbar, cta_mask=0,
  mbarrier_addr=False, gather4=None, src_selector=None, cache_policy=None)``
  requests a TMA load. ``cp_async_bulk_tensor_store(dst, src,
  cache_policy=None)`` and ``cp_reduce_async_bulk_tensor(dst, src,
  cache_policy=None, reduce_op=...)`` request TMA stores and reductions.
* TMA ``descriptor_mode="auto"`` derives a legal descriptor and issue loops
  from layouts. ``descriptor_mode="explicit"`` follows the declared global
  tensor descriptor and issues once. Gather4 row coordinates and source
  selectors are explicit-mode operands. ``cache_hint`` is a static string;
  ``cache_policy`` is a separate runtime operand. They are mutually exclusive.
* ``tcgen05.cp`` copies shared memory to TMEM. ``tcgen05.ld/st`` transfer
  between TMEM and registers at ``scope="warpgroup"``; each warp issues its
  own instruction. The caller supplies waits, fences, and barriers.
* ``mma_sync`` performs register MMA. ``tcgen05.mma`` and
  ``tcgen05.mma_block_scale`` perform asynchronous TMEM accumulation. The
  latter requires both scale-factor tensor operands. Matrix descriptor and
  pointer preparation remain inside lowering.
* ``add/sub/mul/div/max``, ``cvt``, ``fma``, ``sqrt``, ``ex2``, and ``lg2``
  expose tensor arithmetic. Reciprocal is ``div(dst, 1, src)``. Packed
  arithmetic remains available when the operand layouts permit it.

The mathematical compositions ``compose.silu``, ``compose.exp``,
``compose.exp_with_scale_bias``, ``compose.exp2_with_scale_bias``,
``compose.log2_with_scale_bias``, and ``compose.sqrt_with_scale_bias`` are
also ordinary Calls. They expand during lowering and have the distinct
``tile_composite`` category.

Trainium instructions
---------------------

``T.trn.tile`` exposes ``load``, ``store``, ``tensor_copy``, ``matmul``,
``activation``, ``reciprocal``, ``memset``, ``tensortensor``, ``tensorscalar``,
``tensorreduce``, and ``affine_select``. The selected binary instruction must
match the operand layout: broadcasting along a free dimension can require
``tensorscalar`` even when the scalar is supplied by another tensor.

``scalar_tensor_scalar``, ``scalar_tensor_tensor``, ``tensorscalar_reduce``,
and ``activation_reduce`` represent native fused instructions. Static
``opcode``, ``op0``, ``op1``, ``reduce_op``, and ``axes`` qualifiers specify
their arithmetic. Scale, bias, and all tensor operands remain in ``args``.

Named optional ``acc_psum``, ``const_bias``, and ``partial_reduce`` operands
allow caller-provided workspace. The private allocation pass still allocates
and reuses omitted workspace for the instructions that need it. In IR, an
absent operand occupies its fixed slot as an empty ``Tuple``; TVMScript prints
it as ``None``.

Algorithms owned by the caller
------------------------------

CUDA reductions, layout permutation, synchronous global/shared copying, and
scalar fallback copying are no longer tensor primitives or compositions.
Write the loops, temporary registers, and synchronization at the call site.
For an in-place shared-memory permutation, finish all reads into registers,
synchronize the warp, write using the destination layout, and synchronize
again before reusing the storage.

Trainium ``tensor_copy`` rejects partition-axis transposition. A caller using
matrix multiplication for transposition allocates and initializes an identity
tensor, allocates PSUM, calls ``matmul(..., transpose_A=True)``, and optionally
calls ``tensor_copy`` to move the result back to SBUF. Copy lowering does not
allocate an identity or select a matrix multiplication algorithm.

See :doc:`api/tile` for the constructor signatures and
:doc:`arch/tile_dispatch` for the lowering boundary.
