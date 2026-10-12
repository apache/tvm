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

Explicit CUDA tile patterns (prototype)
=======================================

Choose an instruction, prepare zero-copy operand views, then apply it::

    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, memory,
                      scope="thread")

The instruction and modifiers determine the atom contract. The caller supplies
regions whose trailing dimensions have that contract; equal leading dimensions
are repeat axes. Lowering checks the contract and emits one specified PTX
instruction per repeat coordinate, in row-major repeat order. The prototype
unrolls static repeats. It does not infer a different opcode, width, matrix
count, transpose mode, or coordinate permutation.

This is an experimental path alongside the existing CUDA tile operations.
Only the two spellings below are implemented. Other qualifiers or instructions
produce an unsupported-pattern error, even if the PTX dialect supports them.

128-bit global load
-------------------

``ld.global.v4.b32``, ``scope="thread"``:

* Both regions have shape ``(*repeat, 4)`` and scalar 32-bit elements.
* The source is global memory. For each repeat, its four words have addresses
  ``base + 0, base + 1, base + 2, base + 3`` (in words); the base is 16B aligned.
* The destination is four distinct registers of the executing thread. Their
  storage offsets may be separated or permuted. No thread axes are accepted.
* The operation copies raw bits; it performs no numeric conversion.

The earlier example keeps its original float32 tensor layout::

    original = TileLayout(S[(15, 2, 7, 2):(4, 1, 60, 2)])
    A_std = A.rearrange("i b j d -> j i (d b)")
    R_std = R.rearrange("i b j d -> j i (d b)")
    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R_std, A_std,
                      scope="thread")

``A_std`` has shape ``(7,15,4)`` and addresses ``60*j + 4*i + word``.
The repeat axes are ``j`` and ``i``: this emits 7*15 = 105 vector loads.
The destination may instead use original strides ``(8,2,120,4)``; each
instruction then writes slots separated by two, without a remap copy.

ldmatrix with interleaved rows and separated registers
-----------------------------------------------------

``ldmatrix.sync.aligned.m8n8.x2.shared.b16``, ``scope="warp"``:

* Both regions have shape ``(*repeat, 2, 8, 4)``. The suffix is
  ``(matrix, row, word)``.
* Each word is a 32-bit container holding two adjacent 16-bit matrix values.
  The prototype accepts float32/int32/uint32 containers as raw bits.
* In shared memory, each ``(matrix,row)`` is four consecutive words with a
  16B-aligned start. Different rows can be separated; they need not form a
  contiguous 8x8 allocation.
* Destination element ``(matrix,row,word)`` belongs to lane ``4*row+word``.
  Each lane receives two words; the register slots are selected by its layout.
* All 32 lanes must execute the instruction. Shared initialization and any
  preceding synchronization belong to the caller.

Use the original shared shape ``(16,4,2)`` and strides ``(8,1,128)``::

    S_std = SM.rearrange("(r m) w k -> k m r w", r=8, m=2)
    registers = TileLayout(S[(2,2,8,4):(8,2,4 @ laneid,1 @ laneid)])
    # R is declared over 12 caller-allocated words per thread, using registers.
    T.cuda.tile.apply(T.ptx["ldmatrix.sync.aligned.m8n8.x2.shared.b16"],
                      R, S_std, scope="warp")

``S_std[k,m,r,w] = SM[2*r+m,w,k]``. Its strides are ``(128,8,16,1)``:
first matrix = even original rows; second matrix = odd original rows.
For each ``k``, lanes 0..7 provide the eight matrix-0 row addresses, lanes
8..15 provide the eight matrix-1 row addresses, and lanes 16..31 duplicate
those valid addresses. These address providers are distinct from result owners.

For lane ``l``, the complete output mapping is::

    R_raw[0]  = SM[2*(l//4),   l%4, 0]
    R_raw[2]  = SM[2*(l//4)+1, l%4, 0]
    R_raw[8]  = SM[2*(l//4),   l%4, 1]
    R_raw[10] = SM[2*(l//4)+1, l%4, 1]

There are two repeat coordinates, ``k=0,1``, so this is exactly two ``.x2``
instructions. Slots 1,3,4,5,6,7,9,11 remain untouched. Lowering binds a
physical view of the caller's registers and passes these slots directly as
PTX output operands; it allocates no intermediate tile.

Implementation and boundaries
-----------------------------

``python/tvm/backend/cuda/tile/patterns/`` contains:

* ``__init__.py``: normalized instruction name -> pattern, and the ordinary
  ``tirx.cuda.tile.apply`` Call constructor.
* ``common.py``: region starts, symbolic repeat coordinates, alignment and
  storage checks, and unrolled repeat loops.
* ``ld_global.py`` / ``ldmatrix.py``: fixed coordinate contract, validation,
  and one-instruction emission. They import no old CUDA dispatch algorithm.

The IR stores a string instruction operand and tensor regions, with the scope
in existing scope attrs. It survives JSON serialization and TVMScript printing
and reparsing. ``instruction_name`` uses the existing PTX table; tile code does
not inspect the chain's private candidate representation.

The prototype requires positive static region extents and TileLayouts without
replicas or swizzles. Source alignment must be provable from the tensor's
alignment, layout, element offset, and region start. Destination memory strides
must pass a conservative nonoverlap check: after sorting by stride, each stride
covers all preceding dimensions. Exotic injective layouts outside this test
are not yet supported. There is no claim of recognizing every physically
equivalent coordinate expression; the caller prepares the documented views.

The prototype also fixes allocation of TileLayouts with holes and no thread
axes: allocation now covers the memory-axis span. The global gapped-register
example needs 839 slots for 420 values, and the shared example needs 252 words
for 128 values. Previously those allocations used the logical element count.

Executable examples and validation
----------------------------------

The complete kernels, allocation, shared staging, synchronization, and host
checks are in ``tests/python/tirx/tile/cuda/test_explicit_patterns.py``::

    python -m pytest tests/python/tirx/tile/cuda/test_explicit_patterns.py -q

Tests inspect compiled PTX and execute both original layouts, plus the gapped
register global-load variant. The matrix check compares all 12 words in every
lane, including untouched sentinels. Negative cases cover incorrect word order,
misalignment, mismatched repeats, wrong lane ownership, incomplete warps, and
unsupported instructions. A separate GPU case checks nonzero source and
destination region starts together with tensor element offsets.
This is correctness and instruction-shape validation,
not a performance benchmark; the 420-value thread example can spill registers.
