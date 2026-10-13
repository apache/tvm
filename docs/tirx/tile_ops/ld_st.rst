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

ld / st
========

Use ``T.cuda.tile.apply`` to repeat a chosen memory instruction. The operand
order is always destination first: a load takes ``registers, memory`` and a
store takes ``memory, registers``. For example::

    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R, A)
    T.cuda.tile.apply(T.ptx["st.shared.v4.b32"], S, R)

Both calls transfer four 32-bit words per repeat. Global-to-shared copying
through registers consists of these two explicit calls and caller-allocated
register storage. To issue a direct asynchronous transfer, use :doc:`cp_async`.

Contract and instruction selection
----------------------------------

Operands have shape ``(*group,*repeat,word)``. The final extent is the vector
length in the opcode, or one for a scalar instruction. Both operands use
scalar elements with the opcode's storage bit width. Each memory atom must
be consecutive and aligned to the whole vector's byte size. Its start can
vary across repeats. Local register slots can be separated or permuted, but
each must belong to the issuing thread; loaded destinations cannot overlap.

The supported address-space spellings are ``global``, ``shared``,
``shared::cta``, and ``local``. The pointer side of ``ld.local``/``st.local``
is addressable local memory, distinct from the register operands. Legal
types, vector lengths, and qualifiers come from the selected PTX form.

Cache operators such as ``.nc`` and ``.cg`` and prefetch/eviction qualifiers
are part of the opcode. A form with a cache-policy operand takes
``cache_policy=...``. ``pred`` gates execution. A predicated load can use
``preserve_dst=True`` to keep an already initialized destination when the
predicate is false. Lowering does not reduce the width when alignment fails.

Example: a strided tensor and gapped registers
---------------------------------------------------------

The following kernel reads a float32 tensor whose logical axes are not in
physical order. It keeps that layout and prepares a view with four physically
adjacent words at the end. The destination deliberately has twice the storage
stride, so each vector writes separated local slots.

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.tirx.layout import S, TileLayout

    global_layout = TileLayout(S[(15, 2, 7, 2) : (4, 1, 60, 2)])
    register_layout = TileLayout(S[(15, 2, 7, 2) : (8, 2, 120, 4)])


    @T.function
    def strided_load(
        A: T.Tensor((15, 2, 7, 2), "float32", layout=global_layout, align=16),
        Out: T.Tensor((420,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        R = T.alloc_tensor(
            ty_args=[T.Tensor((15, 2, 7, 2), "float32", scope="local", layout=register_layout)]
        )
        av = A.rearrange("i b j d -> j i (d b)")
        rv = R.rearrange("i b j d -> j i (d b)")
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], rv, av)
        for j, i, d, b in T.grid(7, 15, 2, 2):
            Out[60 * j + 4 * i + 2 * d + b] = R[i, b, j, d]

Both views have shape ``(7,15,4)``. For one repeat ``(j,i)``, source word
``w`` lives at ``60*j+4*i+w`` and destination word ``w`` lives at local
offset ``120*j+8*i+2*w``. One instruction binds its four outputs directly to
these four slots. This emits **105 ld.global.v4.b32 instructions**, with no
intermediate tile or register-remapping copy. The final scalar stores are
ordinary caller code used to inspect the result.

Compile and check the complete kernel:

.. code-block:: python

    import numpy as np
    import tvm

    compiled = tvm.tirx.compile(strided_load, target="cuda", pipeline="tirx")
    x = np.arange(420, dtype="float32")
    a = tvm.runtime.tensor(x.reshape(15, 2, 7, 2), tvm.cuda())
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    compiled(a, out)
    np.testing.assert_array_equal(out.numpy(), x)

The raw load in the generated PTX has four output operands, regardless of
the local offsets selected by the view:

.. code-block:: text

    ld.global.v4.b32 {r0, r1, r2, r3}, [address];

Sharing the work across threads
-------------------------------

At CTA scope, prepend ``(warp,lane)``. With 128 threads,
``(4,32,3,4)`` means three four-word instructions per thread. A register
layout can assign the first two axes strides ``1 @ warpid`` and
``1 @ laneid``, and the last two ordinary local strides ``4,1``. Memory
views must expose the same group/repeat coordinates. These axes distribute
the operation; they are not searched for in an arbitrary layout.

Loads and stores do not insert a barrier. If another thread will consume
shared stores, synchronize at that dependency. To change ownership or
reorder shared data in place, finish the old reads before overwriting bytes.

Implementation: ``python/tvm/backend/cuda/tile/patterns/memory.py``.
See :doc:`../cuda_tile_patterns` for group, repeat, and bounds rules.
