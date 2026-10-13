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

add / sub / mul
===============

The opcode chooses the numeric type, rounding, saturation and packing mode.
The operand order is ``dst, lhs, rhs``. For a scalar instruction, each tensor
ends in a singleton register-group axis; for example ``(8,1)`` at thread
scope means eight scalar instructions. A read-only scalar expression may
replace an input tensor and broadcasts over all eight repeats.

Native packed arithmetic
-------------------------

Packed arithmetic still has one register **carrier** per operand:

.. list-table::
   :header-rows: 1

   * - Example opcode
     - Carrier dtype
     - Values per carrier
   * - ``add.rn.f32``
     - float32
     - One float32
   * - ``add.rn.f16x2``
     - uint32 (raw bits)
     - Two float16
   * - ``mul.rn.bf16x2``
     - uint32 (raw bits)
     - Two bfloat16
   * - ``add.rn.f32x2``
     - uint64 (raw bits)
     - Two float32

Not every qualifier is valid for every type or architecture. PTX form
validation checks the spelling and the compiler checks the target. Packing
is a caller decision, not the consequence of an even tensor extent.

Example: eight packed half additions
------------------------------------

This kernel accepts packed input words and leaves the result packed:

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def packed_add(
        A: T.Tensor((8, 1), "uint32"), B: T.Tensor((8, 1), "uint32"), Out: T.Tensor((8, 1), "uint32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = T.alloc_local((8, 1), "uint32", layout="default")
        b = T.alloc_local((8, 1), "uint32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.b32"], a, A)
        T.cuda.tile.apply(T.ptx["ld.global.b32"], b, B)
        T.cuda.tile.apply(T.ptx["add.rn.f16x2"], a, a, b)
        T.cuda.tile.apply(T.ptx["st.global.b32"], Out, a)

.. code-block:: python

    import numpy as np
    import tvm

    x = np.arange(16, dtype="float16").reshape(8, 2)
    y = np.full((8, 2), 2, dtype="float16")
    compiled = tvm.tirx.compile(packed_add, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros((8, 1), dtype="uint32"), tvm.cuda())
    compiled(
        tvm.runtime.tensor(x.view("uint32"), tvm.cuda()),
        tvm.runtime.tensor(y.view("uint32"), tvm.cuda()),
        out,
    )
    np.testing.assert_array_equal(out.numpy().view("float16"), x + y)

The arithmetic expansion is eight ``add.rn.f16x2`` instructions, processing
sixteen half values. It does not insert a pack/unpack conversion. A scalar
constant passed to a packed opcode must already represent the whole packed
carrier; a numeric uint32 constant ``2`` is not the encoding of two half 2s.

Subtraction and multiplication
------------------------------

``sub`` preserves the written operand order, including a scalar on the left.
``mul`` uses the selected floating-point form or the integer ``lo``/``hi``/
``wide`` form and its declared result type. Different result widths require
the matching destination dtype, not a silent reinterpretation.

Tensor operands share repeat extents and must have the correct thread owner.
An input broadcast is a scalar or an explicit zero-stride view. Output
storage cannot overlap across repeats. ``pred`` and ``preserve_dst`` follow
the register instruction contract described in :doc:`mov`.

For exp/SiLU plus scale/bias, define the scalar expression in :doc:`ewise`.
Shared-memory inputs require explicit loads and stores around the arithmetic.
Implementation: ``patterns/register.py`` under ``python/tvm/backend/cuda/tile``.
