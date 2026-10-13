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

cvt
===

``cvt`` performs a numeric conversion selected by the opcode. A dtype view
only reinterprets bits; it is used here to expose the native carrier layout,
not to perform the conversion itself. Destination and source widths, operand
group lengths, rounding, saturation and packing come from the PTX form.

Scalar conversion uses tensor suffix ``(1,)`` on both sides. Packed
``cvt.rn.f16x2.f32`` has three operands: one b32 destination, a float32 source
for its **high** halfword, and a float32 source for its **low** halfword.
All three have suffix ``(1,)``. The pair is not an inferred vectorization of
a scalar cast.

Example: eight float32 values to four packed half pairs
-------------------------------------------------------

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def narrow_pairs(A: T.Tensor((4, 2), "float32"), Out: T.Tensor((4, 1), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = T.alloc_local((4, 2), "float32", layout="default")
        d = T.alloc_local((4, 1), "uint32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], a, A)
        T.cuda.tile.apply(T.ptx["cvt.rn.f16x2.f32"], d, a[:, 1:2], a[:, 0:1])
        T.cuda.tile.apply(T.ptx["st.global.b32"], Out, d)

For each input row ``[x0,x1]``, the output word contains converted ``x0``
in its low half and ``x1`` in its high half. Reversing the two source operands
would exchange the output elements. The expansion emits four packed cvt
instructions, one per row.

.. code-block:: python

    import numpy as np
    import tvm

    x = np.linspace(-2, 2, 8, dtype="float32").reshape(4, 2)
    compiled = tvm.tirx.compile(narrow_pairs, target="cuda", pipeline="tirx")
    out = tvm.runtime.tensor(np.zeros((4, 1), dtype="uint32"), tvm.cuda())
    compiled(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy().view("float16"), x.astype("float16"))

The inverse operation is two scalar ``cvt.f32.f16`` instructions per packed
word; there is no corresponding ``cvt.f32x2.f16x2`` spelling in this API.
Expose the packed source with a uint16 view, add a singleton register-group
axis per half, and apply the scalar opcode. For a composite conversion
expression, including explicit intermediate rounding, use :doc:`ewise`.

Register ownership must agree on both sides even when their storage widths
differ. A wrong carrier width, missing group axis, or overlapping destination
is rejected. There is no implicit staging through shared memory.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
