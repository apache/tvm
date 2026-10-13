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

div / rcp / sqrt
================

Select accuracy, rounding and flush-to-zero behavior in the PTX opcode.
These choices change arithmetic semantics; the tile layer does not replace
a division with multiplication by an approximate reciprocal.

.. list-table::
   :header-rows: 1

   * - Example
     - Operand order
     - Atom
   * - ``div.rn.f32``
     - ``dst, numerator, denominator``
     - One float32 per operand
   * - ``rcp.approx.ftz.f32``
     - ``dst, src``
     - One float32 per operand
   * - ``sqrt.rn.f32``
     - ``dst, src``
     - One float32 per operand

Tensor operands end in ``(1,)``, preceded by execution-group and repeat
axes. Scalar read-only inputs broadcast over those repeats. A constant
numerator is allowed; its position is not exchanged with the denominator.

Example: explicit division followed by square root
--------------------------------------------------

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def root_ratio(A: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A)
        T.cuda.tile.apply(T.ptx["div.rn.f32"], r, r, T.float32(4))
        T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], r, r)
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, r)

For input ``[[4],[16],[36],[64]]``, the result is ``[[1],[2],[3],[4]]``.
The arithmetic expansion contains four ``div.rn.f32`` and four
``sqrt.rn.f32`` calls. Constant folding or instruction simplification later
in the compiler may optimize them; no tile-layout heuristic chooses a form.

To use an approximate reciprocal explicitly, replace the arithmetic with::

    T.cuda.tile.apply(T.ptx["rcp.approx.ftz.f32"], r, r)

This is a distinct numerical contract. A half-precision division composed
from conversion and scalar arithmetic belongs in :doc:`ewise` unless a
registered native form expresses exactly the requested operation.

All tensor storage is local to the executing thread. Views may select gapped
registers, but cannot move values between lanes. No memory staging or barrier
is inserted. Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
