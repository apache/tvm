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

abs / neg
=========

``abs`` and ``neg`` operate independently on local register values. Their
opcode selects the numeric type and applicable qualifiers. Operand order
is ``dst,src``; each scalar tensor operand ends in ``(1,)`` after the group
and repeat axes. An ordinary dtype view does not change the numeric type
of the requested operation.

Example: an explicit two-instruction scalar expression
------------------------------------------------------

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def negative_magnitude(A: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A)
        T.cuda.tile.apply(T.ptx["abs.f32"], r, r)
        T.cuda.tile.apply(T.ptx["neg.f32"], r, r)
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, r)

For finite nonzero inputs this computes ``-abs(A)``. The tile expansion has
four abs calls followed by four neg calls. This order comes from the two
explicit operations, not from a hidden unary-op combination. Later compiler
optimizations may simplify the expression. For a single caller-defined
scalar expression with no exact-instruction requirement, use :doc:`ewise`.

NaN, signed-zero, and integer boundary behavior belongs to the selected PTX
type/form, not to a generic Python ``abs`` reference. Destination storage may
be gapped but must be distinct across repeats. Register ownership cannot
change between source and destination. Predication follows :doc:`mov`.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
No temporary memory, numeric cast, barrier, or packed form is inserted.
