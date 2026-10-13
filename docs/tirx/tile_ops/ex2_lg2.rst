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

ex2 / lg2
=========

These PTX instructions calculate base-two exponential and logarithm. They
are not natural exp/log. Select the explicit approximation and subnormal
policy, for example ``ex2.approx.ftz.f32`` or ``lg2.approx.ftz.f32``.
The tile layer forwards this spelling without selecting a replacement.

Both tensor operands use local registers and end in ``(1,)``; each preceding
repeat selects another scalar input/output. Output repeats cannot alias.
A scalar read-only input broadcasts over all repeats.

Example: four base-two exponentials
-----------------------------------

.. code-block:: python

    from tvm.script import tirx as T


    @T.function
    def exp2_native(A: T.Tensor((4, 1), "float32"), Out: T.Tensor((4, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4, 1), "float32", layout="default")
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A)
        T.cuda.tile.apply(T.ptx["ex2.approx.ftz.f32"], r, r)
        T.cuda.tile.apply(T.ptx["st.global.f32"], Out, r)

Inputs ``[-1,0,1,2]`` give the power-of-two reference ``[0.5,1,2,4]``.
The arithmetic expansion contains four ex2 instructions. For general inputs,
test with tolerances appropriate to the approximate opcode; ``.ftz`` is part
of the chosen handling of subnormal values.

Use ``lg2.approx.ftz.f32`` with the same operand pattern for base-two
logarithms. If the operation is natural exp, SiLU, or an exp/scale/bias
combination, write that scalar expression through :doc:`ewise`. The scalar
compiler then owns its expansion; the call is not documented as one ex2.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
