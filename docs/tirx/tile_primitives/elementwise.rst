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

Elementwise tensor instructions
===============================

``Tx.cuda.tile`` provides ``mov``, ``cvt``, ``add``, ``sub``, ``mul``, ``div``,
``max``, ``fma``, ``sqrt``, ``ex2``, and ``lg2``. The instruction's tensor layout
and execution scope determine how its scalar operation is repeated across the tile.
``mov(dst, 0)`` initializes a tile, and ``div(dst, 1.0, src)`` computes a reciprocal.

The mathematical compositions ``exp``, ``silu``, and the four
``*_with_scale_bias`` functions live in ``Tx.cuda.tile.compose`` and remain opaque
Calls until lowering. Their opcode category is ``tile_composite``.

All tensor operands of an elementwise instruction must be local or all shared.
The corresponding lowerer expands the same operation over the selected storage:

* :doc:`elementwise/reg` uses the register layout's per-thread ownership.
* :doc:`elementwise/smem` constructs a cooperative partition in shared memory.

CUDA reduction algorithms and layout permutation are caller-owned; they have no
replacement in ``compose``.

.. toctree::
   :maxdepth: 1

   elementwise/reg
   elementwise/smem
