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

elementwise → smem
==================

The ``smem`` variant lowers an elementwise op (``sqrt``, ``exp``, ``add``,
``fma``, …) when **all buffer operands are in shared memory**. Scalar inputs are
also accepted where the operation's authoring API permits them. Like the copy
:doc:`../copy/gmem_smem` ``vec_auto`` path it *synthesizes* a
``[outer, threads, vec]``
partition from the execution scope, then applies the op to each (vectorized)
element. Source:
``python/tvm/backend/cuda/tile_primitive/elementwise/smem.py``.

What it accepts
---------------

``is_smem_ewise(spec)`` builds the predicate:

.. code-block:: python

    def check(op_call, sctx):
        if not sctx.is_target("cuda"): return False, "non-cuda target"
        if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"): ...
        ok, reason = _all_threads_active(sctx)              # full scope
        plan, msg = spec.parse(op_call)                     # parse the op's operands
        for br in buffer_regions(plan):
            if not br.buffer.scope().startswith("shared"):  # every buffer operand shared*
                return False, f"operand scope {br.buffer.scope()} != shared*"
            if br.buffer.layout is None: ...
        # + spec.check_extras (dtype rules) and anchor-layout validation

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda``; ``thread`` / ``warp`` / ``warpgroup`` / ``cta`` (all active);
       priority ``10``
   * - operands
     - **every buffer operand** (including the output) in ``shared*``; scalar
       sources are allowed by ``fill``, the binary ops, and ``fma``
   * - op
     - any CUDA elementwise ``OpSpec`` listed on the parent page (unary
       ``sqrt``/``exp``/``zero``…, binary ``add``/``mul``…, ``fma``);
       ``spec.check_extras`` validates the dtype combo
   * - layout
     - operands have layouts. The partition is synthesized from the scope's
       thread count; dtype, logical innermost region extents, and elements per
       thread bound the scheduling chunk width

Demonstration program
----------------------

A CTA takes the elementwise ``sqrt`` of a ``32×32`` ``float32`` shared tile
(adapted from ``test_unary.py`` — here a 256-thread CTA, so the partition is one
round):

.. code-block:: python

    s_layout = TileLayout(S[(32, 32)]); full = (slice(0, 32), slice(0, 32))

    @Tx.prim_func
    def unary_op(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (32, 32), "float32", layout=s_layout)
        Tx.device_entry(); Tx.cta_id([1]); Tx.warp_id([8]); Tx.lane_id([32]); Tx.thread_id([256])
        A_smem = Tx.alloc_buffer((32, 32), "float32", scope="shared", layout=s_layout)
        Tx.tile.cta.copy(A_smem[full], A[full])
        Tx.tile.cta.sqrt(A_smem[full], A_smem[full])   # elementwise smem dispatch
        Tx.tile.cta.copy(A[full], A_smem[full])

Algorithm
---------

**1. Parse the op and check operands.** ``spec.parse`` turns the call into a plan
(inputs, output, the op); the predicate confirms every buffer operand is shared.

**2. Synthesize the partition** from the scope's **thread count** (as
:doc:`../copy/gmem_smem` does): split the region into ``[outer, threads, vec]``.
The candidate width must divide the elements per thread and every operand's
logical innermost region extent. ``_max_layout_vec`` does not inspect physical
layout strides when choosing this width. For the dense identity layout in this
example, ``32×32 = 1024`` ``float32`` over 256 threads gives ``vec = 4`` and
``outer = 1``.

**3. Apply the op per element.** Instead of a copy, each (thread, round) reads its
``vec`` elements, applies the op, and writes back — vectorized:

Generated TIRx IR
-----------------

.. code-block:: python

    for f in Tx.serial(1):                            # outer = 1
        for vec in Tx.vectorized(4):
            A_smem[tid * 4 + vec] = Tx.sqrt(A_smem[tid * 4 + vec])

Generated CUDA
--------------

The ``vec = 4`` element bundle becomes a ``float4`` and the op is applied per
component:

.. code-block:: c++

    float4 v_ = *(float4*)(&A_smem_ptr[tid * 4]);
    __1.x = sqrtf(v_.x);  __1.y = sqrtf(v_.y);
    __1.z = sqrtf(v_.z);  __1.w = sqrtf(v_.w);

(Verified on ``sm_100a`` — the tile equals ``sqrt(A)``.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - op
     - unary → ``sqrtf`` / ``expf`` / … per component; binary → the two inputs
       combined (``a + b``); ``fma`` → ``a * b + c``
   * - dtype
     - bounds the candidate width; elements per thread and logical innermost
       region extents can reduce it, changing the round count.  The current
       width selection does not inspect physical layout contiguity
   * - scope
     - sets the thread axis and count, hence the synthesized partition
