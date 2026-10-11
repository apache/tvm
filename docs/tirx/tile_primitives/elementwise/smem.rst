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

The shared-memory lowerer expands an elementwise instruction (``sqrt``, ``exp``, ``add``,
``fma``, …) when **all buffer operands are in shared memory**. Scalar inputs are
also accepted where the instruction permits them. It synthesizes an
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
   * - target / scope
     - ``cuda``; ``thread`` / ``warp`` / ``warpgroup`` / ``cta`` (all active)
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
---------------------

A CTA takes the elementwise ``sqrt`` of a ``32×32`` ``float32`` shared tile
(adapted from ``test_unary.py`` — here a 256-thread CTA, so the partition is one
round):

.. code-block:: python

    s_layout = TileLayout(S[(32, 32)])
    full = (slice(0, 32), slice(0, 32))


    @Tx.function
    def unary_op(A: Tx.Tensor((32, 32), "float32", layout=s_layout)):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        _index = Tx.cuda.block_idx("x")
        _index = Tx.cuda.warp_id()
        _lane = Tx.cuda.lane_id()
        tid = Tx.cuda.thread_idx("x")
        A_smem = Tx.alloc_tensor((32, 32), "float32", scope="shared", layout=s_layout)
        for k in Tx.serial(4):
            index = tid + k * 256
            value = A[index // 32, index % 32]
            A_smem[index // 32, index % 32] = value
        Tx.cuda.cta_sync()
        Tx.cuda.tile.sqrt(A_smem[full], A_smem[full], scope="cta")  # elementwise smem dispatch
        Tx.cuda.cta_sync()
        for k in Tx.serial(4):
            index = tid + k * 256
            value = A_smem[index // 32, index % 32]
            A[index // 32, index % 32] = value

Algorithm
---------

**1. Parse the op and check operands.** ``spec.parse`` turns the call into a plan
(inputs, output, the op); the predicate confirms every buffer operand is shared.

**2. Synthesize the partition** from the scope's **thread count** (as
the cooperative transfer planner does): split the region into ``[outer, threads, vec]``.
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
