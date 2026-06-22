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

elementwise → reg
=================

The ``reg`` variant lowers an elementwise op (``sqrt``, ``exp``, ``add``,
``fma``, …) when **all operands are register** (``local``) buffers. Like the copy
:doc:`../copy/reg` variant the partition is *induced* by the operands' register
layout — the thread axes are dropped, leaving each thread its private bundle — and
the op is applied to every register in that bundle. Source:
``python/tvm/backend/cuda/operator/tile_primitive/elementwise/reg.py``.

What it accepts
---------------

``is_reg_ewise(spec)`` builds the predicate:

.. code-block:: python

    def check(op_call, sctx):
        if not sctx.is_target("cuda"): return False, "non-cuda target"
        if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"): ...
        ok, reason = _all_threads_active(sctx)
        plan, msg = spec.parse(op_call)
        for br in buffer_regions(plan):
            if br.buffer.scope() != "local":               # every operand register
                return False, f"operand scope {br.buffer.scope()} != local"
            if br.buffer.layout is None: ...
        # + spec.check_extras (dtype rules), pick_anchor + _validate_anchor_layout,
        #   _validate_scope_level_anchor

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda``; ``thread`` / ``warp`` / ``warpgroup`` / ``cta`` (all active);
       priority ``10``
   * - operands
     - **every** operand in ``local`` (registers)
   * - op
     - any registry op (unary / binary / ``fma``); ``spec.check_extras`` validates
       the dtype combo
   * - register layout
     - the anchor register layout must validate, and its thread axis must match the
       scope (it induces the partition)

Demonstration program
----------------------

A warp takes the elementwise ``sqrt`` of a ``32×8`` ``float32`` register tile
(register layout ``S[(32,8):(1@laneid,1)]`` — lane ``i`` owns row ``i``):

.. code-block:: python

    from tvm.tirx.layout import S, TileLayout, laneid

    r_layout = TileLayout(S[(32, 8) : (1 @ laneid, 1)]); fs = (slice(0, 32), slice(0, 8))

    @T.prim_func
    def k(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32, 8), "float32"); B = T.match_buffer(B_ptr, (32, 8), "float32")
        T.device_entry(); T.cta_id([1]); T.lane_id([32]); tid = T.thread_id([32])
        A_smem = T.alloc_buffer((32, 8), "float32", scope="shared", layout=TileLayout(S[(32, 8)]))
        Tx.warp.copy(A_smem[fs], A[fs]); T.cuda.cta_sync()
        R = T.alloc_buffer((32, 8), "float32", scope="local", layout=r_layout)
        Tx.warp.copy(R[fs], A_smem[fs])
        Tx.warp.sqrt(R[fs], R[fs])          # elementwise reg dispatch
        Tx.warp.copy(A_smem[fs], R[fs]); T.cuda.cta_sync()
        Tx.warp.copy(B[fs], A_smem[fs])

Algorithm
---------

**1. Parse and check.** ``spec.parse`` builds the op plan; the predicate confirms
every operand is a register buffer and the anchor register layout is valid.

**2. Induce the partition** from the anchor's thread axis (``laneid`` here): drop
the thread iters, leaving each thread its private bundle (8 elements per lane).

**3. Apply the op per register** — a per-thread loop over the bundle:

Generated TIRx IR
-----------------

.. code-block:: python

    buffer[f] = T.sqrt(buffer_1[f])      # over each register f in the lane's bundle

Generated CUDA
--------------

.. code-block:: c++

    r_local_ptr[f_2] = sqrtf(r_local_ptr[f_2]);   // per-register, private to the lane

(Verified on ``sm_100a`` — the result equals ``sqrt(A)``.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - op
     - unary → ``sqrtf`` / ``expf`` / … per register; binary → ``a + b`` per
       register; ``fma`` → ``a * b + c``
   * - dtype
     - the register element type (``sqrtf`` vs ``hsqrt`` etc.); the bundle size is
       the per-lane element count
   * - register layout
     - the anchor's thread axis sets the partition; a wider per-lane bundle means a
       longer loop
