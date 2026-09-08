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
``fma``, …) when **all buffer operands use ``local`` scope**. Scalar inputs are
also accepted where the operation's authoring API permits them. Like the copy
:doc:`../copy/reg` ``vec_auto`` path the partition is *induced* by the operands'
local-buffer layout — the thread axes are dropped, leaving each thread its
private bundle — and the op is applied to every element in that bundle. The
variant name describes this local-buffer path; final register allocation
remains a CUDA compiler decision. Source:
``python/tvm/backend/cuda/tile_primitive/elementwise/reg.py``.

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
            if br.buffer.scope() != "local":               # every buffer operand local
                return False, f"operand scope {br.buffer.scope()} != local"
            if br.buffer.layout is None: ...
        # + spec.check_extras (dtype rules), pick_anchor + _validate_anchor_layout,
        #   _validate_scope_level_anchor, NumPy-style shape broadcast checks,
        #   and agreement of operand thread/local/replica layout signatures

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope / priority
     - ``cuda``; ``thread`` / ``warp`` / ``warpgroup`` / ``cta`` (all active);
       priority ``10``
   * - operands
     - **every buffer operand** in ``local``; scalar sources are allowed by
       ``fill``, the binary ops, and ``fma``
   * - op
     - any CUDA elementwise ``OpSpec`` listed on the parent page (unary / binary /
       ``fma``); ``spec.check_extras`` validates the dtype combo
   * - local-buffer layout
     - the anchor layout must validate, and its thread axis must match the
       complete scope chain (it induces the partition); all operands must agree
       on their thread, per-thread storage, and replica signatures
   * - shapes
     - every input region is NumPy-style right-aligned broadcast-compatible with
       the destination region

Demonstration program
----------------------

A warp takes the elementwise ``sqrt`` of a ``32×8`` ``float32`` local tile
(local layout ``S[(32,8):(1@laneid,1)]`` — lane ``i`` owns row ``i``):

.. code-block:: python

    from tvm.tirx.layout import S, TileLayout, laneid

    r_layout = TileLayout(S[(32, 8) : (1 @ laneid, 1)]); fs = (slice(0, 32), slice(0, 8))

    @Tx.prim_func
    def k(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (32, 8), "float32"); B = Tx.match_buffer(B_ptr, (32, 8), "float32")
        Tx.device_entry(); Tx.cta_id([1]); Tx.lane_id([32]); tid = Tx.thread_id([32])
        A_smem = Tx.alloc_buffer((32, 8), "float32", scope="shared", layout=TileLayout(S[(32, 8)]))
        Tx.tile.warp.copy(A_smem[fs], A[fs]); Tx.cuda.cta_sync()
        R = Tx.alloc_buffer((32, 8), "float32", scope="local", layout=r_layout)
        Tx.tile.warp.copy(R[fs], A_smem[fs])
        Tx.tile.warp.sqrt(R[fs], R[fs])          # elementwise reg dispatch
        Tx.tile.warp.copy(A_smem[fs], R[fs]); Tx.cuda.cta_sync()
        Tx.tile.warp.copy(B[fs], A_smem[fs])

Algorithm
---------

**1. Parse and check.** ``spec.parse`` builds the op plan; the predicate confirms
every buffer operand is local, validates NumPy-style broadcasting, checks
the anchor against the complete scope chain, and requires compatible
thread/local/replica layout signatures across operands.

**2. Induce the partition** from the anchor's thread axis (``laneid`` here): drop
the thread iters, leaving each thread its private bundle (8 elements per lane).

**3. Apply the op to the per-thread bundle.** Registered packed implementations
are tried widest-first after every operand proves a matching contiguous tail;
otherwise the lowering uses the scalar operation in a per-thread loop.

Generated TIRx IR
-----------------

.. code-block:: python

    buffer[f] = Tx.sqrt(buffer_1[f])      # over each element f in the lane's bundle

Generated CUDA
--------------

.. code-block:: c++

    r_local_ptr[f_2] = sqrtf(r_local_ptr[f_2]);   // per-thread local element

(Verified on ``sm_100a`` — the result equals ``sqrt(A)``.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - op
     - unary → ``sqrtf`` / ``expf`` / … per element; binary → ``a + b`` per
       element; ``fma`` → ``a * b + c``
   * - dtype
     - selects the scalar operation and, where registered, a packed form such as
       ``f32x2`` or a two-element cast
   * - local-buffer layout
     - the anchor's thread axis sets the partition; a wider per-lane bundle means a
       longer loop
