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

reduction → local
=================

The ``local`` variant lowers a reduction (``sum`` / ``max`` / ``min``) when **both
source and destination are ``local`` buffers**. At thread scope it is a
plain sequential reduction over each thread's own elements. Warp scope has two
layout-driven paths: a specialized ``laneid`` shard-to-replica reduction, or a
general local-axis reduction that can optionally add cross-lane shuffle steps.
Source:
``python/tvm/backend/cuda/tile_primitive/reduction/local.py``.

What it accepts
---------------

.. code-block:: python

    @register_dispatch(op_name, "cuda", variant="local", priority=10, when=[
        predicate("storage_scope", _match_reduction_storage_scope, expected_scope=["local"]),
        predicate("local_valid", validate_reduction_local),
    ])

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / priority
     - ``cuda``; priority ``10``
   * - operand scope
     - src **and** dst in ``local``, equal dtype
   * - exec scope
     - ``thread`` (sequential and thread-local); ``warp`` / ``warpgroup``
       require valid non-swizzled ``TileLayout`` values. At warp scope a
       ``laneid`` shard→replica pattern automatically selects the specialized
       shuffle path; otherwise ``thread_reduce=True`` optionally adds shuffles
       to the general path. Warpgroup scope rejects ``thread_reduce=True``
   * - shape
     - axes must be in range. At thread scope, flattened dst size must equal the
       product of the source's non-reduced extents. Wide-scope view reductions
       instead require matching spatial layout dimensions; reduced dimensions
       have dst local extent 1

Demonstration program
----------------------

A single thread reduces a 4-element ``float32`` local vector to a scalar
(thread-wise path, from ``test_reduction.py``):

.. code-block:: python

    @Tx.prim_func
    def test_func(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, [4], "float32", layout=TileLayout(S[(4,)]))
        B = Tx.match_buffer(B_ptr, [1], "float32", layout=TileLayout(S[(1,)]))
        Tx.device_entry(); Tx.cta_id([1]); Tx.thread_id([1])
        A_local = Tx.alloc_buffer([4], "float32", scope="local")
        B_local = Tx.alloc_buffer([1], "float32", scope="local")
        for i in Tx.serial(4): A_local[i] = A[i]
        Tx.tile.sum(B_local, A_local, accum=False)     # reduction local dispatch
        B[0] = B_local[0]

(4 < 8 elements, so this stays on ``local`` rather than the
:doc:`sm100_packed` ``packed_add_sum`` / ``3input_maxmin`` fast paths.)

Algorithm
---------

**Thread-wise** (``_emit_reduction_local_thread_wise``): a spatial loop over the
output positions, each initialized to the op's identity (unless ``accum``), then a
reduction loop accumulating the source — no cross-thread communication:

.. code-block:: python

    for spa in Tx.serial(spatial_len):
        if not accum: dst[spa] = identity
        for red in Tx.serial(reduction_len):
            dst[spa] = op(dst[spa], src[spa, red])

**Specialized shard→replica shuffle** (``_gen_warp_shuffle_reduce``): when the
source has a full-span ``laneid`` shard and the destination has a power-of-two
``laneid`` replica, the implementation copies each lane's corresponding local
values and applies ``Tx.cuda.warp_reduce`` across the replica width. This path is
selected automatically, independently of ``thread_reduce``; it does not first
run the general local-axis loop. With ``accum=True``, the reduced value is then
combined with the old destination.

**General warp/warpgroup view path** (``_emit_reduction_local_view``): reduces
the source's local reduction axes into each destination position. At warp scope,
``thread_reduce=True`` additionally emits explicit
``tvm_warp_shuffle_xor`` steps using ``__activemask()``. Warpgroup scope supports
the local part only.

Generated TIRx IR
-----------------

For the 4-element thread reduction:

.. code-block:: python

    for spa in Tx.serial(1):
        dst[...] = Tx.float32(0)
        for red in Tx.serial(4):
            dst[...] = dst[...] + src[...]        # op = sum

Generated CUDA
--------------

.. code-block:: c++

    for (int red = 0; red < 4; ++red)
      B_local_ptr[0] = B_local_ptr[0] + A_local_ptr[red];

(Verified on ``sm_100a`` — ``B == sum(A)``.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - op
     - ``sum`` → ``+``, ``max`` → ``max``, ``min`` → ``min`` (and the identity)
   * - exec scope
     - ``thread`` → sequential; matching warp shard→replica layout → specialized
       ``warp_reduce``; otherwise warp/warpgroup use the general local view path,
       with optional warp shuffles only when ``thread_reduce=True``
   * - axes / shape
     - set the spatial vs reduction loop extents
   * - accum
     - ``True`` reuses the old dst value instead of the identity
