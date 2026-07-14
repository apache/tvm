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
source and destination are register** (``local``) buffers. At thread scope it is a
plain sequential reduction over each thread's own elements; at warp scope, if the
destination layout carries a ``laneid`` replica, it also folds across lanes with a
``__shfl_xor`` tree. Source:
``python/tvm/backend/cuda/operator/tile_primitive/reduction/local.py``.

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
     - src **and** dst in ``local`` (registers), equal dtype
   * - exec scope
     - ``thread`` (always valid — pure thread-local); ``warp`` / ``warpgroup``
       require a valid (non-swizzled) ``TileLayout``; ``warp`` may additionally
       cross-lane reduce when ``thread_reduce`` and a ``laneid`` shard→replica
       pattern are present
   * - shape
     - dst spatial dims match src; reduced dims have ``local_extent == 1`` on dst

Demonstration program
----------------------

A single thread reduces a 4-element ``float32`` register vector to a scalar
(thread-wise path, from ``test_reduction.py``):

.. code-block:: python

    @T.prim_func
    def test_func(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [4], "float32", layout=TileLayout(S[(4,)]))
        B = T.match_buffer(B_ptr, [1], "float32", layout=TileLayout(S[(1,)]))
        T.device_entry(); T.cta_id([1]); T.thread_id([1])
        A_local = T.alloc_buffer([4], "float32", scope="local")
        B_local = T.alloc_buffer([1], "float32", scope="local")
        for i in T.serial(4): A_local[i] = A[i]
        Tx.sum(B_local, A_local, accum=False)     # reduction local dispatch
        B[0] = B_local[0]

(4 < 8 elements, so this stays on ``local`` rather than the
:doc:`sm100_packed` fast path.)

Algorithm
---------

**Thread-wise** (``_emit_reduction_local_thread_wise``): a spatial loop over the
output positions, each initialized to the op's identity (unless ``accum``), then a
reduction loop accumulating the source — no cross-thread communication:

.. code-block:: python

    for spa in range(spatial_len):
        if not accum: dst[spa] = identity
        for red in range(reduction_len):
            dst[spa] = op(dst[spa], src[spa, red])

**Warp-shuffle** (``_gen_warp_shuffle_reduce``): when the dst layout has a
``laneid`` replica, each lane first reduces its own elements, then
``T.cuda.warp_reduce`` folds across lanes — a ``__shfl_xor`` tree over the **full**
``0xFFFFFFFF`` mask. (This differs from :doc:`shared`, which uses explicit
``tvm_warp_shuffle_xor`` steps over ``__activemask()`` at the *group* width.)

Generated TIRx IR
-----------------

For the 4-element thread reduction:

.. code-block:: python

    for spa in range(1):
        for red in range(4):
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
     - ``thread`` → sequential; ``warp`` with a ``laneid`` replica → adds a
       ``__shfl_xor`` cross-lane tree
   * - axes / shape
     - set the spatial vs reduction loop extents
   * - accum
     - ``True`` reuses the old dst value instead of the identity
