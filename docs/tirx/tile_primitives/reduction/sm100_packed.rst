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

reduction → sm100_packed
========================

The ``sm100_packed`` variant is a **Blackwell-only fast path** (priority **20**, so
it pre-empts :doc:`local`) for a thread-scope reduction of a 1-D ``float32`` vector
of at least 8 elements to a scalar. It uses the SM100 packed math instructions —
``add.f32x2`` for ``sum``, ``max3.f32`` / ``min3.f32`` for ``max`` / ``min`` —
to fold two (or three) lanes of data per instruction. Source:
``python/tvm/backend/cuda/operator/tile_primitive/reduction/sm100_packed.py``.

What it accepts
---------------

All of the following must hold (else the dispatch declines and :doc:`local` runs):

.. code-block:: python

    @register_dispatch(op_name, "cuda", variant=variant_name, priority=20,
        when=[
            predicate("exec_scope", exec_scope_ok, expected_scopes=["thread"]),
            predicate("local_scope", _local_scope_match),       # src & dst local
            predicate("dst_len", _dst_len_ok, expected_len=1),  # reduce to scalar
            predicate("src_ndim", _src_ndim_ok, expected_ndim=1),
            predicate("dtype", _dtype_ok, expected_dtype="float32"),
            predicate("sm_version", sm_version_ok, min_version=100),
            predicate("reduction_len", _reduction_len_ok, min_len=8),
        ])

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / priority
     - ``cuda`` ``sm_100+`` (Blackwell); priority ``20`` (beats ``local``)
   * - exec scope
     - ``thread`` only
   * - operands
     - src & dst ``local``, both ``float32``; dst length ``1``; src **1-D** with
       ``≥ 8`` elements

Demonstration program
----------------------

A single thread sums a 32-element ``float32`` register vector on ``sm_100a`` (from
``test_reduction.py``):

.. code-block:: python

    @T.prim_func
    def test_func(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [32], "float32", layout=TileLayout(S[(32,)]))
        B = T.match_buffer(B_ptr, [1], "float32", layout=TileLayout(S[(1,)]))
        T.device_entry(); T.cta_id([1]); T.thread_id([1])
        A_local = T.alloc_buffer([32], "float32", scope="local")
        B_local = T.alloc_buffer([1], "float32", scope="local")
        for i in T.serial(32): A_local[i] = A[i]
        Tx.sum(B_local, A_local, accum=False)     # -> sm100_packed (len 32 >= 8, fp32, sm100)
        B[0] = B_local[0]

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})

Algorithm
---------

**sum (packed_add_sum).** Keep an 8-wide ``float32`` accumulator. Load the first 8
elements; for each further chunk of 8, pairwise-add it in with four ``add.f32x2``
(each adds two ``float2`` lanes at once); handle the remainder scalar; then collapse
the accumulator ``8 → 4 → 2 → 1`` with three more ``add.f32x2``:

.. code-block:: python

    # final tree (8 -> 4 -> 2 -> 1)
    T.ptx.add_f32x2(T.address_of(local_sum[0]),
                    T.cuda.make_float2(local_sum[0], local_sum[1]),
                    T.cuda.make_float2(local_sum[2], local_sum[3]), ftz=True)
    T.ptx.add_f32x2(T.address_of(local_sum[4]),
                    T.cuda.make_float2(local_sum[4], local_sum[5]),
                    T.cuda.make_float2(local_sum[6], local_sum[7]), ftz=True)
    T.ptx.add_f32x2(T.address_of(local_sum[0]),
                    T.cuda.make_float2(local_sum[0], local_sum[1]),
                    T.cuda.make_float2(local_sum[4], local_sum[5]), ftz=True)
    dst[...] = local_sum[0] + local_sum[1]

**max / min (3input_maxmin).** A 4-wide accumulator folded three-at-a-time with the
``max3.f32`` / ``min3.f32`` instructions.

Generated TIRx IR
-----------------

.. code-block:: python

    T.ptx.add_f32x2(T.address_of(local_sum[0]),
                    T.cuda.make_float2(local_sum[0], local_sum[1]),
                    T.cuda.make_float2(local_sum[2], local_sum[3]))   # ... the 8->4->2->1 tree

Generated CUDA
--------------

.. code-block:: c++

    // packed pairwise add: two float lanes per instruction
    "add.rn.ftz.f32x2 %0, %1, %2;"
    // call: tvm_builtin_ptx_add_f32x2_rn_ftz(&local_sum_ptr[0],
    //          tvm_builtin_make_float2(local_sum_ptr[0], local_sum_ptr[1]),
    //          tvm_builtin_make_float2(local_sum_ptr[2], local_sum_ptr[3]));

(Verified on ``sm_100a`` — ``B == sum(A)`` for a 32-element vector.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - input
     - effect
   * - op
     - ``sum`` → the ``add.f32x2`` packed tree; ``max`` / ``min`` → the
       ``max3.f32`` / ``min3.f32`` 3-input fold
   * - reduction length
     - the chunk-of-8 (sum) / chunk handling and the scalar remainder loop; must be
       ``≥ 8``
   * - accum
     - ``True`` folds the old dst value into the first accumulator slot
   * - anything outside the gate
     - non-fp32, 2-D src, dst length > 1, pre-Blackwell, or ``< 8`` elements → the
       dispatch declines and :doc:`local` handles it
