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

CUDA C++/PTX intrinsics
=======================

When no tile primitive covers what you need, two escape hatches reach the hardware
directly: **call a backend intrinsic** (the ``Tx.cuda.*`` / ``Tx.ptx.*`` namespaces
from ``tvm.backend.cuda``), or **inline raw CUDA** source.

Calling backend intrinsics
--------------------------

``Tx.cuda.*`` and ``Tx.ptx.*`` expose the CUDA backend's device intrinsics directly —
synchronization, mbarriers, reductions, and the PTX data-movement / MMA families:

.. code-block:: python

    Tx.cuda.cta_sync()                    # block barrier (__syncthreads)
    Tx.cuda.warp_sync()                   # __syncwarp
    Tx.cuda.warpgroup_sync(8)             # warpgroup barrier
    Tx.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

    bar = Tx.alloc_shared((1,), "uint64")
    Tx.ptx.mbarrier.init.shared.b64(bar.data, Tx.uint32(1))  # mbarrier for async completion
    Tx.cuda.mbarrier_wait(bar.data, phase)

A complete, runnable example — a warp all-reduce via ``Tx.tvm_warp_shuffle_xor``:

.. code-block:: python

    @Tx.prim_func
    def warp_reduce(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (32,), "float32", align=16)
        Tx.device_entry()
        cta_id = Tx.cta_id([1]); warp_id = Tx.warp_id([1]); lane_id = Tx.lane_id([32])
        v = Tx.alloc_local((1,), "float32"); i = Tx.alloc_local((1,), "int32")
        v[0] = Tx.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            v[0] += Tx.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

The shuffle lowers straight to ``__shfl_xor_sync``:

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

Other families under ``Tx.ptx.*`` / ``Tx.cuda.*``: ``cp.async`` (LDGSTS),
``cp_async.bulk.tensor`` (TMA), ``ldmatrix`` / ``stmatrix``, ``tcgen05.*``
(Blackwell MMA), ``atomic_add``, ``fence`` … See :doc:`../../api/cuda` for CUDA
helpers and :doc:`../../api/ptx` for the registered PTX forms.

Inlining raw CUDA
-----------------

For something with no intrinsic at all, inject a ``__device__`` function from a
source string with ``Tx.cuda.func_call(name, *args, source_code=..., return_type=...)``:

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @Tx.prim_func
    def k(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (256,), "float32")
        B = Tx.match_buffer(B_ptr, (256,), "float32")
        Tx.device_entry(); bx = Tx.cta_id([1]); tx = Tx.thread_id([256])
        B[tx] = Tx.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")

The source is emitted verbatim and the call is wired in:

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
