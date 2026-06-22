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

Compiling and inspecting
========================

Wrap the ``PrimFunc`` in an ``IRModule`` and compile with
``tvm.compile(mod, target=..., tir_pipeline="tirx")``; it runs the TIRx lowering
pipeline and returns an ``Executable`` you call directly. The arch (e.g.
``sm_100a``) is auto-detected from the device, so the target ``"cuda"`` is enough.

.. code-block:: python

    target = tvm.target.Target("cuda")
    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")

``tir_pipeline="tirx"`` selects the TIRx lowering pipeline (``LowerTIRx`` →
tile-primitive dispatch → host/device split → finalize). Compiling inside a
``with target:`` block also works and lets the kernel pick up the target context.

Inspecting the result
---------------------

Read the IR with ``.show()`` / ``.script()``, and read the generated CUDA from the
compiled module.

.. code-block:: python

    scale.show()                          # pretty-print the TIRx (TVMScript)
    print(scale.script())                 # ... the same, as a string

    # the generated CUDA C source, from the compiled Executable:
    print(exe.mod.imports[0].inspect_source())

Debug aids: ``T.print_buffer(C.data, "float32", False, False, 1, (M,))`` emits a
runtime ``printf`` of a buffer into the kernel; ``T.hint("message")`` (statement
or ``with`` block) attaches structured hints that survive a script round-trip.

From simple to complex
----------------------

A natural native progression, each rung adding one capability:

#. **Elementwise** — ``device_entry`` + ``thread_id`` + a guarded store (the first
   kernel).
#. **Shared-memory reduction** — stage into ``T.alloc_shared``, then a
   ``cta_sync``-separated tree (shown in full below). Adds shared memory and a
   block barrier.
#. **Warp / block reduction** — ``T.tvm_warp_shuffle_xor`` or ``T.cuda.cta_sum``
   to combine partial results across lanes/warps (the warp all-reduce in
   :doc:`threads_sync`).
#. **Async pipeline** — ``T.ptx.cp_async`` (or TMA ``cp_async.bulk.tensor``) with
   ``T.ptx.mbarrier.*`` to overlap loads with compute.

Rung 2 in full — a 256-element block sum via a shared-memory tree reduction
(shared buffer, ``cta_sync``, a ``while`` loop, and a thread predicate):

.. code-block:: python

    @T.prim_func
    def block_sum(A_ptr: T.handle, out_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        out = T.match_buffer(out_ptr, (1,), "float32")

        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([256])

        sm = T.alloc_shared((256,), "float32")
        sm[tx] = A[tx]
        T.cuda.cta_sync()

        s = T.alloc_local((1,), "int32")
        s[0] = 128
        while s[0] >= 1:
            if tx < s[0]:
                sm[tx] += sm[tx + s[0]]
            T.cuda.cta_sync()
            s[0] = s[0] // 2

        if tx == 0:
            out[0] = sm[0]

    exe = tvm.compile(tvm.IRModule({"main": block_sum}),
                      target=tvm.target.Target("cuda"), tir_pipeline="tirx")
    a = torch.arange(256, device="cuda", dtype=torch.float32)
    out = torch.zeros(1, device="cuda")
    exe(a, out)                          # out[0] == 32640.0

The full tile-level GEMM/attention ladder (sync → TMA → warp specialization →
2-CTA cluster) is built on top of these and the dispatchable tile primitives in
:doc:`../../tile_primitives`.

Next steps
----------

- :doc:`../../layout` — how buffers map to physical resources (``TileLayout``).
- :doc:`../../tile_primitives` — the dispatchable ops these native idioms lower to.
