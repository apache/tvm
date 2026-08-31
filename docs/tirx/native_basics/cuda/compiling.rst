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
pipeline and returns an ``Executable`` you call directly. With an active CUDA
device, target ``"cuda"`` auto-detects its architecture (for example
``sm_100a``). If no device is available during compilation, TVM warns and falls
back to ``sm_50``; specify ``-arch=...`` when cross-compiling or when the emitted
instructions require a newer architecture.

.. code-block:: python

    target = tvm.target.Target("cuda")
    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")

``tir_pipeline="tirx"`` selects the TIRx lowering pipeline (tile-primitive
dispatch and cleanup inside ``LowerTIRx`` → host/device split → finalize).
Compiling inside a ``with target:`` block also works and lets the kernel pick
up the target context.

Inspecting the result
---------------------

Read the IR with ``.show()`` / ``.script()``, and read the generated CUDA from the
compiled module.

.. code-block:: python

    scale.show()                          # pretty-print the TIRx (TVMScript)
    print(scale.script())                 # ... the same, as a string

    # the generated CUDA C source, from the compiled Executable:
    print(exe.mod.imports[0].inspect_source())

Debug aids: ``Tx.print_buffer(C.data, "float32", False, False, 1, (M,))`` emits a
runtime ``printf`` of a buffer into the kernel; ``Tx.hint("message")`` (statement
or ``with`` block) attaches structured hints that survive a script round-trip.

From simple to complex
----------------------

A natural native progression, each rung adding one capability:

#. **Elementwise** — ``device_entry`` + ``thread_id`` + a guarded store (the first
   kernel).
#. **Shared-memory reduction** — stage into ``Tx.alloc_shared``, then a
   ``cta_sync``-separated tree (shown in full below). Adds shared memory and a
   block barrier.
#. **Warp / block reduction** — ``Tx.tvm_warp_shuffle_xor`` or ``Tx.cuda.cta_sum``
   to combine partial results across lanes/warps (the warp all-reduce in
   :doc:`threads_sync`).
#. **Async pipeline** — ``Tx.ptx.cp.async_`` (or TMA ``cp.async.bulk.tensor``) with
   ``Tx.ptx.mbarrier.*`` / ``Tx.cuda.mbarrier_wait`` to overlap loads with compute.

Rung 2 in full — a 256-element block sum via a shared-memory tree reduction
(shared buffer, ``cta_sync``, a ``while`` loop, and a thread predicate):

.. code-block:: python

    @Tx.prim_func
    def block_sum(A_ptr: Tx.handle, out_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (256,), "float32")
        out = Tx.match_buffer(out_ptr, (1,), "float32")

        Tx.device_entry()
        bx = Tx.cta_id([1])
        tx = Tx.thread_id([256])

        sm = Tx.alloc_shared((256,), "float32")
        sm[tx] = A[tx]
        Tx.cuda.cta_sync()

        s = Tx.alloc_local((1,), "int32")
        s[0] = 128
        while s[0] >= 1:
            if tx < s[0]:
                sm[tx] += sm[tx + s[0]]
            Tx.cuda.cta_sync()
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
