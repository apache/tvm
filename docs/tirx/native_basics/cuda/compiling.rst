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

CUDA compilation uses one immutable ``CompileConfig``. It is available as
``tvm.backend.cuda.CompileConfig``, ``T.cuda.CompileConfig`` and
``txl.cuda.CompileConfig``. Build settings provide defaults; a device entry
can override individual fields:

.. code-block:: python

    from tvm.backend.cuda import CompileConfig

    @T.function
    def pipeline(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            compile_config=T.cuda.CompileConfig(ftz=False),
        ):
            x = T.cuda.thread_idx("x")
            B[x] = A[x] * T.float32(0.5)
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            compile_config=T.cuda.CompileConfig(compiler="nvcc", lineinfo=True),
        ):
            y = T.cuda.thread_idx("x")
            A[y] = B[y] + T.float32(1)

    exe = tvm.compile(pipeline, compile_config=CompileConfig(arch="sm_100a"))

Each entry resolves its configuration before architecture-sensitive lowering.
Entries with different resolved targets or options compile as separate CUDA
modules; their shared device helpers are copied into each group. A single
function can therefore contain kernels with different architectures and
compiler settings. Such kernels must still be compatible with the device on
which the function is executed.

``None`` means unspecified. Explicit ``False``, ``0`` and empty sequences
replace inherited settings. Sequences replace rather than append. Use
``config.with_overrides(ftz=False)`` to derive another immutable configuration.
An explicit ``Target.arch`` and a conflicting build-level ``CompileConfig.arch``
are rejected. Entry-level architecture overrides are allowed.

Online builds can detect the GPU architecture. Offline builds require an
explicit architecture, either in the build configuration or in every device
entry. No fallback architecture is guessed. A generic ``Target("cuda")`` may
remain without an architecture for backend discovery and IR construction;
compilation resolves or validates its architecture before lowering.
Architecture-dependent Python
factories must receive the configuration before tracing and record the chosen
architecture on their entry; later build defaults cannot change that choice.

The default compiler is NVRTC, with fast math enabled and ptxas register usage
level 10. NVRTC produces cubin by default, NVCC produces fatbin, and NVSHMEM
requires cubin. Individual ``ftz``, ``prec_div``, ``prec_sqrt`` and ``fmad``
settings override the fast-math preset. Raw ``nvcc_options``, ``nvrtc_options``
and ``ptxas_options`` are escape hatches for options without a structured field;
repeating a structured option there is an error.

``LaunchConfig`` controls each runtime launch (grid, block, cluster, stream).
``KernelAttributes`` describes CUDA kernel declaration attributes.
``CompileConfig`` controls the CUDA compiler. The three objects have separate
lifetimes and all are accepted explicitly where they apply.

Artifacts and standalone hosts
------------------------------

Compiled CUDA modules save the effective configuration, source and diagnostics.
Source-only fallback artifacts also save their configuration, so replay does
not depend on the producing process's environment. Old binary artifacts remain
loadable; old source-only artifacts without configuration must be regenerated.

``CompileConfig(dump_dir="...")`` writes source, binary, resolved JSON and the
compiler log under a content-derived name. The identity includes source,
effective compiler options and toolchain version; the dump directory is excluded.
Dumping never enables debug or line information implicitly.

For ``target=Target({"kind": "cuda", "arch": "sm_100a"}, host="cuda_host")``,
``tvm.backend.cuda.export_cuda_host(exe.mod)`` returns ordinary C++ host source
with embedded independently compiled device binaries. Compile it with a C++
compiler and CUDA headers, and link with tvm-ffi, cudart and the CUDA driver.
The exported library uses ``cuLaunchKernelEx`` and requires no TVM installation
or device compiler at runtime. Compiling the host does not recompile kernels.

Migration
---------

Replace ``@txl.kernel(arch=...)``, ``Kernel.arch`` and ``tirx.cuda_arch`` with
``compile_config=CompileConfig(arch=...)`` on ``device_entry`` or ``compile``.
Replace compiler/math/ptxas environment variables with the corresponding fields.
``TIRX_PREPARE_CUDA_ARCH`` is removed; CPU benchmark preparation receives an
explicit configuration and defaults to NVCC. Removed environment options raise
an error with the replacement field, instead of being silently ignored.

See :doc:`../../api/cuda_compile` for the complete configuration API.

Inspecting the result
---------------------

Read the IR with ``.show()`` / ``.script()``, and read the generated CUDA from the
compiled module.

.. code-block:: python

    pipeline.show()                       # pretty-print the TIRx (TVMScript)
    print(pipeline.script())              # ... the same, as a string

    # the generated CUDA C source, from the compiled Executable:
    print(exe.mod.imports[0].inspect_source())

``Tx.hint("message")`` (statement or ``with`` block) attaches structured hints
that survive a script round-trip.

From simple to complex
----------------------

A natural native progression, each rung adding one capability:

#. **Elementwise** — ``device_entry`` + ``thread_id`` + a guarded store (the first
   kernel).
#. **Shared-memory reduction** — stage into ``Tx.alloc_shared``, then a
   ``cta_sync``-separated tree (shown in full below). Adds shared memory and a
   block barrier.
#. **Warp / block reduction** — ``Tx.gpu_warp_shuffle_xor`` or ``Tx.cuda.cta_sum``
   to combine partial results across lanes/warps (the warp all-reduce in
   :doc:`threads_sync`).
#. **Async pipeline** — ``Tx.ptx.cp.async_`` (or TMA ``cp.async.bulk.tensor``) with
   ``Tx.ptx.mbarrier.*`` / ``Tx.cuda.mbarrier_wait`` to overlap loads with compute.

Rung 2 in full — a 256-element block sum via a shared-memory tree reduction
(shared buffer, ``cta_sync``, a ``while`` loop, and a thread predicate):

.. code-block:: python

    @Tx.function
    def block_sum(A: Tx.Tensor((256,), "float32"), out: Tx.Tensor((1,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")

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


    exe = tvm.compile(
        tvm.IRModule({"main": block_sum}), target=tvm.target.Target("cuda"), tir_pipeline="tirx"
    )
    a = torch.arange(256, device="cuda", dtype=torch.float32)
    out = torch.zeros(1, device="cuda")
    exe(a, out)  # out[0] == 32640.0

The full tile-level GEMM/attention ladder (sync → TMA → warp specialization →
2-CTA cluster) is built on top of these and the dispatchable tile primitives in
:doc:`../../tile_primitives`.

Next steps
----------

- :doc:`../../layout` — how buffers map to physical resources (``TileLayout``).
- :doc:`../../tile_primitives` — the dispatchable ops these native idioms lower to.
