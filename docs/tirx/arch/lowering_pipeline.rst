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

TIRx lowering pipeline
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` runs an authored TIRx module
through the **tirx pipeline** — an ordered sequence of TIR passes that turns the
high-level constructs you write (tile primitives, ``TileLayout``-typed buffers,
execution-scope ids) into split **host** + **device** functions, which the CUDA
backend then renders to source. The pipeline is defined in
``python/tvm/tirx/compilation_pipeline.py`` (``tirx_pipeline``); this page walks the
passes in order.

Where it sits
-------------

``tvm.compile`` first binds the target, runs the **tirx pipeline** (the module-level
passes below), then applies **finalization** passes separately to the host and
device functions, and finally hands each device function to the CUDA code
generator:

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

The passes
----------

The ``tirx_pipeline`` module pass applies this exact sequence (a few are gated by
``PassContext`` config):

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - What it does
   * - 1
     - ``LowerTIRx``
     - the core lowering — see `Inside LowerTIRx`_ below
   * - 2
     - ``UnifyThreadBinding``
     - merges equivalent thread-axis bindings so each ``threadIdx`` / ``blockIdx``
       axis is declared once
   * - 3
     - ``StmtSimplify``
     - statement-level arithmetic simplification (the arith analyzer)
   * - 4
     - ``LowerTIRxOpaque``
     - lowers remaining opaque TIRx constructs to plain TIR
   * - 5
     - ``FlattenBuffer``
     - flattens multi-dimensional ``BufferLoad`` / ``BufferStore`` to 1-D
   * - 6
     - ``BF16ComputeLegalize``
     - rewrites ``bfloat16`` compute to a legal (f32-up-cast) form
   * - 7
     - ``NarrowDataType(32)``
     - narrows index/loop ``PrimExpr`` dtypes to 32-bit where provably safe
   * - 8
     - ``VectorizeLoop``
     - turns ``T.vectorized`` loops into vector ops (skipped if
       ``tir.disable_vectorize``)
   * - 9
     - ``UnrollLoop``
     - unrolls loops marked ``T.unroll`` (and small constant loops)
   * - 10
     - ``StmtSimplify``
     - simplify again, now that vectorize/unroll exposed constants
   * - 11
     - ``CommonSubexprElim``
     - hoists repeated subexpressions into temporaries (skipped if
       ``tir.disable_cse_tir``)
   * - 12
     - ``FP8ComputeLegalize``
     - rewrites ``float8`` compute to a legal form
   * - 13
     - ``VerifyMemory``
     - checks no host-side code directly dereferences device memory (a safety gate)
   * - 14
     - ``AnnotateEntryFunc``
     - marks the single PrimFunc as the module entry point
   * - 15
     - ``SplitHostDevice``
     - splits each kernel into a **host** function and a **device** function at the
       ``launch_thread`` boundary
   * - 16
     - ``MakePackedAPI``
     - rewrites the host function to the packed-func ABI (the launcher TVM calls)
   * - 17
     - ``FP8StorageLegalize``
     - legalizes ``float8`` storage (packing into supported container types)
   * - 18
     - ``BF16StorageLegalize``
     - legalizes ``bfloat16`` storage

**Finalization** then runs per function kind:

- **host**: ``LowerTVMBuiltin`` (lower ``tvm_*`` builtins), ``LowerIntrin``
  (target-specific intrinsics)
- **device**: ``LowerWarpMemory`` (warp-scoped buffers → shuffles), ``StmtSimplify``,
  ``LowerIntrin``

Inside LowerTIRx
----------------

``LowerTIRx`` is itself a small sequence (``src/tirx/transform/lower_tirx.cc``):

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** replaces every ``TilePrimitiveCall`` (``copy``,
  ``gemm``, ``reduction``, …) with the body emitted by its selected backend
  dispatch — the variant-selection and codegen described in
  :doc:`../tile_primitives`.
- **``LowerTIRxCleanup``** runs the ``LayoutApplier``: it resolves every
  ``TileLayout``-typed buffer access into concrete physical address arithmetic
  (``addr = data + elem_offset + layout.apply(coord)``), flattens the buffers, and
  lowers the execution-scope ids (``T.cta_id`` / ``T.thread_id`` / … →
  ``blockIdx`` / ``threadIdx`` via ``launch_thread``).

So after ``LowerTIRx`` the module is plain TIR: no tile primitives, no
``TileLayout`` indirection, scope ids resolved to thread axes.

A worked example
----------------

Take a one-line scale kernel:

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

**After ``LowerTIRx``** the scope ids are real thread axes and the layout is applied
(``A_1`` / ``B_1`` are the flattened 1-D views):

.. code-block:: python

    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

**After ``SplitHostDevice`` + ``MakePackedAPI``** the one function has become two —
a host launcher and a device kernel:

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

The CUDA backend then renders ``scale_kernel`` to the ``__global__`` function
(``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``).

Reproduce it yourself
---------------------

You can run any prefix of the pipeline by hand to inspect a stage — this is how the
IR snippets across these docs were produced:

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile primitives dispatched, layouts applied
    print(mod.script())               # inspect the lowered TIRx IR

Or compile the whole module and read the generated CUDA:

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
