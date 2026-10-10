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
through the **tirx pipeline** — an ordered sequence of IR passes that turns the
high-level constructs you write (tile primitives, ``TileLayout``-typed buffers,
CUDA index calls) into split **host** + **device** functions, which the CUDA
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
     - ``StmtSimplify``
     - statement-level arithmetic simplification (the sym analyzer)
   * - 3
     - ``LowerTIRxOpaque``
     - lowers remaining opaque constructs to lower-level TIRx forms
   * - 4
     - ``FlattenBuffer``
     - flattens multi-dimensional ``TensorLoad`` / ``TensorStore`` to 1-D
   * - 5
     - ``BF16ComputeLegalize``
     - rewrites ``bfloat16`` compute to a legal (f32-up-cast) form
   * - 6
     - ``NarrowDataType(32)``
     - narrows index/loop ``PrimExpr`` dtypes to 32-bit where provably safe
   * - 7
     - ``VectorizeLoop``
     - turns ``Tx.vectorized`` loops into vector ops (skipped if
       ``tir.disable_vectorize``)
   * - 8
     - ``UnrollLoop``
     - unrolls loops marked ``Tx.unroll`` (and small constant loops)
   * - 9
     - ``StmtSimplify``
     - simplify again, now that vectorize/unroll exposed constants
   * - 10
     - ``CommonSubexprElim``
     - hoists repeated subexpressions into temporaries (skipped if
       ``tir.disable_cse_tir``)
   * - 11
     - ``FP8ComputeLegalize``
     - rewrites ``float8`` compute to a legal form
   * - 12
     - ``VerifyMemory``
     - checks no host-side code directly dereferences device memory (a safety gate)
   * - 13
     - ``AnnotateEntryFunc``
     - marks the single Function as the module entry point
   * - 14
     - ``SplitHostDevice``
     - extracts target-annotated device regions into **device** functions and
       leaves launch calls in the **host** function; the regions originate from
       the explicit ``LaunchConfig`` attached to ``Tx.device_entry``
   * - 15
     - ``LowerIket``
     - lowers CUDA IKET instrumentation after host/device splitting
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

    LowerTIRx = Sequential([ TileDispatch, LowerTIRxCleanup ])

- **``TileDispatch``** replaces every tensor ``Evaluate(Call)`` with the body emitted by its
  instruction lowerer, including delayed mathematical composites as described in
  :doc:`tile_dispatch`. In the same pass it converts ``device_entry`` to a
  ``device_scope`` region retaining launch values as operands, and lowers the
  finite CUDA index calls inside ordinary ``Bind`` statements.
- **``LowerTIRxCleanup``** then runs the ``LayoutApplier``: it resolves every
  ``TileLayout``-typed buffer access into concrete physical address arithmetic
  (``addr = data + elem_offset + layout.apply(coord)``), flattens the buffers,
  and removes buffer offsets that have been folded into the resulting views.

After ``LowerTIRx`` the module remains a ``tvm.tirx.Function``, but contains no
tile primitives or ``TileLayout`` indirection, and CUDA indices read hardware coordinates.  Later TIRx passes lower the remaining opaque constructs and
the target code generators consume ``tirx::Function`` directly; there is no
conversion to the separate ``tvm.tir`` object model.

A worked example
----------------

Take a one-line scale kernel:

.. code-block:: python

    @Tx.function
    def scale(A: Tx.Tensor((256,), "float32"), B: Tx.Tensor((256,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        B[tx] = A[tx] * Tx.float32(2.0)

**After ``LowerTIRx``** the layout is applied and the device region retains its
launch configuration independently of the index bindings:

.. code-block:: python

    with Tx.region("tirx.device_scope", [1, 1, 1, 256, 1, 1], attrs={
        "cuda.launch_fields": ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"],
        "cuda.kernel_options": {},
    }):
        bx: Tx.let = 0
        tx: Tx.let = Tx.cuda.thread_idx("x")
        B_1[tx] = A_1[tx] * Tx.float32(2.0)

**After ``SplitHostDevice`` and the later ``MakePackedAPI`` pass** the one function
has become two —
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
