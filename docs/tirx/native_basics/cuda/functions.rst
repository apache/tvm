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

Defining a function
===================

A kernel is a ``@Tx.prim_func`` (like ``scale`` in :doc:`first_kernel`), or a
``@Tx.jit`` when it has compile-time parameters (see the last section). This
chapter covers the parameter list — how to declare buffers, what types you can
pass, symbolic shapes, and the ``prim_func`` / ``jit`` distinction.

Declaring buffer parameters
---------------------------

There are two equivalent ways to take a tensor parameter:

- **Handle + match_buffer.** Take a ``Tx.handle`` (an opaque data pointer) and bind
  it in the body with ``Tx.match_buffer``. This is the explicit form and the one
  that exposes every descriptor field — ``layout``, ``elem_offset``, ``scope``,
  ``align``, and symbolic shapes:

  .. code-block:: python

      @Tx.prim_func
      def f(A_ptr: Tx.handle, B_ptr: Tx.handle):
          A = Tx.match_buffer(A_ptr, (256,), "float32", align=16)
          B = Tx.match_buffer(B_ptr, (256,), "float32")
          ...

- **Tx.Buffer annotation.** Annotate the parameter directly. This is the concise
  form — equivalent to a handle bound with ``match_buffer`` using the defaults:

  .. code-block:: python

      @Tx.prim_func
      def f(A: Tx.Buffer((256,), "float32"), B: Tx.Buffer((256,), "float32")):
          ...

Both give you a ``Buffer`` you index with ``A[i]`` / ``A[i, j]``. Use ``Tx.Buffer``
for the common case; drop to ``Tx.handle`` + ``match_buffer`` when you need a custom
layout/offset/scope/alignment or a :ref:`symbolic shape <symbolic-shapes>`.

What the parameter list accepts
-------------------------------

A ``PrimFunc`` parameter is one of the following. The third column is what you
pass on the Python side when you call the compiled ``Executable``:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Annotation
     - Is
     - Pass at call time
   * - ``Tx.Buffer((d0, d1), dtype)``
     - a tensor parameter (shape + dtype fixed)
     - a tensor on the right device
   * - ``Tx.handle``
     - an opaque data pointer (bind with ``match_buffer``)
     - a tensor
   * - ``Tx.int32`` / ``Tx.float32`` / …
     - a runtime scalar
     - a Python ``int`` / ``float``
   * - ``Tx.constexpr`` (``@Tx.jit`` only)
     - a compile-time constant
     - supplied to ``.specialize(...)``, **not** at the call

Tensors may be CUDA ``torch`` tensors (zero-copy through TVM FFI tensor
interop) or
``tvm.runtime.tensor(...)``. Arguments are positional and match the parameter
order. For example, a kernel with a scalar parameter::

    @Tx.prim_func
    def scal(A_ptr: Tx.handle, B_ptr: Tx.handle, s: Tx.float32):
        A = Tx.match_buffer(A_ptr, (256,), "float32")
        B = Tx.match_buffer(B_ptr, (256,), "float32")
        Tx.device_entry(); bx = Tx.cta_id([1]); tx = Tx.thread_id([256])
        B[tx] = A[tx] * s

    exe(a, b, 3.0)        # pass the scalar as a Python float

.. _symbolic-shapes:

Symbolic shapes
---------------

For a size that varies at run time, declare a free symbolic extent with
``Tx.int32()`` and use it in the buffer shape. Its value is **inferred from the
passed tensor** at run time, so a *single compiled kernel* handles any size:

.. code-block:: python

    @Tx.prim_func
    def scale_dyn(a: Tx.handle, b: Tx.handle):
        n = Tx.int32()                          # free symbolic extent
        A = Tx.match_buffer(a, (n,), "float32")
        B = Tx.match_buffer(b, (n,), "float32")
        Tx.device_entry()
        bx = Tx.cta_id([1]); tx = Tx.thread_id([1])
        for i in range(n):                     # loop / launch bounds may use n
            B[i] = A[i] * Tx.float32(2.0)

    exe = tvm.compile(tvm.IRModule({"main": scale_dyn}),
                      target=tvm.target.Target("cuda"), tir_pipeline="tirx")
    exe(torch.rand(100, device="cuda"), torch.empty(100, device="cuda"))   # n = 100
    exe(torch.rand(200, device="cuda"), torch.empty(200, device="cuda"))   # n = 200, same kernel

Both ``match_buffer`` calls share ``n``, so the two shapes are constrained equal;
``n`` is never passed explicitly — it comes from the tensor.

In the generated CUDA, ``n`` is just a runtime kernel argument; the host launcher
reads it from the tensor's shape and passes it, and the loop bound uses it
(boilerplate elided):

.. code-block:: c++

    extern "C" __global__ void
    scale_dyn_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr, int n) {
      for (int i = 0; i < n; ++i) {
        B_ptr[i] = A_ptr[i] * 2.0f;
      }
    }

.. note::

   You passed only two tensors, yet the kernel takes a third argument ``n`` — who
   supplies it? A compiled ``Executable`` has two halves: a **host launcher** and
   the **device kernel** above. When you call ``exe(a, b)``, the host launcher
   unpacks the two tensors, reads ``n`` from ``a``'s shape (``a`` was matched as
   ``(n,)``), checks that ``b`` agrees, computes the launch configuration, and then
   invokes the device kernel — forwarding the data pointers **and** the resolved
   ``n`` as explicit arguments. Nothing passes ``n`` by hand; the host side derives
   it from the tensor metadata. ``tirx.transform.SplitHostDevice`` extracts the
   device function, and the later ``tirx.transform.MakePackedAPI`` pass builds
   the packed host-side argument handling.

You can see it in the IR. **Before** the split, the lowered module is a single
merged function (trimmed):

.. code-block:: python

    @Tx.prim_func
    def main(a: Tx.handle, b: Tx.handle):
        n = Tx.int32()
        A = Tx.match_buffer(a, (n,))
        B = Tx.match_buffer(b, (n,))
        with Tx.launch_thread("blockIdx.x", 1), Tx.launch_thread("threadIdx.x", 1):
            for i in range(n):
                B[i] = A[i] * Tx.float32(2.0)

**After** ``SplitHostDevice``, it is two functions — a device kernel that takes
``n`` as a parameter, and a host ``main`` that calls it, forwarding ``n`` (the
trailing ``1, 1`` are the grid/block launch dims):

.. code-block:: python

    @Tx.prim_func   # device
    def scale_dyn_kernel(A_ptr: Tx.handle("float32"), B_ptr: Tx.handle("float32"), n: Tx.int32):
        ...
        for i in range(n):
            B[i] = A[i] * Tx.float32(2.0)

    @Tx.prim_func   # host
    def main(a: Tx.handle, b: Tx.handle):
        n = Tx.int32()
        A = Tx.match_buffer(a, (n,))
        B = Tx.match_buffer(b, (n,))
        Tx.call_packed("scale_dyn_kernel", A.data, B.data, n, 1, 1)   # n forwarded

``MakePackedAPI`` then fills in where ``n`` comes from — reading it from the
argument's shape (essentially ``n = a.shape[0]``) — and adds the dtype / shape /
device checks (e.g. asserting ``B.shape[0] == n``)::

    n = Tx.Cast("int32", Tx.tvm_struct_get(a_shape, 0, 17, "int64"))   # = a.shape[0]

``@Tx.prim_func`` vs ``@Tx.jit``
--------------------------------

- ``@Tx.prim_func`` parses the function immediately into a ``PrimFunc``. Sizes are
  whatever you wrote — concrete ints, or runtime-symbolic vars (above).
- ``@Tx.jit`` **defers** parsing until you call ``.specialize(**constexpr)``:
  parameters annotated ``Tx.constexpr`` are baked in as compile-time constants and
  the result is an ordinary ``PrimFunc``. Use it when you want sizes/flags fixed at
  compile time (so the compiler can unroll, statically size shared memory, etc.).
  Referencing a constexpr inside an annotation (e.g. ``Tx.Buffer((N,), ...)``)
  requires ``from __future__ import annotations`` at the top of the file.

.. code-block:: python

    @Tx.jit
    def add(A: Tx.Buffer((N,), "float32"), B: Tx.Buffer((N,), "float32"),
            C: Tx.Buffer((N,), "float32"), *, N: Tx.constexpr):
        Tx.device_entry(); bx = Tx.cta_id([1]); tx = Tx.thread_id([N])
        C[tx] = A[tx] + B[tx]

    kernel = add.specialize(N=256)   # -> a PrimFunc with N = 256 baked in

So: a **symbolic shape** is one kernel whose size is resolved at run time; a
**constexpr + jit** produces a specialized kernel per value, resolved at compile
time.

Launch parameters
-----------------

``Tx.device_entry()``
~~~~~~~~~~~~~~~~~~~~~

``Tx.device_entry()`` is a flat marker (no ``with``) that starts the authored
device region: parameter binding and shape reads precede it, while the kernel
body follows it. The parser represents the marker as
``AttrStmt("tirx.device_entry", ...)``. ``LowerTIRx`` then removes the marker,
resolves scope ids, and wraps the device body in thread-extent attributes;
target binding and ``SplitHostDevice`` use those resulting device regions to
extract the kernel shown above.

Scope ids
~~~~~~~~~

After ``device_entry`` you declare the thread hierarchy with *scope-id* intrinsics
— each takes its launch extent as a list:

.. code-block:: python

    Tx.device_entry()
    bx, by = Tx.cta_id([GM, GN])     # blockIdx.x / .y  (grid extents)
    warp_id = Tx.warp_id([4])        # cta -> warp
    lane_id = Tx.lane_id([32])       # warp -> thread
    tx = Tx.thread_id([128])         # cta -> flat thread id

Available ids include ``cta_id``, ``thread_id``, ``warp_id``, ``warpgroup_id``,
``warp_id_in_wg``, ``thread_id_in_wg``, ``lane_id``, ``cluster_id``,
``cta_id_in_cluster``, and ``cta_id_in_pair``. (The legacy ``Tx.launch_thread``
exists but native TIRx uses ``device_entry`` + scope ids.)

**Thread-block clusters** (Hopper/Blackwell) are declared with ``cluster_id``
(kernel → cluster) and ``cta_id_in_cluster`` (cluster → cta). The
``cta_id_in_cluster`` extent is the cluster's CTA dimension; its ``preferred=``
argument sets the *preferred* cluster dimension (CUDA 12.8+):

.. code-block:: python

    cid  = Tx.cluster_id([NUM_CLUSTERS])                  # kernel -> cluster (grid of clusters)
    rank = Tx.cta_id_in_cluster([CLUSTER_SIZE],           # cluster -> cta
                               preferred=[CLUSTER_SIZE])
    # -> cluster_dim = CLUSTER_SIZE, preferred_cluster_dim = CLUSTER_SIZE

These become the ``CLUSTER_DIMENSION`` / ``PREFERRED_CLUSTER_DIMENSION`` launch
attributes in the config below. (``cta_id`` and ``cta_id_in_cluster`` also take an
optional ``preferred=``.) In the device code they lower to reads of the cluster
PTX special registers:

.. code-block:: c++

    int cid  = ...;   // mov.u32 %0, %clusterid.x;      (cluster index)
    int rank = ...;   // mov.u32 %0, %cluster_ctarank;  (CTA rank within the cluster)

The cluster *dimensions* themselves are not in the device code — they are set at
launch time via the attributes above.

Launching the kernel
~~~~~~~~~~~~~~~~~~~~~

During lowering the compiler **extracts every launch parameter** the kernel uses —
the grid and block dimensions, plus the dynamic shared-memory size if any — into
the device function's ``tirx.kernel_launch_params`` attribute. For the ``scale``
kernel that list is ``["blockIdx.x", "threadIdx.x"]``; the host launcher computes
each one's extent (from the scope-id extents and any symbolic shapes) and supplies
them alongside the kernel arguments.

By default, the block size also drives the kernel's ``__launch_bounds__``. The
first argument (max threads per block) is set automatically from the thread
extent. To also set the second argument — the minimum blocks per SM, an
occupancy hint — add
``Tx.attr({"tirx.launch_bounds_min_blocks_per_sm": N})`` in the device region (note:
``Tx.attr``, not ``func_attr``):

.. code-block:: python

    Tx.device_entry()
    Tx.attr({"tirx.launch_bounds_min_blocks_per_sm": 2})   # second launch-bounds arg
    bx = Tx.cta_id([1]); tx = Tx.thread_id([256])
    ...

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(256, 2) scale_kernel(...) { ... }

Without the attr the second argument is omitted (just ``__launch_bounds__(256)``).

Some kernels require an exact block and cluster shape instead of an advisory
maximum. Set ``tirx.required_block_size`` to ``1`` to make the thread and
cluster extents a compile-time launch contract:

.. code-block:: python

    Tx.device_entry()
    Tx.attr({"tirx.required_block_size": 1})
    bx, by = Tx.cta_id([4, 2])
    _, cy = Tx.cta_id_in_cluster([1, 2])
    tx = Tx.thread_id([128])
    ...

.. code-block:: c++

    extern "C" __global__ void __block_size__((128, 1, 1), (1, 2, 1)) kernel(...) { ... }

This requires CUDA Toolkit 13 or newer. All thread and cluster dimensions must
be static; CUDA lowers ``__block_size__`` to PTX ``.reqntid`` and checks the
same dimensions at launch. A preferred cluster dimension must be absent or
equal to the required cluster dimension, and each logical block-grid dimension
must be divisible by its cluster dimension.

``tirx.required_block_size`` can be combined with the launch-bounds attributes
when an occupancy hint is also needed; code generation then emits both
``__block_size__`` and ``__launch_bounds__``. It cannot be combined with
``tirx.max_registers``.

At run time the kernel is launched through the **CUDA Driver API**. TVM's CUDA
runtime loads the module (``cuModuleLoadData``), fetches the function
(``cuModuleGetFunction``, cached), and calls ``cuLaunchKernelEx`` with a
``CUlaunchConfig``. Besides the grid/block dims, dynamic shared size, and stream,
the config carries a list of launch *attributes* — the thread-block **cluster
dimension** and **preferred cluster dimension** (Hopper/Blackwell), plus optional
programmatic-dependent-launch and cooperative-launch flags. Kernels with
``tirx.required_block_size`` instead use CUDA's required-block sentinel; their
compile-time cluster shape replaces the ordinary runtime cluster attribute. In
outline, ``src/backend/cuda/runtime/cuda_module.cc`` follows this path:

.. code-block:: c++

    std::array<CUlaunchAttribute, 4> attrs{};
    unsigned int num_attrs = 0;
    bool use_required_block_dimension =
        launch_param_config_.use_required_block_dimension();

    // 1) thread-block cluster dimension
    if (!use_required_block_dimension &&
        launch_param_config_.use_cluster_launch()) {
      CUlaunchAttribute attr{};
      attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attr.value.clusterDim.x = wl.cluster_dim(0);
      attr.value.clusterDim.y = wl.cluster_dim(1);
      attr.value.clusterDim.z = wl.cluster_dim(2);
      attrs[num_attrs++] = attr;
    }
    // 1b) preferred cluster dimension (CUDA 12.8+); (2) programmatic stream
    //     serialization and (3) cooperative launch are appended the same way
    if (!use_required_block_dimension &&
        (wl.preferred_cluster_dim(0) != 1 || wl.preferred_cluster_dim(1) != 1 ||
         wl.preferred_cluster_dim(2) != 1)) {
      CUlaunchAttribute attr{};
      attr.id = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
      attr.value.clusterDim.x = wl.preferred_cluster_dim(0);
      attr.value.clusterDim.y = wl.preferred_cluster_dim(1);
      attr.value.clusterDim.z = wl.preferred_cluster_dim(2);
      attrs[num_attrs++] = attr;
    }

    CUlaunchConfig config{};
    if (use_required_block_dimension) {
      config.gridDimX = wl.grid_dim(0) / wl.cluster_dim(0);
      config.gridDimY = wl.grid_dim(1) / wl.cluster_dim(1);
      config.gridDimZ = wl.grid_dim(2) / wl.cluster_dim(2);
      config.blockDimX = CU_LAUNCH_KERNEL_REQUIRED_BLOCK_DIM;
      config.blockDimY = 1;
      config.blockDimZ = 1;
    } else {
      config.gridDimX = wl.grid_dim(0);
      config.gridDimY = wl.grid_dim(1);
      config.gridDimZ = wl.grid_dim(2);
      config.blockDimX = wl.block_dim(0);
      config.blockDimY = wl.block_dim(1);
      config.blockDimZ = wl.block_dim(2);
    }
    config.sharedMemBytes = wl.dyn_shmem_size;
    config.hStream = strm;
    config.attrs = num_attrs == 0 ? nullptr : attrs.data();
    config.numAttrs = num_attrs;

    CUresult result = cuLaunchKernelEx(&config, fcache_[device_id], void_args, nullptr);

Here ``wl`` is the resolved workload (the grid/block/cluster extents derived from
the launch parameters), ``fcache_[device_id]`` is the cached ``CUfunction``, and
``void_args`` are the kernel arguments — the data pointers plus scalars like the
symbolic ``n``.
