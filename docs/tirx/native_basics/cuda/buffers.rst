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

Buffers and memory
==================

Parameter buffers are bound with ``Tx.match_buffer``; scratch buffers are created
in the body with one of two declaration APIs (below). Index a buffer with
``A[i, j]``, slice it with ``A[m0:m0+BM, 0:BK]`` (a ``BufferRegion``), and take a
pointer with ``A.ptr_to([i, j])`` or the raw data pointer ``A.data``.

Declaring buffers
-----------------

Two fundamental APIs create a buffer:

- ``Tx.alloc_buffer(shape, dtype, scope=..., ...)`` — **allocates new storage**
  (emits an ``AllocBuffer`` node) and returns the ``Buffer``. ``Tx.alloc_shared`` /
  ``Tx.alloc_local`` are just ``alloc_buffer`` with ``scope="shared"`` /
  ``scope="local"``.
- ``Tx.decl_buffer(shape, dtype, data=..., ...)`` — **declares a view** over an
  existing pointer ``data`` (no allocation); use it to alias or reinterpret
  storage — a sub-region of a pool, or a tensor-memory address. With ``data=None``
  it allocates, like ``alloc_buffer``.

A buffer's ``data`` pointer is an immutable ``Var`` (``alloc_buffer`` defines it;
``decl_buffer`` takes one). To back a buffer with a pointer *expression*, assign
it to a name first; the parser creates an immutable pointer binding. See
:doc:`data_types`.

Both share one descriptor; the parameters that matter most:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - Meaning
   * - ``dtype``
     - element type — ``"float32"``, ``"float16"``, ``"float4_e2m1fn"``, …
   * - ``shape``
     - logical shape (a tuple of extents)
   * - ``layout``
     - physical mapping (:doc:`TileLayout <../../layout>`); ``"default"`` = dense
       row-major
   * - ``elem_offset`` / ``allocated_addr``
     - ``elem_offset`` (or ``byte_offset``) places a *view* at an offset into
       ``data``; ``allocated_addr`` carries a pre-assigned address (tensor memory)
   * - ``align``
     - alignment of the data pointer, in bytes

The ``scope`` argument selects the memory space:

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - Shorthand
     - Memory
   * - ``"global"``
     - (default)
     - device global memory
   * - ``"shared"``
     - ``Tx.alloc_shared``
     - static shared memory (``__shared__``)
   * - ``"shared.dyn"``
     - (pool)
     - dynamic shared memory (pooled — see below)
   * - ``"local"``
     - ``Tx.alloc_local``
     - per-thread local storage (usually promoted to registers)
   * - ``"tmem"``
     - (TMEM pool)
     - Blackwell tensor memory (see below)

.. code-block:: python

    A = Tx.match_buffer(A_ptr, (M, K), "float16", align=16)   # parameter buffer
    As = Tx.alloc_shared((BM, BK), "float16")                 # new shared tile
    acc = Tx.alloc_local((4,), "float32")                     # per-thread accumulator
    view = Tx.decl_buffer((BM, BK), "float16", data=As.data)  # a view over As

**A ptr-based buffer is just metadata over a pointer.** For any non-tmem buffer,
the declaration is a pointer plus a layout, and indexing resolves to an address::

    addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]

(``layout.apply`` returns the per-axis mapping; its ``"m"`` component is the
element offset.) So the *same* logical access compiles to different address
arithmetic depending purely on the buffer's metadata. Writing
``B[i, j] = A[i, j] + 1`` over a 4×8 region, with ``B`` declared four ways:

.. code-block:: python

    from tvm.tirx.layout import TileLayout, S

    B = Tx.match_buffer(p, (4, 8), "float32")                                       # row-major
    B = Tx.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))  # column-major
    B = Tx.match_buffer(p, (4, 8), "float32", elem_offset=64)                       # shifted view
    B = Tx.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)])) # row stride 16

each makes ``B[i, j]`` lower to a different index in the generated CUDA (the
``A[i, j]`` load stays ``i*8 + j`` — only ``B``'s metadata changed):

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // row-major:        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // column-major:     j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64:   i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // row stride 16:    i*16 + j

Shared memory
-------------

Shared memory comes in two flavors — **static** (fixed at compile time) and
**dynamic** (sized at launch) — plus a pool helper that manages the dynamic case.

Static
~~~~~~

The simplest shared buffer is a **static** one — ``Tx.alloc_shared`` (that is,
``scope="shared"``), sized at compile time. Stage data into it, ``cta_sync`` so the
whole block sees the writes, then read it back:

.. code-block:: python

    @Tx.prim_func
    def smem_demo(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (128,), "float32")
        B = Tx.match_buffer(B_ptr, (128,), "float32")
        Tx.device_entry()
        bx = Tx.cta_id([1])
        tx = Tx.thread_id([128])
        sm = Tx.alloc_shared((128,), "float32")   # static shared memory
        sm[tx] = A[tx]
        Tx.cuda.cta_sync()
        B[tx] = sm[tx] * Tx.float32(2.0)

It lowers to a plain ``__shared__`` array (generated CUDA, boilerplate elided):

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // Tx.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // Tx.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

Dynamic
~~~~~~~

**Dynamic** shared memory (``scope="shared.dyn"``) is sized per launch (the
``sharedMemBytes`` launch parameter), not at compile time. A kernel may have **only
one** dynamic-shared allocation — the *arena*. So you allocate it once and ``decl``
each buffer as a view into it: ``Tx.decl_buffer`` with ``data=`` the arena pointer
and an ``elem_offset``:

.. code-block:: python

    arena = Tx.alloc_buffer((128,), "float32", scope="shared.dyn")   # the one arena
    As = Tx.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")                 # offset 0
    Bs = Tx.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn") # offset 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    Tx.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

Both views share the single ``extern __shared__`` arena (generated CUDA,
boilerplate elided; arena named ``smem`` for clarity):

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // the one dynamic-shared arena
    smem[tx]      = A_ptr[tx];                       // As — view at offset 0
    smem[tx + 64] = B_ptr[tx];                       // Bs — view at offset 64
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

(Two separate ``alloc_buffer(scope="shared.dyn")`` is an error — *only one dynamic
shared memory allocation is allowed*.) So static shared memory is sized at compile
time (``__shared__ T x[N];``); dynamic shared memory is this one launch-sized arena
with views decl'd at offsets inside it.

.. note::

   **How TVM annotates the dynamic-shared size.** The arena's size is known at
   compile time (here ``128`` floats = ``512`` bytes). During lowering TVM appends
   a ``"tirx.use_dyn_shared_memory"`` tag to the device kernel's
   ``tirx.kernel_launch_params``, and the host launcher computes the total bytes and
   passes them as the last launch argument:

   .. code-block:: python

       # device kernel attribute:
       "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]

       # host-side launch call  (..., gridDim.x, blockDim.x, dyn_shared_bytes):
       Tx.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)

   At run time that ``512`` becomes ``config.sharedMemBytes`` in the
   ``cuLaunchKernelEx`` call. You never set it by hand — it is derived from the
   ``shared.dyn`` allocation's size.

Pool sugar
~~~~~~~~~~

``Tx.SMEMPool`` automates that arena bookkeeping — it bump-allocates the offsets so
you don't ``decl`` views by hand. Beyond ``alloc`` / ``commit``, it offers
per-buffer ``align=``, an ``alloc_tcgen05_mma_AB`` helper that builds an MMA-compatible
swizzle layout for you, and ``move_base_to`` to rewind the cursor and reuse space:

.. code-block:: python

    pool = Tx.SMEMPool()                          # bump allocator over shared.dyn
    As = pool.alloc((BM, BK), "float16", align=128)   # carve a tile
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_tcgen05_mma_AB((BM, BN), "float16")  # MMA-compatible, swizzle inferred
    pool.commit()                                 # finalize the pool's size
    # pool.move_base_to(offset) rewinds the cursor to reuse space

The TMEM pool (`Tensor memory`_, below) is layered on top of an ``SMEMPool``.

Per-thread local storage
------------------------

Per-thread scratch uses ``scope="local"``. Allocate it with
``Tx.alloc_local(shape, dtype)``: it is private to each thread and lowers to a
local array that nvcc/ptxas can promote to registers when its accesses permit.

.. code-block:: python

    r = Tx.alloc_local((4,), "float32")   # per-thread local array
    for k in Tx.unroll(4):
        r[k] = A[tx, k]
    # ... compute on r[0..3] ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          // per-thread array; ptxas may scalarize it
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   The ``alignas(64)`` is the *default* buffer alignment — a buffer's
   ``data_alignment`` defaults to ``runtime::kAllocAlignment`` (64 bytes), and the
   CUDA codegen stamps it onto every allocation, including per-thread ``local``
   arrays. Statically indexed locals are typically promoted to registers by
   nvcc/ptxas (scalar replacement of aggregates), where the alignment has no
   runtime effect. Dynamically indexed arrays may instead spill to addressable
   local memory, so do not rely on register promotion when reasoning about them.

Scalar
~~~~~~

A mutable scalar is represented by a ``local`` buffer with **one element** —
strictly, you don't need a separate concept. You can allocate the buffer and
index ``[0]``:

.. code-block:: python

    phase = Tx.alloc_local((1,), "int32")   # one-element local buffer
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

But writing ``phase[0]`` everywhere is clumsy, so a **scalar** is sugar for exactly
this — a one-element local buffer you read and write **by name**:

.. code-block:: python

    phase: Tx.int32 = 0                 # mutable scalar (sugar for the above)
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = Tx.local_scalar("int32")        # explicit form; assign by name (s = ..., not s[0])
    acc: Tx.float32 = 0.0               # a type-annotated assignment also makes one

The two are not just similar — they parse to **structurally identical TIRx**. The
sugar is resolved entirely in the parser: ``phase: Tx.int32`` *is* that one-element
``local`` buffer, and ``phase`` / ``phase += 1`` *are* ``phase[0]`` /
``phase[0] += 1``. ``tvm.ir.assert_structural_equal`` on the two kernels passes, and
the printer even renders the explicit ``alloc_local`` + ``[0]`` form **back** as the
scalar form — so once parsing is done there is no difference at all. Both therefore
lower to the same ``alignas(64) int phase_ptr[1];``; the scalar just lets you drop
the ``[0]``. (``Tx.local_scalar`` / ``Tx.shared_scalar`` / ``Tx.alloc_scalar`` choose
the scope explicitly.)

.. note::

   **Why not a** ``Var``\ **?** A TIRx ``Var`` is *immutable* — a single static
   binding (it is exactly what ``Tx.let`` produces, below). A scalar needs to be
   *mutable* — you reassign it in loops and accumulators — so it must be backed by a
   one-element buffer you can store into repeatedly, not a ``Var``.

``let``
~~~~~~~

A ``Tx.let`` binding is **immutable** — a single TIRx ``Bind`` statement (a named
value, not a buffer). Use it for derived constants:

.. code-block:: python

    n: Tx.let = M * K               # immutable Bind
    half: Tx.let[Tx.int32] = N // 2  # ... with an explicit type

It lowers to a **plain scalar C variable** — not a buffer (no array, no ``[0]``).
For ``half: Tx.let = m * 2`` (with a runtime ``m``):

.. code-block:: c++

    int half = m * 2;     // the `let` -> a const-like local

Because the value is immutable, the simplifier is free to propagate and CSE it, so
at the use sites you often see ``m * 2`` substituted directly (or shared through a
common-subexpression temporary) rather than a reference to ``half``.

.. note::

   **Why have an immutable binding at all?** Because the value cannot change, the
   arithmetic analyzer binds the var to it (``analyzer.Bind(var, value)`` when it
   simplifies a TIRx ``Bind``), so facts proven about the value — constant bounds, the
   modular set (divisibility / alignment), ranges — **propagate through every use**.
   That feeds index simplification, bounds-check elimination, and
   alignment/vectorization decisions. A *mutable* scalar is a memory load
   (``buf[0]``): the analyzer cannot assume it stays constant, so none of those
   properties carry through. A ``let`` is also a pure value — no allocation, and
   free to inline / substitute / CSE — whereas a scalar is a one-element buffer with
   load/store semantics.

Tensor memory
-------------

Blackwell *tensor memory* is not a plain scratch scope: it must be explicitly
reserved and freed with the warp-uniform ``Tx.ptx.tcgen05.alloc`` /
``tcgen05.dealloc`` intrinsics, and each tensor is a view into it declared with
``Tx.decl_buffer(..., scope="tmem", allocated_addr=<address>, layout=<tmem layout>)``.
The ``allocated_addr`` is the allocated tensor-memory base address plus any desired
column offset. It is mandatory — the tensor-core dispatch asserts it — so
``Tx.alloc_buffer(scope="tmem")`` (which does **not** set it) will not work. Unlike
shared memory, tensor memory is not directly addressable: it is read and written
only through ``tcgen05`` ``mma`` / ``ld`` / ``st`` / ``cp``.

By hand, one warp issues the allocation into a shared slot, you ``decl`` each
tensor as a view at a column offset, and one warp frees it at the end:

.. code-block:: python

    addr = Tx.alloc_shared((1,), "uint32")             # slot for the allocated base
    if warp_id == alloc_warp:                         # tcgen05.alloc is warp-uniform
        Tx.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
            Tx.address_of(addr), Tx.uint32(512))
    acc = Tx.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                        allocated_addr=addr[0], layout=tmem_layout)  # allocated base
    # ... use acc as a gemm_async / copy_async operand ...
    if warp_id == alloc_warp:
        Tx.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
        Tx.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
            addr[0], Tx.uint32(512))

You add any column offsets to ``addr[0]`` and manage the ``tmem_layout`` (the
matching datapath A–G layout) yourself. This is the sequence the pool below emits.

Pool
~~~~

``Tx.TMEMPool`` wraps the warp-uniform alloc/dealloc and column bump-allocation.
Pass an explicit layout to ``alloc``, or use its operand-specific allocation
helpers when the MMA datapath determines the layout:

.. code-block:: python

    tmem_addr = pool.alloc((1,), "uint32")          # pool = the kernel's smem pool
    tmem_pool = Tx.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                           tmem_addr=tmem_addr)
    # Choose the layout required by the instruction that consumes the buffer:
    acc = tmem_pool.alloc((CTA_M, 512), "float32")  # Layout D when CTA_M=128
    # Layout B: PTX M=128, cta_group=2, 64 logical rows in this CTA.
    # acc = tmem_pool.alloc_tcgen05_mma_D(
    #     (64, N), "float32", M=128, cta_group=2)
    tmem_pool.commit()                               # emits tcgen05.alloc (one warp)
    # ... use acc ...
    tmem_pool.dealloc()                              # emits tcgen05.dealloc (one warp)

See the :doc:`../../tile_primitives` walkthroughs for full examples.

Buffer APIs
-----------

A ``Buffer`` is metadata over a pointer (see *Declaring buffers* above), so most of
its methods are *compile-time* reshapes/reinterprets that change index arithmetic
or hand you a pointer — they emit no runtime op of their own. The common ones:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - What it is
   * - ``B.data``
     - the raw data pointer (a ``Var``); prints as ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - a typed pointer to an element (``address_of``); prints as ``&B_ptr[…]``
   * - ``B.vload([i], dtype="float32x4")`` / ``B.vstore([i], v)``
     - a vectorized load / store; prints as ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - reinterpret the same storage under a new shape/layout (no copy)
   * - ``B.local(*shape, layout=…)``
     - the calling thread's private storage slice of a ``local`` buffer,
       in physical storage order by default
   * - ``B.permute(*dims)``
     - a view with axes permuted (a transposed layout)
   * - ``B.access_ptr(mask, …)``
     - a masked access pointer (the ``tvm_access_ptr`` builtin), for passing a
       region to an intrinsic

**Pointers — ``ptr_to`` / ``data``.** ``ptr_to`` is how you hand an element address
to an intrinsic or inline function; ``data`` is the base pointer:

.. code-block:: python

    B[tx] = Tx.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          // ptr_to([tx]) -> &A_ptr[tx];  A.data -> A_ptr

The pointer returned by ``ptr_to`` has the buffer's element type and storage
scope. This remains true when the buffer is a typed view over a byte-addressed
allocation pool; the pool's raw backing-pointer type does not leak through the
element address.

**Vectorized access — ``vload`` / ``vstore``.** Move several elements as one wide
transfer (see also :doc:`data_types`):

.. code-block:: python

    B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**Reshape / reinterpret — ``view`` / ``permute``.** Both are pure metadata; the
data pointer is unchanged, only the index arithmetic differs. ``A.view(64, 4)``
sees the 256-element buffer as ``64×4``; ``A.permute(1, 0)`` transposes the axes:

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j]  -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 // view: row-major 64x4 index
    At_ptr[(j * 4) + i]                       // permute: swapped strides

**Per-thread view — ``local``.** Decomposes a thread-axis ``local`` layout into the
calling thread's storage bundle (used pervasively by the tile primitives).
Both forms expose the raw physical storage span by default, including layout
gaps and offsets: ``R.local()`` infers a flat 1-D span, while
``R.local(d0, d1, ...)`` is a row-major reshape whose product must equal that
span.  Pass an explicit ``layout=`` only when storage-iterator coordinates are
required.  This mediated form is an escape hatch: the supplied layout
interprets the requested shape, so that shape is not constrained to the raw
span.  Without ``layout=``, omitting the shape always infers a one-dimensional
physical storage span.  Only the explicit-``layout=`` compatibility form with
no shape infers a one-dimensional shape from the logical storage size:

.. code-block:: python

    R  = Tx.alloc_buffer((32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
    R_flat = R.local()       # this lane's 8 local elements, physical order
    R_2d = R.local(2, 4)     # the same elements, row-major 2x4 reshape

.. code-block:: c++

    alignas(64) float R_flat_ptr[8];          // the lane's private local elements
