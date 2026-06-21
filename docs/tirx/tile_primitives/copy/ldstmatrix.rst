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

copy → ldstmatrix
=================

The ``ldstmatrix`` variant lowers a ``copy`` between **register and shared** memory
to the warp-collective PTX ``ldmatrix`` / ``stmatrix`` instructions: one
instruction moves ``num`` 8×8 16-bit matrix tiles between shared memory and the
warp's registers, with the hardware performing the lane↔element shuffle that an
MMA fragment needs. It only applies when the register and shared **layouts match
the m8n8 fragment geometry**; otherwise the copy falls back to :doc:`reg`. Source:
``python/tvm/backend/cuda/operator/tile_primitive/copy/ld_stmatrix.py``.

What it accepts
---------------

The predicate is lean — scope, a valid copy, and a register↔shared pair:

.. code-block:: python

    def _is_ldstmatrix(op_call, sctx):
        if not sctx.is_target("cuda"):
            return False, "non-cuda target"
        if sctx.scope_kind not in ("warp", "warpgroup", "cta"):
            return False, f"unsupported exec_scope {sctx.scope_kind} (need warp, warpgroup, or cta)"
        for check in (
            lambda: _all_threads_active(sctx),
            lambda: _is_valid_copy(op_call, sctx),
            lambda: _scope_allowed(op_call, sctx, allowed_pairs=_REG_SMEM_PAIRS),  # (local, shared*)
        ):
            ok, msg = check()
            if not ok:
                return False, msg
        return True, None

The **real** gate is the layout fit, applied during emit. Both this variant and
:doc:`reg` are priority 10 and both accept ``local ↔ shared``; ``ldstmatrix`` is
tried first and **declines** (via ``fail(...)``) if the layouts are not ldmatrix
fragments, leaving ``reg`` to handle the copy:

.. code-block:: python

    # _emit: try the widest matrix count that fits, else decline
    for num in (4, 2, 1):
        chosen = _try_num(r, s, num)
        if chosen is not None:
            break
    if chosen is None:
        fail("ldstmatrix layout doesn't fit any num ∈ {4,2,1}")

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / scope
     - ``cuda``; ``warp`` / ``warpgroup`` / ``cta`` (needs a full warp), all active
   * - memory pair
     - ``_REG_SMEM_PAIRS`` = ``(local, shared*)`` / ``(shared*, local)``
   * - dtype
     - 16-bit (``.b16``) — ldmatrix/stmatrix move 8 fp16 = 16 B per lane per tile
   * - layout fit
     - both operands regroup to ``[T/32, 8, 4, M/(2·num), num, 2]`` with the
       register side equal to the m8n8 fragment pattern and the shared side row- or
       column-major with 16-B-aligned tile strides (``_try_num``), for some
       ``num ∈ {4, 2, 1}``

Demonstration program
----------------------

A warp loads ``num = 2`` row-major matrix tiles (``M, N = 8, 16`` fp16) shared →
register, from ``test_ld_stmatrix.py`` (register layout = the m8n8 fragment,
``S[(8,4,2,2):(4@laneid, 1@laneid, 2, 1)]``):

.. code-block:: python

    from tvm.tirx.layout import S, TileLayout, laneid

    num = 2; M, N = 8, num * 8
    r_layout = TileLayout(S[(8, 4, num, 2) : (4 @ laneid, 1 @ laneid, 2, 1)])
    s_layout = TileLayout(S[(8, 4, num, 2) : (num * 8, 2, 8, 1)])     # row-major
    full = (slice(0, 8), slice(0, 4), slice(0, num), slice(0, 2))

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (M, N), "float16")
        B = T.match_buffer(B_ptr, (M, N), "float16")
        T.device_entry(); T.cta_id([1]); T.lane_id([32]); tid = T.thread_id([32])
        A_smem = T.alloc_buffer((8, 4, num, 2), "float16", scope="shared", layout=s_layout)
        # ... stage A into A_smem (row = tid//4, cp = tid%4) ...
        T.cuda.cta_sync()
        R = T.alloc_buffer((8, 4, num, 2), "float16", scope="local", layout=r_layout)
        Tx.warp.copy(R[full], A_smem[full])     # shared -> register  (ldmatrix)
        # ... write R back out to B ...

Algorithm
---------

**1. Regroup both layouts to the matrix geometry.** ``_try_num(r, s, num)`` groups
each layout's iters into ``[T/32, 8, 4, M/(2·num), num, 2]``: the warp-replication
outer, the **8** rows of a tile, the **4** lane-column-pairs, ``m_outer`` tiles
along M, the ``num`` tiles, and the inner **2** (the ``.b16`` element pair). If the
group fails, the layout isn't a fragment → ``None``.

**2. The register side must be the exact m8n8 fragment.** The 8/4/2 register
strides must be ``(4, 1, 1)`` — i.e. the canonical ldmatrix fragment where lane
``i`` holds row ``i//4``, column-pair ``i%4``:

.. code-block:: python

    r8, r4, _r_num_iters, r2 = rs
    if (r8, r4, r2) != (4, 1, 1):
        return None

**3. The shared side decides** ``trans`` **and the per-tile stride** ``p``.
Row-major shared (``s4, s2 == 2, 1``, ``s8`` a positive multiple of 8) → plain
``ldmatrix`` with ``p = s8``; column-major (``s8 == 1``, ``s4 == 2·s2``,
``s2`` a multiple of 8) → the ``.trans`` form:

.. code-block:: python

    if (s4, s2) == (2, 1) and s8 > 0 and s8 % 8 == 0:
        return (rg, rsep, sg, ssep, False, s8, num)     # trans=False, p=s8
    if s8 == 1 and s2 > 0 and s2 % 8 == 0 and s4 == 2 * s2:
        return (rg, rsep, sg, ssep, True,  s2, num)     # trans=True,  p=s2

The 8-multiple checks enforce 16-byte alignment (8 fp16) for every tile and every
``m_outer`` advance, since each lane's ``.b16`` access reads 16 bytes.

**4. Emit one instruction per** ``m_outer`` **tile group.** Each lane contributes
its shared address (tile offset + ``(laneid % 8) · p``) and ``num`` register
handles:

.. code-block:: python

    for mm in T.unroll(m_outer):
        smem_ptr = _ptr_off(s_buf.ptr_to(s_zero), _smem_off(mm, tile_off + (laneid % 8) * p))
        handles  = [r_local.ptr_to([...]) for i in range(num)]
        if direction == "ld":
            T.ptx.ldmatrix(trans, num, ".b16", smem_ptr, *handles)
        else:
            T.ptx.stmatrix(trans, num, ".b16", smem_ptr, *handles, shape="m8n8", space="shared")

(This is the one copy variant that **does** use ``T.unroll`` — ``m_outer`` is tiny.)

Generated TIRx IR
-----------------

For the demo (``num = 2``, ``M = 8`` ⇒ ``m_outer = 1``):

.. code-block:: python

    for mm in T.unroll(1):
        T.ptx.ldmatrix(T.bool(False), 2, ".b16", smem_ptr,
                       T.address_of(r_local[0]), T.address_of(r_local[2]))

Generated CUDA
--------------

.. code-block:: c++

    __forceinline__ __device__ void ptx_ldmatrix_2_b16_0(void* smem_ptr, void* dst0, void* dst1) {
      // ...
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
      // ...
    }
    // call site (per lane):
    ptx_ldmatrix_2_b16_0(smem_ptr, &r_local_ptr[0], &r_local_ptr[2]);

``num = 2`` becomes ``.x2`` with two destination registers; the warp's 32 lanes
cooperatively supply the 8 source rows and receive the shuffled fragment.

How inputs change the algorithm
-------------------------------

``num`` (the matrix count that fits) selects the instruction width and the number
of register handles; ``trans`` (set by the shared layout) selects the transposing
form:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - input
     - emitted
     - PTX
   * - ``num = 1``
     - ``.x1``
     - ``ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];``
   * - ``num = 2``
     - ``.x2``
     - ``ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];``
   * - ``num = 4``
     - ``.x4``
     - ``ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];``
   * - ``trans = True``
     - ``.trans``
     - ``ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];``

A larger M raises ``m_outer`` (more unrolled instructions per lane); the ``st``
direction emits ``stmatrix`` with the same width/trans logic. If no ``num`` fits,
the copy is handled by :doc:`reg` instead.
