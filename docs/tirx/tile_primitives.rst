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

Tile Primitives
===============

.. note::

   This page documents the tile-primitive surface and dispatch as it exists in
   the source today; signatures and variants may change.

Tile primitives are the dispatchable, hardware-level operations a TIRx kernel
issues — data movement (``copy``, ``copy_async``), matrix multiply (``gemm``,
``gemm_async``), reductions, elementwise math, and a few fused/compose forms.
A primitive call is recorded as an **unresolved** ``TilePrimitiveCall`` IR node;
the compiler later *dispatches* it — selecting a concrete lowering from the
primitive, the execution scope, the operand layouts, the target, and an optional
explicit hint — and replaces it with native IR (loops, address arithmetic,
synchronization, and backend intrinsics).

Calling convention
------------------

The examples use one TIRx dialect alias and reach tile primitives through its
``tile`` namespace::

    from tvm.script import tirx as Tx

``Tx`` is an ordinary Python module alias, not an injected language keyword.
Under ``Tx.tile``, the next namespace prefix selects the **cooperation
scope**:

- ``Tx.tile.<name>(...)`` — unqualified, runs at **thread** scope.
- ``Tx.tile.warp.<name>`` / ``Tx.tile.wg.<name>`` (alias
  ``Tx.tile.warpgroup``) / ``Tx.tile.cta.<name>`` /
  ``Tx.tile.cluster.<name>`` / ``Tx.tile.thread.<name>`` — bind a wider scope.

Most primitive constructors also carry ``workspace: dict[str, Buffer] | None``,
``dispatch: str | None`` (force a named lowering variant), and ``**kwargs``
collected into a ``config`` dict that tunes the chosen lowering.  ``ScopedOp``
fills the underlying ``scope`` argument from the namespace prefix; select a
scope with ``Tx.tile.warp`` / ``Tx.tile.wg`` / ``Tx.tile.cta`` rather than
passing it to the callable directly. Operands are ``Buffer`` / ``BufferRegion``
values, each carrying a :doc:`TileLayout <layout>` that dispatch reads.

Primitive catalog
-----------------

The 31 primitives, grouped. Signatures show the operands plus the usual
``workspace``/``dispatch``/``**kwargs`` tail (abbreviated ``...``); ``select``
and the ``compose_op`` frame have narrower signatures.  The underlying IR node
also stores a scope, selected through the namespace prefix as described above.

Data movement
~~~~~~~~~~~~~~

::

    copy(dst, src, ...)            # synchronous element copy src -> dst
    copy_async(dst, src, ...)      # asynchronous copy (caller commits/waits)
    permute_layout(dst, src, ...)  # rearrange under a different layout (may alias)

Matrix multiply
~~~~~~~~~~~~~~~~

::

    gemm(D, A, B, C, transpose_A=False, transpose_B=False,
         alpha=1.0, beta=0.0, ...)          # D = alpha*A*B + beta*C (register mma)
    gemm_async(C, A, B, SFA=None, SFB=None,
               transA=False, transB=False, accum=False, ...)  # async / block-scaled

Fill / memset / zero
~~~~~~~~~~~~~~~~~~~~~~

::

    fill(dst, value, ...)        # fill region with a scalar
    memset(dst, value, ...)      # set all elements to a value
    zero(dst, src=None, ...)     # zero out (in place if src omitted)

Cast and elementwise
~~~~~~~~~~~~~~~~~~~~~~

::

    cast(dst, src=None, ...)                         # dtype cast (buffer form)
    sqrt / exp / exp2 / log2(dst, src=None, bias=None, scale=None, ...)
    reciprocal(dst, src=None, ...)                   # dst = 1/src
    silu(dst, src, ...)                              # dst = src*sigmoid(src)
    add / sub / mul / fdiv(dst, src1, src2, ...)     # element-wise arithmetic
    maximum / minimum(dst, src1, src2, ...)          # element-wise max / min
    fma(dst, src, scale, bias, ...)                  # dst = src*scale + bias
    select(dst, true_value, false_value, pred)              # dst = pred ? t : f

Reductions
~~~~~~~~~~

::

    sum / max / min(dst, src, axes=-1, accum=False, ...)   # reduce over axes

Fused / compose
~~~~~~~~~~~~~~~~

::

    binary_reduce(...)   # binary op then reduce, fused
    unary_reduce(...)    # unary (with bias/scale) then reduce
    binary_chain(...)    # chain two binary ops
    reduce_negate(...)   # reduce then negate
    compose_op(...)      # frame/context manager to group primitives

Dispatch config
---------------

A call is materialized as a ``TilePrimitiveCall`` node whose fields carry
everything dispatch needs (``python/tvm/tirx/tile_primitive.py``):

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Field
     - Type
     - Meaning
   * - ``op``
     - ``tvm.Op``
     - primitive identity, e.g. ``tirx.tile.copy_async``
   * - ``args``
     - ``Array``
     - operands (regions / scalars), in the order shown above
   * - ``workspace``
     - ``Map[str, Buffer]``
     - pre-allocated scratch buffers
   * - ``config``
     - ``Map[str, Any]``
     - open-ended tuning bag (table below)
   * - ``dispatch``
     - ``Optional[str]``
     - forced variant name; ``None`` = auto-select
   * - ``scope``
     - ``ExecScope``
     - cooperation scope (default ``thread``)

``config`` has **no central schema** — each key is read only by the dispatch
variant(s) that need it (via ``config.get(...)``); a key meant for another
primitive is simply ignored. Only ``dispatch`` is generic. The keys observed in
the CUDA backend, by consumer:

.. list-table::
   :header-rows: 1
   :widths: 18 30 18 34

   * - Key
     - Used by
     - Type / values
     - Meaning
   * - ``vec_len``
     - ``copy`` / ``copy_async`` (vectorized variants)
     - int | None
     - vectorization width for the copy
   * - ``mbar``
     - ``copy_async``: ``tma`` (g2s), ``dsmem``
     - mbarrier handle
     - completion barrier
   * - ``cta_group``
     - ``copy_async``: ``tma``, ``smem→tmem``; ``gemm_async``: ``tcgen05``
     - ``1`` | ``2``
     - CTA-group; ``2`` routes completion to the cluster
   * - ``cta_mask``
     - ``copy_async``: ``tma`` (g2s)
     - int | None
     - multicast CTA mask
   * - ``cache_hint``
     - ``copy_async``: ``tma``
     - ``"evict_normal"`` | ``""``
     - L2 cache eviction hint
   * - ``oob``
     - ``copy_async``: ``tma``
     - ``"zero"`` | ``"nan"`` | None
     - out-of-bounds fill policy (``nan`` is float-only)
   * - ``use_tma_reduce``
     - ``copy_async``: ``tma`` (s2g)
     - str (e.g. ``"add"``) | None
     - TMA store-with-reduction mode
   * - ``prefetch_tensormap``
     - ``copy_async``: ``tma``
     - bool
     - prefetch the tensor map at kernel entry
   * - ``remote_cta_id``
     - ``copy_async``: ``dsmem``
     - int | PrimExpr
     - target CTA for a cross-CTA shared→shared copy
   * - ``descI``
     - ``gemm_async``: ``tcgen05``
     - uint32 | None
     - pre-encoded MMA instruction descriptor
   * - ``thread_reduce``
     - ``reduction``: ``local`` (warp scope)
     - bool
     - per-thread shuffle reduction
   * - ``rounding_mode``
     - ``elementwise``: binary ops
     - ``"rn"`` | ``"rz"`` | ...
     - FP rounding mode for the packed form

Three dispatch inputs are **implicit**, not config keys: the **execution scope**
(set by the namespace, then refined against the active thread set tracked through
control flow into ``inter``/``intra`` maps and a ``scope_kind``), the **operand
layouts** (each ``Buffer.layout``), and the **target** (the dispatch table is
keyed by its kind, e.g. ``"cuda"``).

See also
--------

- :doc:`layout` — the ``TileLayout`` model dispatch reads from operands.
- :doc:`api/tile` — exact ``Tx.tile.*`` signatures.
- :doc:`arch/tile_dispatch` — dispatch selection, extension points, and the
  target-specific variants for each primitive.
- :doc:`overview` — execution scope, tensor layout, and tile primitive dispatch
  as the three core constructs.
