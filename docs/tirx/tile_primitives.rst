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

The C++ registry currently defines 31 operation names.  This programming guide
groups them by purpose; the :doc:`API reference <api/tile>` is the single source
for their exact Python signatures.

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Group
     - Operations
     - Purpose
   * - Data movement
     - ``copy``, ``copy_async``, ``permute_layout``
     - synchronous and asynchronous transfer, or rearrangement between layouts
   * - Matrix multiply
     - ``gemm``, ``gemm_async``
     - synchronous register MMA or asynchronous/backend-specific MMA, including
       block scaling where supported
   * - Initialization
     - ``fill``, ``memset``, ``zero``
     - initialize a tile or region
   * - Unary and cast
     - ``cast``, ``sqrt``, ``exp``, ``exp2``, ``log2``, ``reciprocal``, ``silu``
     - per-element conversion or unary math
   * - Binary and ternary
     - ``add``, ``sub``, ``mul``, ``fdiv``, ``maximum``, ``minimum``, ``fma``,
       ``select``
     - per-element arithmetic and selection
   * - Reductions
     - ``sum``, ``max``, ``min``
     - reduce selected axes, optionally accumulating into the destination
   * - Fused and composed
     - ``binary_reduce``, ``unary_reduce``, ``binary_chain``, ``reduce_negate``,
       ``compose_op``
     - combine several primitive operations for backends that dispatch them as
       one unit

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

``config`` has **no central schema**.  Each dispatch implementation defines and
validates the keys it consumes.  Some implementations ignore unrelated keys,
while others (notably the TMA variants) reject unknown keys.  Only ``dispatch``
is interpreted generically by the dispatcher.  The current target
implementations consume the following keys:

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Primitive / variant
     - Keys consumed
     - Notes
   * - ``copy``: ``vec_16b`` / ``vec_32b`` / ``vec_64b`` /
       ``vec_128b`` / ``vec_256b``
     - ``cache``, ``l1_evict``, ``l2_evict``, ``prefetch_size``
     - Explicit, thread-scope fixed-width copies.  Cache controls require a
       global-memory source.
   * - ``copy_async``: ``ldgsts``
     - ``prefetch_size``, ``predicate``, ``fill_mode``, ``direct``
     - ``direct=True`` requires thread scope and an exact 4-, 8-, or 16-byte
       region.
   * - ``copy_async``: ``tma_auto`` / ``tma_explicit``
     - ``cache_hint``, ``cta_group``, ``cta_mask``, ``mbar``,
       ``mbarrier_addr``, ``oob``, ``prefetch_tensormap``,
       ``tensormap_l2_promotion``, ``tma_dtype``, ``use_tma_reduce``
     - ``tma_explicit`` additionally accepts ``gather4`` and ``src_selector``.
       Direction-specific constraints are documented in
       :doc:`tile_primitives/copy_async/tma`.
   * - ``copy_async``: ``dsmem``
     - ``remote_cta_id``, ``mbar``
     - Both are required: the destination CTA id and its completion barrier.
   * - ``copy_async``: ``smem->tmem``
     - ``shape``, ``multicast``, ``cta_group``
     - ``decompress`` is detected but currently rejected as unsupported.
   * - ``gemm_async``: ``tcgen05``
     - ``cta_group``, ``mma_m``, ``mma_n``, ``descI``, ``is_AB_tf32``,
       ``weight_stationary``, ``smem_desc``
     - ``mma_m`` and ``mma_n`` are an all-or-nothing pair.  ``descI`` is
       accepted only by the block-scaled path.
   * - ``sum`` / ``max`` / ``min``: ``local``
     - ``thread_reduce``
     - Enables the per-thread shuffle-reduction mode where supported.
   * - binary elementwise: ``reg`` / ``smem``
     - ``rounding_mode``
     - Passed to packed floating-point forms that expose a rounding mode.
   * - Trainium tile implementations
     - ``max_inst_size``
     - Instruction-size limit used by copy, elementwise, select, reduction,
       and composed-op implementations.

Vector widths selected by ``vec_auto`` and ``ldgsts`` are derived internally
from dtype, alignment, layout, and execution scope; ``vec_len`` is not a user
configuration key.

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
