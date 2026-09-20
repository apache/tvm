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

copy_async → tma_auto / tma_explicit
=====================================

The ``tma_auto`` and ``tma_explicit`` variants lower ``copy_async`` between
global and shared memory to CUDA Tensor Memory Accelerator instructions.  Both
variants:

* require the call to be in a single-thread execution scope (the caller
  normally selects the issuing thread);
* construct and cache ``cuTensorMap`` descriptors on the host with
  ``cuTensorMapEncodeTiled``;
* use the same hardware validator, descriptor cache, prefetch path, and PTX
  emitter; and
* require the source and destination regions to contain the same total number
  of bytes.  Their ranks and per-dimension shapes need not match.

The variants differ only in how the TensorMap and issue count are planned.

``tma_auto``
------------

``tma_auto`` derives the largest legal TMA box from the shared-memory iteration
order.  It is intended for ordinary full-tile copies whose layout relationship
can be proven statically:

.. code-block:: python

    Tx.tile.copy_async(
        A_smem[:, :],
        A[tile_m : tile_m + 64, tile_k : tile_k + 64],
        dispatch="tma_auto",
        mbar=mbar.ptr_to([0]),
    )

The shared slice must contain only memory iterators and must form one complete
contiguous stride chain.  The planner groups the global layout against that
chain, selects the maximum hardware-legal shared prefix for the TensorMap box,
and emits mixed-radix issue loops for the remaining dimensions.

Every raw candidate runs the same address-preserving dimension
canonicalization: address-free unit dimensions are removed where legal, and
adjacent contiguous dimensions are merged when the inner dimension is copied
in full from coordinate zero.  When an innermost contiguous, fully boxed,
coordinate-zero pair is blocked only because its merged box would exceed 256,
canonicalization may first apply one byte-preserving descriptor-unit promotion
and retry that same boundary, but only if that single promotion makes the merge
legal and a farther outer dimension exists.  This avoids skipping outward and
changing the TensorMap tiling of the contiguous inner chain.  Promotion is
otherwise a repair step used only when the canonical candidate fails a
repairable hardware rule.  It preserves byte strides, payload size,
transaction size, global base, and shared pointer, and is not used for
reductions, TF32, packed dtypes, interleave, OOB fill, gather4, or issue-driven
innermost axes.

``tma_auto`` does not accept ``gather4`` or ``src_selector`` and only accepts the
default no-OOB contract (``oob=None``).  Symbolic facts that affect layout
mapping, prefix selection, or repair must be proven.  A dynamic ``globalDim``
whose range is otherwise unknown is the exception: the runtime TensorMap
encoder checks it is in ``(0, 2^32]`` immediately before the CUDA Driver call.
Use ``tma_explicit`` for explicit OOB behavior or when other descriptor facts
are only known at runtime.

``tma_explicit``
----------------

``tma_explicit`` maps the supplied global Buffer or view directly:

* Buffer/view shape becomes ``globalDim``;
* layout strides become byte ``globalStrides``;
* region start becomes the instruction coordinates;
* region extent becomes ``boxDim``; and
* Buffer data plus ``elem_offset`` becomes the TensorMap base.

It never regroups, compresses, promotes, shrinks, or splits a copy.  One
``Tx.tile.copy_async`` call emits exactly one TMA instruction, so a caller must
explicitly tile a wider transfer:

.. code-block:: python

    for atom in Tx.unroll(8):
        Tx.tile.copy_async(
            O[:, atom * 64 : (atom + 1) * 64],
            O_smem[:, atom * 64 : (atom + 1) * 64],
            dispatch="tma_explicit",
        )

The sliced shared layout must canonicalize to a trivial box after its pointer
offset and swizzle are extracted.  Global rank and memory-layout rank must
match.  A statically illegal value is rejected; symbolic global shapes,
strides, and alignment are retained for validation by the runtime encoder.

Gather4
~~~~~~~

Gather4 is an explicit global-to-shared operation on SM100 or newer.  It
requires a two-dimensional TensorMap and exactly four absolute row
coordinates.  Public axis zero is the four-row payload, and PTX receives
coordinates in ``{column, row0, row1, row2, row3}`` order:

.. code-block:: python

    Tx.tile.copy_async(
        K_smem[0:4, :],
        K[0:1, :],
        dispatch="tma_explicit",
        mbar=mbar.ptr_to([0]),
        gather4=[row0, row1, row2, row3],
    )

Longer gathers must be written as multiple four-row calls.  Each destination
slice must be four-row box-linear and each source row must have the descriptor's
declared byte stride.

Descriptor selection
~~~~~~~~~~~~~~~~~~~~

``src_selector`` selects among alternate global Buffers or views while reusing
the main operand's region and gather coordinates:

.. code-block:: python

    Tx.tile.copy_async(
        K_smem[0:4, :],
        K_main[0:1, :],
        dispatch="tma_explicit",
        mbar=mbar.ptr_to([0]),
        gather4=[row0, row1, row2, row3],
        src_selector=[
            (use_extra, K_extra.sub[base_row:, :]),
            (use_backup, K_backup),
        ],
    )

Conditions use first-true priority and the main Buffer is the default.  Every
candidate gets its own validated and encoded TensorMap.  Candidates may have
different bases, global shapes, and strides, but must have the same descriptor
dtype, rank, box, swizzle, and transfer byte count.  Lowering selects a
pointer-typed descriptor and emits one TMA instruction; it does not select
coordinates or generate instruction-level branches.

``prefetch_tensormap=True`` deduplicates and prefetches only the main
descriptor.  Selector candidates are not prefetched automatically.

Common configuration
--------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Meaning
   * - ``mbar`` / ``mbarrier_addr``
     - ``mbar`` is the required completion barrier for global-to-shared copies.
       ``mbarrier_addr=False`` leaves it in generic-pointer form; ``True`` or
       any TIR expression requests the PTX shared-address operand form, so
       lowering converts ``mbar`` once before the instruction. The expression
       is a form-selection marker, not a replacement barrier operand. Neither
       option is valid for shared-to-global.
   * - ``cta_group`` / ``cta_mask``
     - ``cta_group`` is ``1`` or ``2``. Group 2 requires SM100+ and a
       precomputed shared mbarrier address. ``cta_mask`` is a uint16 integer or
       TIR expression and is valid only for global-to-shared multicast.
   * - ``cache_hint``
     - ``evict_normal``, ``evict_first``, ``evict_last``, or a runtime
       ``uint64`` cache-policy expression.
   * - ``prefetch_tensormap``
     - Deduplicated device-side prefetch of the main descriptor. Requires the
       ``warp_id_in_cta`` launch parameter.
   * - ``tensormap_l2_promotion``
     - ``None`` (the default, meaning ``L2::128B``), integer ``0`` through
       ``3``, ``none``, ``L2::none``, ``L2::64B``, ``L2::128B``, or
       ``L2::256B``.
   * - ``tma_dtype``
     - ``tf32`` or ``tfloat32`` for a float32 descriptor conversion.
   * - ``use_tma_reduce``
     - Shared-to-global reduction operation: ``add``, ``min``, ``max``,
       ``inc``, ``dec``, ``and``, ``or``, or ``xor``.
   * - ``oob``
     - ``tma_explicit`` global-to-shared only. ``None`` is the default zero-fill
       encoding; an explicit ``zero`` selects the same encoding, while ``nan``
       is limited to non-packed floating-point descriptors. ``tma_auto``
       requires ``oob=None`` and rejects an explicitly supplied mode.

Hardware validation
-------------------

TensorMap dimensions use CUDA API order: dimension zero is innermost and
``globalStrides`` omits its unit stride.  Layout order is reversed once when
the descriptor specification is constructed.

The shared validator checks, among other rules:

* rank 1 through 5 and target SM90 or newer;
* legal scalar descriptor dtype and conversion dtype;
* 16-byte global base alignment, raised where packed or interleaved modes
  require 32 bytes, and 64-byte TensorMap object alignment;
* ``globalDim`` in ``(0, 2^32]``;
* each explicit byte stride non-negative, aligned, and less than ``2^40``;
* ``boxDim`` in ``[1, 256]`` and element stride in ``[1, 8]``;
* unit innermost stride and ordinary inner-box bytes divisible by 16;
* legal interleave, swizzle, L2 promotion, and OOB combinations;
* swizzled inner boxes no larger than their 32-, 64-, or 128-byte atom; and
* direction, reduction, mbarrier, CTA group, coordinate count, and load-mode
  combinations.

Packed sub-byte descriptors are validated in their actual TensorMap encoding
units rather than by truncating ``bits // 8``.

For ``tma_auto``, a statically disproven rule always rejects the candidate.
Unknown planner-sensitive rules also reject it; only the dynamic
``globalDim`` range is deferred to the runtime encoder.  ``tma_explicit``
retains unknown descriptor values for runtime validation because it does not
choose among alternative boxes or issue loops.

Completion
----------

The dispatch emits the TMA instruction but no completion operation.
Global-to-shared callers initialize the barrier, call
``arrive.expect_tx`` with the transferred byte count, and wait for its phase.
Shared-to-global callers use the bulk-group commit/wait operations required by
their surrounding algorithm.
