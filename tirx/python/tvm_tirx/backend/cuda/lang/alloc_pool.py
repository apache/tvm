# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Shared-memory bump allocator for CUDA TIRx kernels."""

from __future__ import annotations

import functools
import operator

from tvm import DataType

# ---------------------------------------------------------------------------
# ir_builder helpers — imported lazily to avoid circular deps at module level
# ---------------------------------------------------------------------------

_ir = None


def _get_ir():
    global _ir
    if _ir is None:
        from tvm.tirx.script.ir_builder import ir as _mod

        _ir = _mod
    return _ir


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_POOL_UNSET = object()


def _shape_product(shape):
    return functools.reduce(operator.mul, shape, 1)


def _auto_swizzle_mode(dtype):
    """Select the default MMA swizzle mode for a shared-memory allocation."""
    from tvm.backend.cuda.layout import SwizzleMode

    del dtype
    return SwizzleMode.SWIZZLE_128B_ATOM


def _swizzle_atom_bytes(swizzle_mode):
    """Return the row width (in bytes) of one swizzle atom for *swizzle_mode*."""
    from tvm.backend.cuda.layout import SwizzleMode

    return {
        SwizzleMode.SWIZZLE_NONE: 0,
        SwizzleMode.SWIZZLE_32B_ATOM: 32,
        SwizzleMode.SWIZZLE_64B_ATOM: 64,
        SwizzleMode.SWIZZLE_128B_ATOM: 128,
    }[swizzle_mode]


def _suggest_swizzle_for_row_bytes(row_bytes):
    """Pick the largest valid swizzle mode whose atom row fits within *row_bytes*."""

    for atom_bytes, mode in (
        (128, "SWIZZLE_128B_ATOM"),
        (64, "SWIZZLE_64B_ATOM"),
        (32, "SWIZZLE_32B_ATOM"),
    ):
        if row_bytes >= atom_bytes and row_bytes % atom_bytes == 0:
            return mode
    return "SWIZZLE_NONE"


def _validate_mma_alloc_shape(shape, dtype, swizzle_mode):
    """Validate that *shape* / *dtype* / *swizzle_mode* are mutually compatible.

    ``mma_shared_layout`` tiles a swizzle atom of shape ``[8, swizzle_bytes / dtype_bytes]``
    over the last two logical dimensions of *shape*. If the row width or row count of
    the request is smaller than (or not a multiple of) the atom, the underlying
    ``Layout.tile_to`` lowers to a ``floordiv``/``floormod`` by zero and raises an
    opaque internal "Divide by zero" diagnostic from ``tile_tile_ops.cc``. Catch the
    misconfiguration here so callers see *what* is wrong and *how* to fix it.

    Validation skipped when *swizzle_mode* is ``SWIZZLE_NONE`` (no atom).
    """
    from tvm.backend.cuda.layout import SwizzleMode

    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return

    if len(shape) < 2:
        raise ValueError(
            f"alloc_tcgen05_mma_AB shape={tuple(shape)} has fewer than 2 dimensions; "
            f"swizzled MMA layouts tile over the last two dims (rows, cols). "
            f"Use swizzle_mode='none' for 1-D allocations."
        )

    # Only validate concrete int dims; symbolic dims fall through (the analyzer
    # in C++ will still ICHECK on them, but at least we don't false-positive).
    rows = shape[-2]
    cols = shape[-1]
    if not (isinstance(rows, int) and isinstance(cols, int)):
        return

    dtype_bytes = DataType(dtype).bits // 8
    if dtype_bytes == 0:
        # Sub-byte dtype (e.g. float4); ``cols`` is already in element units, so
        # use a fractional check expressed via bits.
        col_bits = cols * DataType(dtype).bits
        atom_bits = _swizzle_atom_bytes(swizzle_mode) * 8
        if col_bits < atom_bits or col_bits % atom_bits != 0:
            row_bytes = col_bits // 8 if col_bits % 8 == 0 else col_bits / 8
            atom_bytes = _swizzle_atom_bytes(swizzle_mode)
            suggestion = _suggest_swizzle_for_row_bytes(col_bits // 8 if col_bits >= 8 else 0)
            raise ValueError(
                f"alloc_tcgen05_mma_AB shape={tuple(shape)} with dtype={dtype!r} produces "
                f"{row_bytes}B rows, which is incompatible with the {atom_bytes}B "
                f"swizzle atom selected by {swizzle_mode.name}. "
                f"Use swizzle_mode=SwizzleMode.{suggestion}, or widen shape[-1] "
                f"to a multiple of "
                f"{(atom_bits + DataType(dtype).bits - 1) // DataType(dtype).bits} elements."
            )
    else:
        row_bytes = cols * dtype_bytes
        atom_bytes = _swizzle_atom_bytes(swizzle_mode)
        if row_bytes < atom_bytes or row_bytes % atom_bytes != 0:
            suggestion = _suggest_swizzle_for_row_bytes(row_bytes)
            min_cols = atom_bytes // dtype_bytes
            raise ValueError(
                f"alloc_tcgen05_mma_AB shape={tuple(shape)} with dtype={dtype!r} produces "
                f"{row_bytes}B rows, which is incompatible with the {atom_bytes}B "
                f"swizzle atom selected by {swizzle_mode.name}. "
                f"Use swizzle_mode=SwizzleMode.{suggestion}, or widen shape[-1] "
                f"to a multiple of {min_cols} elements (>= {atom_bytes}B at {dtype})."
            )

    # Atom rows is always 8 (see ``mma_atom_shape`` in backend/cuda/layout.py).
    atom_rows = 8
    if rows < atom_rows or rows % atom_rows != 0:
        raise ValueError(
            f"alloc_tcgen05_mma_AB shape={tuple(shape)} has shape[-2]={rows}, but the "
            f"{swizzle_mode.name} atom requires shape[-2] to be a positive "
            f"multiple of {atom_rows}. Use swizzle_mode='none', or widen shape[-2] "
            f"to a multiple of {atom_rows}."
        )


def _meta_class(cls):
    """Apply @meta_class decorator from ir_builder."""
    return _get_ir().meta_class(cls)


# ---------------------------------------------------------------------------
# SMEMPool
# ---------------------------------------------------------------------------


@_meta_class
class SMEMPool:
    """Bump allocator over a contiguous shared memory region.

    Parameters
    ----------
    ptr : Var or None, optional
        If omitted, an ``alloc_tensor(ty_args=[Tensor([0], "uint8", scope="shared.dyn")])`` is
        created automatically and ``commit()`` must be called after all
        allocations to emit the size annotation.
        If a ``Var`` is provided, the caller manages the backing buffer and
        ``commit()`` is a no-op.
    """

    def __init__(self, ptr=_POOL_UNSET):
        ir = _get_ir()
        if ptr is _POOL_UNSET:
            self.buf = ir._bind_tensor(
                ir.alloc_tensor(ty_args=[ir.Tensor([0], "uint8", scope="shared.dyn")])
            )
            self.ptr = self.buf.data
            self._owns_buffer = True
        else:
            self.buf = None
            self.ptr = ptr
            self._owns_buffer = False
        self.offset = 0
        self.max_offset = 0

    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=0,
        layout="default",
    ):
        ir = _get_ir()
        if align > 0:
            self.offset = (self.offset + align - 1) // align * align
        res = ir._bind_tensor(
            ir.decl_tensor(
                self.ptr,
                ty_args=[
                    ir.Tensor(
                        shape,
                        dtype,
                        strides=strides,
                        byte_offset=self.offset,
                        scope=scope,
                        align=align,
                        layout=layout,
                    )
                ],
            )
        )
        # Advance in bits then round up to bytes so sub-byte dtypes (e.g.
        # float4_e2m1fn = 4 bits) still bump the cursor instead of leaving it
        # at 0 (bits // 8) and silently overlapping the next allocation.
        self.offset += (_shape_product(shape) * DataType(dtype).bits + 7) // 8
        if self._owns_buffer:
            self.max_offset = max(self.max_offset, self.offset)
        return res

    def alloc_tcgen05_mma_AB(self, shape, dtype="float16", swizzle_mode="auto", align=1024):
        """Allocate MMA-compatible shared memory with an inferred swizzle layout."""
        from tvm.backend.cuda.layout import (
            SwizzleMode,
            mma_shared_layout,
        )

        if isinstance(swizzle_mode, str):
            if swizzle_mode == "auto":
                swizzle_mode = _auto_swizzle_mode(dtype)
            elif swizzle_mode == "none":
                swizzle_mode = SwizzleMode.SWIZZLE_NONE
            else:
                raise ValueError(
                    f"Unsupported swizzle_mode={swizzle_mode!r}; expected 'auto', 'none', "
                    "or SwizzleMode"
                )
        _validate_mma_alloc_shape(shape, dtype, swizzle_mode)
        layout = mma_shared_layout(dtype, swizzle_mode, shape)
        return self.alloc(shape, dtype, align=align, layout=layout)

    def move_base_to(self, offset):
        self.offset = offset
        if self._owns_buffer:
            self.max_offset = max(self.max_offset, self.offset)

    def commit(self, size=None):
        """Record the pool resource requirement on the enclosing kernel entry.

        Must be called after all ``alloc()`` / ``move_base_to()`` calls.

        Parameters
        ----------
        size : int, optional
            Explicit shared memory size in bytes.  When *None* (the default),
            the high-water mark ``max_offset`` tracked by the allocator is used.
        """
        if not self._owns_buffer:
            return
        resolved = size if size is not None else self.max_offset
        assert resolved >= self.max_offset, (
            f"Specified smem size ({resolved}) is smaller than "
            f"the pool high-water mark ({self.max_offset})"
        )
        from tvm.tirx.script.ir_builder import _ffi_api

        _ffi_api.CudaSharedMemoryRequirement(resolved)
