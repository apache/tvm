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
"""SMEM and TMEM bump-allocator pools for TIRX kernels."""

from __future__ import annotations

import functools
import operator

from tvm import DataType
from tvm.tirx.layout import S, TCol, TileLayout, TLane

# ---------------------------------------------------------------------------
# ir_builder helpers — imported lazily to avoid circular deps at module level
# ---------------------------------------------------------------------------

_ir = None


def _get_ir():
    global _ir
    if _ir is None:
        from tvm.tirx.script.builder import ir as _mod

        _ir = _mod
    return _ir


def _get_frame():
    from tvm.tirx.script.builder import frame

    return frame


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_POOL_UNSET = object()


def _default_tmem_layout(rows, cols):
    return TileLayout(S[(rows, cols) : (1 @ TLane, 1 @ TCol)])


def _emit_stmt(expr):
    ir = _get_ir()
    ir.add_to_parent(ir.evaluate(expr))


def _shape_product(shape):
    return functools.reduce(operator.mul, shape, 1)


def _auto_swizzle_mode(dtype):
    """Select the default MMA swizzle mode for a shared-memory allocation."""
    from tvm.tirx.operator.tile_primitive.cuda.tma_utils import SwizzleMode

    del dtype
    return SwizzleMode.SWIZZLE_128B_ATOM


def _swizzle_atom_bytes(swizzle_mode):
    """Return the row width (in bytes) of one swizzle atom for *swizzle_mode*."""
    from tvm.tirx.operator.tile_primitive.cuda.tma_utils import SwizzleMode

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
    from tvm.tirx.operator.tile_primitive.cuda.tma_utils import SwizzleMode

    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return

    if len(shape) < 2:
        raise ValueError(
            f"alloc_mma shape={tuple(shape)} has fewer than 2 dimensions; "
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
                f"alloc_mma shape={tuple(shape)} with dtype={dtype!r} produces "
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
                f"alloc_mma shape={tuple(shape)} with dtype={dtype!r} produces "
                f"{row_bytes}B rows, which is incompatible with the {atom_bytes}B "
                f"swizzle atom selected by {swizzle_mode.name}. "
                f"Use swizzle_mode=SwizzleMode.{suggestion}, or widen shape[-1] "
                f"to a multiple of {min_cols} elements (>= {atom_bytes}B at {dtype})."
            )

    # Atom rows is always 8 (see ``mma_atom_shape`` in tma_utils.py).
    atom_rows = 8
    if rows < atom_rows or rows % atom_rows != 0:
        raise ValueError(
            f"alloc_mma shape={tuple(shape)} has shape[-2]={rows}, but the "
            f"{swizzle_mode.name} atom requires shape[-2] to be a positive "
            f"multiple of {atom_rows}. Use swizzle_mode='none', or widen shape[-2] "
            f"to a multiple of {atom_rows}."
        )


# ---------------------------------------------------------------------------
# TMEMRegion
# ---------------------------------------------------------------------------


def _meta_class(cls):
    """Apply @meta_class decorator from ir_builder."""
    return _get_ir().meta_class(cls)


@_meta_class
class TMEMRegion:
    """Parse-time staged view over a TMEM buffer.

    Parameters
    ----------
    buf : Buffer
        The underlying TMEM buffer (e.g. f32 or f16 view).
    col_start : int
        First column of stage 0 in *buf*'s column space.
    width : int
        Number of columns per stage.
    stages : int
        Number of pipeline stages (default 1).
    stride : int or None
        Column distance between consecutive stages.  When *None* (default),
        equals *width* (stages are packed back-to-back).
    """

    def __init__(self, buf, col_start, width, stages=1, stride=None):
        self.buf = buf
        self.col_start = col_start
        self.width = width
        self.stages = stages
        self.stride = width if stride is None else stride

    def _stage_base(self, stage):
        return self.col_start + stage * self.stride

    def __getitem__(self, item):
        if isinstance(item, tuple):
            assert len(item) == 2, "TMEMRegion expects region[stage] or region[stage, start:stop]"
            stage, col_slice = item
            assert isinstance(col_slice, slice), "TMEMRegion tuple indexing requires a slice"
            base = self._stage_base(stage)
            start = 0 if col_slice.start is None else col_slice.start
            stop = self.width if col_slice.stop is None else col_slice.stop
            return self.buf[:, base + start : base + stop : col_slice.step]
        base = self._stage_base(item)
        return self.buf[:, base : base + self.width]


# ---------------------------------------------------------------------------
# TMEMPool
# ---------------------------------------------------------------------------


@_meta_class
class TMEMPool:
    """Bump allocator over TMEM columns."""

    def __init__(
        self,
        pool,
        total_cols=512,
        *,
        cta_group=1,
        alloc_warp=0,
        dealloc_warp=None,
        tmem_addr=None,
        sync_after_alloc=True,
    ):
        # tcgen05 alloc/dealloc are warp-uniform PTX instructions: every lane
        # in the chosen warp must participate, and exactly one warp in the
        # CTA must execute them. The pool emits its own
        # ``if thread_rank() // 32 == target_warp: with Tx.warp(): tcgen05.alloc(...)``
        # guard, using ``Tx.cuda.thread_rank()`` (cooperative_groups thread
        # rank) so callers don't have to declare the CTA's thread layout.
        self.pool = pool
        self.total_cols = total_cols
        self.cta_group = cta_group
        self.alloc_warp = alloc_warp
        self.dealloc_warp = alloc_warp if dealloc_warp is None else dealloc_warp
        self.sync_after_alloc = sync_after_alloc
        self.offset = 0
        self.max_offset = 0
        self._committed = False
        self._addr_buf = pool.alloc([1], "uint32", align=4) if tmem_addr is None else tmem_addr

    def _addr_slot(self):
        try:
            return self._addr_buf[0]
        except TypeError:
            return self._addr_buf

    @property
    def addr(self):
        return self._addr_slot()

    def _emit_warp_guard(self, Tx, target_warp, emit):
        with Tx.If(Tx.cuda.thread_rank() // 32 == target_warp):
            with Tx.Then():
                with Tx.warp():
                    emit()

    def _resolve_cols(self, shape, dtype, cols, layout=None):
        if cols is not None:
            return cols
        bits = DataType(dtype).bits
        if layout is not None:
            # span("TCol") is in *element* (buffer dtype) units; one TMEM cell
            # holds 32 bits regardless of the element type.
            tcol_elems = int(layout.span("TCol"))
            tcol_bits = tcol_elems * bits
            assert tcol_bits % 32 == 0, (
                f"layout TCol span={tcol_elems} elems x {bits}b is not 32-bit aligned"
            )
            return tcol_bits // 32
        assert len(shape) == 2, "TMEMPool.alloc() requires cols= for non-2D TMEM buffers"
        total_bits = _shape_product(shape) * bits
        rows = shape[0]
        assert total_bits % (32 * rows) == 0, (
            f"Cannot infer TMEM columns from shape={shape}, dtype={dtype!r}; "
            "please pass cols= explicitly"
        )
        return total_bits // (32 * rows)

    def alloc(self, shape, dtype="float32", *, layout=None, cols=None):
        ir = _get_ir()
        cols = self._resolve_cols(shape, dtype, cols, layout)
        col_start = self.offset
        col_end = col_start + cols
        assert col_end <= self.total_cols, f"TMEM overflow: {col_end} > {self.total_cols}"
        if layout is None:
            assert len(shape) == 2, "TMEMPool.alloc() requires layout= for non-2D TMEM buffers"
            layout = _default_tmem_layout(shape[0], shape[1])
        res = ir.decl_buffer(shape, dtype, scope="tmem", allocated_addr=col_start, layout=layout)
        self.offset = col_end
        self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset
        return res

    def alloc_sf(self, shape, dtype, *, sf_per_mma, sf_reuse=1):
        """Allocate a tcgen05 block-scaled SF TMEM buffer with an inferred layout.

        ``shape`` last two dims are ``(rows, SF_K * sf_reuse)`` (the last dim is
        what gemm dispatch iterates over). When ``shape`` has 3 dims, the first
        is treated as a pipe-depth outer.
        """
        from tvm.tirx.operator.tile_primitive.cuda.gemm_async.tcgen05 import sf_tmem_layout

        if len(shape) == 2:
            pipe_depth, rows, last = None, shape[0], shape[1]
        elif len(shape) == 3:
            pipe_depth, rows, last = shape[0], shape[1], shape[2]
        else:
            raise ValueError(
                f"alloc_sf expects 2D (rows, SF_K*sf_reuse) or 3D "
                f"(pipe_depth, rows, SF_K*sf_reuse); got shape={shape}"
            )
        assert last % sf_reuse == 0, (
            f"alloc_sf: shape last dim {last} must be divisible by sf_reuse={sf_reuse}"
        )
        SF_K = last // sf_reuse
        layout = sf_tmem_layout(
            rows=rows, SF_K=SF_K, sf_per_mma=sf_per_mma, sf_reuse=sf_reuse, pipe_depth=pipe_depth
        )
        return self.alloc(shape, dtype, layout=layout)

    def move_base_to(self, col):
        self.offset = col
        self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset

    def region(self, buf, col_start, width, stages=1, stride=None):
        """Create a staged region view over *buf*.

        Parameters
        ----------
        buf : Buffer
            TMEM buffer returned by ``alloc()``.
        col_start : int
            First column of stage 0 (in *buf*'s column units).
        width : int
            Columns per stage.
        stages : int
            Pipeline depth.
        stride : int or None
            Column distance between consecutive stages (default = *width*).
        """
        return TMEMRegion(buf, col_start, width, stages, stride)

    def commit(self):
        assert not self._committed, "TMEMPool.commit() can only be called once"
        from tvm.script import tirx as Tx

        def emit_alloc():
            _emit_stmt(
                Tx.ptx.tcgen05.alloc(
                    Tx.address_of(self.addr), n_cols=self.total_cols, cta_group=self.cta_group
                )
            )
            if self.sync_after_alloc:
                _emit_stmt(Tx.cuda.warp_sync())

        self._emit_warp_guard(Tx, self.alloc_warp, emit_alloc)
        self._committed = True

    def dealloc(self):
        from tvm.script import tirx as Tx

        def emit_dealloc():
            _emit_stmt(Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=self.cta_group))
            _emit_stmt(
                Tx.ptx.tcgen05.dealloc(self.addr, n_cols=self.total_cols, cta_group=self.cta_group)
            )

        self._emit_warp_guard(Tx, self.dealloc_warp, emit_dealloc)


# ---------------------------------------------------------------------------
# SMEMPool
# ---------------------------------------------------------------------------


@_meta_class
class SMEMPool:
    """Bump allocator over a contiguous shared memory region.

    Parameters
    ----------
    ptr : Var or None, optional
        If omitted, an ``alloc_buffer([0], "uint8", scope="shared.dyn")`` is
        created automatically and ``commit()`` must be called after all
        allocations to emit the size annotation.
        If a ``Var`` is provided, the caller manages the backing buffer and
        ``commit()`` is a no-op.
    """

    def __init__(self, ptr=_POOL_UNSET):
        ir = _get_ir()
        if ptr is _POOL_UNSET:
            self.buf = ir.alloc_buffer([0], "uint8", scope="shared.dyn")
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
        scope="global",
        align=0,
        buffer_type="",
        axis_separators=None,
        layout="default",
    ):
        ir = _get_ir()
        if align > 0:
            self.offset = (self.offset + align - 1) // align * align
        res = ir.decl_buffer(
            shape,
            dtype,
            self.ptr,
            strides,
            None,
            self.offset,
            scope,
            align,
            0,
            buffer_type,
            axis_separators,
            layout,
        )
        self.offset += functools.reduce(lambda x, y: x * y, shape) * (DataType(dtype).bits // 8)
        if self._owns_buffer:
            self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset
        return res

    def alloc_mma(self, shape, dtype="float16", swizzle_mode="auto", align=1024):
        """Allocate MMA-compatible shared memory with an inferred swizzle layout."""
        from tvm.tirx.operator.tile_primitive.cuda.tma_utils import (
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
            self.max_offset = self.offset if self.offset > self.max_offset else self.max_offset

    def commit(self, size=None):
        """Emit pool size annotation into the IR.

        Must be called after all ``alloc()`` / ``move_base_to()`` calls.

        Parameters
        ----------
        size : int, optional
            Explicit shared memory size in bytes.  When *None* (the default),
            the high-water mark ``max_offset`` tracked by the allocator is used.
        """
        if not self._owns_buffer:
            return
        ir = _get_ir()
        frame_mod = _get_frame()
        resolved = size if size is not None else self.max_offset
        assert resolved >= self.max_offset, (
            f"Specified smem size ({resolved}) is smaller than "
            f"the pool high-water mark ({self.max_offset})"
        )
        attr_frame = ir.attr(self.ptr, "tirx.pool_max_bytes", resolved)
        if isinstance(attr_frame, frame_mod.AttrFrame):
            from functools import partial

            attr_frame.add_callback(partial(attr_frame.__exit__, None, None, None))
            attr_frame.__enter__()
