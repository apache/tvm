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

"""Op-agnostic elementwise schema.

All elementwise ops (unary / binary / cast / fma) live in one ``ALL_OPS``
table. Each entry is an ``OpSpec`` with a ``parse(op_call) -> Plan`` and a
``compute(src_vals, extras, dst_dtype) -> raw_value``. Schedules iterate
``Plan.srcs`` without knowing the arity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tvm.ir.expr import PrimExpr
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, TilePrimitiveCall
from tvm.tirx.expr import FloatImm


@dataclass
class SrcSpec:
    """One operand of an elementwise op.

    Either a buffer-region (per-element load) or a scalar PrimExpr.
    ``index_fn``, if given, computes per-element indices for broadcasting
    cases (e.g. binary src2 with extent=1 dims):
        index_fn(dst_indices, dst_start, dst_extent, src_start, src_extent) -> list[Expr]
    Default is the standard ``get_indices`` over the src's own region.
    """

    buf_region: BufferRegion | None = None
    scalar: PrimExpr | None = None
    index_fn: Callable | None = None

    @property
    def is_scalar(self) -> bool:
        return self.scalar is not None

    @property
    def buffer(self):
        return self.buf_region.buffer if self.buf_region is not None else None


@dataclass
class Plan:
    """Parsed elementwise op ready for a schedule to consume."""

    dst: BufferRegion
    srcs: list[SrcSpec]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpSpec:
    """Metadata for an elementwise op.

    Schedules consult ``vec_emit_factory`` first: given (op_call, plan, sctx, vec_len)
    it may return a fully-built PrimFunc using a PTX/CUDA intrinsic (e.g.
    ``add.<rm>.ftz.f32x2``). If it returns None, the schedule falls back to a
    scalar ``Tx.vectorized`` loop driven by ``compute``.
    """

    name: str  # TIRx op short name, e.g. "exp" / "add" / "fma" / "cast"
    parse: Callable[[TilePrimitiveCall], tuple[Plan | None, str | None]]
    compute: Callable[[list, dict, str], Any]
    # extras dtype checker, optional: (extras, compute_dtype) -> (ok, msg)
    check_extras: Callable | None = None
    # Optional vector-intrinsic emit factory: (op_call, plan, sctx, vec_len)
    # -> PrimFunc | None. Called by each schedule before scalar emit. The
    # factory is responsible for ALL applicability checks (dtype, vec_len,
    # sm version, broadcasting, scope) and must return None if the intrinsic
    # cannot be used — the schedule will then emit the scalar fallback.
    vec_emit_factory: Callable | None = None


# -----------------------------------------------------------------------------
# Parse helpers — one per op family. They produce Plan/None+msg without touching
# scope/layout (those checks live in the schedule validators).
# -----------------------------------------------------------------------------
def _parse_unary(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    """Parse Tx.<unary>(dst, src[, bias, scale]).

    src can be a BufferRegion or a PrimExpr (scalar fill).
    bias can be a BufferRegion (per-element) or FloatImm (constant) or None.
    scale is FloatImm or None (defaults to 1.0).

    Produces:
      Plan(dst, srcs=[SrcSpec(main src), optional SrcSpec(bias_buf)],
           extras={scale: ..., bias_const: ... or None})
    """
    _dst: BufferRegion = op.args[0]
    _src = op.args[1]
    _bias = op.args[2] if len(op.args) > 2 else None
    _scale = op.args[3] if len(op.args) > 2 else None

    srcs: list[SrcSpec] = []
    if isinstance(_src, BufferRegion):
        srcs.append(SrcSpec(buf_region=_src))
    elif isinstance(_src, PrimExpr):
        srcs.append(SrcSpec(scalar=_src))
    else:
        return None, f"unsupported src type {type(_src).__name__}"

    extras: dict[str, Any] = {
        "scale": _scale,
        "bias_const": _bias if isinstance(_bias, FloatImm) else None,
    }
    if isinstance(_bias, BufferRegion):
        srcs.append(SrcSpec(buf_region=_bias))
        extras["has_bias_buf"] = True
    else:
        extras["has_bias_buf"] = False
    return Plan(dst=_dst, srcs=srcs, extras=extras), None


def _check_unary_extras(extras: dict, compute_dtype: str) -> tuple[bool, str | None]:
    scale = extras.get("scale")
    if scale is not None and scale.dtype != compute_dtype:
        return False, f"scale dtype {scale.dtype} != compute dtype {compute_dtype}"
    bias_const = extras.get("bias_const")
    if bias_const is not None and bias_const.dtype != compute_dtype:
        return False, f"bias_const dtype {bias_const.dtype} != compute dtype {compute_dtype}"
    return True, None


def _unary_with_bias_scale(raw_op):
    """Wrap a unary raw op (e.g. Tx.exp) into a compute that applies bias/scale.

    raw_op: lambda v: <expr>   (applied AFTER scale+bias if any)
    Returns: lambda src_vals, extras, dt: <expr>
    """

    def compute(src_vals, extras, dt):
        x = src_vals[0]
        scale = extras.get("scale")
        if scale is not None:
            x = x * scale
        if extras.get("has_bias_buf"):
            x = x + src_vals[1]
        elif extras.get("bias_const") is not None:
            x = x + extras["bias_const"]
        return raw_op(x)

    return compute


# Compute callbacks for unary ops.
def _compute_zero(src_vals, extras, dt):
    return 0.0


def _compute_fill(src_vals, extras, dt):
    return src_vals[0]


def _compute_reciprocal(src_vals, extras, dt):
    x = src_vals[0]
    return Tx.FloatImm(x.dtype, 1.0) / x


def _compute_silu(src_vals, extras, dt):
    # NOTE: silu doesn't apply bias/scale in the legacy table — preserve that.
    x = src_vals[0]
    return x / (Tx.FloatImm(x.dtype, 1.0) + Tx.exp(Tx.FloatImm(x.dtype, 0.0) - x))


# -----------------------------------------------------------------------------
# Binary: Tx.<op>(dst, src1, src2) with optional broadcasting + constant rhs.
# -----------------------------------------------------------------------------
def _binary_broadcast_index_fn(dst_indices, dst_start, dst_extent, src_start, src_extent):
    """Compute src2 indices when src2 has extent=1 broadcasting dims."""
    len_diff = len(dst_extent) - len(src_extent)
    return [
        (
            (dst_indices[i + len_diff] - dst_start[i + len_diff]) + src_start[i]
            if src_extent[i] != 1
            else src_start[i]
        )
        for i in range(len(src_extent))
    ]


def _binary_is_commutative(op_name: str) -> bool:
    return op_name in ("add", "mul")


def _parse_binary_for(op_name: str):
    """Build a parse(op_call) -> (Plan, msg) for a specific binary op name."""

    def parse(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
        _dst: BufferRegion = op.args[0]
        _src1 = op.args[1]
        _src2 = op.args[2]

        # Reject both-constant (degenerate).
        s1_scalar = not isinstance(_src1, BufferRegion)
        s2_scalar = not isinstance(_src2, BufferRegion)
        if s1_scalar and s2_scalar:
            return None, "both inputs are constants"

        # Move constant to rhs (commute if allowed; else reject).
        if s1_scalar:
            if not _binary_is_commutative(op_name):
                return None, f"non-commutative op {op_name} cannot have constant lhs"
            _src1, _src2 = _src2, _src1
            s1_scalar, s2_scalar = False, True

        # If rhs is a smaller buffer (broadcast), and op is commutative, optionally swap.
        if not s2_scalar:
            import functools
            import operator

            s1_n = functools.reduce(operator.mul, [r.extent for r in _src1.region], 1)
            s2_n = functools.reduce(operator.mul, [r.extent for r in _src2.region], 1)
            if s1_n < s2_n:
                if not _binary_is_commutative(op_name):
                    return None, f"non-commutative op {op_name} cannot swap to broadcast"
                _src1, _src2 = _src2, _src1

        srcs: list[SrcSpec] = [SrcSpec(buf_region=_src1)]
        if s2_scalar:
            srcs.append(SrcSpec(scalar=_src2))
        else:
            # If src2 is broadcasting (any extent=1 dims smaller than src1's), attach
            # a broadcast index_fn that derives src2 indices from dst's.
            s1_ext = [r.extent for r in _src1.region]
            s2_ext = [r.extent for r in _src2.region]
            needs_broadcast = (len(s2_ext) != len(s1_ext)) or (
                any(e != 1 for e in s2_ext)
                and (
                    any(
                        int(s2_ext[i]) == 1 and int(s1_ext[-len(s2_ext) + i]) != 1
                        for i in range(len(s2_ext))
                    )
                )
            )
            srcs.append(
                SrcSpec(
                    buf_region=_src2,
                    index_fn=_binary_broadcast_index_fn if needs_broadcast else None,
                )
            )
        extras: dict[str, Any] = {}
        rm = op.config.get("rounding_mode", None)
        if rm is not None:
            extras["rounding_mode"] = rm
        return Plan(dst=_dst, srcs=srcs, extras=extras), None

    return parse


# Compute callbacks for binary ops.
def _compute_add(src_vals, extras, dt):
    return src_vals[0] + src_vals[1]


def _compute_sub(src_vals, extras, dt):
    return src_vals[0] - src_vals[1]


def _compute_mul(src_vals, extras, dt):
    return src_vals[0] * src_vals[1]


def _compute_fdiv(src_vals, extras, dt):
    return src_vals[0] / src_vals[1]


# -----------------------------------------------------------------------------
# Packed f32x2 vector intrinsic emit (sm_100+, f32, vec_len=2) for add/sub/mul.
# This carries rounding_mode (PTX attr) that scalar `a+b` cannot express.
#
# The underlying PTX ops are ``Tx.ptx.{add,sub,mul}_f32x2(d, a, b, ...)`` which
# take packed-as-u64 register operands. We provide local adapters that accept
# (4 scalar inputs + d_addr + rounding_mode) so the call sites here read more
# directly; the adapters pack the scalars via ``Tx.cuda.make_float2``.
# -----------------------------------------------------------------------------


def _f32x2_adapter(op_name):
    """Return a callable with the old (a1, a2, b1, b2, d, rounding_mode=) shape
    that internally invokes the new DPS ``Tx.ptx.{op}_f32x2`` API."""
    op_func = getattr(Tx.ptx, f"{op_name}_f32x2")

    def _emit(a1, a2, b1, b2, d, rounding_mode):
        return op_func(
            d,
            Tx.cuda.make_float2(a1, a2),
            Tx.cuda.make_float2(b1, b2),
            rounding=rounding_mode,
            ftz=True,
        )

    return _emit


_PACKED_F32X2_PTX = {
    "add": _f32x2_adapter("add"),
    "sub": _f32x2_adapter("sub"),
    "mul": _f32x2_adapter("mul"),
}


def _fma_f32x2_adapter(a1, a2, b1, b2, c1, c2, d, rounding_mode):
    """Adapter: (6 scalar inputs + d_addr + rounding_mode) → new DPS API."""
    return Tx.ptx.fma_f32x2(
        d,
        Tx.cuda.make_float2(a1, a2),
        Tx.cuda.make_float2(b1, b2),
        Tx.cuda.make_float2(c1, c2),
        rounding=rounding_mode,
        ftz=True,
    )


def _make_binary_packed_f32x2_factory(op_name: str):
    """Build a vec_emit_factory for binary add/sub/mul on f32 vec_len=2."""

    op_func_f32x2 = _PACKED_F32X2_PTX[op_name]

    def factory(op_call, plan, sctx, vec_len):
        # Importing here to avoid module-level cycles with cuda.common.
        from ..common import get_st_extent, sm_version_ok
        from ..layout_utils import get_local_region

        # ---- applicability -----------------------------------------------
        # NOTE: this emit always processes 2 elements per chunk via the PTX
        # packed-f32x2 intrinsic, regardless of the schedule's vec_len choice
        # (codegen does not auto-fuse vec_len=4 + 4 scalar adds into packed).
        if plan.dst.buffer.dtype != "float32":
            return None
        if not sm_version_ok(op_call, sctx, min_version=100)[0]:
            return None
        # Two emit modes:
        #   thread-scope : flat per-thread buffers; index buf[fused] directly
        #   wg/warp scope: collective tile with layout; need buf.local(*shape)
        #                  to get the per-thread reg slice, then index that.
        if sctx.is_thread:
            use_view = False
        elif sctx.scope_kind in ("warp", "warpgroup", "cta"):
            use_view = True
            # All buffer srcs + dst must have non-trivial layout for view.
            if plan.dst.buffer.layout is None or plan.dst.buffer.layout.is_trivial():
                return None
            for s in plan.srcs:
                if not s.is_scalar and (
                    s.buf_region.buffer.layout is None or s.buf_region.buffer.layout.is_trivial()
                ):
                    return None
        else:
            return None
        # All buffer srcs must be f32; const srcs must be f32 too.
        for s in plan.srcs:
            if s.is_scalar:
                if s.scalar.dtype != "float32":
                    return None
            else:
                if s.buf_region.buffer.dtype != "float32":
                    return None
                if s.index_fn is not None:
                    # Broadcasting not supported by this packed intrinsic.
                    return None
        if len(plan.srcs) != 2:
            return None

        dst = plan.dst.buffer
        dst_st_raw, dst_ext_raw = get_st_extent(plan.dst)
        s1, s2 = plan.srcs[0], plan.srcs[1]
        rm = plan.extras.get("rounding_mode", "rz")
        s1_buf = None if s1.is_scalar else s1.buf_region.buffer
        s2_buf = None if s2.is_scalar else s2.buf_region.buffer
        s1_scalar_val = s1.scalar if s1.is_scalar else None
        s2_scalar_val = s2.scalar if s2.is_scalar else None
        if s1.is_scalar and s2.is_scalar:
            return None  # degenerate, parse already rejects this

        import functools
        import operator

        from ..common import get_indices

        if not use_view:
            # ---- thread-scope: index raw buffer directly -------------------
            total = functools.reduce(operator.mul, dst_ext_raw, 1)
            try:
                if int(total) % 2 != 0:
                    return None
            except (TypeError, ValueError):
                return None
            n_chunks = int(total) // 2
            dst_st, dst_ext = dst_st_raw, dst_ext_raw
            s1_st, s1_ext = (None, None) if s1.is_scalar else get_st_extent(s1.buf_region)
            s2_st, s2_ext = (None, None) if s2.is_scalar else get_st_extent(s2.buf_region)

            if not s1.is_scalar and s2.is_scalar:

                @Tx.prim_func(check_well_formed=False)
                def impl():
                    for s in Tx.serial(0, n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        s1_idx_a = Tx.meta_var(get_indices(2 * s, s1_st, s1_ext))
                        s1_idx_b = Tx.meta_var(get_indices(2 * s + 1, s1_st, s1_ext))
                        op_func_f32x2(
                            s1_buf[tuple(s1_idx_a)],
                            s1_buf[tuple(s1_idx_b)],
                            s2_scalar_val,
                            s2_scalar_val,
                            Tx.address_of(dst[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

                return impl

            if s1.is_scalar and not s2.is_scalar:

                @Tx.prim_func(check_well_formed=False)
                def impl():
                    for s in Tx.serial(0, n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        s2_idx_a = Tx.meta_var(get_indices(2 * s, s2_st, s2_ext))
                        s2_idx_b = Tx.meta_var(get_indices(2 * s + 1, s2_st, s2_ext))
                        op_func_f32x2(
                            s1_scalar_val,
                            s1_scalar_val,
                            s2_buf[tuple(s2_idx_a)],
                            s2_buf[tuple(s2_idx_b)],
                            Tx.address_of(dst[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

                return impl

            @Tx.prim_func(check_well_formed=False)
            def impl():
                for s in Tx.serial(0, n_chunks):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                    s1_idx_a = Tx.meta_var(get_indices(2 * s, s1_st, s1_ext))
                    s1_idx_b = Tx.meta_var(get_indices(2 * s + 1, s1_st, s1_ext))
                    s2_idx_a = Tx.meta_var(get_indices(2 * s, s2_st, s2_ext))
                    s2_idx_b = Tx.meta_var(get_indices(2 * s + 1, s2_st, s2_ext))
                    op_func_f32x2(
                        s1_buf[tuple(s1_idx_a)],
                        s1_buf[tuple(s1_idx_b)],
                        s2_buf[tuple(s2_idx_a)],
                        s2_buf[tuple(s2_idx_b)],
                        Tx.address_of(dst[tuple(dst_idx)]),
                        rounding_mode=rm,
                    )

            return impl

        # ---- wg/warp/cta-scope: collective tile -> per-thread reg view ------
        # Use get_local_region to get the per-thread (shape, st, ext).
        dst_info = get_local_region(dst.layout, list(dst.shape), dst_st_raw, dst_ext_raw)
        if dst_info is None:
            return None
        dst_local_shape, dst_local_st, dst_local_ext = dst_info
        local_total = functools.reduce(operator.mul, dst_local_ext, 1)
        try:
            if int(local_total) % 2 != 0:
                return None
        except (TypeError, ValueError):
            return None
        n_chunks = int(local_total) // 2

        def _src_local_info(src):
            if src.is_scalar:
                return None
            b = src.buf_region.buffer
            st, ext = get_st_extent(src.buf_region)
            info = get_local_region(b.layout, b.shape, st, ext)
            return info

        s1_info = _src_local_info(s1)
        s2_info = _src_local_info(s2)
        if (not s1.is_scalar and s1_info is None) or (not s2.is_scalar and s2_info is None):
            return None
        s1_local_shape = s1_info[0] if s1_info else None
        s1_local_st = s1_info[1] if s1_info else None
        s1_local_ext = s1_info[2] if s1_info else None
        s2_local_shape = s2_info[0] if s2_info else None
        s2_local_st = s2_info[1] if s2_info else None
        s2_local_ext = s2_info[2] if s2_info else None

        if not s1.is_scalar and s2.is_scalar:

            @Tx.prim_func(check_well_formed=False)
            def impl():
                with Tx.thread():
                    dst_view = dst.local(*dst_local_shape)
                    s1_view = s1_buf.local(*s1_local_shape)
                    for s in Tx.unroll(n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_local_st, dst_local_ext))
                        s1_idx_a = Tx.meta_var(get_indices(2 * s, s1_local_st, s1_local_ext))
                        s1_idx_b = Tx.meta_var(get_indices(2 * s + 1, s1_local_st, s1_local_ext))
                        op_func_f32x2(
                            s1_view[tuple(s1_idx_a)],
                            s1_view[tuple(s1_idx_b)],
                            s2_scalar_val,
                            s2_scalar_val,
                            Tx.address_of(dst_view[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

            return impl

        if s1.is_scalar and not s2.is_scalar:

            @Tx.prim_func(check_well_formed=False)
            def impl():
                with Tx.thread():
                    dst_view = dst.local(*dst_local_shape)
                    s2_view = s2_buf.local(*s2_local_shape)
                    for s in Tx.unroll(n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_local_st, dst_local_ext))
                        s2_idx_a = Tx.meta_var(get_indices(2 * s, s2_local_st, s2_local_ext))
                        s2_idx_b = Tx.meta_var(get_indices(2 * s + 1, s2_local_st, s2_local_ext))
                        op_func_f32x2(
                            s1_scalar_val,
                            s1_scalar_val,
                            s2_view[tuple(s2_idx_a)],
                            s2_view[tuple(s2_idx_b)],
                            Tx.address_of(dst_view[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

            return impl

        @Tx.prim_func(check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_view = dst.local(*dst_local_shape)
                s1_view = s1_buf.local(*s1_local_shape)
                s2_view = s2_buf.local(*s2_local_shape)
                for s in Tx.unroll(n_chunks):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_local_st, dst_local_ext))
                    s1_idx_a = Tx.meta_var(get_indices(2 * s, s1_local_st, s1_local_ext))
                    s1_idx_b = Tx.meta_var(get_indices(2 * s + 1, s1_local_st, s1_local_ext))
                    s2_idx_a = Tx.meta_var(get_indices(2 * s, s2_local_st, s2_local_ext))
                    s2_idx_b = Tx.meta_var(get_indices(2 * s + 1, s2_local_st, s2_local_ext))
                    op_func_f32x2(
                        s1_view[tuple(s1_idx_a)],
                        s1_view[tuple(s1_idx_b)],
                        s2_view[tuple(s2_idx_a)],
                        s2_view[tuple(s2_idx_b)],
                        Tx.address_of(dst_view[tuple(dst_idx)]),
                        rounding_mode=rm,
                    )

        return impl

    return factory


# -----------------------------------------------------------------------------
# Cast: Tx.cast(dst, src) -- arity 1, no bias/scale, dst dtype != src dtype.
# -----------------------------------------------------------------------------
def _parse_cast(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    _dst: BufferRegion = op.args[0]
    _src = op.args[1]
    if not isinstance(_src, BufferRegion):
        return None, "cast src must be a buffer region"
    return Plan(dst=_dst, srcs=[SrcSpec(buf_region=_src)], extras={}), None


def _compute_cast(src_vals, extras, dt):
    # Outer Tx.cast(..., dst.dtype) in the schedule already does the cast.
    return src_vals[0]


# Cast vec2 packed CUDA intrinsics. Each value is the CUDA builtin name that
# converts one packed-2 source to one packed-2 dest in a single instruction.
_VEC2_CAST_INTRINSICS = {
    ("float32", "float16"): "__float22half2_rn",
    ("float16", "float32"): "__half22float2",
    ("bfloat16", "float32"): "__bfloat1622float2",
    ("float32", "bfloat16"): "__float22bfloat162_rn",
}
_DTYPE_X2_NAME = {"float32": "float2", "float16": "half2", "bfloat16": "nv_bfloat162"}


def _is_contiguous_region(analyzer, st, ext, shape):
    """[st:st+ext] is a contiguous block in row-major ``shape``."""
    found_break = False
    for i in reversed(range(len(st))):
        is_full = analyzer.can_prove_equal(st[i], 0) and analyzer.can_prove_equal(ext[i], shape[i])
        if found_break:
            if not analyzer.can_prove_equal(ext[i], 1):
                return False
        else:
            if not is_full:
                found_break = True
    return True


def _linear_offset(st, shape):
    """Row-major linear offset of position ``st`` in buffer of given ``shape``."""
    offset = 0
    stride = 1
    for i in reversed(range(len(st))):
        offset = offset + st[i] * stride
        stride = stride * shape[i]
    return offset


def _make_cast_vec2_factory():
    """Cast vec_emit using CUDA packed-pair intrinsics (e.g. __float22half2_rn)."""

    def factory(op_call, plan, sctx, vec_len):
        from tvm.arith import Analyzer

        from ..common import get_indices, get_st_extent
        from ..layout_utils import get_local_region

        if len(plan.srcs) != 1 or plan.srcs[0].is_scalar:
            return None
        src = plan.srcs[0]
        if src.index_fn is not None:
            return None
        src_dtype = src.buf_region.buffer.dtype
        dst_dtype = plan.dst.buffer.dtype
        intrinsic = _VEC2_CAST_INTRINSICS.get((src_dtype, dst_dtype))
        if intrinsic is None:
            return None

        import functools
        import operator

        dst = plan.dst.buffer
        dst_st, dst_ext = get_st_extent(plan.dst)
        src_buf = src.buf_region.buffer
        src_st, src_ext = get_st_extent(src.buf_region)

        src_dtypex2 = _DTYPE_X2_NAME[src_dtype]
        dst_dtypex2 = _DTYPE_X2_NAME[dst_dtype]
        func_name = f"tvm_builtin_cast_{src_dtype}x2_{dst_dtype}x2"
        source_code = (
            f"\n__forceinline__ __device__ void {func_name}(void* dst, void* src) {{\n"
            f"    (({dst_dtypex2}*)dst)[0] = {intrinsic}((({src_dtypex2}*)src)[0]);\n"
            "}\n"
        )

        if sctx.is_thread:
            total = functools.reduce(operator.mul, dst_ext, 1)
            try:
                if int(total) % 2 != 0:
                    return None
            except (TypeError, ValueError):
                return None
            n_chunks = int(total) // 2

            @Tx.prim_func(check_well_formed=False)
            def impl_thread():
                # (no Tx.thread wrap; outer scope is already thread)
                for s in Tx.serial(0, n_chunks):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                    src_idx = Tx.meta_var(get_indices(2 * s, src_st, src_ext))
                    Tx.cuda.func_call(
                        func_name,
                        Tx.address_of(dst[tuple(dst_idx)]),
                        Tx.address_of(src_buf[tuple(src_idx)]),
                        source_code=source_code,
                    )

            return impl_thread

        if sctx.scope_kind not in ("warp", "warpgroup", "cta", "cluster"):
            return None

        # Per-thread vec2 cast at collective scope. Mirrors HEAD's
        # cast/local_view fast path: open Tx.thread, view each buffer as a
        # flat per-thread 1D array, issue cuda intrinsic per pair.
        src_has_layout = src_buf.layout is not None and not src_buf.layout.is_trivial()
        dst_has_layout = dst.layout is not None and not dst.layout.is_trivial()
        if not (src_has_layout or dst_has_layout):
            return None

        if src_has_layout:
            src_info = get_local_region(src_buf.layout, list(src_buf.shape), src_st, src_ext)
            if not src_info:
                return None
            src_local_shape, src_local_st, src_local_ext = src_info
        else:
            src_local_shape = list(src_buf.shape)
            src_local_st = list(src_st)
            src_local_ext = list(src_ext)

        if dst_has_layout:
            dst_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_ext)
            if not dst_info:
                return None
            dst_local_shape, dst_local_st, dst_local_ext = dst_info
        else:
            dst_local_shape = list(dst.shape)
            dst_local_st = list(dst_st)
            dst_local_ext = list(dst_ext)

        src_local_total = functools.reduce(operator.mul, src_local_ext, 1)
        dst_local_total = functools.reduce(operator.mul, dst_local_ext, 1)
        try:
            src_total_i = int(src_local_total)
            dst_total_i = int(dst_local_total)
        except (TypeError, ValueError):
            return None
        if src_total_i != dst_total_i or dst_total_i % 2 != 0:
            return None
        n2 = dst_total_i // 2

        analyzer = Analyzer()
        if not _is_contiguous_region(analyzer, src_local_st, src_local_ext, src_local_shape):
            return None
        if not _is_contiguous_region(analyzer, dst_local_st, dst_local_ext, dst_local_shape):
            return None
        src_off = _linear_offset(src_local_st, src_local_shape)
        dst_off = _linear_offset(dst_local_st, dst_local_shape)
        try:
            if int(src_off) % 2 != 0 or int(dst_off) % 2 != 0:
                return None
        except (TypeError, ValueError):
            if not (
                analyzer.can_prove_equal(src_off % 2, 0)
                and analyzer.can_prove_equal(dst_off % 2, 0)
            ):
                return None

        src_full_size = functools.reduce(operator.mul, src_local_shape, 1)
        dst_full_size = functools.reduce(operator.mul, dst_local_shape, 1)

        @Tx.prim_func(check_well_formed=False)
        def impl_collective():
            with Tx.thread():
                base_src = Tx.decl_buffer(
                    (src_full_size,), src_buf.dtype, src_buf.data, scope=src_buf.scope()
                )
                base_dst = Tx.decl_buffer((dst_full_size,), dst.dtype, dst.data, scope=dst.scope())
                for s in Tx.serial(0, n2):
                    src_idx = Tx.meta_var(src_off + s * 2)
                    dst_idx = Tx.meta_var(dst_off + s * 2)
                    Tx.cuda.func_call(
                        func_name,
                        Tx.address_of(base_dst[dst_idx]),
                        Tx.address_of(base_src[src_idx]),
                        source_code=source_code,
                    )

        return impl_collective

    return factory


# -----------------------------------------------------------------------------
# FMA: Tx.fma(dst, a, b, c) -- compute = a*b + c.
# -----------------------------------------------------------------------------
def _parse_fma(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    _dst: BufferRegion = op.args[0]
    args = op.args[1:4]
    srcs: list[SrcSpec] = []
    for a in args:
        if isinstance(a, BufferRegion):
            srcs.append(SrcSpec(buf_region=a))
        else:
            srcs.append(SrcSpec(scalar=a))
    return Plan(dst=_dst, srcs=srcs, extras={}), None


def _compute_fma(src_vals, extras, dt):
    return src_vals[0] * src_vals[1] + src_vals[2]


def _make_fma_packed_f32x2_factory():
    """FMA vec_emit for sm_100+ f32: Tx.ptx.fma_packed_f32x2."""

    def factory(op_call, plan, sctx, vec_len):
        from ..common import get_indices, get_st_extent, sm_version_ok
        from ..layout_utils import get_local_region

        if plan.dst.buffer.dtype != "float32":
            return None
        if not sm_version_ok(op_call, sctx, min_version=100)[0]:
            return None
        # Two emit modes:
        if sctx.is_thread:
            use_view = False
        elif sctx.scope_kind in ("warp", "warpgroup", "cta"):
            use_view = True
            if plan.dst.buffer.layout is None or plan.dst.buffer.layout.is_trivial():
                return None
            for s in plan.srcs:
                if not s.is_scalar and (
                    s.buf_region.buffer.layout is None or s.buf_region.buffer.layout.is_trivial()
                ):
                    return None
        else:
            return None
        if len(plan.srcs) != 3:
            return None
        a, b, c = plan.srcs
        if a.is_scalar or a.buf_region.buffer.dtype != "float32":
            return None
        for s in (b, c):
            if s.is_scalar:
                if s.scalar.dtype != "float32":
                    return None
            else:
                if s.buf_region.buffer.dtype != "float32":
                    return None
                if s.index_fn is not None:
                    return None
        if a.index_fn is not None:
            return None

        import functools
        import operator

        dst = plan.dst.buffer
        dst_st_raw, dst_ext_raw = get_st_extent(plan.dst)
        rm = plan.extras.get("rounding_mode", "rz")
        a_buf = a.buf_region.buffer
        a_st_raw, a_ext_raw = get_st_extent(a.buf_region)

        b_is_buf = not b.is_scalar
        c_is_buf = not c.is_scalar
        b_buf = b.buf_region.buffer if b_is_buf else None
        c_buf = c.buf_region.buffer if c_is_buf else None
        b_st_raw, b_ext_raw = get_st_extent(b.buf_region) if b_is_buf else (None, None)
        c_st_raw, c_ext_raw = get_st_extent(c.buf_region) if c_is_buf else (None, None)
        b_scalar = b.scalar if not b_is_buf else None
        c_scalar = c.scalar if not c_is_buf else None

        if not use_view:
            # thread-scope: use raw region st/ext, index buffer directly
            dst_st, dst_ext = dst_st_raw, dst_ext_raw
            a_st, a_ext = a_st_raw, a_ext_raw
            b_st, b_ext = b_st_raw, b_ext_raw
            c_st, c_ext = c_st_raw, c_ext_raw
            total = functools.reduce(operator.mul, dst_ext, 1)
            try:
                if int(total) % 2 != 0:
                    return None
            except (TypeError, ValueError):
                return None
            n_chunks = int(total) // 2
        else:
            # wg/warp/cta-scope: build per-thread local views + use local st/ext.
            dst_info = get_local_region(dst.layout, list(dst.shape), dst_st_raw, dst_ext_raw)
            a_info = get_local_region(a_buf.layout, a_buf.shape, a_st_raw, a_ext_raw)
            if dst_info is None or a_info is None:
                return None
            b_info = (
                get_local_region(b_buf.layout, b_buf.shape, b_st_raw, b_ext_raw)
                if b_is_buf
                else None
            )
            c_info = (
                get_local_region(c_buf.layout, c_buf.shape, c_st_raw, c_ext_raw)
                if c_is_buf
                else None
            )
            if (b_is_buf and b_info is None) or (c_is_buf and c_info is None):
                return None
            dst_local_shape, dst_st, dst_ext = dst_info
            a_local_shape, a_st, a_ext = a_info
            b_local_shape, b_st, b_ext = b_info if b_info else (None, None, None)
            c_local_shape, c_st, c_ext = c_info if c_info else (None, None, None)
            local_total = functools.reduce(operator.mul, dst_ext, 1)
            try:
                if int(local_total) % 2 != 0:
                    return None
            except (TypeError, ValueError):
                return None
            n_chunks = int(local_total) // 2

        # Four shape combos depending on whether b and c are buffers or scalars,
        # x two scope modes (thread = direct buf indexing, wg = .local(*shape) view).
        # TVMScript can't handle Python closure calls inside the IR body so each
        # combo gets its own @Tx.prim_func.
        if b_is_buf and c_is_buf:
            if not use_view:

                @Tx.prim_func(check_well_formed=False)
                def impl():
                    for s in Tx.serial(0, n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                        a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                        b_idx_a = Tx.meta_var(get_indices(2 * s, b_st, b_ext))
                        b_idx_b = Tx.meta_var(get_indices(2 * s + 1, b_st, b_ext))
                        c_idx_a = Tx.meta_var(get_indices(2 * s, c_st, c_ext))
                        c_idx_b = Tx.meta_var(get_indices(2 * s + 1, c_st, c_ext))
                        _fma_f32x2_adapter(
                            a_buf[tuple(a_idx_a)],
                            a_buf[tuple(a_idx_b)],
                            b_buf[tuple(b_idx_a)],
                            b_buf[tuple(b_idx_b)],
                            c_buf[tuple(c_idx_a)],
                            c_buf[tuple(c_idx_b)],
                            Tx.address_of(dst[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

                return impl

            @Tx.prim_func(check_well_formed=False)
            def impl():
                with Tx.thread():
                    dst_view = dst.local(*dst_local_shape)
                    a_view = a_buf.local(*a_local_shape)
                    b_view = b_buf.local(*b_local_shape)
                    c_view = c_buf.local(*c_local_shape)
                    for s in Tx.unroll(n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                        a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                        b_idx_a = Tx.meta_var(get_indices(2 * s, b_st, b_ext))
                        b_idx_b = Tx.meta_var(get_indices(2 * s + 1, b_st, b_ext))
                        c_idx_a = Tx.meta_var(get_indices(2 * s, c_st, c_ext))
                        c_idx_b = Tx.meta_var(get_indices(2 * s + 1, c_st, c_ext))
                        _fma_f32x2_adapter(
                            a_view[tuple(a_idx_a)],
                            a_view[tuple(a_idx_b)],
                            b_view[tuple(b_idx_a)],
                            b_view[tuple(b_idx_b)],
                            c_view[tuple(c_idx_a)],
                            c_view[tuple(c_idx_b)],
                            Tx.address_of(dst_view[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

            return impl

        if b_is_buf and not c_is_buf:
            if not use_view:

                @Tx.prim_func(check_well_formed=False)
                def impl():
                    for s in Tx.serial(0, n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                        a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                        b_idx_a = Tx.meta_var(get_indices(2 * s, b_st, b_ext))
                        b_idx_b = Tx.meta_var(get_indices(2 * s + 1, b_st, b_ext))
                        _fma_f32x2_adapter(
                            a_buf[tuple(a_idx_a)],
                            a_buf[tuple(a_idx_b)],
                            b_buf[tuple(b_idx_a)],
                            b_buf[tuple(b_idx_b)],
                            c_scalar,
                            c_scalar,
                            Tx.address_of(dst[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

                return impl

            @Tx.prim_func(check_well_formed=False)
            def impl():
                with Tx.thread():
                    dst_view = dst.local(*dst_local_shape)
                    a_view = a_buf.local(*a_local_shape)
                    b_view = b_buf.local(*b_local_shape)
                    for s in Tx.unroll(n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                        a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                        b_idx_a = Tx.meta_var(get_indices(2 * s, b_st, b_ext))
                        b_idx_b = Tx.meta_var(get_indices(2 * s + 1, b_st, b_ext))
                        _fma_f32x2_adapter(
                            a_view[tuple(a_idx_a)],
                            a_view[tuple(a_idx_b)],
                            b_view[tuple(b_idx_a)],
                            b_view[tuple(b_idx_b)],
                            c_scalar,
                            c_scalar,
                            Tx.address_of(dst_view[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

            return impl

        if not b_is_buf and c_is_buf:
            if not use_view:

                @Tx.prim_func(check_well_formed=False)
                def impl():
                    for s in Tx.serial(0, n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                        a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                        c_idx_a = Tx.meta_var(get_indices(2 * s, c_st, c_ext))
                        c_idx_b = Tx.meta_var(get_indices(2 * s + 1, c_st, c_ext))
                        _fma_f32x2_adapter(
                            a_buf[tuple(a_idx_a)],
                            a_buf[tuple(a_idx_b)],
                            b_scalar,
                            b_scalar,
                            c_buf[tuple(c_idx_a)],
                            c_buf[tuple(c_idx_b)],
                            Tx.address_of(dst[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

                return impl

            @Tx.prim_func(check_well_formed=False)
            def impl():
                with Tx.thread():
                    dst_view = dst.local(*dst_local_shape)
                    a_view = a_buf.local(*a_local_shape)
                    c_view = c_buf.local(*c_local_shape)
                    for s in Tx.unroll(n_chunks):
                        dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                        a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                        a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                        c_idx_a = Tx.meta_var(get_indices(2 * s, c_st, c_ext))
                        c_idx_b = Tx.meta_var(get_indices(2 * s + 1, c_st, c_ext))
                        _fma_f32x2_adapter(
                            a_view[tuple(a_idx_a)],
                            a_view[tuple(a_idx_b)],
                            b_scalar,
                            b_scalar,
                            c_view[tuple(c_idx_a)],
                            c_view[tuple(c_idx_b)],
                            Tx.address_of(dst_view[tuple(dst_idx)]),
                            rounding_mode=rm,
                        )

            return impl

        # Both b and c scalar
        if not use_view:

            @Tx.prim_func(check_well_formed=False)
            def impl():
                for s in Tx.serial(0, n_chunks):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                    a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                    a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                    _fma_f32x2_adapter(
                        a_buf[tuple(a_idx_a)],
                        a_buf[tuple(a_idx_b)],
                        b_scalar,
                        b_scalar,
                        c_scalar,
                        c_scalar,
                        Tx.address_of(dst[tuple(dst_idx)]),
                        rounding_mode=rm,
                    )

            return impl

        @Tx.prim_func(check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_view = dst.local(*dst_local_shape)
                a_view = a_buf.local(*a_local_shape)
                for s in Tx.unroll(n_chunks):
                    dst_idx = Tx.meta_var(get_indices(2 * s, dst_st, dst_ext))
                    a_idx_a = Tx.meta_var(get_indices(2 * s, a_st, a_ext))
                    a_idx_b = Tx.meta_var(get_indices(2 * s + 1, a_st, a_ext))
                    _fma_f32x2_adapter(
                        a_view[tuple(a_idx_a)],
                        a_view[tuple(a_idx_b)],
                        b_scalar,
                        b_scalar,
                        c_scalar,
                        c_scalar,
                        Tx.address_of(dst_view[tuple(dst_idx)]),
                        rounding_mode=rm,
                    )

        return impl

    return factory


# -----------------------------------------------------------------------------
# Registry: one table, no per-arity buckets.
# -----------------------------------------------------------------------------
ALL_OPS: dict[str, OpSpec] = {
    "zero": OpSpec(
        name="zero", parse=_parse_unary, compute=_compute_zero, check_extras=_check_unary_extras
    ),
    "fill": OpSpec(
        name="fill", parse=_parse_unary, compute=_compute_fill, check_extras=_check_unary_extras
    ),
    "reciprocal": OpSpec(
        name="reciprocal",
        parse=_parse_unary,
        compute=_compute_reciprocal,
        check_extras=_check_unary_extras,
    ),
    "sqrt": OpSpec(
        name="sqrt",
        parse=_parse_unary,
        compute=_unary_with_bias_scale(Tx.sqrt),
        check_extras=_check_unary_extras,
    ),
    "exp": OpSpec(
        name="exp",
        parse=_parse_unary,
        compute=_unary_with_bias_scale(Tx.exp),
        check_extras=_check_unary_extras,
    ),
    "exp2": OpSpec(
        name="exp2",
        parse=_parse_unary,
        compute=_unary_with_bias_scale(Tx.exp2),
        check_extras=_check_unary_extras,
    ),
    "silu": OpSpec(
        name="silu",
        parse=_parse_unary,
        compute=_compute_silu,
        check_extras=_check_unary_extras,
    ),
    "add": OpSpec(
        name="add",
        parse=_parse_binary_for("add"),
        compute=_compute_add,
        vec_emit_factory=_make_binary_packed_f32x2_factory("add"),
    ),
    "sub": OpSpec(
        name="sub",
        parse=_parse_binary_for("sub"),
        compute=_compute_sub,
        vec_emit_factory=_make_binary_packed_f32x2_factory("sub"),
    ),
    "mul": OpSpec(
        name="mul",
        parse=_parse_binary_for("mul"),
        compute=_compute_mul,
        vec_emit_factory=_make_binary_packed_f32x2_factory("mul"),
    ),
    "fdiv": OpSpec(name="fdiv", parse=_parse_binary_for("fdiv"), compute=_compute_fdiv),
    "cast": OpSpec(
        name="cast",
        parse=_parse_cast,
        compute=_compute_cast,
        vec_emit_factory=_make_cast_vec2_factory(),
    ),
    "fma": OpSpec(
        name="fma",
        parse=_parse_fma,
        compute=_compute_fma,
        vec_emit_factory=_make_fma_packed_f32x2_factory(),
    ),
}
