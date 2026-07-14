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
"""Builtin ops in TIRX"""

import functools
from collections.abc import Callable

import tvm.tirx.operator as tirx_op
from tvm.ir import Op
from tvm.tirx import Buffer, BufferRegion, Expr
from tvm.tirx.exec_scope import _SCOPE_KIND_TO_NAME, ExecScope
from tvm.tirx.expr import FloatImm
from tvm.tirx.lang.alloc_pool import SMEMPool, TMEMPool, TMEMStages
from tvm.tirx.predicate import Predicate

from . import _ffi_api, frame
from .ir import decl_buffer, meta_class


def _normalize_scope(scope) -> ExecScope:
    """Normalize a scope selector to an ``ExecScope``.

    Accepts an ``ExecScope`` (passed through), a scope-name ``str``
    (e.g. ``"warp"``, normalized via the FFI ctor / ``StringToScopeKind``),
    or an ``int`` ``ScopeKind`` value. ``None`` resolves to the default
    ``thread`` scope, keeping the default in one place.
    """
    if scope is None:
        return ExecScope("thread")
    if isinstance(scope, ExecScope):
        return scope
    if isinstance(scope, str):
        return ExecScope(scope)
    if isinstance(scope, int):
        return ExecScope(_SCOPE_KIND_TO_NAME[scope])
    raise TypeError(f"Cannot interpret {scope!r} as an execution scope")


class ScopedOp:
    """Make a tile-primitive op callable at the default ``thread`` scope.

    A bare ``Tx.copy(...)`` emits a call at ``thread`` scope. To cooperate at a
    wider scope, reach the op through a scope namespace -- ``Tx.warp.copy(...)``,
    ``Tx.wg.sum(...)``, ``Tx.cta.fill(...)`` (see :class:`ScopeNamespace`).

    The wrapped ``fn`` must accept a keyword-only ``scope`` parameter that it
    threads into the constructed ``TilePrimitiveCall``.
    """

    def __init__(self, fn):
        self._fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, scope=ExecScope("thread"), **kwargs)

    def _bind(self, scope: ExecScope):
        """Return a callable that emits this op at ``scope``.

        Used by :class:`ScopeNamespace`; not part of the user-facing surface.
        """
        return lambda *args, **kwargs: self._fn(*args, scope=scope, **kwargs)


class ScopeNamespace:
    """Bind a cooperation scope to every tile primitive reached through it.

    ``Tx.cluster`` / ``Tx.cta`` / ``Tx.wg`` (warpgroup) / ``Tx.warp`` are the
    instances exposed on the ``Tx`` surface. Attribute access resolves a
    tile-primitive op name against the public ``Tx`` surface (registered and
    dynamic ops alike) and binds this namespace's scope, so
    ``Tx.warp.copy(dst, src)`` emits a copy at warp scope and
    ``Tx.cta.sum(out, x)`` reduces at CTA scope. A bare ``Tx.copy(...)`` (no
    namespace prefix) stays at the default ``thread`` scope.
    """

    def __init__(self, scope, label: str):
        self._scope = _normalize_scope(scope)
        self._label = label

    def __repr__(self):
        return f"<Tx.{self._label}: {self._scope.name}-scope tile primitives>"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        from tvm.tirx.script import tile as _tile_script

        op = getattr(_tile_script, name)
        if not isinstance(op, ScopedOp):
            # AttributeError (not TypeError) so hasattr()/getattr(..., default)
            # degrade gracefully on a scope namespace.
            raise AttributeError(
                f"'Tx.{self._label}.{name}' is not a tile primitive; the "
                f"'Tx.{self._label}.' scope prefix applies only to tile primitives"
            )
        return op._bind(self._scope)


# Scope-prefix namespaces: ``Tx.warp.copy(...)`` / ``Tx.wg.sum(...)`` /
# ``Tx.cta.fill(...)`` / ``Tx.cluster.copy(...)``. ``wg`` == warpgroup. A bare
# ``Tx.copy(...)`` (no prefix) runs at the default ``thread`` scope.
cluster = ScopeNamespace("cluster", "cluster")
cta = ScopeNamespace("cta", "cta")
wg = ScopeNamespace("warpgroup", "wg")
warpgroup = ScopeNamespace("warpgroup", "warpgroup")  # full-name alias of ``wg``
warp = ScopeNamespace("warp", "warp")
thread = ScopeNamespace("thread", "thread")


def _is_buffer_or_region(x):
    return isinstance(x, Buffer | BufferRegion)


def _to_region(buffer: BufferRegion | Buffer):
    if isinstance(buffer, Buffer):
        return buffer[[slice(None, None, None) for _ in range(len(buffer.shape))]]
    assert isinstance(buffer, BufferRegion)
    return buffer


def _wrap_elem_in_tuple(e):
    if isinstance(e, tuple | list):
        return e
    return (e,)


f_insert = _ffi_api.TilePrimitiveCall  # pylint: disable=no-member


@ScopedOp
def zero(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer | None = None,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Zero out all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for zero result.
        When src is omitted, also used as the source (in-place).

    src : Union[BufferRegion, Buffer], optional
        The source buffer region. If omitted, dst is used (in-place).

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if src is None:
        src = dst
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(
        tirx_op.Zero(dst, src, workspace=workspace, config=config, dispatch=dispatch, scope=scope)
    )


@ScopedOp
def sqrt(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer | None = None,
    bias: BufferRegion | Buffer | FloatImm | None = None,
    scale: FloatImm | None = None,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Sqrt all elements in src and store to dst.

    dst = sqrt(src * scale + bias)  (if scale or bias are provided)

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sqrt result.
        When src is omitted, also used as the source (in-place).

    src : Union[BufferRegion, Buffer], optional
        The source buffer region. If omitted, dst is used (in-place).

    bias : Optional[Union[BufferRegion, Buffer, FloatImm]]
        The bias of the sqrt src. Only supported on Trn.

    scale : Optional[FloatImm]
        The scale of the sqrt src. Only supported on Trn.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    # Expression-form overload: ``sqrt(value)`` returns the underlying expression.
    from tvm import tirx as _tirx

    if not _is_buffer_or_region(dst):
        return _tirx.sqrt(dst)
    if src is None:
        src = dst
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)
    return f_insert(
        tirx_op.Sqrt(
            dst,
            src,
            bias,
            scale,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def add(
    dst: BufferRegion | Buffer,
    src1: BufferRegion | Buffer | FloatImm,
    src2: BufferRegion | Buffer | FloatImm,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Add data from src1 and src2, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for add result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirx_op.Add(
            dst, src1, src2, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def sub(
    dst: BufferRegion | Buffer,
    src1: BufferRegion | Buffer,
    src2: BufferRegion | Buffer | FloatImm,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Sub data from src2 to src1, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sub result.

    src1 : Union[BufferRegion, Buffer]
        The source buffer region 1.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirx_op.Sub(
            dst, src1, src2, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def mul(
    dst: BufferRegion | Buffer,
    src1: BufferRegion | Buffer | FloatImm,
    src2: BufferRegion | Buffer | FloatImm,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Multiply data from src1 and src2, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for mul result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirx_op.Mul(
            dst, src1, src2, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def fdiv(
    dst: BufferRegion | Buffer,
    src1: BufferRegion | Buffer,
    src2: BufferRegion | Buffer | FloatImm,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """(Float) Div data from src2 to src1, store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for div result.

    src1 : Union[BufferRegion, Buffer]
        The source buffer region 1.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirx_op.FDiv(
            dst, src1, src2, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def fma(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer,
    scale: BufferRegion | Buffer | Expr,
    bias: BufferRegion | Buffer | Expr,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Fused multiply-add: dst = src * scale + bias.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region.

    src : Union[BufferRegion, Buffer]
        The input buffer region.

    scale : Union[BufferRegion, Buffer, Expr]
        The scale factor (buffer region or scalar).

    bias : Union[BufferRegion, Buffer, Expr]
        The bias term (buffer region or scalar).

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    if isinstance(scale, Buffer):
        scale = _to_region(scale)
    if isinstance(bias, Buffer):
        bias = _to_region(bias)
    return f_insert(
        tirx_op.FMA(
            dst,
            src,
            scale,
            bias,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def cast(
    dst,
    src=None,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Cast — overloaded.

    1. ``cast(value, dtype)`` — expression-level cast: returns ``T.cast(value, dtype)``.
       Also accepts ``cast(value, dtype=...)`` as a kwarg form.
    2. ``cast(dst, src, workspace=..., dispatch=...)`` — buffer-level Cast operator.
    """
    # Expression-level cast: src is a dtype (str / DataType) — emit T.cast(value, dtype).
    from tvm import tirx as _tirx

    # Accept ``T.cast(value, dtype=...)`` (kwarg) in addition to the
    # ``T.cast(value, dtype)`` positional form.
    if src is None and "dtype" in kwargs:
        src = kwargs.pop("dtype")
    if src is None or isinstance(src, str) or hasattr(src, "with_lanes"):
        # Treat as expression cast: dst=value, src=dtype.
        return _tirx.Cast(src, dst)
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(
        tirx_op.Cast(dst, src, workspace=workspace, config=config, dispatch=dispatch, scope=scope)
    )


@ScopedOp
def copy(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Copy data from src to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(
        tirx_op.Copy(dst, src, workspace=workspace, config=config, dispatch=dispatch, scope=scope)
    )


@ScopedOp
def copy_async(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(
        tirx_op.CopyAsync(
            dst, src, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def gemm_async(
    C: BufferRegion | Buffer,
    A: BufferRegion | Buffer,
    B: BufferRegion | Buffer,
    SFA: BufferRegion | Buffer | None = None,
    SFB: BufferRegion | Buffer | None = None,
    transA: bool = False,
    transB: bool = False,
    accum: bool = False,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """General matrix multiplication asynchronously.

    Parameters
    ----------
    C : Union[BufferRegion, Buffer]
        The buffer of matrix C.

    A : Union[BufferRegion, Buffer]
        The buffer of matrix A.

    B : Union[BufferRegion, Buffer]
        The buffer of matrix B.

    SFA : Optional[Union[BufferRegion, Buffer]]
        The scale factor buffer for matrix A (block-scaled MMA only).

    SFB : Optional[Union[BufferRegion, Buffer]]
        The scale factor buffer for matrix B (block-scaled MMA only).

    transA : bool
        False if A is K-major (MxK), True if A is MN-major (KxM).

    transB : bool
        False if B is K-major (NxK), True if B is MN-major (KxN).

    accum : bool
        Whether C is accumulated.
        C = A * B if accum is False, otherwise C += A * B.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    C = _to_region(C)
    A = _to_region(A)
    B = _to_region(B)
    if (SFA is None) != (SFB is None):
        raise ValueError("SFA and SFB must both be provided or both be None")
    if SFA is not None and SFB is not None:
        SFA = _to_region(SFA)
        SFB = _to_region(SFB)
        return f_insert(
            tirx_op.GemmAsync(
                C,
                A,
                B,
                SFA,
                SFB,
                transA,
                transB,
                accum,
                workspace=workspace,
                config=config,
                dispatch=dispatch,
                scope=scope,
            )
        )
    return f_insert(
        tirx_op.GemmAsync(
            C,
            A,
            B,
            transA,
            transB,
            accum,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def fill(
    dst: BufferRegion | Buffer,
    value: Expr,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Fill the buffer region with the value.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region.

    value : Expr
        The value to be filled.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    return f_insert(
        tirx_op.Fill(dst, value, workspace=workspace, config=config, dispatch=dispatch, scope=scope)
    )


@ScopedOp
def gemm(
    D: BufferRegion | Buffer,
    A: BufferRegion | Buffer,
    B: BufferRegion | Buffer,
    C: BufferRegion | Buffer,
    transpose_A: bool = False,
    transpose_B: bool = False,
    alpha: Expr = 1.0,
    beta: Expr = 0.0,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """General matrix multiplication.

    D = A * B * alpha + C * beta

    Parameters
    ----------
    D : Union[BufferRegion, Buffer]
        The buffer of matrix D.

    A : Union[BufferRegion, Buffer]
        The buffer of matrix A.

    B : Union[BufferRegion, Buffer]
        The buffer of matrix B.

    C : Union[BufferRegion, Buffer]
        The buffer of matrix C.

    transpose_A : bool
        Whether to transpose A.

    transpose_B : bool
        Whether to transpose B.

    alpha : Expr
        The scalar alpha.

    beta : Expr
        The scalar beta.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    D = _to_region(D)
    A = _to_region(A)
    B = _to_region(B)
    C = _to_region(C)
    return f_insert(
        tirx_op.Gemm(
            D,
            A,
            B,
            C,
            transpose_A,
            transpose_B,
            alpha,
            beta,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def sum(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer,
    axes: int | tuple[int] = -1,
    accum: bool = False,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """
    Sum all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for sum result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    axes : Union[int, Tuple[int]]
        The axis to sum over.

    accum : bool
        Whether dst is accumulated.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    axes = _wrap_elem_in_tuple(axes)
    return f_insert(
        tirx_op.Sum(
            dst,
            src,
            axes,
            accum,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def max(
    dst,
    src=None,
    axes: int | tuple[int] = -1,
    accum: bool = False,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Max — overloaded.

    1. ``max(a, b)`` — expression: returns ``tirx.max(a, b)``.
    2. ``max(dst, src, axes=, accum=)`` — reduction operator over buffers.
    """
    from tvm import tirx as _tirx

    if not isinstance(dst, BufferRegion | Buffer) or not isinstance(src, BufferRegion | Buffer):
        # Expression-level max
        return _tirx.max(dst, src)
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    axes = _wrap_elem_in_tuple(axes)
    return f_insert(
        tirx_op.Max(
            dst,
            src,
            axes,
            accum,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def min(
    dst,
    src=None,
    axes: int | tuple[int] = -1,
    accum: bool = False,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Min — overloaded.

    1. ``min(a, b)`` — expression: returns ``tirx.min(a, b)``.
    2. ``min(dst, src, axes=, accum=)`` — reduction operator over buffers.
    """
    from tvm import tirx as _tirx

    if not isinstance(dst, BufferRegion | Buffer) or not isinstance(src, BufferRegion | Buffer):
        return _tirx.min(dst, src)
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    axes = _wrap_elem_in_tuple(axes)
    return f_insert(
        tirx_op.Min(
            dst,
            src,
            axes,
            accum,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def reciprocal(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer | None = None,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Reciprocal all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for reciprocal result.
        When src is omitted, also used as the source (in-place).

    src : Union[BufferRegion, Buffer], optional
        The source buffer region. If omitted, dst is used (in-place).

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    # Expression-form overload: ``reciprocal(value)`` returns the underlying expression.
    from tvm import tirx as _tirx

    if not _is_buffer_or_region(dst):
        return _tirx.reciprocal(dst)
    if src is None:
        src = dst
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(
        tirx_op.Reciprocal(
            dst, src, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def silu(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Compute SiLU (x * sigmoid(x)) for all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for SiLU result.

    src : Union[BufferRegion, Buffer]
        The source buffer region.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    # Expression-form overload: ``silu(value)`` returns the underlying expression.
    from tvm import tirx as _tirx

    if not _is_buffer_or_region(dst):
        return _tirx.silu(dst)
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    return f_insert(
        tirx_op.SiLU(dst, src, workspace=workspace, config=config, dispatch=dispatch, scope=scope)
    )


@ScopedOp
def memset(
    dst: BufferRegion | Buffer,
    value: Expr,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Set all elements in dst to value.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for memset.

    value : Expr
        The value to be set.

    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    return f_insert(
        tirx_op.Memset(
            dst, value, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def maximum(
    dst: BufferRegion | Buffer,
    src1: BufferRegion | Buffer | FloatImm,
    src2: BufferRegion | Buffer | FloatImm,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Maximum all elements in src1 and src2 and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for maximum result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirx_op.Maximum(
            dst, src1, src2, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def minimum(
    dst: BufferRegion | Buffer,
    src1: BufferRegion | Buffer | FloatImm,
    src2: BufferRegion | Buffer | FloatImm,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Minimum all elements in src1 and src2 and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for minimum result.

    src1 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 1, or float.

    src2 : Union[BufferRegion, Buffer, FloatImm]
        The source buffer region 2, or float.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    if isinstance(src1, Buffer):
        src1 = _to_region(src1)
    if isinstance(src2, Buffer):
        src2 = _to_region(src2)
    return f_insert(
        tirx_op.Minimum(
            dst, src1, src2, workspace=workspace, config=config, dispatch=dispatch, scope=scope
        )
    )


@ScopedOp
def exp(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer | None = None,
    bias: BufferRegion | Buffer | FloatImm | None = None,
    scale: FloatImm | None = None,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Exponentiate all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for exp result.
        When src is omitted, also used as the source (in-place).

    src : Union[BufferRegion, Buffer], optional
        The source buffer region. If omitted, dst is used (in-place).

    bias : Optional[Union[BufferRegion, Buffer, FloatImm]]
        The bias of the exp src. Only supported on Trn.

    scale : Optional[FloatImm]
        The scale of the exp src. Only supported on Trn.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    # Expression-form overload: ``exp(value)`` returns the underlying expression.
    from tvm import tirx as _tirx

    if not _is_buffer_or_region(dst):
        return _tirx.exp(dst)
    if src is None:
        src = dst
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)
    return f_insert(
        tirx_op.Exp(
            dst,
            src,
            bias,
            scale,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def exp2(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer | None = None,
    bias: BufferRegion | Buffer | FloatImm | None = None,
    scale: FloatImm | None = None,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Compute base-2 exponential (2^x) of all elements in src and store to dst.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for exp2 result.
        When src is omitted, also used as the source (in-place).

    src : Union[BufferRegion, Buffer], optional
        The source buffer region. If omitted, dst is used (in-place).

    bias : Optional[Union[BufferRegion, Buffer, FloatImm]]
        The bias of the exp2 src.

    scale : Optional[FloatImm]
        The scale of the exp2 src.

    workspace : Dict[str, Buffer]
        The workspace of the operator.
    """
    # Expression-form overload: ``exp2(value)`` returns the underlying expression.
    from tvm import tirx as _tirx

    if not _is_buffer_or_region(dst):
        return _tirx.exp2(dst)
    if src is None:
        src = dst
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    dst = _to_region(dst)
    src = _to_region(src)
    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)
    return f_insert(
        tirx_op.Exp2(
            dst,
            src,
            bias,
            scale,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


def compose_op(
    workspace: dict[str, Buffer] | None = None, dispatch: str | None = None, **kwargs
) -> frame.ComposeOpFrame:
    """Compose a TIRx op.

    Parameters
    ----------
    workspace : Optional[Dict[str, Buffer]]
        The workspace of the operator

    Returns
    -------
    res : frame.ComposeOpFrame
        The result ComposeOpFrame.
    """
    if workspace is None:
        workspace = {}
    config = kwargs or {}
    return _ffi_api.ComposeOp(workspace, config, dispatch)  # pylint: disable=no-member


@ScopedOp
def binary_reduce(
    binary_output: BufferRegion | Buffer,
    reduce_output: BufferRegion | Buffer,
    binary_input1: BufferRegion | Buffer | FloatImm,
    binary_input2: BufferRegion | Buffer | FloatImm,
    binary_op: str | Op,
    reduce_op: str | Op,
    reduce_axes: int | tuple[int] = -1,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Combine a binary operation with a reduction operation.

    Parameters
    ----------
    binary_output : Union[BufferRegion, Buffer]
        The destination buffer region for binary operation result.

    reduce_output : Union[BufferRegion, Buffer]
        The destination buffer region for reduction result.

    binary_input1 : Union[BufferRegion, Buffer, FloatImm]
        The first source input for binary operation.

    binary_input2 : Union[BufferRegion, Buffer, FloatImm]
        The second source input for binary operation.

    binary_op : Union[str, Op]
        The binary operation to perform.

    reduce_op : Union[str, Op]
        The reduction operation to perform.

    reduce_axes : Union[int, Tuple[int]]
        The axes to reduce over.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    config : Dict[str, Any]
        The scheduler configuration.
    """
    if workspace is None:
        workspace = {}
    binary_output = _to_region(binary_output)
    reduce_output = _to_region(reduce_output)
    if isinstance(binary_input1, Buffer):
        binary_input1 = _to_region(binary_input1)
    if isinstance(binary_input2, Buffer):
        binary_input2 = _to_region(binary_input2)
    reduce_axes = _wrap_elem_in_tuple(reduce_axes)

    if isinstance(binary_op, str):
        binary_op = tirx_op.get_tirx_op(binary_op)
    if isinstance(reduce_op, str):
        reduce_op = tirx_op.get_tirx_op(reduce_op)

    config = kwargs or {}
    return f_insert(
        tirx_op.BinaryReduce(
            binary_output,
            reduce_output,
            binary_input1,
            binary_input2,
            binary_op,
            reduce_op,
            reduce_axes,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def unary_reduce(
    unary_output: BufferRegion | Buffer,
    reduce_output: BufferRegion | Buffer,
    unary_input: BufferRegion | Buffer,
    unary_op: str | Op,
    reduce_op: str | Op,
    bias: BufferRegion | Buffer | FloatImm | None = None,
    scale: FloatImm | None = None,
    reduce_axes: int | tuple[int] = -1,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Combine a unary operation with a reduction operation.

    Parameters
    ----------
    unary_output : Union[BufferRegion, Buffer]
        The destination buffer region for unary operation result.

    reduce_output : Union[BufferRegion, Buffer]
        The destination buffer region for reduction result.

    unary_input : Union[BufferRegion, Buffer]
        The source input for unary operation.

    unary_op : Union[str, Op]
        The unary operation to perform.

    reduce_op : Union[str, Op]
        The reduction operation to perform.

    bias : Optional[Union[BufferRegion, Buffer, FloatImm]]
        The bias to apply before unary operation.

    scale : Optional[FloatImm]
        The scale to apply before unary operation.

    reduce_axes : Union[int, Tuple[int]]
        The axes to reduce over.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    config : Dict[str, Any]
        The scheduler configuration.
    """
    if workspace is None:
        workspace = {}
    unary_output = _to_region(unary_output)
    reduce_output = _to_region(reduce_output)
    unary_input = _to_region(unary_input)

    if bias is not None and isinstance(bias, Buffer):
        bias = _to_region(bias)

    reduce_axes = _wrap_elem_in_tuple(reduce_axes)

    if isinstance(unary_op, str):
        unary_op = tirx_op.get_tirx_op(unary_op)
    if isinstance(reduce_op, str):
        reduce_op = tirx_op.get_tirx_op(reduce_op)

    config = kwargs or {}
    return f_insert(
        tirx_op.UnaryReduce(
            unary_output,
            reduce_output,
            unary_input,
            unary_op,
            reduce_op,
            bias,
            scale,
            reduce_axes,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def binary_chain(
    output: BufferRegion | Buffer,
    data: BufferRegion | Buffer,
    operand0: BufferRegion | Buffer | FloatImm,
    operand1: BufferRegion | Buffer | FloatImm,
    op0: str | Op,
    op1: str | Op,
    reverse1: bool = False,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Chain multiple binary operations together.

    if not reverse1:
        output = (operand0 op0 data) op1 operand1
    else:
        output = operand1 op1 (operand0 op0 data)

    Parameters
    ----------
    output : Union[BufferRegion, Buffer]
        The destination buffer region for the result.

    data : Union[BufferRegion, Buffer]
        The input data to operate on.

    operand0 : Union[BufferRegion, Buffer, FloatImm]
        The first operand to combine with data.

    operand1 : Union[BufferRegion, Buffer, FloatImm]
        The second operand to use in chained operation.

    op0 : Union[str, Op]
        The first binary operation to perform.

    op1 : Union[str, Op]
        The second binary operation to perform.

    reverse1 : bool
        Whether to reverse the order of the second binary operation.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    config : Dict[str, Any]
        The scheduler configuration.
    """
    if workspace is None:
        workspace = {}
    output = _to_region(output)
    data = _to_region(data)

    if isinstance(operand0, Buffer):
        operand0 = _to_region(operand0)
    if isinstance(operand1, Buffer):
        operand1 = _to_region(operand1)

    if isinstance(op0, str):
        op0 = tirx_op.get_tirx_op(op0)
    if isinstance(op1, str):
        op1 = tirx_op.get_tirx_op(op1)

    config = kwargs or {}
    return f_insert(
        tirx_op.BinaryChain(
            output,
            data,
            operand0,
            operand1,
            op0,
            op1,
            reverse1,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def reduce_negate(
    output: BufferRegion | Buffer,
    input: BufferRegion | Buffer,
    reduce_op: str | Op,
    reduce_axes: int | tuple[int] = -1,
    accum: bool = False,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Negate the result of a reduction operation.

    Parameters
    ----------
    output : Union[BufferRegion, Buffer]
        The destination buffer region for the negated reduction result.

    input : Union[BufferRegion, Buffer]
        The input buffer region to reduce.

    reduce_axes : Union[int, Tuple[int]]
        The axes to reduce over.

    accum : bool
        Whether to accumulate the result into the output.

    reduce_op : Union[str, Op]
        The reduction operation to perform before negation.

    workspace : Dict[str, Buffer]
        The workspace of the operator.

    config : Dict[str, Any]
        The scheduler configuration.
    """
    if workspace is None:
        workspace = {}
    output = _to_region(output)
    input = _to_region(input)
    reduce_axes = _wrap_elem_in_tuple(reduce_axes)

    if isinstance(reduce_op, str):
        reduce_op = tirx_op.get_tirx_op(reduce_op)

    config = kwargs or {}
    return f_insert(
        tirx_op.ReduceNegate(
            output,
            input,
            reduce_axes,
            accum,
            reduce_op,
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


@ScopedOp
def select(
    dst: BufferRegion | Buffer,
    true_value: BufferRegion | Buffer | FloatImm,
    false_value: BufferRegion | Buffer | FloatImm,
    pred: Predicate | Callable[..., Expr],
    scope: ExecScope | None = None,
):
    """Select between two values based on a predicate.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        The destination buffer region for the result.

    true_value : Union[BufferRegion, Buffer, FloatImm]
        The value to select if the predicate is true.

    false_value : Union[BufferRegion, Buffer, FloatImm]
        The value to select if the predicate is false.

    pred : Union[Predicate, Callable[..., Expr]]
        The predicate to evaluate. The callable should take the same number of arguments as the dimensions of the destination buffer.
    """  # noqa: E501
    dst = _to_region(dst)
    if isinstance(true_value, Buffer):
        true_value = _to_region(true_value)
    if isinstance(false_value, Buffer):
        false_value = _to_region(false_value)
    if not isinstance(pred, Predicate):
        pred = Predicate(pred)
    return f_insert(tirx_op.Select(dst, true_value, false_value, pred, scope=scope))


def reshape(buffer: Buffer, shape: list[Expr]):
    # auto-infer the shape if shape has only one -1
    # for example, if buffer.shape is (1024, 1024) and shape is (128, -1, 2), then the new shape will be (128, 4, 2)  # noqa: E501
    shape = list(shape)
    if -1 in shape and shape.count(-1) == 1:
        size = functools.reduce(lambda x, y: x * y, buffer.shape)
        n_size = functools.reduce(lambda x, y: x * y, [s for s in shape if s != -1], 1)
        shape[shape.index(-1)] = size // n_size
    else:
        assert functools.reduce(lambda x, y: x * y, shape) == functools.reduce(
            lambda x, y: x * y, buffer.shape
        ), (
            "The shape of the buffer "
            + str(buffer.shape)
            + " and the new shape "
            + str(shape)
            + " are not compatible"
        )

    assert buffer.buffer_type == 1
    return decl_buffer(
        shape,
        buffer.dtype,
        buffer.data,
        buffer.strides,
        buffer.elem_offset,
        None,
        buffer.scope(),
        buffer.data_alignment,
        buffer.offset_factor,
        "",
        buffer.axis_separators,
        buffer.layout,
    )


@ScopedOp
def permute_layout(
    dst: BufferRegion | Buffer,
    src: BufferRegion | Buffer,
    workspace: dict[str, Buffer] | None = None,
    dispatch: str | None = None,
    scope: ExecScope | None = None,
    **kwargs,
):
    """Move data so the buffer's bytes are arranged under a different layout.

    Logical shape is preserved (``dst.shape == src.shape``); only the
    byte placement changes (``dst.layout != src.layout``). ``dst`` and
    ``src`` may alias the same SMEM (in-place) or be two distinct buffers.

    Parameters
    ----------
    dst : Union[BufferRegion, Buffer]
        Destination view (carries the target layout).
    src : Union[BufferRegion, Buffer]
        Source view (carries the current layout).
    workspace : Dict[str, Buffer]
        Optional workspace for the operator.
    dispatch : Optional[str]
        Force a specific dispatch variant by name.
    """

    # Promote Buffer to BufferRegion covering the full extent, matching the
    # convention used by ``Tx.<dynamic>`` fallback registration.
    from tvm.tirx import Buffer as _TBuffer

    def _to_region(b):
        if isinstance(b, _TBuffer):
            slices = [slice(None) for _ in range(len(b.shape))]
            return b[slices]
        return b

    config = kwargs or {}
    return f_insert(
        tirx_op.PermuteLayout(
            _to_region(dst),
            _to_region(src),
            workspace=workspace,
            config=config,
            dispatch=dispatch,
            scope=scope,
        )
    )


__all__ = [
    "SMEMPool",
    "ScopeNamespace",
    "ScopedOp",
    "TMEMPool",
    "TMEMStages",
    "add",
    "binary_chain",
    "binary_reduce",
    "cast",
    "cluster",
    "compose_op",
    "copy",
    "copy_async",
    "cta",
    "exp",
    "exp2",
    "fdiv",
    "fill",
    "fma",
    "gemm",
    "gemm_async",
    "max",
    "maximum",
    "memset",
    "meta_class",
    "min",
    "minimum",
    "mul",
    "permute_layout",
    "reciprocal",
    "reduce_negate",
    "select",
    "silu",
    "sqrt",
    "sub",
    "sum",
    "thread",
    "unary_reduce",
    "warp",
    "warpgroup",
    "wg",
    "zero",
]
