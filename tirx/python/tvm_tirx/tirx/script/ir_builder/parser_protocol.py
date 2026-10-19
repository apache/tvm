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
"""TIRx implementation of the shared source-to-builder protocol.

Hooks construct native frames and statements using this dialect's IR and operations.
For example, generated ``X.if_(condition)`` creates the native conditional frame;
``X.then_()`` and ``X.else_()`` enter its branches. See the corresponding shared
``tvm_tirx.script.ir_builder.parser_protocol`` hooks for operand and loc contracts.
"""

from __future__ import annotations

import builtins as _python
from collections.abc import Sequence
from typing import Any

import tvm_ffi

import tvm_tirx
from tvm_tirx import ir as _ir
from tvm_tirx import tirx as _tir
from tvm_tirx.ir import IntImm, is_prim_expr
from tvm_tirx.ir.type import _PrimTypeArrayAnnotation
from tvm_tirx.script.ir_builder import _ffi_api as _shared_ffi_api
from tvm_tirx.script.ir_builder import base as _base
from tvm_tirx.script.ir_builder import frame as _shared_frame
from tvm_tirx.script.ir_builder.base import AlreadyEmitted
from tvm_tirx.script.ir_builder.base import IRBuilder as _IRBuilder

# isort: off
from tvm_tirx.script.ir_builder.stmt import (
    _enter_concise as _enter_concise,
    add_to_parent as add_to_parent,
    assert_ as assert_,
    break_ as break_,
    continue_ as continue_,
    else_ as else_,
    evaluate as evaluate,
    grid as grid,
    if_ as if_,
    region as region,
    return_ as return_,
    then_ as then_,
    unpack_ as unpack_,
    while_ as while_,
)

# isort: on
from tvm_tirx.tirx import Expr

from . import _ffi_api, frame
from . import ir as _native
from . import op as _op
from .op import and_ as and_
from .op import eq_ as eq_
from .op import ge_ as ge_
from .op import gt_ as gt_
from .op import if_then_else_ as if_then_else_
from .op import le_ as le_
from .op import lt_ as lt_
from .op import ne_ as ne_
from .op import not_ as not_
from .op import or_ as or_

_Loc = _base.LocationEntry | _ir.Location | None

# --------------------------------------
# Function
# --------------------------------------


def function(
    is_private: bool = False,
    persistent: bool = False,
    *,
    private: bool | None = None,
) -> frame.FunctionFrame:
    """The primitive function statement.

    Parameters
    ----------
    is_private : bool
        Whether the Function is annotated as private.
    persistent : bool
        Whether this is a persistent kernel.
    private : bool
        Alias for ``is_private`` (used in decorator syntax).

    Returns
    -------
    res : frame.FunctionFrame
        The FunctionFrame.
    """
    if private is not None:
        is_private = private
    return _ffi_api.Function(is_private, persistent, False)  # type: ignore[attr-defined] # pylint: disable=no-member


def function_(
    *,
    private: bool = False,
    persistent: bool = False,
    decl: bool = False,
    loc: _Loc = None,
) -> frame.FunctionFrame:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.function_`.

    Private/persistent options pass to the native TIRx function frame.
    The same frame supports declaration and body entry.
    """
    native = _ffi_api.Function(private, persistent, decl)
    return _base.at_(loc, native)


def arg_(name: str, annotation: Any, *, loc: _Loc = None) -> _ir.Var:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.arg_`."""
    if getattr(annotation, "__tvm_optional_annotation__", None) is not None:
        raise TypeError("T.Optional is only supported by @T.jit")
    if callable(annotation) and not isinstance(annotation, _ir.Expr | _ir.Type):
        annotation = annotation()
    if isinstance(annotation, _ir.Type):
        annotation = _ir.Var(name, annotation)
    return _ffi_api.Arg(name, _base.at_(loc, annotation))


def func_name_(name: str) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.func_name_`."""
    return _ffi_api.FuncName(name)


def func_ret_type_(annotation: Any, *, loc: _Loc = None) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.func_ret_type_`."""
    annotation = _base._return_annotation(annotation)
    if callable(annotation) and not isinstance(annotation, _ir.Expr | _ir.Type):
        annotation = annotation()
    if isinstance(annotation, _ir.Expr):
        annotation = annotation.ty
    return _ffi_api.FuncRet(_ir.Type.missing() if annotation is None else annotation)


def func_attr(attrs: dict[str, Any]) -> None:
    """The Function annotation statement.

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotations of the Function.
    """
    _ffi_api.FuncAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def check_well_formed_(function: _tir.Function) -> None:
    """Validate a completed TIRx function."""
    try:
        _tir.analysis.check_well_formed(function)
    except Exception as error:
        raise ValueError(
            "Program is not well-formed. If this is deliberate, set "
            f"check_well_formed=False in the top-level decorator.\n{error}"
        ) from error


# --------------------------------------
# Bindings
# --------------------------------------


def resolve_global_info_(content: Any) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.resolve_global_info_`.

    TIRx does not define global-info selectors.
    """
    raise NotImplementedError("TIRx does not support global-info lookup")


def call_global_var_(function: _ir.GlobalVar, args: Sequence[Any]) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.call_global_var_`."""
    return _op._call_global(function, *args)


def _name(value: Any, name: str | None, loc: _Loc) -> Any:
    if name is not None:
        _IRBuilder.name(name, value)
    return _base.at_(loc, value)


def bind_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    loc: _Loc = None,
    value_loc: _Loc = None,
    name_loc: _Loc = None,
    frame_value: bool = False,
) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.bind_`.

    Returned Vars, including tensors, and metadata retain identity, names and locs.
    Other expressions create native Bind nodes; value_loc belongs to the RHS.
    Explicit typed bindings and frame targets retain their separate contracts.
    """
    name_loc = loc if name_loc is None else name_loc
    if frame_value:
        if isinstance(value, _python.list | _python.tuple | tvm_ffi.Array):
            for index, item in enumerate(value):
                bind_(
                    item,
                    name=None if name is None else f"{name}_{index}",
                    loc=loc,
                    name_loc=name_loc,
                    frame_value=True,
                )
        elif isinstance(value, _ir.Var | _tir.Layout):
            _name(value, name, name_loc)
        elif isinstance(value, _ir.TensorLoad) and _tir.is_tensor_var(value.source):
            _name(value.source, name, name_loc)
        return value
    if isinstance(ty, _native.LetAnnotation):
        if value is _base.MISSING:
            raise ValueError("An immutable binding requires an initializer")
        value = _op._as_expr(value)
        if not isinstance(value, _ir.Var):
            _base.at_(value_loc, value)
        variable = _name(ty.as_var(rhs_dtype=value.ty), name, name_loc)
        _base.with_at_group_(loc, lambda: emit_(_ir.Bind(variable, value)))
        return variable
    if ty is not None:
        annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr | _ir.Type) else ty
        annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        value = _op._as_expr(value)
        if not isinstance(value, _ir.Var):
            _base.at_(value_loc, value)
        variable = _ir.Var(name or "", annotation)
        _base.with_at_group_(loc, lambda: emit_(_ir.Bind(variable, value)))
        return _name(variable, name, name_loc)
    if value is _base.MISSING:
        raise ValueError("An uninitialized binding requires a scalar type annotation")
    if isinstance(value, _base.AlreadyEmitted):
        return _base.at_(value_loc, value)
    if isinstance(value, _base.IRBuilderFrame):
        frame_loc = value_loc if value_loc is not None else loc
        return _name(_enter_concise(_base.at_(frame_loc, value)), name, name_loc)
    # a = existing_var and a = producer() share the same runtime value rule.
    # A Var already owns its declaration, including a newly constructed tensor view.
    if isinstance(value, _ir.Var):
        return value
    if isinstance(value, _ir.TensorRegion):
        return value
    if not isinstance(value, _ir.Expr | _python.int | _python.float | _python.bool | str):
        return value
    value = _base.at_(value_loc, _op._as_expr(value))
    variable = _ir.Var("v", value.ty)
    _base.with_at_group_(loc, lambda: emit_(_ir.Bind(variable, value)))
    return _name(variable, name, name_loc)


def decl_mutable_cell_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    loc: _Loc = None,
    name_loc: _Loc = None,
) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.decl_mutable_cell_`.

    Primitive annotations allocate scalar local storage; vector annotations
    allocate their declared shape. Var declaration producers retain their own effects.
    """
    name_loc = loc if name_loc is None else name_loc
    if _tir.is_tensor_var(value) and ty is not None:
        if isinstance(ty, _native.LetAnnotation):
            annotation = ty.as_var(rhs_dtype=value.ty).ty
        else:
            annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr | _ir.Type) else ty
            annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        _ir.assert_structural_equal(value.ty, annotation)
        return _name(value, name, name_loc)
    if isinstance(ty, _PrimTypeArrayAnnotation):
        if value is not _base.MISSING:
            raise ValueError("Vector annotation does not support an initializer")
        return _name(
            _base.with_at_group_(loc, lambda: _native.alloc_local(ty.shape, ty.dtype)),
            name,
            name_loc,
        )
    if ty is not None:
        annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr | _ir.Type) else ty
        annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        if not isinstance(annotation, _ir.PrimType) or str(annotation) == "handle":
            raise TypeError("Mutable scalar annotations require a primitive scalar type")
        storage = _base.with_at_group_(loc, lambda: _native.local_scalar(str(annotation)))
        if value is not _base.MISSING:
            set_mutable_cell_(storage, value, loc=loc)
    else:
        storage = value
    if isinstance(storage, _ir.TensorLoad):
        _name(storage.source, name, name_loc)
    elif _tir.is_tensor_var(storage):
        _name(storage, name, name_loc)
    else:
        raise TypeError("A mutable declaration requires scalar or vector storage")
    return storage


def set_mutable_cell_(
    target: _ir.TensorLoad | _ir.Var, value: Any, *, loc: _Loc = None
) -> _base.AlreadyEmitted[tvm_tirx.ir.Stmt]:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.set_mutable_cell_`.

    Updates emit a scalar tensor store. Targets must denote scalar storage.
    """
    if isinstance(target, _ir.TensorLoad):
        return _base.at_(loc, tensor_store(target.source, list(target.indices), value))
    elif (
        _tir.is_tensor_var(target)
        and len(target.ty.shape) == 1
        and isinstance(target.ty.shape[0], tvm_tirx.ir.IntImm)
        and target.ty.shape[0].value == 1
    ):
        return _base.at_(loc, tensor_store(target, [0], value))
    else:
        raise TypeError("A mutable assignment requires scalar storage")


def emit_(value: Any, *, loc: _Loc = None) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.emit_`.

    Native statements emit once; receipts are already emitted. Vars, layouts
    and meta_class instances are inert. Sequences are consumed
    elementwise; concise frames close with their owning parent. Other values use
    native expression conversion, retaining its errors for unsupported host values.
    """
    if isinstance(value, _base.AlreadyEmitted):
        _base.at_(loc, value)
        return None
    if (
        value is None
        or isinstance(value, str | _ir.Var | _tir.Layout)
        or getattr(type(value), "_is_meta_class", False)
    ):
        return
    if isinstance(value, list | tuple | tvm_ffi.Array):
        for item in value:
            emit_(item, loc=loc)
        return
    if isinstance(value, _base.IRBuilderFrame):
        _enter_concise(_base.at_(loc, value))
    elif hasattr(value, "frames"):
        for frame in value.frames:
            _enter_concise(_base.at_(loc, frame))
    elif isinstance(value, tvm_tirx.ir.Stmt):
        add_to_parent(_base.at_(loc, value))
        if isinstance(value, _ir.Bind) and _tir.is_tensor_var(value.var):
            _native._record_meta_resource(value.var)
    else:
        # Native conversion owns Python literals; annotate the exact expression
        # it stored, as well as the statement, without converting or emitting twice.
        emitted = evaluate(value)
        _base.at_(loc, emitted.value.value)
        _base.at_(loc, emitted)


def setitem_(
    target: Any, key: Any, value: Any, *, loc: _Loc = None
) -> _base.AlreadyEmitted[tvm_tirx.ir.Stmt]:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.setitem_`."""
    return _base.at_(loc, tensor_store(target, key, value))


def setattr_(
    target: Any, name: str, value: Any, *, loc: _Loc = None
) -> _base.AlreadyEmitted[tvm_tirx.ir.Stmt] | None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.setattr_`."""
    previous = getattr(target, name, _base.MISSING)
    tensor = previous.source if isinstance(previous, _ir.TensorLoad) else previous
    if _tir.is_tensor_var(tensor):
        shape = tensor.ty.shape
        if len(shape) == 1 and _python.bool(shape[0] == 1):
            return set_mutable_cell_(previous, value, loc=loc)
    _python.setattr(target, name, value)


def tensor_store(
    dest: Expr,
    indices: list[Expr | slice],
    value: Expr,
) -> AlreadyEmitted[tvm_tirx.ir.Stmt]:
    """Emit a tensor store and return a receipt for the stored statement.

    Parameters
    ----------
    dest : Expr
        The destination expression.

    indices : List[Union[Expr, slice]]
        The indices location to be stored.

    value : Expr
        The value to be stored.

    Returns
    -------
    result : AlreadyEmitted[Stmt]
        Receipt for the exact stored statement; consuming it does not emit again.

    """
    from tvm_tirx.sym import Analyzer  # pylint: disable=import-outside-toplevel

    if not isinstance(indices, list | tuple | tvm_ffi.Array):
        indices = [indices]

    expr_indices = []
    for index in indices:
        if isinstance(index, slice):
            step = 1 if index.step is None else index.step
            lanes = Analyzer().simplify(  # pylint: disable=redefined-outer-name
                (index.stop - index.start + step - 1) // step
            )
            if lanes == 1:
                expr_indices.append(index.start)
            else:
                expr_indices.append(_op.ramp(index.start, step, lanes))
        else:
            expr_indices.append(index)
    if isinstance(value, bool) and dest.ty.dtype == "bool":
        value = IntImm("bool", value)
    return AlreadyEmitted(_ffi_api.TensorStore(dest, expr_indices, value))


# --------------------------------------
# Special
# --------------------------------------
# Syntax markers and declaration policies are registered by the source namespace.
# They are consumed by the parser before runtime builder calls.

# --------------------------------------
# Control
# --------------------------------------


def for_(
    iterable: Any, *, names: str | Sequence[str] | None = None, loc: _Loc = None
) -> _shared_frame.ForFrame:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.for_`.

    A single native loop returns its scalar Var; multiple loops return their
    sequence. frame.vars remains the stable sequence for source unpacking.
    """
    if isinstance(iterable, _python.range):
        iterable = serial(iterable.start, iterable.stop, step=iterable.step)
    if not isinstance(iterable, _shared_frame.ForFrame):
        raise TypeError("A primitive for loop requires an iteration specification")
    iterable.set_names(names)
    return _base.at_(loc, iterable)


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> _shared_frame.ForFrame:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.range_`."""
    if len(args) == 1:
        # serial constructs the omitted zero in the stop expression's dtype.
        # A Python zero would promote narrow typed bounds to int32.
        return serial(args[0], annotations=annotations)
    elif len(args) == 2:
        args = (*args, None)
    elif len(args) != 3:
        raise TypeError("range expects one to three arguments")
    if isinstance(args[2], _python.int) and args[2] == 0:
        raise ValueError("range step cannot be zero")
    return serial(args[0], args[1], step=args[2], annotations=annotations)


def _for(
    kind: _ir.ForKind,
    start: Expr,
    stop: Expr | None = None,
    *,
    step: Expr | None = None,
    dtype: str | None = None,
    annotations: dict[str, Any] | None = None,
) -> _shared_frame.ForFrame:
    if stop is None:
        stop = start
        # Preserve the endpoint dtype when the lower bound is omitted.
        start = IntImm(start.ty, 0) if is_prim_expr(start) else 0
    return _shared_ffi_api.For(kind, start, stop, step, dtype, annotations)


def serial(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    unroll: bool | int | None = None,
    dtype: str | None = None,
) -> _shared_frame.ForFrame:
    """The serial For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : Expr
        The optional step value of iteration.

    unroll : bool or int, optional
        If True, adds ``{"tirx.pragma_unroll": True}`` annotation, which asks CUDA codegen
        to emit ``#pragma unroll`` while preserving the loop as a C++ ``for``.
        If False, adds ``{"tirx.disable_unroll": True}`` annotation.
        If a positive integer, emits ``#pragma unroll N``. Boolean values are
        handled separately from integers, so ``False`` keeps disabling unrolling.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds. Bounds that do not already have this
        dtype are converted (literals are retyped, other expressions get a Cast).

    Returns
    -------
    res : _shared_frame.ForFrame
        The ForFrame.
    """
    if unroll is not None:
        annotations = dict(annotations) if annotations else {}
        if isinstance(unroll, bool):
            if unroll:
                annotations["tirx.pragma_unroll"] = True
            else:
                annotations["tirx.disable_unroll"] = True
        elif isinstance(unroll, int):
            if unroll < 1:
                raise ValueError("unroll must be a positive integer")
            annotations["tirx.pragma_unroll"] = unroll
        else:
            raise TypeError("unroll must be a bool, a positive integer, or None")
    return _for(_ir.ForKind.DEFAULT, start, stop, step=step, dtype=dtype, annotations=annotations)


def parallel(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    dtype: str | None = None,
) -> _shared_frame.ForFrame:
    """The parallel For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement. On CPU,
        ``{"codegen.parallel_stride_pattern": True}`` assigns iterations cyclically:
        worker ``t`` executes ``t, t + P, t + 2 * P, ...`` for ``P`` workers.
        Absent or false selects contiguous chunks. Each loop selects its own
        policy. When migrating ``pragma_parallel_stride_pattern``, annotate
        every intended loop, including later loops in the same launch outside
        the former attribute body.

    step : Expr
        The optional step value of iteration.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds.

    Returns
    -------
    res : _shared_frame.ForFrame
        The ForFrame.
    """
    return _for(_ir.ForKind.PARALLEL, start, stop, step=step, dtype=dtype, annotations=annotations)


def vectorized(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    dtype: str | None = None,
) -> _shared_frame.ForFrame:
    """The vectorized For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : Expr
        The optional step value of iteration.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds.

    Returns
    -------
    res : _shared_frame.ForFrame
        The ForFrame.
    """
    return _for(
        _ir.ForKind.VECTORIZED, start, stop, step=step, dtype=dtype, annotations=annotations
    )


def unroll(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    dtype: str | None = None,
) -> _shared_frame.ForFrame:
    """The unrolled For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : Expr
        The optional step value of iteration.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds.

    Returns
    -------
    res : _shared_frame.ForFrame
        The ForFrame.
    """
    return _for(_ir.ForKind.UNROLLED, start, stop, step=step, dtype=dtype, annotations=annotations)


def device_entry(
    *launch_values,
    launch=None,
    kernel_attrs=None,
    backend_config=None,
    attrs=None,
    body_params=None,
) -> _shared_frame.RegionFrame:
    """Enter a device kernel with independent CUDA launch and compile settings.

    LaunchConfig values are ordinary region operands. BackendConfig supplies
    static per-entry overrides. Canonical operands and attributes are accepted
    for IR reconstruction. Other backends may use the argument-free entry.
    """
    if launch is not None and (launch_values or attrs is not None):
        raise ValueError("device_entry launch cannot be combined with canonical operands or attrs")
    attributes = dict(attrs) if attrs is not None else {}
    if backend_config is not None:
        from tvm_tirx.backend.config import backend_config_json

        if "backend_config" in attributes:
            raise ValueError("device_entry backend_config cannot duplicate canonical attrs")
        attributes["backend_config"] = backend_config_json(backend_config)
    if launch is None:
        if kernel_attrs is not None:
            raise ValueError("device_entry kernel_attrs require a launch configuration")
        return region("tirx.device_entry", launch_values, attrs=attributes, body_params=body_params)
    from tvm_tirx.backend.cuda.launch._impl import pack_kernel_attrs, pack_launch

    names, values = pack_launch(launch)
    attributes.update(
        {"cuda.launch_fields": names, "cuda.kernel_attrs": pack_kernel_attrs(kernel_attrs, launch)}
    )
    return region("tirx.device_entry", values, attrs=attributes, body_params=body_params)


def device_context(
    device_type: Expr, device_id: Expr, *, attrs=None, body_params=None
) -> _shared_frame.RegionFrame:
    """Supply lexical device context for allocation and packed-call lowering.

    This region does not change the active runtime device.
    """
    return region(
        "tirx.device_context", [device_type, device_id], attrs=attrs, body_params=body_params
    )


def compute_scope(name: str, *, attrs=None, body_params=None) -> _shared_frame.RegionFrame:
    """Outline the body as a named CPU compute helper."""
    return region("tirx.compute_scope", [name], attrs=attrs, body_params=body_params)


def parallel_launch(*, attrs=None, body_params=None) -> _shared_frame.RegionFrame:
    """Launch a CPU worker team around parallel loops and team barriers."""
    return region("tirx.parallel_launch", [], attrs=attrs, body_params=body_params)


def launch_thread(
    thread_tag: str, extent: Expr, *, attrs=None, body_params=None
) -> _shared_frame.RegionFrame:
    """Launch a hardware or virtual thread with a fresh lexical variable.

    The extent determines the variable's scalar integer type. Tags starting with
    ``vthread`` denote virtual threads. This frame works with any TIR builder
    parent, including standalone statement construction.

    Examples
    --------
    .. code-block:: python

        with T.launch_thread("threadIdx.x", 32) as tx:
            T.evaluate(tx)
    """
    return region("tirx.launch_thread", [thread_tag, extent], attrs=attrs, body_params=body_params)


# --------------------------------------
# Operators
# --------------------------------------
# Operator hooks are re-exported directly from the concrete op module above.

func_ret = func_ret_type_
emit = emit_

__all__ = [
    "add_to_parent",
    "and_",
    "arg_",
    "assert_",
    "bind_",
    "break_",
    "call_global_var_",
    "check_well_formed_",
    "compute_scope",
    "continue_",
    "decl_mutable_cell_",
    "device_context",
    "device_entry",
    "else_",
    "emit",
    "emit_",
    "eq_",
    "evaluate",
    "for_",
    "func_attr",
    "func_name_",
    "func_ret",
    "func_ret_type_",
    "function",
    "function_",
    "ge_",
    "grid",
    "gt_",
    "if_",
    "if_then_else_",
    "launch_thread",
    "le_",
    "lt_",
    "ne_",
    "not_",
    "or_",
    "parallel",
    "parallel_launch",
    "range_",
    "region",
    "resolve_global_info_",
    "return_",
    "serial",
    "set_mutable_cell_",
    "setattr_",
    "setitem_",
    "tensor_store",
    "then_",
    "unpack_",
    "unroll",
    "vectorized",
    "while_",
]
