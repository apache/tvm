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
"""Parser-generated binding and statement operations for tirx."""

import builtins as _python
from functools import partial as _partial

import tvm_ffi as _ffi

from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import ir as _I
from tvm.script.ir_builder.base import MISSING as _MISSING
from tvm.script.ir_builder.base import BypassBind as _BypassBind
from tvm.script.ir_builder.base import IRBuilderFrame as _NativeFrame
from tvm.script.ir_builder.base import _construction_span
from tvm.script.ir_builder.base import at as _at
from tvm.script.ir_builder.type_var_frame import TypeVarDecl as _TypeVarDecl
from tvm.script.ir_builder.type_var_frame import TypeVarFrame as _TypeVarFrame

from .. import builder as _builder
from . import frame as _frame
from . import ir as _native


def _name(value, name, span):
    if name is not None:
        _IRBuilder.name(name, value)
    return _at(span, value)


def _enter_concise(frame):
    native = frame.native if isinstance(frame, _builder._Frame) else frame
    native.add_callback(_partial(frame.__exit__, None, None, None))
    return frame.__enter__()


def bind_(
    value=_MISSING,
    *,
    ty=None,
    name=None,
    span=None,
    name_span=None,
    previous=_MISSING,
    declaration=False,
    frame_value=False,
):
    """Construct a named binding under the active primitive function's policy."""
    if isinstance(value, _BypassBind):
        # Scope declarations already own their native binding.  Supply only the
        # missing source name; generic bypass values retain the immediate path.
        if isinstance(value, _native._ScopeIdResult) and _ir.is_prim_var(value.value):
            if not value.value.name and name is not None:
                _IRBuilder.name(name, value.value)
        return value.value
    name_span = span if name_span is None else name_span
    # Dtype constructors return anonymous Vars.
    # Block axes and environment threads are owned by native frames;
    # replacing their Vars would disconnect references from those registrations.
    # Other anonymous declarations share the function-owned type-variable frame.
    if ty is None and not frame_value and _ir.is_prim_var(value):
        for frame in reversed(_IRBuilder.current().frames):
            if isinstance(frame, _frame.SBlockFrame) and _python.any(
                axis.var.same_as(value) for axis in frame.iter_vars
            ):
                if name is not None and _python.any(
                    axis.var.name == name and not axis.var.same_as(value)
                    for axis in frame.iter_vars
                ):
                    raise ValueError(f"Duplicate block axis name {name!r}")
                return _name(value, name, name_span)
        if _python.any(
            isinstance(frame, _frame.PrimFuncFrame)
            and _python.any(thread.same_as(value) for thread in frame.env_threads)
            for frame in _IRBuilder.current().frames
        ):
            return _name(value, name, name_span)
        if not value.name:
            return _TypeVarFrame.current().resolve(name, value.ty, span=name_span)
    if isinstance(value, _TypeVarDecl):
        return _TypeVarFrame.current().resolve(name, value.ty, span=name_span)
    with _construction_span(span):
        if frame_value:
            if isinstance(value, _frame.SBlockFrame):
                raise TypeError("A block does not introduce an as-target value")
            if isinstance(value, _python.list | _python.tuple | _ir.Array):
                for index, item in enumerate(value):
                    bind_(
                        item,
                        name=None if name is None else f"{name}_{index}",
                        span=span,
                        name_span=name_span,
                        frame_value=True,
                    )
            elif isinstance(value, _ir.Var | _tir.IterVar | _tir.Layout):
                _name(value, name, name_span)
            elif isinstance(value, _ir.TensorLoad) and _tir.is_buffer_var(value.source):
                _name(value.source, name, name_span)
            return value
        if previous is not _MISSING and _tir.is_buffer_var(previous):
            shape = previous.ty.shape
            if len(shape) == 1 and _builder.bool(shape[0] == 1):
                if value is _MISSING:
                    raise ValueError("A reassignment requires an initializer")
                _builder.buffer_store(previous, value, [0])
                return previous
        if previous is not _MISSING and isinstance(getattr(previous, "ty", None), _ir.PointerType):
            raise ValueError(f"Pointer variable {name!r} cannot be reassigned")
        if previous is not _MISSING and (
            _tir.is_buffer_var(previous)
            or isinstance(previous, _tir.IterVar)
            or _python.any(
                isinstance(frame, _frame.SBlockFrame)
                and _python.any(axis.var.same_as(previous) for axis in frame.iter_vars)
                for frame in _IRBuilder.current().frames
            )
        ):
            raise ValueError(f"Cannot rebind buffer or block axis {name!r}")
        if declaration:
            if not _ir.is_prim_var(value):
                raise TypeError("A symbol declaration requires a concrete primitive variable")
            if ty is not None:
                annotation = ty() if callable(ty) else ty
                annotation = (
                    annotation.ty if isinstance(annotation, _ir.Expr | _TypeVarDecl) else annotation
                )
                if not _ffi.structural_equal(annotation, value.ty):
                    raise TypeError("The symbol declaration has an incompatible type")
            if previous is not _MISSING:
                if not _ir.is_prim_var(previous) or not _ffi.structural_equal(
                    previous.ty, value.ty
                ):
                    raise TypeError("The symbol declaration has an incompatible signature dtype")
                return previous
            return _name(value, name, name_span)
        if previous is not _MISSING and isinstance(previous, _ir.TensorLoad):
            if value is _MISSING:
                raise ValueError("A reassignment requires an initializer")
            _builder.buffer_store(previous.source, value, list(previous.indices))
            return previous
        if isinstance(value, _I.meta_var):
            return value.value
        if isinstance(ty, _native.LocalVectorAnnotation):
            if value is not _MISSING:
                raise ValueError("Vector annotation does not support an initializer")
            return _name(_native.alloc_local(ty.shape, ty.dtype), name, name_span)
        if isinstance(ty, _native.LetAnnotation):
            if value is _MISSING:
                raise ValueError("An immutable binding requires an initializer")
            value = _builder._as_expr(value)
            variable = _name(ty.as_var(rhs_dtype=value.ty), name, name_span)
            _native.Bind(value, var=variable)
            return variable
        if ty is not None:
            annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
            annotation = (
                annotation.ty if isinstance(annotation, _ir.Expr | _TypeVarDecl) else annotation
            )
            if not isinstance(annotation, _ir.PrimType) or str(annotation) == "handle":
                raise TypeError("Mutable scalar annotations require a primitive scalar type")
            result = _native.local_scalar(str(annotation)).scalar
            _name(result.source, name, name_span)
            if value is not _MISSING:
                _builder.buffer_store(result.source, value, [0])
            return result
        if value is _MISSING:
            raise ValueError("An uninitialized binding requires a scalar type annotation")
        if (
            isinstance(value, _ir.TensorLoad)
            and _tir.is_buffer_var(value.source)
            and not value.source.name
            and len(value.source.ty.shape) == 1
            and isinstance(value.source.ty.shape[0], _tir.IntImm)
            and value.source.ty.shape[0].value == 1
        ):
            _name(value.source, name, name_span)
            return value
        if isinstance(value, _native.scalar_wrapper):
            _name(value.scalar.source, name, name_span)
            return value.scalar
        if isinstance(value, _NativeFrame | _builder._Frame):
            return _name(_enter_concise(value), name, name_span)
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                bind_(item, name=None if name is None else f"{name}_{index}", span=span)
            return value
        if getattr(type(value), "_is_meta_class", False):
            if name is not None:
                _native.name_meta_class_value(name, value)
            return value
        if _tir.is_buffer_var(value) or isinstance(value, _tir.IterVar | _tir.Layout):
            return _name(value, name, name_span)
        if isinstance(value, _ir.Var) and not value.name:
            return _name(value, name, name_span)
        if isinstance(value, _ir.TensorRegion):
            return value
        if not isinstance(value, _ir.Expr | _python.int | _python.float | _python.bool | str):
            return value
        value = _builder._as_expr(value)
        if _ir.is_prim_expr(value):
            result = _native.local_scalar(str(value.ty.dtype)).scalar
            _name(result.source, name, name_span)
            _builder.buffer_store(result.source, value, [0])
            return result
        return _name(_native.Bind(value), name, name_span)


def emit_(value, *, span=None):
    """Consume an expression statement, including effect-only calls."""
    from tvm.script.ir_builder.base import BypassEmit

    if isinstance(value, _BypassBind):
        # Binding bypass does not imply emission bypass. Consume the declared
        # value normally, including each result of a multi-axis declaration.
        values = value.value if isinstance(value.value, (list, tuple)) else (value.value,)
        for item in values:
            emit_(item, span=span)
        return None
    if isinstance(value, BypassEmit):
        return None
    if value is None or isinstance(value, str | _ir.Var):
        return
    with _construction_span(span):
        if isinstance(value, _NativeFrame | _builder._Frame):
            _enter_concise(value)
        elif hasattr(value, "frames"):
            for frame in value.frames:
                _enter_concise(frame)
        elif isinstance(value, _tir.Stmt):
            _native.add_to_parent(value)
        else:
            _native.evaluate(value)
