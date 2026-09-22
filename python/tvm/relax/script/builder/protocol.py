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
"""Parser-generated binding and statement operations for relax."""

import builtins as _python

import tvm_ffi as _ffi

from tvm import ir as _ir
from tvm import relax as _relax
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import ir as _I
from tvm.script.ir_builder.base import MISSING as _MISSING
from tvm.script.ir_builder.base import BypassBind as _BypassBind
from tvm.script.ir_builder.base import _construction_span
from tvm.script.ir_builder.base import at as _at
from tvm.script.ir_builder.base import source_span as _source_span
from tvm.script.ir_builder.type_var_frame import TypeVarDecl as _TypeVarDecl
from tvm.script.ir_builder.type_var_frame import TypeVarFrame as _TypeVarFrame

from .. import builder as _builder
from . import _ffi_api


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
    """Emit a Relax binding or retain a named frame-owned value."""
    if isinstance(value, _BypassBind):
        return value.value
    name_span = _source_span(span if name_span is None else name_span)
    # Shared dtype constructors return anonymous primitive Vars. Reuse the
    # signature's canonical symbol for declarations, while named aliases and
    # computed primitive expressions retain ordinary Relax binding semantics.
    if not frame_value and _ir.is_prim_var(value) and not value.name:
        # A matching explicit annotation still denotes the same declaration.
        # Mismatches follow the existing native binding validation below.
        ty = None if ty is None else _builder._type(ty)
        if ty is None or _ffi.structural_equal(ty, value.ty):
            return _TypeVarFrame.current().resolve(name, value.ty, span=name_span)
    if isinstance(value, _TypeVarDecl):
        return _TypeVarFrame.current().resolve(name, value.ty, span=name_span)
    if frame_value:
        if isinstance(value, _python.list | _python.tuple | _ir.Array):
            for index, item in enumerate(value):
                bind_(
                    item,
                    name=None if name is None else f"{name}_{index}",
                    span=_source_span(span),
                    name_span=name_span,
                    frame_value=True,
                )
        elif isinstance(value, _ir.Var):
            if name is not None:
                _IRBuilder.name(name, value)
            _at(name_span if name_span is not None else span, value)
        return value
    if declaration:
        if not _ir.is_prim_var(value):
            raise TypeError("A symbol declaration requires a concrete primitive variable")
        if ty is not None and not _ffi.structural_equal(_builder._type(ty), value.ty):
            raise TypeError("The symbol declaration has an incompatible type")
        if previous is not _MISSING:
            if not _ir.is_prim_var(previous) or not _ffi.structural_equal(previous.ty, value.ty):
                raise TypeError("The symbol declaration has an incompatible signature dtype")
            return previous
        if name is not None:
            _IRBuilder.name(name, value)
        return _at(name_span if name_span is not None else span, value)
    if value is _MISSING:
        raise ValueError("Relax bindings require an initializer")
    if isinstance(value, _I.meta_var):
        return value.value
    ty = None if ty is None else _builder._type(ty)
    value = _builder._value(value, ty)
    with _construction_span(span):
        if isinstance(value, _relax.MatchCast):
            if ty is not None and not _ffi.structural_equal(ty, value.ty):
                raise TypeError("The binding annotation differs from the match-cast type")
            result = _ffi_api.EmitMatchCastWithSpan(value.value, value.ty, name_span)
        elif isinstance(value, _relax.Expr):
            result = _ffi_api.EmitWithSpan(value, ty, name_span)
        else:
            return value
    if name is not None:
        _IRBuilder.name(name, result)
    return _at(name_span if name_span is not None else span, result)


def emit_(value, *, span=None):
    """Emit a void expression statement."""
    from tvm.script.ir_builder.base import BypassEmit

    if isinstance(value, BypassEmit):
        return None
    if value is None:
        return
    if not isinstance(value, _relax.Expr):
        raise TypeError(f"Unsupported expression statement value: {type(value).__name__}")
    result = bind_(value, name="_", span=span)
    if not isinstance(result.ty, _ir.TupleType) or len(result.ty.fields) != 0:
        raise ValueError(
            "Non-void expressions must be bound to a variable; "
            f"expression of type {result.ty} was used as a statement"
        )
