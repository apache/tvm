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
"""S-TIR construction reuses common TIRx operations with its own frames."""

from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.script.ir_builder import base as _base
from tvm.tirx.script.builder import *
from tvm.tirx.script.builder import _native
from tvm.tirx.script.builder import arg as _shared_arg
from tvm.tirx.script.builder import bind_ as _shared_bind

from . import _ffi_api
from .frame import PrimFuncFrame as _PrimFuncFrame
from .frame import SBlockFrame as _SBlockFrame
from .ir import (
    axis,
    block_name_suffix_context,
    init,
    reads,
    sblock,
    sblock_alloc_buffer,
    sblock_attr,
    where,
    writes,
)


def prim_func(is_private=False, persistent=False, *, private=None):
    """Create an S-TIR function frame."""
    return _ffi_api.PrimFunc(is_private if private is None else private, persistent)


def function_(*, private=False, persistent=False, decl=False, span=None):
    """Enter an S-TIR declaration or definition frame."""
    native = _ffi_api.DeclFunction(private, persistent) if decl else prim_func(private, persistent)
    return _base.at_(span, native)


def arg(name, annotation, *, span=None):
    """Normalize eagerly constructed parameter buffers for S-TIR."""
    if _tir.is_buffer_var(annotation) and annotation.ty.layout is not None:
        ty = annotation.ty
        annotation = _native.buffer(
            ty.shape,
            ty.dtype,
            strides=ty.strides,
            elem_offset=ty.elem_offset,
            scope=ty.storage_scope,
            align=ty.data_alignment,
            offset_factor=ty.offset_factor,
            layout=None,
            allocated_addr=list(ty.allocated_addr),
            buffer_name=name,
        )
    return _shared_arg(name, annotation, span=span)


def bind_(value=_base.MISSING, **options):
    """Block frames cannot introduce an as-target value."""
    if options.get("frame_value") and isinstance(value, _SBlockFrame):
        raise TypeError("A block does not introduce an as-target value")
    return _shared_bind(value, **options)


def check_well_formed_(function):
    """Validate a completed S-TIR function."""
    from tvm.s_tir import analysis

    try:
        analysis.verify_well_formed(_ir.IRModule.from_expr(function))
    except Exception as error:
        raise ValueError(
            "Program is not well-formed. If this is deliberate, set "
            f"check_well_formed=False in the top-level decorator.\n{error}"
        ) from error


def _check_module_well_formed(module):
    """Validate S-TIR functions independently of other module dialects."""
    from tvm.s_tir import analysis

    functions = {
        gv: fn
        for gv, fn in module.functions.items()
        if isinstance(fn, _tir.PrimFunc) and fn.attrs.get("s_tir", False)
    }
    if functions:
        analysis.verify_well_formed(_ir.IRModule(functions))
