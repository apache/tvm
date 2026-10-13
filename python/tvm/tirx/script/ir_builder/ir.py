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
"""Concrete TIRx types, tensors, allocations and construction metadata."""

import contextlib
import functools
import inspect
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from tvm import DataType, ir
from tvm import ir as _ir
from tvm import tirx as tir
from tvm.ir import Range, TensorRegion, Type
from tvm.ir.location import UNKNOWN_LOC, Location
from tvm.ir.type import _init_type_api, _normalize_type
from tvm.runtime import MemoryScope
from tvm.script.ir_builder.base import MISSING
from tvm.script.ir_builder.ir import Lambda, meta_var
from tvm.script.parser.protocol_registry import (
    register_mutable_decl as _register_mutable_decl,
)
from tvm.script.parser.protocol_registry import (
    register_scalar_annotation as _register_scalar_annotation,
)
from tvm.target import Target

# pylint: disable=unused-import
from tvm.tirx import Expr, IndexMap, Var, is_tensor_var, tile
from tvm.tirx.exec_scope import ExecScope

# import tirx.expr for direct ir construction to pass structural_equal comparison
from tvm.tirx.expr import (
    FloatImm,
    IntImm,
    TensorLoad,
)
from tvm.tirx.layout import Axis, ComposeLayout, Iter, Layout, R, S, TileLayout

from . import _ffi_api

_SCOPE_GLOBAL = MemoryScope("global")
_SCOPE_LOCAL = MemoryScope("local")
_SCOPE_SHARED = MemoryScope("shared")
_SCOPE_SHARED_DYN = MemoryScope("shared.dyn")

# pylint: enable=unused-import


def _get_layout(layout: str | Layout | None, shape: list[Expr], scope: str) -> Layout | None:
    if layout is MISSING or layout is None:
        return None
    if isinstance(layout, Layout):
        return layout
    if layout == "default":
        return TileLayout(S[tuple(shape)])
    raise TypeError("layout must be a Layout, None, or 'default'")


def _normalize_prim_type(dtype) -> ir.PrimType:
    if isinstance(dtype, ir.PrimType):
        return dtype
    if callable(dtype):
        value = dtype()
        ty = getattr(value, "ty", None)
        if isinstance(ty, ir.PrimType):
            return ty
        type_annotation = getattr(value, "type_annotation", None)
        if isinstance(type_annotation, ir.PrimType):
            return type_annotation
    return ir.PrimType(dtype)


def _get_elem_offset(elem_offset, byte_offset, dtype: str):
    assert elem_offset is None or byte_offset is None, (
        "elem_offset and byte_offset cannot be set at the same time"
    )
    if elem_offset is not None:
        return elem_offset
    if byte_offset is None:
        return None
    return byte_offset * 8 // (DataType(_normalize_prim_type(dtype).dtype).bits)


_meta_construction_state = threading.local()

_THIS_FILE = __file__


class _MetaResourceRecord:
    """Resource created while constructing a meta_class instance."""

    def __init__(
        self, value: Any, filename: str, lineno: int, colno: int | None, code: str
    ) -> None:
        self.value = value
        self.filename = filename
        self.lineno = lineno
        self.colno = colno
        self.code = code


class _MetaConstructionScope:
    """Thread-local construction scope for a single meta_class __init__ call."""

    def __init__(self, instance: Any, cls: type) -> None:
        self.instance = instance
        self.cls = cls
        self.created: list[_MetaResourceRecord] = []

    def record(self, value: Any, frame_info: inspect.FrameInfo) -> None:
        positions = getattr(frame_info, "positions", None)
        colno = None
        if positions is not None and positions.col_offset is not None:
            colno = positions.col_offset + 1
        code = frame_info.code_context[0].strip() if frame_info.code_context else ""
        self.created.append(
            _MetaResourceRecord(
                value=value,
                filename=frame_info.filename,
                lineno=frame_info.lineno,
                colno=colno,
                code=code,
            )
        )


def _meta_construction_stack() -> list[_MetaConstructionScope]:
    stack = getattr(_meta_construction_state, "stack", None)
    if stack is None:
        stack = []
        _meta_construction_state.stack = stack
    return stack


def _current_meta_construction_scope() -> _MetaConstructionScope | None:
    stack = _meta_construction_stack()
    return stack[-1] if stack else None


@contextlib.contextmanager
def _with_meta_construction_scope(instance: Any, cls: type):
    scope = _MetaConstructionScope(instance, cls)
    stack = _meta_construction_stack()
    stack.append(scope)
    try:
        yield scope
    finally:
        stack.pop()


def _record_meta_resource(value: Any, skip_frames: int = 2) -> None:
    scope = _current_meta_construction_scope()
    if scope is not None:
        stack = inspect.stack(context=1)
        frame_info = None
        for candidate in stack[2:]:
            if candidate.filename != _THIS_FILE:
                frame_info = candidate
                break
        if frame_info is None:
            frame_info = stack[min(skip_frames + 1, len(stack) - 1)]
        scope.record(value, frame_info)


_tensor_type = _register_mutable_decl("tirx.Tensor", syntax="parameter")(tir.TensorType)


def Tuple(*fields: Type) -> Type:  # pylint: disable=invalid-name
    """Construct a tuple type for a TIRx function or binding annotation."""
    normalized_fields = []
    for field in fields:
        if isinstance(field, Type) or (isinstance(field, type) and issubclass(field, Type)):
            field = _normalize_type(field)
        elif callable(field) and not isinstance(field, Expr):
            field = field()
        if isinstance(field, Expr):
            field = field.ty
        normalized_fields.append(_normalize_type(field))
    return ir.TupleType(normalized_fields)


def elected():
    """Stub that rejects the removed ``T.elected()`` sugar.

    Write the explicit form instead::

        if T.cuda.elect_sync():
            ...                         # thread is the default scope
    """
    raise RuntimeError(
        "T.elected() is no longer available. Write explicitly: "
        "`if T.cuda.elect_sync(): ...` (thread is the default scope)"
    )


def alloc_tensor(
    placement=None,
    *,
    ty_args,
    attrs=None,
    annotations: dict[str, Any] | None = None,
    loc: Location = UNKNOWN_LOC,
    ty=None,
) -> ir.Call:
    """Allocate the tensor described by one explicit ``Tensor`` type argument.

    Optional placement is an ordinary tuple operand. Allocation annotations
    belong to ``attrs``; ``annotations`` accepts a Python dictionary whose
    scalar values are normalized to typed IR constants.
    """
    if attrs is not None and annotations is not None:
        raise TypeError("alloc_tensor cannot mix attrs and annotations")

    def normalize(value):
        if isinstance(value, bool):
            return tir.IntImm("bool", int(value))
        if isinstance(value, int):
            return tir.IntImm("int32", value)
        if isinstance(value, float):
            return tir.FloatImm("float32", value)
        return value

    if attrs is None:
        normalized = {key: normalize(value) for key, value in (annotations or {}).items()}
        if "tensor_allocated_addr" in normalized:
            if placement is not None:
                raise ValueError("Allocation placement was specified twice")
            placement = normalized.pop("tensor_allocated_addr")
        attrs = ir.DictAttrs(normalized)
    args = []
    if placement is not None:
        if not isinstance(placement, ir.Tuple):
            placement = placement if isinstance(placement, list | tuple | ir.Array) else [placement]
            placement = ir.Tuple([normalize(value) for value in placement])
        args.append(placement)
    return ir.Call("tirx.alloc_tensor", args, attrs=attrs, ty_args=ty_args, loc=loc, ty=ty)


def _bind_tensor(value):
    from tvm.script.ir_builder.stmt import bind  # pylint: disable=import-outside-toplevel

    tensor = bind(value)
    _record_meta_resource(tensor, skip_frames=2)
    return tensor


def _alloc_tensor(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    byte_offset=None,
    scope=_SCOPE_GLOBAL,
    align=-1,
    offset_factor=0,
    layout=MISSING,
    allocated_addr=None,
    annotations=None,
):
    """Implement separately named allocation conveniences using the canonical Op."""
    scope = MemoryScope(scope)
    if allocated_addr is not None:
        if scope in (_SCOPE_GLOBAL, _SCOPE_SHARED, _SCOPE_SHARED_DYN, _SCOPE_LOCAL):
            raise ValueError(f"{scope} does not support allocation placement")
        if data is not None or elem_offset is not None or byte_offset is not None or offset_factor:
            raise ValueError(
                "Allocation placement cannot be combined with pointer or offset metadata"
            )
    tensor_type = _tensor_type(
        shape,
        dtype,
        scope,
        strides,
        elem_offset,
        align,
        offset_factor,
        layout,
        data=data,
        byte_offset=byte_offset,
    )
    return _bind_tensor(
        alloc_tensor(allocated_addr, ty_args=[tensor_type], annotations=annotations)
    )


class LetAnnotation:
    """Marker for an immutable Bind, created by ``T.let`` or ``T.let[type]``.

    Usage in TVMScript::

        x: T.let[T.int32] = expr  # Bind with an explicit type
        x: T.let = expr          # Bind with an inferred RHS type
    """

    def __init__(self, type_spec=None):
        self.type_spec = type_spec

    def __class_getitem__(cls, item):
        return LetAnnotation(item)

    def __getitem__(self, item):
        return LetAnnotation(item)

    def as_var(self, rhs_dtype=None):
        """Resolve to a tir.Var."""
        if self.type_spec is not None:
            if isinstance(self.type_spec, ir.Var):
                return self.type_spec  # Already a Var (e.g. T.handle(...))
            elif isinstance(self.type_spec, Type):
                return Var("", self.type_spec)
            elif callable(self.type_spec):
                return self.type_spec()
            else:
                raise TypeError(f"Invalid type for T.let: {self.type_spec}")
        elif rhs_dtype is not None:
            rhs_ty = rhs_dtype if isinstance(rhs_dtype, Type) else ir.PrimType(rhs_dtype)
            return Var("", rhs_ty)
        else:
            raise TypeError("T.let requires either a type or an RHS value")


let = LetAnnotation()  # Singleton for T.let (no subscript)


def decl_tensor(
    data,
    *,
    ty_args,
    attrs=None,
    loc: Location = UNKNOWN_LOC,
    ty=None,
) -> ir.Call:
    """Declare a pointer-backed tensor with one explicit ``Tensor`` type argument."""
    return ir.Call("tirx.decl_tensor", [data], attrs=attrs, ty_args=ty_args, loc=loc, ty=ty)


alloc_shared = functools.partial(_alloc_tensor, scope="shared")

_register_mutable_decl("tirx.alloc_shared")(alloc_shared)

alloc_local = functools.partial(_alloc_tensor, scope="local")

_register_mutable_decl("tirx.alloc_local")(alloc_local)

smem = _register_mutable_decl("tirx.smem")(alloc_shared)

tmem = functools.partial(_alloc_tensor, scope="tmem")


@_register_mutable_decl("tirx.alloc_scalar")
def alloc_scalar(
    dtype: str = "float32",
    scope: str | MemoryScope = _SCOPE_GLOBAL,
    *,
    annotations: dict[str, Any] | None = None,
) -> _ir.TensorLoad:
    """Allocate a zero-dimensional tensor (scalar), with optional allocation annotations."""
    buf = _alloc_tensor(shape=(1,), dtype=dtype, scope=scope, layout=None, annotations=annotations)
    assert is_tensor_var(buf)
    scalar = buf[0]
    return scalar


@_register_mutable_decl("tirx.decl_scalar")
def decl_scalar(dtype, data, scope, elem_offset=None, byte_offset=None) -> _ir.TensorLoad:
    """Declare a zero-dimensional tensor (scalar) from a pointer."""
    buf = _bind_tensor(
        decl_tensor(
            data,
            ty_args=[
                _tensor_type(
                    (1,),
                    dtype,
                    scope=scope,
                    elem_offset=_get_elem_offset(elem_offset, byte_offset, dtype),
                    align=-1,
                    layout=None,
                )
            ],
        )
    )
    assert is_tensor_var(buf)
    scalar = buf[0]
    return scalar


@_register_mutable_decl("tirx.shared_scalar")
def shared_scalar(
    dtype: str = "float32", *, annotations: dict[str, Any] | None = None
) -> _ir.TensorLoad:
    """Allocate a zero-dimensional tensor in shared memory."""
    return alloc_scalar(dtype=dtype, scope="shared", annotations=annotations)


@_register_mutable_decl("tirx.local_scalar")
def local_scalar(
    dtype: str = "float32", *, annotations: dict[str, Any] | None = None
) -> _ir.TensorLoad:
    """Allocate a zero-dimensional tensor in local memory."""
    return alloc_scalar(dtype=dtype, scope="local", annotations=annotations)


def _is_meta_class_instance(value: Any) -> bool:
    return getattr(type(value), "_is_meta_class", False)


def _meta_resource_for_value(value: Any) -> Any | None:
    if isinstance(value, _ir.TensorLoad):
        return value.source
    if is_tensor_var(value):
        return value
    return None


def _same_meta_resource(lhs: Any, rhs: Any) -> bool:
    same_as = getattr(lhs, "same_as", None)
    if same_as is not None:
        try:
            return bool(same_as(rhs))
        except TypeError:
            pass
    return lhs is rhs


def _collect_meta_resources(value: Any, visited: set[int] | None = None) -> list[Any]:
    if visited is None:
        visited = set()
    obj_id = id(value)
    if obj_id in visited:
        return []
    visited.add(obj_id)

    resource = _meta_resource_for_value(value)
    if resource is not None:
        return [resource]
    if _is_meta_class_instance(value):
        owned = []
        for field_name, field_value in vars(value).items():
            if field_name.startswith("_tirx_"):
                continue
            owned.extend(_collect_meta_resources(field_value, visited))
        return owned
    if isinstance(value, list | tuple):
        owned = []
        for item in value:
            owned.extend(_collect_meta_resources(item, visited))
        return owned
    if isinstance(value, dict):
        owned = []
        for item in value.values():
            owned.extend(_collect_meta_resources(item, visited))
        return owned
    return []


def _format_unowned_meta_resource_error(cls: type, record: _MetaResourceRecord, total: int) -> str:
    count = "" if total == 1 else f" ({total} total)"
    location = f"{record.filename}:{record.lineno}"
    if record.colno is not None:
        location = f"{location}:{record.colno}"
    message = [
        f"TIRx meta_class constructor created an unowned resource{count}.",
        f"  class: {cls.__name__}",
        f"  location: {location}",
    ]
    if record.code:
        message.extend(["", f"  {record.code}", "  ^ resource must be assigned to self.<field>"])
    message.extend(
        [
            "",
            "Resources created in a meta_class constructor must be reachable from the",
            "constructed instance.",
            "unowned resource at "
            f"{location}: assign it to self.<field>, or move the allocation into a "
            "parser-owned assignment.",
        ]
    )
    return "\n".join(message)


def _validate_meta_construction_scope(scope: _MetaConstructionScope) -> None:
    if not scope.created:
        object.__setattr__(scope.instance, "_tirx_meta_owned_resources", [])
        return
    created_resources = [record.value for record in scope.created]
    owned_resources = _collect_meta_resources(scope.instance)
    missing = [
        record
        for record in scope.created
        if not any(_same_meta_resource(record.value, owned) for owned in owned_resources)
    ]
    if missing:
        raise ValueError(_format_unowned_meta_resource_error(scope.cls, missing[0], len(missing)))
    object.__setattr__(scope.instance, "_tirx_meta_owned_resources", created_resources)


def _prim_type(dtype: str) -> ir.PrimType:
    """Export a primitive type with its scalar and local-declaration syntax."""
    value = ir.PrimType(dtype)
    _register_scalar_annotation(f"tirx.{dtype}", value, dtype=dtype)
    _register_mutable_decl(f"tirx.{dtype}", syntax="annotation")(value)
    return value


def static_assert(x: Any, message: str = ""):
    assert x, message


int8 = _prim_type("int8")

int16 = _prim_type("int16")

int32 = _prim_type("int32")

int64 = _prim_type("int64")

int8x2 = _prim_type("int8x2")

int16x2 = _prim_type("int16x2")

int32x2 = _prim_type("int32x2")

int64x2 = _prim_type("int64x2")

int8x4 = _prim_type("int8x4")

int16x4 = _prim_type("int16x4")

int32x4 = _prim_type("int32x4")

int64x4 = _prim_type("int64x4")

int8x8 = _prim_type("int8x8")

int16x8 = _prim_type("int16x8")

int32x8 = _prim_type("int32x8")

int64x8 = _prim_type("int64x8")

int8x16 = _prim_type("int8x16")

int16x16 = _prim_type("int16x16")

int32x16 = _prim_type("int32x16")

int64x16 = _prim_type("int64x16")

int8x32 = _prim_type("int8x32")

int16x32 = _prim_type("int16x32")

int32x32 = _prim_type("int32x32")

int64x32 = _prim_type("int64x32")

int8x64 = _prim_type("int8x64")

int16x64 = _prim_type("int16x64")

int32x64 = _prim_type("int32x64")

int64x64 = _prim_type("int64x64")

uint8 = _prim_type("uint8")

uint16 = _prim_type("uint16")

uint32 = _prim_type("uint32")

uint64 = _prim_type("uint64")

uint8x2 = _prim_type("uint8x2")

uint16x2 = _prim_type("uint16x2")

uint32x2 = _prim_type("uint32x2")

uint64x2 = _prim_type("uint64x2")

uint8x4 = _prim_type("uint8x4")

uint16x4 = _prim_type("uint16x4")

uint32x4 = _prim_type("uint32x4")

uint64x4 = _prim_type("uint64x4")

uint8x8 = _prim_type("uint8x8")

uint16x8 = _prim_type("uint16x8")

uint32x8 = _prim_type("uint32x8")

uint64x8 = _prim_type("uint64x8")

uint8x16 = _prim_type("uint8x16")

uint16x16 = _prim_type("uint16x16")

uint32x16 = _prim_type("uint32x16")

uint64x16 = _prim_type("uint64x16")

uint8x32 = _prim_type("uint8x32")

uint16x32 = _prim_type("uint16x32")

uint32x32 = _prim_type("uint32x32")

uint64x32 = _prim_type("uint64x32")

uint8x64 = _prim_type("uint8x64")

uint16x64 = _prim_type("uint16x64")

uint32x64 = _prim_type("uint32x64")

uint64x64 = _prim_type("uint64x64")

float16 = _prim_type("float16")

float32 = _prim_type("float32")

float64 = _prim_type("float64")

float16x2 = _prim_type("float16x2")

float32x2 = _prim_type("float32x2")

float64x2 = _prim_type("float64x2")

float16x4 = _prim_type("float16x4")

float32x4 = _prim_type("float32x4")

float64x4 = _prim_type("float64x4")

float16x8 = _prim_type("float16x8")

float32x8 = _prim_type("float32x8")

float64x8 = _prim_type("float64x8")

float16x16 = _prim_type("float16x16")

float32x16 = _prim_type("float32x16")

float64x16 = _prim_type("float64x16")

float16x32 = _prim_type("float16x32")

float32x32 = _prim_type("float32x32")

float64x32 = _prim_type("float64x32")

float16x64 = _prim_type("float16x64")

float32x64 = _prim_type("float32x64")

float64x64 = _prim_type("float64x64")

float8_e3m4 = _prim_type("float8_e3m4")

float8_e3m4x2 = _prim_type("float8_e3m4x2")

float8_e3m4x4 = _prim_type("float8_e3m4x4")

float8_e3m4x8 = _prim_type("float8_e3m4x8")

float8_e3m4x16 = _prim_type("float8_e3m4x16")

float8_e3m4x32 = _prim_type("float8_e3m4x32")

float8_e3m4x64 = _prim_type("float8_e3m4x64")

float8_e4m3 = _prim_type("float8_e4m3")

float8_e4m3x2 = _prim_type("float8_e4m3x2")

float8_e4m3x4 = _prim_type("float8_e4m3x4")

float8_e4m3x8 = _prim_type("float8_e4m3x8")

float8_e4m3x16 = _prim_type("float8_e4m3x16")

float8_e4m3x32 = _prim_type("float8_e4m3x32")

float8_e4m3x64 = _prim_type("float8_e4m3x64")

float8_e4m3b11fnuz = _prim_type("float8_e4m3b11fnuz")

float8_e4m3b11fnuzx2 = _prim_type("float8_e4m3b11fnuzx2")

float8_e4m3b11fnuzx4 = _prim_type("float8_e4m3b11fnuzx4")

float8_e4m3b11fnuzx8 = _prim_type("float8_e4m3b11fnuzx8")

float8_e4m3b11fnuzx16 = _prim_type("float8_e4m3b11fnuzx16")

float8_e4m3b11fnuzx32 = _prim_type("float8_e4m3b11fnuzx32")

float8_e4m3b11fnuzx64 = _prim_type("float8_e4m3b11fnuzx64")

float8_e4m3fn = _prim_type("float8_e4m3fn")

float8_e4m3fnx2 = _prim_type("float8_e4m3fnx2")

float8_e4m3fnx4 = _prim_type("float8_e4m3fnx4")

float8_e4m3fnx8 = _prim_type("float8_e4m3fnx8")

float8_e4m3fnx16 = _prim_type("float8_e4m3fnx16")

float8_e4m3fnx32 = _prim_type("float8_e4m3fnx32")

float8_e4m3fnx64 = _prim_type("float8_e4m3fnx64")

float8_e4m3fnuz = _prim_type("float8_e4m3fnuz")

float8_e4m3fnuzx2 = _prim_type("float8_e4m3fnuzx2")

float8_e4m3fnuzx4 = _prim_type("float8_e4m3fnuzx4")

float8_e4m3fnuzx8 = _prim_type("float8_e4m3fnuzx8")

float8_e4m3fnuzx16 = _prim_type("float8_e4m3fnuzx16")

float8_e4m3fnuzx32 = _prim_type("float8_e4m3fnuzx32")

float8_e4m3fnuzx64 = _prim_type("float8_e4m3fnuzx64")

float8_e5m2 = _prim_type("float8_e5m2")

float8_e5m2x2 = _prim_type("float8_e5m2x2")

float8_e5m2x4 = _prim_type("float8_e5m2x4")

float8_e5m2x8 = _prim_type("float8_e5m2x8")

float8_e5m2x16 = _prim_type("float8_e5m2x16")

float8_e5m2x32 = _prim_type("float8_e5m2x32")

float8_e5m2x64 = _prim_type("float8_e5m2x64")

float8_e5m2fnuz = _prim_type("float8_e5m2fnuz")

float8_e5m2fnuzx2 = _prim_type("float8_e5m2fnuzx2")

float8_e5m2fnuzx4 = _prim_type("float8_e5m2fnuzx4")

float8_e5m2fnuzx8 = _prim_type("float8_e5m2fnuzx8")

float8_e5m2fnuzx16 = _prim_type("float8_e5m2fnuzx16")

float8_e5m2fnuzx32 = _prim_type("float8_e5m2fnuzx32")

float8_e5m2fnuzx64 = _prim_type("float8_e5m2fnuzx64")

float8_e8m0fnu = _prim_type("float8_e8m0fnu")

float8_e8m0fnux2 = _prim_type("float8_e8m0fnux2")

float8_e8m0fnux4 = _prim_type("float8_e8m0fnux4")

float8_e8m0fnux8 = _prim_type("float8_e8m0fnux8")

float8_e8m0fnux16 = _prim_type("float8_e8m0fnux16")

float8_e8m0fnux32 = _prim_type("float8_e8m0fnux32")

float8_e8m0fnux64 = _prim_type("float8_e8m0fnux64")

float6_e2m3fn = _prim_type("float6_e2m3fn")

float6_e2m3fnx2 = _prim_type("float6_e2m3fnx2")

float6_e2m3fnx4 = _prim_type("float6_e2m3fnx4")

float6_e2m3fnx8 = _prim_type("float6_e2m3fnx8")

float6_e2m3fnx16 = _prim_type("float6_e2m3fnx16")

float6_e2m3fnx32 = _prim_type("float6_e2m3fnx32")

float6_e2m3fnx64 = _prim_type("float6_e2m3fnx64")

float6_e3m2fn = _prim_type("float6_e3m2fn")

float6_e3m2fnx2 = _prim_type("float6_e3m2fnx2")

float6_e3m2fnx4 = _prim_type("float6_e3m2fnx4")

float6_e3m2fnx8 = _prim_type("float6_e3m2fnx8")

float6_e3m2fnx16 = _prim_type("float6_e3m2fnx16")

float6_e3m2fnx32 = _prim_type("float6_e3m2fnx32")

float6_e3m2fnx64 = _prim_type("float6_e3m2fnx64")

float4_e2m1fn = _prim_type("float4_e2m1fn")

float4_e2m1fnx2 = _prim_type("float4_e2m1fnx2")

float4_e2m1fnx4 = _prim_type("float4_e2m1fnx4")

float4_e2m1fnx8 = _prim_type("float4_e2m1fnx8")

float4_e2m1fnx16 = _prim_type("float4_e2m1fnx16")

float4_e2m1fnx32 = _prim_type("float4_e2m1fnx32")

float4_e2m1fnx64 = _prim_type("float4_e2m1fnx64")

bfloat16 = _prim_type("bfloat16")

# Shorthand aliases
f16 = _register_scalar_annotation("tirx.f16", float16, dtype="float16")
_register_mutable_decl("tirx.f16", syntax="annotation")(f16)
f32 = _register_scalar_annotation("tirx.f32", float32, dtype="float32")
_register_mutable_decl("tirx.f32", syntax="annotation")(f32)
f64 = _register_scalar_annotation("tirx.f64", float64, dtype="float64")
_register_mutable_decl("tirx.f64", syntax="annotation")(f64)
bf16 = _register_scalar_annotation("tirx.bf16", bfloat16, dtype="bfloat16")
_register_mutable_decl("tirx.bf16", syntax="annotation")(bf16)
i8 = _register_scalar_annotation("tirx.i8", int8, dtype="int8")
_register_mutable_decl("tirx.i8", syntax="annotation")(i8)
i16 = _register_scalar_annotation("tirx.i16", int16, dtype="int16")
_register_mutable_decl("tirx.i16", syntax="annotation")(i16)
i32 = _register_scalar_annotation("tirx.i32", int32, dtype="int32")
_register_mutable_decl("tirx.i32", syntax="annotation")(i32)
i64 = _register_scalar_annotation("tirx.i64", int64, dtype="int64")
_register_mutable_decl("tirx.i64", syntax="annotation")(i64)
u8 = _register_scalar_annotation("tirx.u8", uint8, dtype="uint8")
_register_mutable_decl("tirx.u8", syntax="annotation")(u8)
u16 = _register_scalar_annotation("tirx.u16", uint16, dtype="uint16")
_register_mutable_decl("tirx.u16", syntax="annotation")(u16)
u32 = _register_scalar_annotation("tirx.u32", uint32, dtype="uint32")
_register_mutable_decl("tirx.u32", syntax="annotation")(u32)
u64 = _register_scalar_annotation("tirx.u64", uint64, dtype="uint64")

_register_mutable_decl("tirx.u64", syntax="annotation")(u64)


boolean = _prim_type("bool")


def handle(
    dtype: str | None = None,
    storage_scope: str | MemoryScope = _SCOPE_GLOBAL,
) -> Var:
    """Create a TIR var that represents a pointer.

    Parameters
    ----------
    dtype: str | None
        The data type of the pointer. If omitted, construct an opaque handle.

    storage_scope: str
        The storage scope of the pointer.

    Returns
    -------
    res : Expr
        The new tirx.Var with type handle or casted expression with type handle.
    """
    if dtype in ("TensorMap", "tensormap", "CUtensorMap", "cuTensorMap"):
        return _ffi_api.TensorMap()  # type: ignore[attr-defined] # pylint: disable=no-member
    return _ffi_api.Handle(  # type: ignore[attr-defined] # pylint: disable=no-member
        dtype,
        storage_scope,
    )


# TensorMap denotes the descriptor element type; Ptr supplies its pointer wrapper.
TensorMap = tir.TensorMapType


void = _prim_type("void")


def ptr(dtype: str, storage_scope: str | MemoryScope = _SCOPE_GLOBAL) -> Var:
    """The pointer declaration function.

    Parameters
    ----------
    dtype : str
        The data type of the pointer.

    storage_scope : str
        The storage scope of the pointer.

    Returns
    -------
    res : Var
        The pointer.
    """
    return _ffi_api.Ptr(dtype, storage_scope)  # type: ignore[attr-defined] # pylint: disable=no-member


def index_map(
    mapping: Callable,
    *,
    inverse_index_map: Callable | None = None,
    index_dtype: str = "int64",
) -> IndexMap:
    """Create a TIR Index mapping"""
    return IndexMap.from_func(mapping, inverse_index_map=inverse_index_map, index_dtype=index_dtype)


def target(
    target_config: dict | str,
    host: dict | str | Target | None = None,
) -> Target:
    """
    Create a target

    Parameters
    ----------
    target_config : Union[Dict, str]
        The target configuration.

    host : Optional[Union[Dict, str, Target]]
        The target configuration.

    Returns
    -------
    res : Target
        The target.
    """
    if not isinstance(target_config, str | dict):
        raise ValueError(
            f"T.target expected a config dict or string, but got {type(target_config)}"
        )
    if host is not None and not isinstance(host, str | dict | Target):
        raise ValueError(
            "T.target expected the host to be "
            "a config dict, string, or T.target, "
            f"but got {type(host)}"
        )
    if isinstance(target_config, dict) and "host" in target_config and host is not None:
        raise ValueError(
            "T.target expects to either receive the host "
            "as part of the target's config dictionary, "
            "or as a separate argument, but not both."
        )
    return Target(target_config, host)


if TYPE_CHECKING:
    C = TypeVar("C")

    def meta_class(cls: C) -> C:
        return cls

else:

    def _install_meta_class(cls):
        if cls.__dict__.get("_tirx_meta_class_installed", False):
            cls._is_meta_class = True
            return cls

        original_init = getattr(cls, "__init__", object.__init__)
        original_init_subclass = getattr(cls, "__init_subclass__", None)

        def __init__(self, *args, **kwargs):
            with _with_meta_construction_scope(self, type(self)) as scope:
                original_init(self, *args, **kwargs)
                _validate_meta_construction_scope(scope)

        @classmethod
        def __init_subclass__(subcls, **kwargs):
            if original_init_subclass is not None:
                original_init_subclass(**kwargs)
            _install_meta_class(subcls)

        cls.__init__ = __init__
        cls.__init_subclass__ = __init_subclass__
        cls._is_meta_class = True
        cls._tirx_meta_class_installed = True
        return cls

    def meta_class(cls):
        """Decorator for utility classes used inside @T.function.

        Instances of decorated classes are treated as parser meta values.
        """
        return _install_meta_class(cls)


# These are aliases of canonical bindings, including their constructor signatures.
Ptr = ir.PtrType
Tensor = tir.TensorType

__all__ = [
    "Axis",
    "ComposeLayout",
    "ExecScope",
    "FloatImm",
    "IntImm",
    "Iter",
    "Lambda",
    "Layout",
    "LetAnnotation",
    "MemoryScope",
    "Ptr",
    "R",
    "Range",
    "S",
    "Tensor",
    "TensorLoad",
    "TensorMap",
    "TensorRegion",
    "TileLayout",
    "Tuple",
    "Var",
    "alloc_local",
    "alloc_scalar",
    "alloc_shared",
    "alloc_tensor",
    "bf16",
    "bfloat16",
    "boolean",
    "decl_scalar",
    "decl_tensor",
    "f16",
    "f32",
    "f64",
    "float4_e2m1fn",
    "float4_e2m1fnx2",
    "float4_e2m1fnx4",
    "float4_e2m1fnx8",
    "float4_e2m1fnx16",
    "float4_e2m1fnx32",
    "float4_e2m1fnx64",
    "float6_e2m3fn",
    "float6_e2m3fnx2",
    "float6_e2m3fnx4",
    "float6_e2m3fnx8",
    "float6_e2m3fnx16",
    "float6_e2m3fnx32",
    "float6_e2m3fnx64",
    "float6_e3m2fn",
    "float6_e3m2fnx2",
    "float6_e3m2fnx4",
    "float6_e3m2fnx8",
    "float6_e3m2fnx16",
    "float6_e3m2fnx32",
    "float6_e3m2fnx64",
    "float8_e3m4",
    "float8_e3m4x2",
    "float8_e3m4x4",
    "float8_e3m4x8",
    "float8_e3m4x16",
    "float8_e3m4x32",
    "float8_e3m4x64",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3b11fnuzx2",
    "float8_e4m3b11fnuzx4",
    "float8_e4m3b11fnuzx8",
    "float8_e4m3b11fnuzx16",
    "float8_e4m3b11fnuzx32",
    "float8_e4m3b11fnuzx64",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e4m3fnuzx2",
    "float8_e4m3fnuzx4",
    "float8_e4m3fnuzx8",
    "float8_e4m3fnuzx16",
    "float8_e4m3fnuzx32",
    "float8_e4m3fnuzx64",
    "float8_e4m3fnx2",
    "float8_e4m3fnx4",
    "float8_e4m3fnx8",
    "float8_e4m3fnx16",
    "float8_e4m3fnx32",
    "float8_e4m3fnx64",
    "float8_e4m3x2",
    "float8_e4m3x4",
    "float8_e4m3x8",
    "float8_e4m3x16",
    "float8_e4m3x32",
    "float8_e4m3x64",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e5m2fnuzx2",
    "float8_e5m2fnuzx4",
    "float8_e5m2fnuzx8",
    "float8_e5m2fnuzx16",
    "float8_e5m2fnuzx32",
    "float8_e5m2fnuzx64",
    "float8_e5m2x2",
    "float8_e5m2x4",
    "float8_e5m2x8",
    "float8_e5m2x16",
    "float8_e5m2x32",
    "float8_e5m2x64",
    "float8_e8m0fnu",
    "float8_e8m0fnux2",
    "float8_e8m0fnux4",
    "float8_e8m0fnux8",
    "float8_e8m0fnux16",
    "float8_e8m0fnux32",
    "float8_e8m0fnux64",
    "float16",
    "float16x2",
    "float16x4",
    "float16x8",
    "float16x16",
    "float16x32",
    "float16x64",
    "float32",
    "float32x2",
    "float32x4",
    "float32x8",
    "float32x16",
    "float32x32",
    "float32x64",
    "float64",
    "float64x2",
    "float64x4",
    "float64x8",
    "float64x16",
    "float64x32",
    "float64x64",
    "handle",
    "i8",
    "i16",
    "i32",
    "i64",
    "index_map",
    "int8",
    "int8x2",
    "int8x4",
    "int8x8",
    "int8x16",
    "int8x32",
    "int8x64",
    "int16",
    "int16x2",
    "int16x4",
    "int16x8",
    "int16x16",
    "int16x32",
    "int16x64",
    "int32",
    "int32x2",
    "int32x4",
    "int32x8",
    "int32x16",
    "int32x32",
    "int32x64",
    "int64",
    "int64x2",
    "int64x4",
    "int64x8",
    "int64x16",
    "int64x32",
    "int64x64",
    "let",
    "local_scalar",
    "meta_class",
    "meta_var",
    "ptr",
    "shared_scalar",
    "smem",
    "static_assert",
    "target",
    "tile",
    "tmem",
    "u8",
    "u16",
    "u32",
    "u64",
    "uint8",
    "uint8x2",
    "uint8x4",
    "uint8x8",
    "uint8x16",
    "uint8x32",
    "uint8x64",
    "uint16",
    "uint16x2",
    "uint16x4",
    "uint16x8",
    "uint16x16",
    "uint16x32",
    "uint16x64",
    "uint32",
    "uint32x2",
    "uint32x4",
    "uint32x8",
    "uint32x16",
    "uint32x32",
    "uint32x64",
    "uint64",
    "uint64x2",
    "uint64x4",
    "uint64x8",
    "uint64x16",
    "uint64x32",
    "uint64x64",
    "void",
]


_init_type_api("tirx", __name__)
