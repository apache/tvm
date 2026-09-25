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
"""Concrete TIRx types, buffers, allocations and construction metadata."""

import contextlib
import functools
import inspect
import threading
from collections.abc import Callable
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Any, TypeVar

from tvm import ir as _ir
from tvm.ir import TensorRegion
from tvm.script.ir_builder.base import annotation_constructor as _annotation_constructor
from tvm.script.ir_builder.base import at as _at
from tvm.script.parser.protocol_registry import (
    result_span as _result_span,
)

# isort: off
from typing import Literal

# isort: on


from tvm import DataType, ir
from tvm import tirx as tir
from tvm.ir import TensorLoad, Type, is_prim_expr
from tvm.script.ir_builder.base import IRBuilder
from tvm.script.ir_builder.ir import meta_var
from tvm.script.parser.protocol_registry import (
    mutable_cell_decl as _mutable_cell_decl,
)
from tvm.script.parser.protocol_registry import (
    register_scalar_annotation as _register_scalar_annotation,
)
from tvm.target import Target

# pylint: disable=unused-import
from tvm.tirx import Buffer, Expr, IndexMap, is_buffer_var, type_annotation
from tvm.tirx.exec_scope import ExecScope, ScopeIdDef, Var

# import tirx.expr for direct ir construction to pass structural_equal comparison
from tvm.tirx.expr import (
    BufferLoad,
    FloatImm,
    IntImm,
    IterVar,
)
from tvm.tirx.layout import (
    ComposeLayout,
    Iter,
    Layout,
    R,
    S,
    TileLayout,
    wg_local_layout,
)

from . import _ffi_api, frame

# pylint: enable=unused-import


def _get_layout(layout: str | Layout | None, shape: list[Expr], scope: str) -> Layout | None:
    if layout is None:
        return None
    if isinstance(layout, Layout):
        return layout
    assert isinstance(layout, str)
    if layout == "default":
        if IRBuilder.is_in_scope():
            for function_frame in reversed(list(IRBuilder.current().frames)):
                if isinstance(function_frame, frame.PrimFuncFrame):
                    return function_frame.default_buffer_layout(shape, scope)
        if scope in ["trn.sbuf", "trn.psum"]:
            return None
        return TileLayout(S[tuple(shape)])
    shape = tuple(shape)
    if scope == "trn.sbuf":
        layout = TileLayout.trainium(layout, shape)
    elif scope == "trn.psum":
        layout = TileLayout.trainium(layout, shape).to_psum()
    return layout


def _normalize_prim_type(dtype) -> ir.PrimType:
    if isinstance(dtype, ir.PrimType):
        return dtype
    dtype_str = getattr(dtype, "_dtype_str", None)
    if dtype_str is not None:
        return ir.PrimType(dtype_str)
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


@_result_span("T.Buffer")
@_mutable_cell_decl("T.Buffer", syntax="parameter")
@_annotation_constructor
def buffer(
    shape: list[Expr] | tuple[Expr] | Expr | Integral,
    dtype: str = "float32",
    data: Var = None,
    strides: list[Expr] | None = None,
    elem_offset: Expr = None,
    byte_offset: Expr = None,
    scope: str = "global",
    align: int = 0,
    offset_factor: int = 0,
    layout: str | Layout | None = "default",
    allocated_addr: int | tuple[int, ...] | None = None,
    buffer_name: str = "",
    *,
    span=None,
) -> Buffer:
    """The buffer declaration function.

    Parameters
    ----------
    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The shape of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[Expr]
        The strides of each dimension.

    elem_offset : Expr
        The offset in terms of number of dtype elements (including lanes).

    byte_offset : Expr, optional
        The offset in bytes, as an alternative to elem_offset.

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    layout : str or Layout, optional
        The buffer layout; "default" selects the layout for the buffer scope.

    allocated_addr : int or tuple of int, optional
        Addresses assigned to the buffer allocation.

    buffer_name : str
        The name of the buffer.

    Returns
    -------
    res : Buffer
        The declared buffer.
    """
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, Integral) else shape
    shape = tuple(shape)
    if strides is None:
        strides = []
    if allocated_addr is None:
        allocated_addr = []
    if not isinstance(allocated_addr, list | tuple):
        allocated_addr = [allocated_addr]
    result = _ffi_api.Buffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        dtype,
        buffer_name,
        data,
        strides,
        _get_elem_offset(elem_offset, byte_offset, dtype),
        scope,
        align,
        offset_factor,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )
    return _at(span, result)


def Tuple(*fields: Type) -> Type:  # pylint: disable=invalid-name
    """Construct a tuple type for a TIRx function or binding annotation."""
    normalized_fields = []
    for field in fields:
        if callable(field) and not isinstance(field, Expr):
            field = field()
        if isinstance(field, Expr):
            field = field.ty
        normalized_fields.append(field)
    return ir.TupleType(normalized_fields)


@_mutable_cell_decl("T.match_buffer")
def match_buffer(
    param: Var | TensorLoad | TensorRegion,
    shape: list[Expr] | tuple[Expr] | Expr | Integral = None,
    dtype: str = "float32",
    data: Var = None,
    strides: list[Expr] | None = None,
    elem_offset: Expr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    layout: str | Layout | None = "default",
    allocated_addr: Expr | int | tuple[Expr | int, ...] | None = None,
) -> Buffer:
    """The buffer match function.

    Note
    ----
    This function will perform different behavior, depending on the type of param.
    If the param is a var in function parameter, it will create a buffer from DLTensor.
    Else if the param is a subregion of other buffers, then create a subregion match inside a block.

    Example
    -------
    Match buffer from function parameter

    .. code-block:: python

        A = T.match_buffer(a, (128, 128), dtype="float32")

    Match buffer from Buffer subregion

    .. code-block:: python

        A = T.match_buffer(B[0:128, i * 128 : i * 128 + 128], (128, 128), dtype="float32")

    Parameters
    ----------
    param : Union[Var, TensorLoad, TensorRegion]
        The parameter of the PrimFunc to match.

    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[Expr]
        The strides of each dimension.

    elem_offset : Expr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    layout: Optional[Union[str, Layout]]
        The layout of the buffer.

    allocated_addr : Expr or int or tuple of Expr or int, optional
        Addresses assigned to the buffer allocation.

    Returns
    -------
    res : Buffer
        The matched buffer.
    """
    if isinstance(param, TensorRegion) and not is_buffer_var(param.source):
        raise TypeError("match_buffer requires a TensorRegion with a BufferVar source")
    if shape is None:
        if isinstance(param, TensorRegion):
            dtype = param.source.ty.dtype
            shape = [region.extent for region in param.region]
        else:
            raise ValueError("Shape must be specified when binding input param")
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, Integral) else shape
    if strides is None:
        strides = []
    if allocated_addr is None:
        allocated_addr = []
    if not isinstance(allocated_addr, list | tuple):
        allocated_addr = [allocated_addr]
    result = _ffi_api.MatchBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        param,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )
    return result


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


def scope_id(
    extents: list[Expr | int] | None, parent: str, cur: str, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Declare scope IDs between execution levels and return their native variables.

    One dimension returns a variable; multiple dimensions return a tuple in
    declaration order. ``None`` defers extent inference to LowerTIRx.
    """
    ret = _ffi_api.ScopeId(extents, parent, "T.scope_id", cur, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def cluster_id(
    extents: list[Expr | int] | None = None, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Define a kernel→cluster scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling ScopeIdDef closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.ClusterId(extents, "kernel", dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def cta_id(
    extents: list[Expr | int] | None = None, preferred=None, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Define a kernel→cta scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling ScopeIdDef closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.CtaId(extents, "kernel", preferred, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def cta_id_in_cluster(
    extents: list[Expr | int] | None = None, preferred=None, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Define a cluster→cta scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling ScopeIdDef closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.CtaId(extents, "cluster", preferred, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def cta_id_in_pair(dtype: str = "int32") -> Var:
    """Return the native CTA index within its two-CTA pair."""
    ret = _ffi_api.CtaIdInPair(dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    return ret[0]


def warpgroup_id(
    extents: list[Expr | int] | None = None, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Define a cta→warpgroup scope id. Pass ``None`` (the default) to defer
    the extent; it will be inferred at LowerTIRx from sibling closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.WarpgroupId(extents, "cta", dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def warp_id(extents: list[Expr | int] | None = None, dtype: str = "int32") -> Var | tuple[Var, ...]:
    """Define a cta→warp scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.WarpId(extents, "cta", dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def warp_id_in_wg(
    extents: list[Expr | int] | None = None, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Define a warpgroup→warp scope id. Pass ``None`` (the default) to defer
    the extent; it will be inferred at LowerTIRx from sibling closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.WarpId(extents, "warpgroup", dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def lane_id(extents: list[Expr | int] | None = None, dtype: str = "int32") -> Var | tuple[Var, ...]:
    """Define a warp→thread scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.ThreadId(extents, "warp", dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def thread_id(
    extents: list[Expr | int] | None = None, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Define a cta→thread scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.ThreadId(extents, "cta", dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


def thread_id_in_wg(
    extents: list[Expr | int] | None = None, dtype: str = "int32"
) -> Var | tuple[Var, ...]:
    """Define a warpgroup→thread scope id. Pass ``None`` (the default) to defer
    the extent; it will be inferred at LowerTIRx from sibling closure.

    ``dtype`` selects the dtype of the introduced vars (``"int32"`` or ``"uint32"``).
    """
    ret = _ffi_api.ThreadId(extents, "warpgroup", dtype)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


@_mutable_cell_decl("T.alloc_buffer")
def alloc_buffer(
    shape: list[Expr] | tuple[Expr] | Expr | Integral,
    dtype: str = "float32",
    data: Var | None = None,
    strides: list[Expr] | None = None,
    elem_offset: Expr | None = None,
    byte_offset: Expr | None = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    layout: str | Layout | None = "default",
    allocated_addr: int | tuple[int, ...] | None = None,
    annotations: dict[str, Any] | None = None,
) -> Buffer:
    """Statement-level buffer allocation (creates an AllocBuffer IR node).

    Emits an AllocBuffer statement and returns the Buffer directly::

        buf = T.alloc_buffer((128, 128))


    Parameters
    ----------
    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The shape of the buffer to allocate.
    dtype : str
        The data type of the buffer elements.
    scope : str
        The storage scope of the buffer (e.g., "global", "shared").
    data : Optional[Var]
        Optional explicit data pointer.
    strides : Optional[List[Expr]]
        Optional strides.
    elem_offset : Optional[Expr]
        Optional element offset.
    byte_offset : Optional[Expr]
        Optional byte offset.
    align : int
        Alignment requirement in bytes.
    offset_factor : int
        Offset factor.
    layout : Optional[Union[str, Layout]]
        Optional layout.
    allocated_addr : Optional[Union[int, Tuple[int, ...]]]
        Optional pre-allocated address metadata.
    annotations : Optional[Dict[str, Any]]
        Optional annotations for the allocation.

    Returns
    -------
    res : Buffer
        The allocated buffer.
    """
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, Integral) else shape
    buf = buffer(
        shape=shape,
        dtype=dtype,
        data=data,
        strides=strides,
        elem_offset=elem_offset,
        byte_offset=byte_offset,
        scope=scope,
        align=align,
        offset_factor=offset_factor,
        layout=layout,
        allocated_addr=allocated_addr,
        buffer_name="",
    )
    _record_meta_resource(buf, skip_frames=2)

    # AllocBuffer.annotations holds typed IR values. The C++ side stores
    # alignment / shape-like ints as ``IntImm(int32, ...)``; if the user
    # (or a parsed-source round-trip) passes a bare Python int, normalize
    # it so structural equality is preserved against the LowerOpaqueBlock
    # output. Booleans must stay as IntImm("bool", ...).
    def _normalize_ann_value(v):
        if isinstance(v, bool):
            return tir.IntImm("bool", int(v))
        if isinstance(v, int):
            return tir.IntImm("int32", v)
        if isinstance(v, float):
            return tir.FloatImm("float32", v)
        return v

    norm_annotations = {k: _normalize_ann_value(v) for k, v in (annotations or {}).items()}
    _ffi_api.AddToParent(tir.AllocBuffer(buf, norm_annotations))  # type: ignore[attr-defined] # pylint: disable=no-member
    return buf


def wg_reg_tile(elem_per_thread: int, dtype: str = "float32") -> Buffer:
    """Warpgroup-wide ``(128, elem_per_thread)`` register tile in local scope.

    Sugar for the recurring pattern::

        T.alloc_buffer(
            (128, elem_per_thread), dtype,
            layout=wg_local_layout(elem_per_thread),
            scope="local",
        )

    Used to stage a tcgen05 load: each of the 128 threads in a warpgroup
    owns one row of ``elem_per_thread`` contiguous elements.
    """
    return alloc_buffer(
        (128, elem_per_thread),
        dtype,
        layout=wg_local_layout(elem_per_thread),
        scope="local",
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
            elif callable(self.type_spec):
                return self.type_spec()  # e.g. T.int32() -> Var
            elif isinstance(self.type_spec, Type):
                return Var("", self.type_spec)
            else:
                raise TypeError(f"Invalid type for T.let: {self.type_spec}")
        elif rhs_dtype is not None:
            rhs_ty = rhs_dtype if isinstance(rhs_dtype, Type) else ir.PrimType(rhs_dtype)
            return Var("", rhs_ty)
        else:
            raise TypeError("T.let requires either a type or an RHS value")


let = LetAnnotation()  # Singleton for T.let (no subscript)


@_mutable_cell_decl("T.LocalVectorAnnotation", syntax="annotation")
class LocalVectorAnnotation:
    """Marker for local vector/tensor allocation via type annotation subscript.

    Created when a DtypeConstructor is subscripted, e.g. ``T.float32[N]`` or
    ``T.float32[M, N]``. The declaration protocol recognizes this annotation
    and allocates local storage with ``T.alloc_local(shape=..., dtype=...)``.
    """

    __slots__ = ("dtype", "shape")

    def __init__(self, dtype: str, shape: tuple):
        self.dtype = dtype
        self.shape = shape


class DtypeConstructor:
    """Callable + subscriptable dtype object.

    Replaces the plain functions previously returned by ``func_gen``.

    * ``T.float32()``        — same FFI call as before (returns ``Var``).
    * ``T.float32[N]``       — returns ``LocalVectorAnnotation("float32", (N,))``.
    * ``T.float32[M, N]``    — returns ``LocalVectorAnnotation("float32", (M, N))``.
    * ``x: T.float32``       — parser calls this object, gets a ``Var``.
    """

    def __init__(self, ffi_name: str, dtype_str: str):
        self._ffi_name = ffi_name
        self._dtype_str = dtype_str

    def __call__(
        self,
        expr: "Expr | Literal['inf', '-inf', 'nan'] | int | float | None" = None,
    ) -> "Expr":
        if isinstance(expr, str):
            expr = float(expr)
        return getattr(_ffi_api, self._ffi_name)(expr)

    def __getitem__(self, shape):
        if isinstance(shape, tuple):
            return LocalVectorAnnotation(self._dtype_str, shape)
        return LocalVectorAnnotation(self._dtype_str, (shape,))

    def __repr__(self):
        return f"DtypeConstructor({self._dtype_str!r})"


@_mutable_cell_decl("T.decl_buffer")
def decl_buffer(
    shape,
    dtype="float32",
    data=None,
    strides=None,
    elem_offset=None,
    byte_offset=None,
    scope="global",
    align=0,
    offset_factor=0,
    layout="default",
    allocated_addr=None,
) -> Buffer:
    """Create a buffer declaration node.

    When ``data`` is provided, creates a DeclBuffer (alias to existing data).
    When ``data`` is None, creates an AllocBuffer (new allocation).

    Parameters
    ----------
    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[Expr]
        The strides of each dimension.

    elem_offset : Expr
        The offset in terms of number of dtype elements (including lanes).

    byte_offset : Expr
        The offset in terms of number of bytes.

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    layout : Layout
        The layout of the buffer.

    Returns
    -------
    res : Buffer
        The declared buffer.
    """
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, Integral) else shape
    shape = tuple(shape)
    if strides is None:
        strides = []
    dtype = _normalize_prim_type(dtype)
    decl_frame = _ffi_api.DeclBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        dtype,
        "",
        data,
        strides,
        _get_elem_offset(elem_offset, byte_offset, dtype),
        scope,
        align,
        offset_factor,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )
    if isinstance(decl_frame, frame.DeclBufferFrame):
        decl_frame.add_callback(partial(decl_frame.__exit__, None, None, None))
        buf = decl_frame.__enter__()
    else:
        buf = decl_frame
    _record_meta_resource(buf, skip_frames=2)
    return buf


alloc_shared = functools.partial(alloc_buffer, scope="shared")


_mutable_cell_decl("T.alloc_shared")(alloc_shared)


alloc_local = functools.partial(alloc_buffer, scope="local")


_mutable_cell_decl("T.alloc_local")(alloc_local)


smem = _mutable_cell_decl("T.smem")(alloc_shared)


tmem = functools.partial(alloc_buffer, scope="tmem")


def alloc_tcgen05_ldst_frag(instr_shape, tensor_shape, dtype):
    """Allocate a register fragment for ``tcgen05.{ld,st}`` atoms.

    Sizes the per-thread storage, allocates ``local`` scope memory, and returns
    a 2-D view of shape ``tensor_shape`` with a matching ``tcgen05_atom_layout``.
    Pass the result to ``Tx.wg.copy_async`` (with a matching TMEM
    buffer) to trigger the corresponding dispatch path.

    Parameters
    ----------
    instr_shape : str
        ``"32x32b"`` (M=128 fragment, 128 row warpgroup tile, layout
        ``(128, K):(1@tid_in_wg, 1)``); or ``"16x64b"`` / ``"16x128b"`` /
        ``"16x256b"`` (M=64 fragments, 64 row warpgroup tile with the
        per-shape per-lane register decomposition).
    tensor_shape : tuple[int, int]
        Logical fragment shape ``(frag_rows, K)`` in element units. ``frag_rows``
        is ``128`` for ``.32x32b`` and ``64`` for the ``.16x*b`` shapes. The
        fp32 Layout B readback image also uses ``("32x32b", (64, N))``.
    dtype : str
        ``"float32"``, ``"float16"``, or ``"bfloat16"``.

    Returns
    -------
    Buffer
        2-D view of shape ``tensor_shape`` whose layout matches
        ``tcgen05_atom_layout(instr_shape, tensor_shape, dtype)``.

    Examples
    --------
    M=128 readback (existing dispatch):
        ``frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "float32")``
        ``Tx.wg.copy_async(frag[:, :], tmem[:, 0:64])``

    M=64 readback (.16x64b dispatch):
        ``frag = T.alloc_tcgen05_ldst_frag("16x64b", (64, 64), "float32")``
        ``Tx.wg.copy_async(frag[:, :], tmem[0:64, 0:64])``

    Datapath B readback (cta_group=2, per-CTA M=64):
        ``C = tmem_pool.alloc((64, 128), "float32", datapath="B")``
        ``frag = T.alloc_tcgen05_ldst_frag("32x32b", (64, 128), "float32")``
        ``Tx.wg.copy_async(frag[:, :], C[:, :])``
    """
    from tvm.tirx.layout import tcgen05_atom_layout  # local import to avoid cycle

    rows, cols = tensor_shape
    bits = DataType(dtype).bits
    # Per-warpgroup total bits = 64 rows x K cols x bits. Divided across 128
    # threads gives per-thread bits; convert to element count.
    per_thread_bits = (rows * cols * bits) // 128
    if per_thread_bits % bits != 0:
        raise ValueError(
            f"alloc_tcgen05_ldst_frag tensor_shape={tensor_shape} dtype={dtype!r} "
            f"does not evenly divide across 128 threads"
        )
    per_thread_elems = per_thread_bits // bits

    layout = tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
    flat = alloc_local((per_thread_elems,), dtype)
    return flat.view(rows, cols, layout=layout)


def alloc_cast_frag(src, dtype):
    """Allocate a register frag holding ``src`` value-cast to ``dtype``.

    Inherits ``src``'s logical shape and its ``(lane, register)`` layout — only
    the element dtype changes — so ``Tx.cast(dst, src)`` is a per-thread
    element-wise cast with no cross-lane movement. ``.permute(...)`` the result
    to the axis order a downstream consumer (e.g. ``stmatrix`` via
    ``Tx.copy(dispatch="ldstmatrix")``) expects.

    Parameters
    ----------
    src : Buffer
        Source register frag (e.g. from ``alloc_tcgen05_ldst_frag``).
    dtype : str
        Destination element dtype.

    Returns
    -------
    Buffer
        Fresh ``local`` frag, ``src.shape`` shaped, ``src.layout``, dtype-cast.
    """
    rows, cols = src.ty.shape
    per_thread_elems = (rows * cols) // 128
    flat = alloc_local((per_thread_elems,), dtype)
    return flat.view(rows, cols, layout=src.ty.layout)


@_mutable_cell_decl("T.alloc_scalar")
def alloc_scalar(dtype: str = "float32", scope: str = "global") -> TensorLoad:
    """Allocate a zero-dimensional buffer (scalar)."""
    buf = alloc_buffer(shape=(1,), dtype=dtype, scope=scope, layout=TileLayout(S[1]))
    assert is_buffer_var(buf)
    scalar = buf[0]
    return scalar


@_mutable_cell_decl("T.decl_scalar")
def decl_scalar(dtype, data, scope, elem_offset=None, byte_offset=None) -> TensorLoad:
    """Declare a zero-dimensional buffer (scalar) from a pointer."""
    buf = decl_buffer(
        shape=(1,),
        dtype=dtype,
        data=data,
        scope=scope,
        elem_offset=_get_elem_offset(elem_offset, byte_offset, dtype),
        strides=None,
        align=-1,
        offset_factor=0,
        layout=TileLayout(S[1]),
    )
    assert is_buffer_var(buf)
    scalar = buf[0]
    return scalar


@_mutable_cell_decl("T.shared_scalar")
def shared_scalar(dtype: str = "float32") -> TensorLoad:
    """Allocate a zero-dimensional buffer in shared memory."""
    return alloc_scalar(dtype=dtype, scope="shared")


@_mutable_cell_decl("T.local_scalar")
def local_scalar(dtype: str = "float32") -> TensorLoad:
    """Allocate a zero-dimensional buffer in local memory."""
    return alloc_scalar(dtype=dtype, scope="local")


def _is_meta_class_instance(value: Any) -> bool:
    return getattr(type(value), "_is_meta_class", False)


def _meta_resource_for_value(value: Any) -> Any | None:
    if isinstance(value, TensorLoad):
        return value.source
    if is_buffer_var(value):
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


def _ffi_name_to_dtype(name: str) -> str:
    """Convert an FFI type name to its TVM dtype string.

    Examples: "Float32" -> "float32", "Int8x4" -> "int8x4",
    "Float8E4M3" -> "float8_e4m3", "Float8E4M3B11FNUZ" -> "float8_e4m3b11fnuz".
    """
    import re

    # Insert underscore before E-notation in float8 names (E3M4, E4M3, etc.)
    s = re.sub(r"(?<=[a-z0-9])E(\d)", r"_e\1", name, flags=re.IGNORECASE)
    return s.lower()


def func_gen(name: str):
    """Generate a DtypeConstructor for each Expr dtype.

    Parameters
    ----------
    name: str
        The ffi function name to call, e.g. "Float32", "Int32".
    """
    dtype = _ffi_name_to_dtype(name)
    constructor = DtypeConstructor(name, dtype)
    _register_scalar_annotation(f"T.{dtype}", constructor, dtype=dtype)
    _mutable_cell_decl(f"T.{dtype}", syntax="annotation")(constructor)
    return constructor


def static_assert(x: Any, message: str = ""):
    assert x, message


int8 = func_gen("Int8")


int16 = func_gen("Int16")


int32 = func_gen("Int32")


int64 = func_gen("Int64")


int8x2 = func_gen("Int8x2")


int16x2 = func_gen("Int16x2")


int32x2 = func_gen("Int32x2")


int64x2 = func_gen("Int64x2")


int8x4 = func_gen("Int8x4")


int16x4 = func_gen("Int16x4")


int32x4 = func_gen("Int32x4")


int64x4 = func_gen("Int64x4")


int8x8 = func_gen("Int8x8")


int16x8 = func_gen("Int16x8")


int32x8 = func_gen("Int32x8")


int64x8 = func_gen("Int64x8")


int8x16 = func_gen("Int8x16")


int16x16 = func_gen("Int16x16")


int32x16 = func_gen("Int32x16")


int64x16 = func_gen("Int64x16")


int8x32 = func_gen("Int8x32")


int16x32 = func_gen("Int16x32")


int32x32 = func_gen("Int32x32")


int64x32 = func_gen("Int64x32")


int8x64 = func_gen("Int8x64")


int16x64 = func_gen("Int16x64")


int32x64 = func_gen("Int32x64")


int64x64 = func_gen("Int64x64")


uint8 = func_gen("UInt8")


uint16 = func_gen("UInt16")


uint32 = func_gen("UInt32")


uint64 = func_gen("UInt64")


uint8x2 = func_gen("UInt8x2")


uint16x2 = func_gen("UInt16x2")


uint32x2 = func_gen("UInt32x2")


uint64x2 = func_gen("UInt64x2")


uint8x4 = func_gen("UInt8x4")


uint16x4 = func_gen("UInt16x4")


uint32x4 = func_gen("UInt32x4")


uint64x4 = func_gen("UInt64x4")


uint8x8 = func_gen("UInt8x8")


uint16x8 = func_gen("UInt16x8")


uint32x8 = func_gen("UInt32x8")


uint64x8 = func_gen("UInt64x8")


uint8x16 = func_gen("UInt8x16")


uint16x16 = func_gen("UInt16x16")


uint32x16 = func_gen("UInt32x16")


uint64x16 = func_gen("UInt64x16")


uint8x32 = func_gen("UInt8x32")


uint16x32 = func_gen("UInt16x32")


uint32x32 = func_gen("UInt32x32")


uint64x32 = func_gen("UInt64x32")


uint8x64 = func_gen("UInt8x64")


uint16x64 = func_gen("UInt16x64")


uint32x64 = func_gen("UInt32x64")


uint64x64 = func_gen("UInt64x64")


float16 = func_gen("Float16")


float32 = func_gen("Float32")


float64 = func_gen("Float64")


float16x2 = func_gen("Float16x2")


float32x2 = func_gen("Float32x2")


float64x2 = func_gen("Float64x2")


float16x4 = func_gen("Float16x4")


float32x4 = func_gen("Float32x4")


float64x4 = func_gen("Float64x4")


float16x8 = func_gen("Float16x8")


float32x8 = func_gen("Float32x8")


float64x8 = func_gen("Float64x8")


float16x16 = func_gen("Float16x16")


float32x16 = func_gen("Float32x16")


float64x16 = func_gen("Float64x16")


float16x32 = func_gen("Float16x32")


float32x32 = func_gen("Float32x32")


float64x32 = func_gen("Float64x32")


float16x64 = func_gen("Float16x64")


float32x64 = func_gen("Float32x64")


float64x64 = func_gen("Float64x64")


float8_e3m4 = func_gen("Float8E3M4")


float8_e3m4x2 = func_gen("Float8E3M4x2")


float8_e3m4x4 = func_gen("Float8E3M4x4")


float8_e3m4x8 = func_gen("Float8E3M4x8")


float8_e3m4x16 = func_gen("Float8E3M4x16")


float8_e3m4x32 = func_gen("Float8E3M4x32")


float8_e3m4x64 = func_gen("Float8E3M4x64")


float8_e4m3 = func_gen("Float8E4M3")


float8_e4m3x2 = func_gen("Float8E4M3x2")


float8_e4m3x4 = func_gen("Float8E4M3x4")


float8_e4m3x8 = func_gen("Float8E4M3x8")


float8_e4m3x16 = func_gen("Float8E4M3x16")


float8_e4m3x32 = func_gen("Float8E4M3x32")


float8_e4m3x64 = func_gen("Float8E4M3x64")


float8_e4m3b11fnuz = func_gen("Float8E4M3B11FNUZ")


float8_e4m3b11fnuzx2 = func_gen("Float8E4M3B11FNUZx2")


float8_e4m3b11fnuzx4 = func_gen("Float8E4M3B11FNUZx4")


float8_e4m3b11fnuzx8 = func_gen("Float8E4M3B11FNUZx8")


float8_e4m3b11fnuzx16 = func_gen("Float8E4M3B11FNUZx16")


float8_e4m3b11fnuzx32 = func_gen("Float8E4M3B11FNUZx32")


float8_e4m3b11fnuzx64 = func_gen("Float8E4M3B11FNUZx64")


float8_e4m3fn = func_gen("Float8E4M3FN")


float8_e4m3fnx2 = func_gen("Float8E4M3FNx2")


float8_e4m3fnx4 = func_gen("Float8E4M3FNx4")


float8_e4m3fnx8 = func_gen("Float8E4M3FNx8")


float8_e4m3fnx16 = func_gen("Float8E4M3FNx16")


float8_e4m3fnx32 = func_gen("Float8E4M3FNx32")


float8_e4m3fnx64 = func_gen("Float8E4M3FNx64")


float8_e4m3fnuz = func_gen("Float8E4M3FNUZ")


float8_e4m3fnuzx2 = func_gen("Float8E4M3FNUZx2")


float8_e4m3fnuzx4 = func_gen("Float8E4M3FNUZx4")


float8_e4m3fnuzx8 = func_gen("Float8E4M3FNUZx8")


float8_e4m3fnuzx16 = func_gen("Float8E4M3FNUZx16")


float8_e4m3fnuzx32 = func_gen("Float8E4M3FNUZx32")


float8_e4m3fnuzx64 = func_gen("Float8E4M3FNUZx64")


float8_e5m2 = func_gen("Float8E5M2")


float8_e5m2x2 = func_gen("Float8E5M2x2")


float8_e5m2x4 = func_gen("Float8E5M2x4")


float8_e5m2x8 = func_gen("Float8E5M2x8")


float8_e5m2x16 = func_gen("Float8E5M2x16")


float8_e5m2x32 = func_gen("Float8E5M2x32")


float8_e5m2x64 = func_gen("Float8E5M2x64")


float8_e5m2fnuz = func_gen("Float8E5M2FNUZ")


float8_e5m2fnuzx2 = func_gen("Float8E5M2FNUZx2")


float8_e5m2fnuzx4 = func_gen("Float8E5M2FNUZx4")


float8_e5m2fnuzx8 = func_gen("Float8E5M2FNUZx8")


float8_e5m2fnuzx16 = func_gen("Float8E5M2FNUZx16")


float8_e5m2fnuzx32 = func_gen("Float8E5M2FNUZx32")


float8_e5m2fnuzx64 = func_gen("Float8E5M2FNUZx64")


float8_e8m0fnu = func_gen("Float8E8M0FNU")


float8_e8m0fnux2 = func_gen("Float8E8M0FNUx2")


float8_e8m0fnux4 = func_gen("Float8E8M0FNUx4")


float8_e8m0fnux8 = func_gen("Float8E8M0FNUx8")


float8_e8m0fnux16 = func_gen("Float8E8M0FNUx16")


float8_e8m0fnux32 = func_gen("Float8E8M0FNUx32")


float8_e8m0fnux64 = func_gen("Float8E8M0FNUx64")


float6_e2m3fn = func_gen("Float6E2M3FN")


float6_e2m3fnx2 = func_gen("Float6E2M3FNx2")


float6_e2m3fnx4 = func_gen("Float6E2M3FNx4")


float6_e2m3fnx8 = func_gen("Float6E2M3FNx8")


float6_e2m3fnx16 = func_gen("Float6E2M3FNx16")


float6_e2m3fnx32 = func_gen("Float6E2M3FNx32")


float6_e2m3fnx64 = func_gen("Float6E2M3FNx64")


float6_e3m2fn = func_gen("Float6E3M2FN")


float6_e3m2fnx2 = func_gen("Float6E3M2FNx2")


float6_e3m2fnx4 = func_gen("Float6E3M2FNx4")


float6_e3m2fnx8 = func_gen("Float6E3M2FNx8")


float6_e3m2fnx16 = func_gen("Float6E3M2FNx16")


float6_e3m2fnx32 = func_gen("Float6E3M2FNx32")


float6_e3m2fnx64 = func_gen("Float6E3M2FNx64")


float4_e2m1fn = func_gen("Float4E2M1FN")


float4_e2m1fnx2 = func_gen("Float4E2M1FNx2")


float4_e2m1fnx4 = func_gen("Float4E2M1FNx4")


float4_e2m1fnx8 = func_gen("Float4E2M1FNx8")


float4_e2m1fnx16 = func_gen("Float4E2M1FNx16")


float4_e2m1fnx32 = func_gen("Float4E2M1FNx32")


float4_e2m1fnx64 = func_gen("Float4E2M1FNx64")


bfloat16 = func_gen("BFloat16")

# Shorthand aliases
f16 = _register_scalar_annotation("T.f16", float16, dtype="float16")
_mutable_cell_decl("T.f16", syntax="annotation")(f16)
f32 = _register_scalar_annotation("T.f32", float32, dtype="float32")
_mutable_cell_decl("T.f32", syntax="annotation")(f32)
f64 = _register_scalar_annotation("T.f64", float64, dtype="float64")
_mutable_cell_decl("T.f64", syntax="annotation")(f64)
bf16 = _register_scalar_annotation("T.bf16", bfloat16, dtype="bfloat16")
_mutable_cell_decl("T.bf16", syntax="annotation")(bf16)
i8 = _register_scalar_annotation("T.i8", int8, dtype="int8")
_mutable_cell_decl("T.i8", syntax="annotation")(i8)
i16 = _register_scalar_annotation("T.i16", int16, dtype="int16")
_mutable_cell_decl("T.i16", syntax="annotation")(i16)
i32 = _register_scalar_annotation("T.i32", int32, dtype="int32")
_mutable_cell_decl("T.i32", syntax="annotation")(i32)
i64 = _register_scalar_annotation("T.i64", int64, dtype="int64")
_mutable_cell_decl("T.i64", syntax="annotation")(i64)
u8 = _register_scalar_annotation("T.u8", uint8, dtype="uint8")
_mutable_cell_decl("T.u8", syntax="annotation")(u8)
u16 = _register_scalar_annotation("T.u16", uint16, dtype="uint16")
_mutable_cell_decl("T.u16", syntax="annotation")(u16)
u32 = _register_scalar_annotation("T.u32", uint32, dtype="uint32")
_mutable_cell_decl("T.u32", syntax="annotation")(u32)
u64 = _register_scalar_annotation("T.u64", uint64, dtype="uint64")

_mutable_cell_decl("T.u64", syntax="annotation")(u64)


def boolean(expr: Expr | None = None) -> Expr:
    """Construct a new tirx.Var with type boolean or cast expression to type boolean.

    Parameters
    ----------
    expr: Expr
        The expression to be cast.

    Returns
    -------
    res : Expr
        The new tirx.Var with type boolean or casted expression with type boolean.
    """
    return _ffi_api.Boolean(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def handle(
    dtype: str | None = None,
    storage_scope: str = "global",
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


def TensorMap() -> Var:  # pylint: disable=invalid-name
    """Create a TIRx var that represents a CUDA tensor-map descriptor.

    The host/runtime ABI passes a handle to descriptor storage. CUDA kernel
    codegen lowers this type to ``const __grid_constant__ CUtensorMap`` when it
    appears as a kernel parameter.
    """
    return _ffi_api.TensorMap()  # type: ignore[attr-defined] # pylint: disable=no-member


def void(expr: Expr | None = None) -> Expr:
    """Construct a new tirx.Var with type void or cast expression to type void.

    Parameters
    ----------
    expr: Expr
        The expression to be cast.

    Returns
    -------
    res : Expr
        The new tirx.Var with type void or casted expression with type void.
    """
    return _ffi_api.Void(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def ptr(dtype: str, storage_scope: str = "global") -> Var:
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


def iter_var(v: Var | str, dom: ir.Range, iter_type: str, thread_tag: str) -> IterVar:
    """The iteration variable.

    Parameters
    ----------
    v : Union[Var, str]
        The internal variable that is used for iteration.

    dom : Range
        The domain of the iteration.

    iter_type : str
        The iteration type.

    thread_tag : str
        The thread type tag.

    Returns
    -------
    res : IterVar
        The iteration variable.
    """
    iter_type = getattr(IterVar, iter_type)
    return IterVar(dom, v, iter_type, thread_tag)


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


def Range(begin: Expr, end: Expr) -> ir.Range:  # pylint: disable=invalid-name
    """
    Create a Range object.

    Parameters
    ----------
    begin : Expr
        The begin value of the range.

    end : Optional[Expr]
        The end value of the range.
    """
    return ir.Range(begin, end)


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
        """Decorator for utility classes used inside @T.prim_func.

        Instances of decorated classes are treated as parser meta values.
        """
        return _install_meta_class(cls)


def Ptr(dtype, storage_scope="global", *, span=None):
    """The pointer declaration function.

    Parameters
    ----------
    dtype : str, Type or callable
        The data type of the pointer.

    storage_scope : str
        The storage scope of the pointer.

    span : SpanEntry, Span or None, optional
        Source location attached to the constructed IR.

    Returns
    -------
    res : Var
        The pointer.
    """
    if callable(dtype) and not isinstance(dtype, _ir.Expr):
        dtype = dtype()
    if isinstance(dtype, _ir.Expr):
        dtype = dtype.ty
    if isinstance(dtype, _ir.PrimType):
        dtype = dtype.dtype
    return _at(span, ptr(dtype, storage_scope))


Buffer = buffer
_mutable_cell_decl("T.buffer", syntax="parameter")(buffer)

__all__ = [
    "Buffer",
    "BufferLoad",
    "ComposeLayout",
    "DtypeConstructor",
    "ExecScope",
    "FloatImm",
    "IntImm",
    "Iter",
    "IterVar",
    "Layout",
    "LetAnnotation",
    "LocalVectorAnnotation",
    "Ptr",
    "R",
    "Range",
    "S",
    "ScopeIdDef",
    "TensorMap",
    "TileLayout",
    "Tuple",
    "Var",
    "alloc_buffer",
    "alloc_cast_frag",
    "alloc_local",
    "alloc_scalar",
    "alloc_shared",
    "alloc_tcgen05_ldst_frag",
    "bf16",
    "bfloat16",
    "boolean",
    "buffer",
    "cluster_id",
    "cta_id",
    "cta_id_in_cluster",
    "cta_id_in_pair",
    "decl_buffer",
    "decl_scalar",
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
    "iter_var",
    "lane_id",
    "let",
    "local_scalar",
    "match_buffer",
    "meta_class",
    "meta_var",
    "ptr",
    "scope_id",
    "shared_scalar",
    "smem",
    "static_assert",
    "target",
    "thread_id",
    "thread_id_in_wg",
    "tmem",
    "type_annotation",
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
    "warp_id",
    "warp_id_in_wg",
    "warpgroup_id",
    "wg_reg_tile",
]
