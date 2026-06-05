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
"""IRBuilder for TIR"""

import contextlib
import functools
import inspect
import threading
from collections.abc import Callable
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, Union

# isort: off
from typing import Literal

# isort: on

from tvm_ffi.core import String

from tvm import DataType, ir
from tvm import tirx as tir
from tvm.ir import Type
from tvm.ir import register_op_attr as _register_op_attr
from tvm.ir.base import deprecated
from tvm.runtime import convert
from tvm.script.ir_builder.base import IRBuilder
from tvm.target import Target

# pylint: disable=unused-import
from tvm.target.codegen import llvm_lookup_intrinsic_id
from tvm.tirx import Buffer, BufferRegion, IndexMap, PrimExpr, type_annotation
from tvm.tirx import op as _tir_op
from tvm.tirx.exec_scope import ExecScope, ScopeIdDef, Var

# import tirx.expr for direct ir construction to pass structural_equal comparison
from tvm.tirx.expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Broadcast,
    BufferLoad,
    Call,
    CallEffectKind,
    Cast,
    CommReducer,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    IterVar,
    Max,
    Min,
    Mod,
    Mul,
    Not,
    Or,
    ProducerLoad,
    Ramp,
    Reduce,
    Select,
    Shuffle,
    SizeVar,
    StringImm,
    Sub,
)
from tvm.tirx.generic import cast
from tvm.tirx.layout import (
    ComposeLayout,
    Iter,
    Layout,
    R,
    S,
    SwizzleLayout,
    TileLayout,
    wg_local_layout,
)

from . import _ffi_api, frame, utils
from .external_kernel import call_kernel

# pylint: enable=unused-import


def _current_s_tir() -> bool:
    """Return True if the innermost enclosing PrimFuncFrame has ``s_tir=True``.

    Gates the parser's default layout fill: ``s_tir=True`` PrimFuncs leave
    ``layout=None`` (so s_tir-style passes that don't touch layout round-trip
    cleanly); ``s_tir=False`` (default, tirx) get ``DefaultLayout(shape)``.
    """
    from tvm.script.ir_builder.base import IRBuilder  # local import to avoid cycle

    if not IRBuilder.is_in_scope():
        return False
    builder = IRBuilder.current()
    for f in reversed(list(builder.frames)):
        if isinstance(f, frame.PrimFuncFrame):
            return bool(f.s_tir)
    return False


def _get_layout(layout: str | Layout | None, shape: list[PrimExpr], scope: str) -> Layout | None:
    if layout is None:
        return None
    if isinstance(layout, Layout):
        return layout
    assert isinstance(layout, str)
    if layout == "default":
        if _current_s_tir():
            return None
        if scope in ["trn.sbuf", "trn.psum"]:
            return None
        return TileLayout(S[tuple(shape)])
    shape = tuple(shape)
    if scope == "trn.sbuf":
        layout = TileLayout.trainium(layout, shape)
    elif scope == "trn.psum":
        layout = TileLayout.trainium(layout, shape).to_psum()
    return layout


def _get_elem_offset(elem_offset, byte_offset, dtype: str):
    assert elem_offset is None or byte_offset is None, (
        "elem_offset and byte_offset cannot be set at the same time"
    )
    if elem_offset is not None:
        return elem_offset
    if byte_offset is None:
        return None
    return byte_offset * 8 // (DataType(dtype).bits)


_block_name_suffix = threading.local()
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


def _get_sblock_name_suffix() -> str:
    """Get the current block name suffix for macro expansion."""
    return getattr(_block_name_suffix, "value", "")


@contextlib.contextmanager
def block_name_suffix_context(block_suffix: str):
    """Context manager to set block name suffix during macro expansion.

    Parameters
    ----------
    block_suffix : str
        The suffix to append to block names (e.g., "_1", "_2").

    Yields
    ------
    None
    """
    old_suffix = getattr(_block_name_suffix, "value", "")
    _block_name_suffix.value = block_suffix
    try:
        yield
    finally:
        _block_name_suffix.value = old_suffix


def buffer(
    shape: list[PrimExpr] | tuple[PrimExpr] | PrimExpr | Integral,
    dtype: str = "float32",
    data: Var = None,
    strides: list[PrimExpr] | None = None,
    elem_offset: PrimExpr = None,
    byte_offset: PrimExpr = None,
    scope: str = "global",
    align: int = 0,
    offset_factor: int = 0,
    buffer_type: str = "",
    axis_separators: list[int] | None = None,
    layout: str | Layout | None = "default",
    allocated_addr: int | tuple[int, ...] | None = None,
    buffer_name: str = "",
) -> Buffer:
    """The buffer declaration function.

    Parameters
    ----------
    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[PrimExpr]
        The strides of each dimension.

    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    buffer_type : str
        The buffer type.

    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.

    buffer_name : str
        The name of the buffer.

    Returns
    -------
    res : Buffer
        The declared buffer.
    """
    shape = (shape,) if isinstance(shape, PrimExpr | Integral) else shape
    if strides is not None:
        strides = [Var(s, "int32") if isinstance(s, str) else s for s in strides]
    else:
        strides = []
    if allocated_addr is None:
        allocated_addr = []
    if not isinstance(allocated_addr, list | tuple):
        allocated_addr = [allocated_addr]
    return _ffi_api.Buffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        dtype,
        buffer_name,
        data,
        strides,
        _get_elem_offset(elem_offset, byte_offset, dtype),
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )


@deprecated("T.buffer_decl(...)", "T.Buffer(...)")
def buffer_decl(*args, **kwargs):
    return buffer(*args, **kwargs)


def prim_func(
    is_private: bool = False,
    s_tir: bool = False,
    persistent: bool = False,
    *,
    private: bool | None = None,
) -> frame.PrimFuncFrame:
    """The primitive function statement.

    Parameters
    ----------
    is_private : bool
        Whether the PrimFunc is annotated as private.
    s_tir : bool
        Whether this PrimFunc uses s_tir (apache-derived TIR) semantics:
        parser fills layout=None on buffers, ScriptComplete wraps body in a
        root SBlock. Default (False) selects tirx semantics: parser fills
        ``DefaultLayout(shape)`` and no root-block wrapping.
    persistent : bool
        Whether this is a persistent kernel.
    private : bool
        Alias for ``is_private`` (used in decorator syntax).

    Returns
    -------
    res : frame.PrimFuncFrame
        The PrimFuncFrame.
    """
    if private is not None:
        is_private = private
    return _ffi_api.PrimFunc(is_private, s_tir, persistent)  # type: ignore[attr-defined] # pylint: disable=no-member


def arg(name: str, obj: Var | Buffer) -> Var | Buffer:
    """The PrimFunc arguments adding function.

    Parameters
    ----------
    name : str
        The name of the argument.

    var : Union[Var, Buffer]
        The argument of Var or Buffer.

    Returns
    -------
    res : Union[Var, Buffer]
        The argument.
    """
    return _ffi_api.Arg(name, obj)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_name(name: str) -> None:
    """The PrimFunc naming statement.

    Parameters
    ----------
    name : str
        The name of the PrimFunc.
    """
    _ffi_api.FuncName(name)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_attr(attrs: dict[str, Any]) -> None:
    """The PrimFunc annotation statement.

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotations of the PrimFunc.
    """
    _ffi_api.FuncAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_ret(ret_type: Type) -> Type:
    """The PrimFunc return type statement.

    Parameters
    ----------
    ret_type : Type
        The return type of the PrimFunc.

    Returns
    -------
    res : Type
        The return type.
    """
    return _ffi_api.FuncRet(ret_type)  # type: ignore[attr-defined] # pylint: disable=no-member


def match_buffer(
    param: Var | BufferLoad | BufferRegion,
    shape: list[PrimExpr] | tuple[PrimExpr] | PrimExpr | Integral = None,
    dtype: str = "float32",
    data: Var = None,
    strides: list[PrimExpr] | None = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: list[int] | None = None,
    layout: str | Layout | None = "default",
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
    param : Union[Var, BufferLoad, BufferRegion]
        The parameter of the PrimFunc to match.

    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[PrimExpr]
        The strides of each dimension.

    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    buffer_type : str
        The buffer type.

    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.

    layout: Optional[Union[str, Layout]]
        The layout of the buffer.

    Returns
    -------
    res : Buffer
        The matched buffer.
    """
    if shape is None:
        if isinstance(param, BufferRegion):
            dtype = param.buffer.dtype
            shape = [region.extent for region in param.region]
        else:
            raise ValueError("Shape must be specified when binding input param")
    shape = (shape,) if isinstance(shape, PrimExpr | Integral) else shape
    if strides is not None:
        idx_dtype = shape[0].dtype if isinstance(shape[0], PrimExpr) else "int32"
        strides = [Var(s, idx_dtype) if isinstance(s, str) else s for s in strides]
    else:
        strides = []
    return _ffi_api.MatchBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        param,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
        _get_layout(layout, shape, scope),
    )


def sblock(name: str = "", no_realize: bool = False, exec_scope: str = "") -> frame.SBlockFrame:
    """The sblock declaration statement.

    Parameters
    ----------
    name : str
        The name of the sblock.

    no_realize : bool
        The flag whether to construct SBlockRealize or SBlock.

    exec_scope : str
        The execution scope of the block.

    Returns
    -------
    res : frame.SBlockFrame
        The SBlockFrame.
    """
    if isinstance(name, list):
        # tir+
        return _ffi_api.ScopeSlice(name, no_realize)
    block_suffix = _get_sblock_name_suffix()
    if block_suffix and name:
        name = name + block_suffix
    return _ffi_api.Block(name, no_realize, exec_scope)  # type: ignore[attr-defined] # pylint: disable=no-member


def device_entry() -> None:
    """Mark the device-region entry within the enclosing PrimFunc body.

    Flat marker (no ``with``). Subsequent statements in the function body
    accumulate into an ``AttrStmt("tirx.device_entry", True, body=...)``;
    the wrapping is closed by the PrimFunc frame at function end.

    Anything written before this marker is host code (e.g. ``T.match_buffer``);
    anything after is device code.

    Example::

        @T.prim_func
        def kernel(...):
            A = T.match_buffer(...)
            T.device_entry()           # device region starts here
            bx = T.cta_id([SM_COUNT])  # standalone scope-id def
            ...
    """
    attr_frame = _ffi_api.DeviceEntry()  # type: ignore[attr-defined] # pylint: disable=no-member
    attr_frame.__enter__()
    # No return: the frame is registered on the IRBuilder stack; the
    # PrimFunc frame's exit drains it.


def elected():
    """Stub that rejects the removed ``T.elected()`` sugar.

    Write the explicit form instead::

        if T.ptx.elect_sync():
            ...                         # thread is the default scope
    """
    raise RuntimeError(
        "T.elected() is no longer available. Write explicitly: "
        "`if T.ptx.elect_sync(): ...` (thread is the default scope)"
    )


def scope_id(extents: list[PrimExpr | int] | None, parent: str, cur: str) -> Var | list[Var]:
    ret = _ffi_api.ScopeId(extents, parent, "T.scope_id", cur)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def cluster_id(extents: list[PrimExpr | int] | None = None) -> Var | list[Var]:
    """Define a kernel→cluster scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling ScopeIdDef closure."""
    ret = _ffi_api.ClusterId(extents, "kernel")  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def cta_id(extents: list[PrimExpr | int] | None = None, preferred=None) -> Var | list[Var]:
    """Define a kernel→cta scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling ScopeIdDef closure."""
    ret = _ffi_api.CtaId(extents, "kernel", preferred)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def cta_id_in_cluster(
    extents: list[PrimExpr | int] | None = None, preferred=None
) -> Var | list[Var]:
    """Define a cluster→cta scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling ScopeIdDef closure."""
    ret = _ffi_api.CtaId(extents, "cluster", preferred)  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def cta_id_in_pair() -> Var:
    ret = _ffi_api.CtaIdInPair()  # type: ignore[attr-defined] # pylint: disable=no-member
    return ret[0]


def warpgroup_id(extents: list[PrimExpr | int] | None = None) -> Var | list[Var]:
    """Define a cta→warpgroup scope id. Pass ``None`` (the default) to defer
    the extent; it will be inferred at LowerTIRx from sibling closure."""
    ret = _ffi_api.WarpgroupId(extents, "cta")  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def warp_id(extents: list[PrimExpr | int] | None = None) -> Var | list[Var]:
    """Define a cta→warp scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling closure."""
    ret = _ffi_api.WarpId(extents, "cta")  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def warp_id_in_wg(extents: list[PrimExpr | int] | None = None) -> Var | list[Var]:
    """Define a warpgroup→warp scope id. Pass ``None`` (the default) to defer
    the extent; it will be inferred at LowerTIRx from sibling closure."""
    ret = _ffi_api.WarpId(extents, "warpgroup")  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def lane_id(extents: list[PrimExpr | int] | None = None) -> Var | list[Var]:
    """Define a warp→thread scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling closure."""
    ret = _ffi_api.ThreadId(extents, "warp")  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def thread_id(extents: list[PrimExpr | int] | None = None) -> Var | list[Var]:
    """Define a cta→thread scope id. Pass ``None`` (the default) to defer the
    extent; it will be inferred at LowerTIRx from sibling closure."""
    ret = _ffi_api.ThreadId(extents, "cta")  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def thread_id_in_wg(extents: list[PrimExpr | int] | None = None) -> Var | list[Var]:
    """Define a warpgroup→thread scope id. Pass ``None`` (the default) to defer
    the extent; it will be inferred at LowerTIRx from sibling closure."""
    ret = _ffi_api.ThreadId(extents, "warpgroup")  # type: ignore[attr-defined] # pylint: disable=no-member
    if len(ret) == 1:
        return ret[0]
    return ret


def init() -> frame.BlockInitFrame:
    """The block initialization statement.

    Returns
    -------
    res : frame.BlockInitFrame
        The BlockInitFrame.
    """
    return _ffi_api.Init()  # type: ignore[attr-defined] # pylint: disable=no-member


def where(predicate: PrimExpr | int) -> None:
    """The block predicate statement.

    Parameters
    ----------
    predicate : Union[PrimExpr, Literal[0, 1]]
        The predicate condition.
    """
    if isinstance(predicate, bool):
        predicate = IntImm("bool", predicate)
    if isinstance(predicate, int):
        if predicate in [0, 1]:
            predicate = IntImm("bool", predicate)
        else:
            raise ValueError(f"Invalid value for predicate: {predicate}")
    _ffi_api.Where(predicate)  # type: ignore[attr-defined] # pylint: disable=no-member


def reads(*buffer_slices: list[BufferRegion | BufferLoad]) -> None:
    """The block buffer region reading statement.

    Parameters
    ----------
    buffer_slices : List[Union[BufferRegion, BufferLoad]]
        The array of buffer regions to read.
    """
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore[assignment]
        else:
            buffer_slices = [buffer_slices[0]]
    else:
        buffer_slices = list(buffer_slices)  # type: ignore[assignment]
    _ffi_api.Reads(buffer_slices)  # type: ignore[attr-defined] # pylint: disable=no-member


def writes(*buffer_slices: list[BufferRegion | BufferLoad]) -> None:
    """The block buffer region writing statement.

    Parameters
    ----------
    buffer_slices : List[Union[BufferRegion, BufferLoad]]
        The array of buffer regions to write.
    """
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore[assignment]
        else:
            buffer_slices = [buffer_slices[0]]
    else:
        buffer_slices = list(buffer_slices)  # type: ignore[assignment]
    _ffi_api.Writes(buffer_slices)  # type: ignore[attr-defined] # pylint: disable=no-member


def sblock_attr(attrs: dict[str, Any]) -> None:
    """The block annotation statement (for non-tirx SBlock usage).

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotation of the block.
    """
    return _ffi_api.BlockAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def alloc_buffer(
    shape: list[PrimExpr] | tuple[PrimExpr] | PrimExpr | Integral,
    dtype: str = "float32",
    data: Var | None = None,
    strides: list[PrimExpr] | None = None,
    elem_offset: PrimExpr | None = None,
    byte_offset: PrimExpr | None = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: list[int] | None = None,
    layout: str | Layout | None = "default",
    allocated_addr: int | tuple[int, ...] | None = None,
    annotations: dict[str, Any] | None = None,
) -> Buffer:
    """Statement-level buffer allocation (creates an AllocBuffer IR node).

    Emits an AllocBuffer statement and returns the Buffer directly::

        buf = T.alloc_buffer((128, 128))

    For SBlock-level buffer allocation (added to SBlock.alloc_buffers),
    use T.sblock_alloc_buffer() instead.

    Parameters
    ----------
    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The shape of the buffer to allocate.
    dtype : str
        The data type of the buffer elements.
    scope : str
        The storage scope of the buffer (e.g., "global", "shared").
    data : Optional[Var]
        Optional explicit data pointer.
    strides : Optional[List[PrimExpr]]
        Optional strides.
    elem_offset : Optional[PrimExpr]
        Optional element offset.
    byte_offset : Optional[PrimExpr]
        Optional byte offset.
    align : int
        Alignment requirement in bytes.
    offset_factor : int
        Offset factor.
    buffer_type : str
        Buffer type.
    axis_separators : Optional[List[int]]
        Optional axis separators.
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
    shape = (shape,) if isinstance(shape, PrimExpr | Integral) else shape
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
        buffer_type=buffer_type,
        axis_separators=axis_separators,
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


def sblock_alloc_buffer(
    shape: list[PrimExpr] | tuple[PrimExpr] | PrimExpr | Integral,
    dtype: str = "float32",
    data: Var = None,
    strides: list[PrimExpr] | None = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: list[int] | None = None,
    layout: str | Layout | None = "default",
    allocated_addr: int | tuple[int, ...] | None = None,
) -> Buffer:
    """SBlock-level buffer allocation function.

    Parameters
    ----------
    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.
    dtype : str
        The data type in the content of the buffer.
    data : Var
        The pointer to the head of the data.
    strides : List[PrimExpr]
        The strides of each dimension.
    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).
    scope : str
        The optional storage scope of buffer data pointer.
    align : int
        The alignment requirement of data pointer in bytes.
    offset_factor : int
        The factor of elem_offset field.
    buffer_type : str
        The buffer type.
    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.

    layout: Optional[Union[str, Layout]]
        The layout of the buffer.

    allocated_addr: Optional[Union[int, Tuple[int]]]
        The address of the allocated buffer. Might be multi-dimensional.
        There can be pooled storage scopes on some devices. For example,
        the Trainium device has a pooled storage scope for the SRAN buffers. ("trn.sbuf")
        CUDA has a pooled storage scope for the shared memory ("shared.dyn")

    Returns
    -------
    res : Buffer
        The allocated buffer.
    """
    shape = (shape,) if isinstance(shape, PrimExpr | Integral) else shape
    if strides is not None:
        strides = [Var(s, "int32") if isinstance(s, str) else s for s in strides]
    else:
        strides = []
    if axis_separators is None:
        axis_separators = []
    if allocated_addr is None:
        allocated_addr = []
    if not isinstance(allocated_addr, list | tuple):
        allocated_addr = [allocated_addr]
    alloc_frame = _ffi_api.SBlockAllocBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )
    if isinstance(alloc_frame, frame.AllocBufferFrame):
        alloc_frame.add_callback(partial(alloc_frame.__exit__, None, None, None))
        buf = alloc_frame.__enter__()
    else:
        buf = alloc_frame
    _record_meta_resource(buf, skip_frames=2)
    return buf


def _as_range(dom: ir.Range | list[PrimExpr]) -> ir.Range:
    """The range constructor.

    Parameters
    ----------
    dom : Union[Range, List[PrimExpr]]
        The domain.

    Returns
    -------
    res : Range
        The Range.
    """
    if isinstance(dom, ir.Range):
        return dom
    if isinstance(dom, list | tuple):
        from tvm.arith import Analyzer  # pylint: disable=import-outside-toplevel

        extent = Analyzer().simplify(dom[1] - dom[0])
        if isinstance(extent, tir.IntImm):
            return ir.Range.from_min_extent(dom[0], extent)
        return ir.Range(dom[0], dom[1])
    if hasattr(dom, "dtype"):
        return ir.Range(IntImm(dom.dtype, 0), dom)
    return ir.Range(0, dom)


class axis:  # pylint: disable=invalid-name
    """The axis class"""

    @staticmethod
    def spatial(
        dom: ir.Range | list[PrimExpr] | tuple[PrimExpr],
        binding: PrimExpr,
        dtype: str = "int32",
    ) -> Var:
        """The spatial block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisSpatial(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def reduce(
        dom: ir.Range | list[PrimExpr] | tuple[PrimExpr],
        binding: PrimExpr,
        dtype: str = "int32",
    ) -> Var:
        """The reduced block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisReduce(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def scan(
        dom: ir.Range | list[PrimExpr] | tuple[PrimExpr],
        binding: PrimExpr,
        dtype: str = "int32",
    ) -> Var:
        """The scanning block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisScan(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def opaque(
        dom: ir.Range | list[PrimExpr] | tuple[PrimExpr],
        binding: PrimExpr,
        dtype: str = "int32",
    ) -> Var:
        """The opaque block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisOpaque(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def remap(kinds: str, bindings: list[PrimExpr], dtype: str = "int32") -> list[Var] | Var:
        """The block axis remapping function.

        Parameters
        ----------
        kinds : str
            The types of the iteration variables.

        bindings : List[PrimExpr]
            The binding values of the iteration variables.

        dtype : str
            The data types of the iteration variables.

        Returns
        -------
        res : Var
            The iteration variables.
        """
        iter_vars = _ffi_api.AxisRemap(  # type: ignore[attr-defined] # pylint: disable=no-member
            kinds, bindings, dtype
        )
        return iter_vars[0] if len(iter_vars) == 1 else iter_vars

    S = spatial  # pylint: disable=invalid-name
    R = reduce  # pylint: disable=invalid-name


def serial(
    start: PrimExpr,
    stop: PrimExpr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: PrimExpr | None = None,
    unroll: bool | None = None,
) -> frame.ForFrame:
    """The serial For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : PrimExpr
        The optional step value of iteration.

    unroll : bool, optional
        If True, adds ``{"pragma_unroll": True}`` annotation, which asks CUDA codegen
        to emit ``#pragma unroll`` while preserving the loop as a C++ ``for``.
        If False, adds ``{"disable_unroll": True}`` annotation.
        Shorthand for ``annotations={"disable_unroll": True}``.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if unroll is not None:
        annotations = dict(annotations) if annotations else {}
        if unroll:
            annotations["pragma_unroll"] = True
        else:
            annotations["disable_unroll"] = True
    if stop is None:
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0
    return _ffi_api.Serial(start, stop, annotations, step)  # type: ignore[attr-defined] # pylint: disable=no-member


def parallel(
    start: PrimExpr,
    stop: PrimExpr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: PrimExpr | None = None,
) -> frame.ForFrame:
    """The parallel For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : PrimExpr
        The optional step value of iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0
    return _ffi_api.Parallel(start, stop, annotations, step)  # type: ignore[attr-defined] # pylint: disable=no-member


def vectorized(
    start: PrimExpr,
    stop: PrimExpr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: PrimExpr | None = None,
) -> frame.ForFrame:
    """The vectorized For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : PrimExpr
        The optional step value of iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0
    return _ffi_api.Vectorized(start, stop, annotations, step)  # type: ignore[attr-defined] # pylint: disable=no-member


def unroll(
    start: PrimExpr,
    stop: PrimExpr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: PrimExpr | None = None,
) -> frame.ForFrame:
    """The unrolled For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : PrimExpr
        The optional step value of iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0
    return _ffi_api.Unroll(start, stop, annotations, step)  # type: ignore[attr-defined] # pylint: disable=no-member


def thread_binding(
    start: PrimExpr,
    stop: PrimExpr = None,
    thread: str | None = None,
    *,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """The thread-binding For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    thread : str
        The thread for loop variable to bind.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if thread is None:
        if not isinstance(stop, str):
            raise ValueError("Thread cannot be None for thread_binding")
        thread = stop
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0
    elif stop is None:
        stop = start
        if hasattr(start, "dtype"):
            start = IntImm(start.dtype, 0)
        else:
            start = 0
    return _ffi_api.ThreadBinding(  # type: ignore[attr-defined] # pylint: disable=no-member
        start, stop, thread, annotations
    )


def grid(*extents: tuple[PrimExpr | tuple[PrimExpr, PrimExpr]]) -> frame.ForFrame:
    """The grid For statement.

    Parameters
    ----------
    extents : Tuple[Union[PrimExpr, Tuple[PrimExpr, PrimExpr]]]
        If a single PrimExpr is provided, it is used as the extent of the iteration.
        If a tuple of two PrimExpr is provided, the first is the start of the iteration,
        and the second is the extent of the iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    # Convert integer extents to IntImm
    # TODO(@bohan): fix this after FFI refactor
    processed_extents = []
    for extent in extents:
        if isinstance(extent, tuple):
            start, extent = extent
            start = IntImm("int32", start) if isinstance(start, int) else start
            extent = IntImm("int32", extent) if isinstance(extent, int) else extent
            processed_extents.append((start, extent))
        else:
            processed_extents.append(IntImm("int32", extent) if isinstance(extent, int) else extent)
    extents = tuple(processed_extents)
    return _ffi_api.Grid(extents)  # type: ignore[attr-defined] # pylint: disable=no-member


def Assert(condition: PrimExpr, message, error_kind: str = "RuntimeError") -> frame.AssertFrame:  # pylint: disable=invalid-name
    """Create an assertion statement.

    Parameters
    ----------
    condition : PrimExpr
        The PrimExpr to test.

    message : str or list[str]
        The error message when the assertion fails. Can be a single string
        or a list of string parts (fragments stored separately in the IR
        for binary size reduction through string reuse).

    error_kind : str
        The error kind (e.g. "RuntimeError", "TypeError", "ValueError").

    Returns
    -------
    res : frame.AssertFrame
        The result AssertFrame.
    """
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    if not isinstance(message, list | tuple):
        message = [message]
    return _ffi_api.Assert(condition, error_kind, message)  # type: ignore[attr-defined] # pylint: disable=no-member


def Bind(  # pylint: disable=invalid-name
    value: PrimExpr,
    type_annotation: Type | None = None,  # pylint: disable=redefined-outer-name
    *,
    var: Var | None = None,  # pylint: disable=redefined-outer-name
) -> Var:
    """Create a Bind (variable binding).

    Emits a flat Bind statement to the current frame and returns the bound variable.

    Parameters
    ----------
    value : PrimExpr
        The value to be bound.
    type_annotation : Optional[Type] = None
        The type annotation of the binding. Usually it is used for fine-grained var typing,
        particularly, PointerType.
    var : Optional[Var] = None
        The variable to bind. If not specified, a new variable will be created.

    Returns
    -------
    var : Var
        The bound variable.
    """
    if type_annotation is not None:
        if callable(type_annotation):
            type_annotation = type_annotation()
        if isinstance(type_annotation, Var):
            type_annotation = type_annotation.type_annotation
    return _ffi_api.Bind(value, type_annotation, var)  # type: ignore[attr-defined] # pylint: disable=no-member


def Let(  # pylint: disable=invalid-name
    expr: PrimExpr,
    where: dict[Var, PrimExpr],  # pylint: disable=redefined-outer-name
) -> PrimExpr:
    """Create a Let expression binding"""
    assert len(where) == 1, "T.Let only allows `where` to have exactly one element"
    var, value = next(iter(where.items()))  # pylint: disable=redefined-outer-name
    return tir.Let(var, value, expr)


bind = Bind


class LetAnnotation:
    """Marker for explicit LetStmt. Created by T.let or T.let[type].
    Usage in TVMScript:
        x: T.let[T.int32] = expr   # LetStmt with explicit type
        x: T.let = expr             # LetStmt with auto-typed RHS
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
            if isinstance(self.type_spec, Var):
                return self.type_spec  # Already a Var (e.g. T.handle(...))
            elif callable(self.type_spec):
                return self.type_spec()  # e.g. T.int32() -> Var
            elif isinstance(self.type_spec, Type):
                return Var("", self.type_spec)
            else:
                raise TypeError(f"Invalid type for T.let: {self.type_spec}")
        elif rhs_dtype is not None:
            return Var("", ir.PrimType(rhs_dtype))
        else:
            raise TypeError("T.let requires either a type or an RHS value")


let = LetAnnotation()  # Singleton for T.let (no subscript)


class LocalVectorAnnotation:
    """Marker for local vector/tensor allocation via type annotation subscript.

    Created when a DtypeConstructor is subscripted, e.g. ``T.float32[N]`` or
    ``T.float32[M, N]``.  The parser's ``visit_ann_assign`` recognises this
    object and lowers it to ``T.alloc_local(shape=..., dtype=...)``.
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
        expr: "None | PrimExpr | Literal['inf', '-inf', 'nan'] | int | float" = None,
        *,
        is_size_var: bool = False,
    ) -> "PrimExpr":
        if isinstance(expr, str):
            expr = float(expr)
        return getattr(_ffi_api, self._ffi_name)(expr, is_size_var)

    def __getitem__(self, shape):
        if isinstance(shape, tuple):
            return LocalVectorAnnotation(self._dtype_str, shape)
        return LocalVectorAnnotation(self._dtype_str, (shape,))

    def __repr__(self):
        return f"DtypeConstructor({self._dtype_str!r})"


def allocate(
    extents: list[PrimExpr],
    dtype: str,
    scope: str = "global",
    condition: PrimExpr = None,
    annotations=None,
) -> frame.AllocateFrame:
    """Allocate node.

    Parameters
    ----------
    extents : List[PrimExpr]
        The extents of the allocate.

    dtype : str
        The data type of the buffer.

    scope : str
        The storage scope.

    condition : PrimExpr
        The condition.

    annotations: Optional[Mapping[str, Object]]
        Additional annotation hints.
    """
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.Allocate(  # type: ignore[attr-defined] # pylint: disable=no-member
        extents, dtype, scope, condition, annotations
    )


def attr(
    node_or_dict: Any, attr_key: str | None = None, value: PrimExpr | str | None = None
) -> Union[frame.AttrFrame, "utils._FrameScope"]:
    """Create an attribute node, or multiple attribute nodes from a dict.

    Usage 1 — single attr::

        with T.attr(node, key, value):
            ...

    Usage 2 — dict sugar (node defaults to ``T.int32(0)``)::

        with T.attr({"key1": value1, "key2": value2}):
            ...

    Parameters
    ----------
    node_or_dict : Any
        If a dict, each key-value pair becomes an AttrStmt with
        ``node=T.int32(0)``.  Otherwise the node to annotate.

    attr_key : str, optional
        Attribute type key (required when ``node_or_dict`` is not a dict).

    value : Union[PrimExpr, str], optional
        The attribute value (required when ``node_or_dict`` is not a dict).

    Returns
    -------
    res : Union[frame.AttrFrame, _FrameScope]
        A single AttrFrame, or a _FrameScope wrapping multiple AttrFrames.
    """
    if isinstance(node_or_dict, dict):
        frames = []
        for k, v in node_or_dict.items():
            if isinstance(v, bool):
                v = IntImm("bool", v)
            frames.append(
                _ffi_api.Attr(  # type: ignore[attr-defined]
                    convert(IntImm("int32", 0)), k, convert(v)
                )
            )
        if len(frames) == 1:
            return frames[0]
        return utils._FrameScope(frames)
    else:
        if attr_key is None or value is None:
            raise ValueError("T.attr(node, attr_key, value) requires all three arguments")
        node_or_dict = convert(node_or_dict)
        value = convert(value)
        return _ffi_api.Attr(node_or_dict, attr_key, value)  # type: ignore[attr-defined] # pylint: disable=no-member


def hint(message: str = "", **attrs) -> frame.HintFrame:
    """Universal directive primitive for the sketch language.

    Parameters
    ----------
    message : str
        Free-form directive string that the agent interprets.
    **attrs
        Optional structured key-value attributes for known patterns.

    Returns
    -------
    res : frame.HintFrame
        Usable as context manager (with T.hint("msg"):) or bare statement (T.hint("msg")).
    """
    return _ffi_api.Hint(message, attrs or {})  # type: ignore[attr-defined] # pylint: disable=no-member


def While(condition: PrimExpr) -> frame.WhileFrame:  # pylint: disable=invalid-name
    """Create a while node.

    Parameters
    ----------
    condition : PrimExpr
        The termination condition of the loop.

    Returns
    -------
    res : frame.WhileFrame
        The result WhileFrame.
    """
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.While(condition)  # type: ignore[attr-defined] # pylint: disable=no-member


def Break() -> None:  # pylint: disable=invalid-name
    """Create a break node."""
    return _ffi_api.Break()  # type: ignore[attr-defined] # pylint: disable=no-member


def Continue() -> None:  # pylint: disable=invalid-name
    """Create a continue node."""
    return _ffi_api.Continue()  # type: ignore[attr-defined] # pylint: disable=no-member


def If(condition: PrimExpr) -> frame.IfFrame:  # pylint: disable=invalid-name
    """Create an if node.

    Parameters
    ----------
    condition : PrimExpr
        The condition of if statement, executes the true branch if the condition is true,
        otherwise jump into the false branch.

    Returns
    -------
    res : frame.IfFrame
        The result IfFrame.
    """
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.If(condition)  # type: ignore[attr-defined] # pylint: disable=no-member


def Then() -> frame.ThenFrame:  # pylint: disable=invalid-name
    """Create a then.

    Returns
    -------
    res : frame.ThenFrame
        The result ThenFrame.
    """
    return _ffi_api.Then()  # type: ignore[attr-defined] # pylint: disable=no-member


def Else() -> frame.ElseFrame:  # pylint: disable=invalid-name
    """Create an else.

    Returns
    -------
    res : frame.ElseFrame
        The result ElseFrame.
    """
    return _ffi_api.Else()  # type: ignore[attr-defined] # pylint: disable=no-member


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
    buffer_type="",
    axis_separators=None,
    layout="default",
    allocated_addr=None,
) -> Buffer:
    """Create a buffer declaration node.

    When ``data`` is provided, creates a DeclBuffer (alias to existing data).
    When ``data`` is None, creates an AllocBuffer (new allocation).

    Parameters
    ----------
    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[PrimExpr]
        The strides of each dimension.

    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).

    byte_offset : PrimExpr
        The offset in terms of number of bytes.

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    buffer_type : str
        The buffer type.

    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.

    layout : Layout
        The layout of the buffer.

    Returns
    -------
    res : Buffer
        The declared buffer.
    """
    shape = (shape,) if isinstance(shape, PrimExpr | Integral) else shape
    if strides is not None:
        strides = [Var(s, "int32") if isinstance(s, str) else s for s in strides]
    else:
        strides = []
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
        buffer_type,
        axis_separators,
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
alloc_local = functools.partial(alloc_buffer, scope="local")
smem = alloc_shared
tmem = functools.partial(alloc_buffer, scope="tmem")


def alloc_tcgen05_ldst_frag(instr_shape, tensor_shape, dtype):
    """Allocate a register fragment for ``tcgen05.{ld,st}`` atoms.

    Sizes the per-thread storage, allocates ``local`` scope memory, and returns
    a 2-D view of shape ``tensor_shape`` with a matching ``tcgen05_atom_layout``.
    Pass the result to ``Tx.copy_async`` (with a ``(128, W)``-shaped TMEM
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
        is ``128`` for ``.32x32b`` and ``64`` for the ``.16x*b`` shapes.
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
        ``Tx.copy_async(frag[:, :], tmem[:, 0:64])``

    M=64 readback (.16x64b dispatch):
        ``frag = T.alloc_tcgen05_ldst_frag("16x64b", (64, 64), "float32")``
        ``Tx.copy_async(frag[:, :], tmem[0:64, 0:64])``
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
    rows, cols = src.shape
    per_thread_elems = (rows * cols) // 128
    flat = alloc_local((per_thread_elems,), dtype)
    return flat.view(rows, cols, layout=src.layout)


if TYPE_CHECKING:
    ScalarT = TypeVar("ScalarT")

    # Keep type checking/linting simple by treating wrapper as identity.
    def scalar_wrapper(x: ScalarT) -> ScalarT:
        return x

else:

    class scalar_wrapper:
        """Internal wrapper to allow IRBuilder auto-naming on scalar assignment."""

        def __init__(self, scalar: BufferLoad):
            assert isinstance(scalar, BufferLoad)
            self.scalar = scalar

        def __getattr__(self, name: str) -> Any:
            return getattr(self.scalar, name)

        def __add__(self, other):
            return self.scalar + other

        def __radd__(self, other):
            return other + self.scalar

        def __sub__(self, other):
            return self.scalar - other

        def __rsub__(self, other):
            return other - self.scalar

        def __mul__(self, other):
            return self.scalar * other

        def __rmul__(self, other):
            return other * self.scalar

        def __truediv__(self, other):
            return self.scalar / other

        def __rtruediv__(self, other):
            return other / self.scalar

        def __floordiv__(self, other):
            return self.scalar // other

        def __rfloordiv__(self, other):
            return other // self.scalar

        def __mod__(self, other):
            return self.scalar % other

        def __rmod__(self, other):
            return other % self.scalar

        def __lt__(self, other):
            return self.scalar < other

        def __le__(self, other):
            return self.scalar <= other

        def __gt__(self, other):
            return self.scalar > other

        def __ge__(self, other):
            return self.scalar >= other

        def __eq__(self, other):
            return self.scalar == other

        def __ne__(self, other):
            return self.scalar != other

        def __and__(self, other):
            return self.scalar & other

        def __rand__(self, other):
            return other & self.scalar

        def __or__(self, other):
            return self.scalar | other

        def __ror__(self, other):
            return other | self.scalar

        def __xor__(self, other):
            return self.scalar ^ other

        def __rxor__(self, other):
            return other ^ self.scalar

        def __neg__(self):
            return -self.scalar

        def __invert__(self):
            return ~self.scalar


def alloc_scalar(dtype: str = "float32", scope: str = "global") -> BufferLoad:
    """Allocate a zero-dimensional buffer (scalar)."""
    buf = alloc_buffer(shape=(1,), dtype=dtype, scope=scope, layout=TileLayout(S[1]))
    assert isinstance(buf, Buffer)
    scalar = buf[0]
    if _current_meta_construction_scope() is not None:
        return scalar
    return scalar_wrapper(scalar)


def decl_scalar(dtype, data, scope, elem_offset=None, byte_offset=None) -> BufferLoad:
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
        buffer_type="default",
        axis_separators=None,
        layout=TileLayout(S[1]),
    )
    assert isinstance(buf, Buffer)
    scalar = buf[0]
    if _current_meta_construction_scope() is not None:
        return scalar
    return scalar_wrapper(scalar)


def shared_scalar(dtype: str = "float32") -> BufferLoad:
    """Allocate a zero-dimensional buffer in shared memory."""
    return alloc_scalar(dtype=dtype, scope="shared")


def local_scalar(dtype: str = "float32") -> BufferLoad:
    """Allocate a zero-dimensional buffer in local memory."""
    return alloc_scalar(dtype=dtype, scope="local")


def _is_meta_class_instance(value: Any) -> bool:
    return getattr(type(value), "_is_meta_class", False)


def _sanitize_meta_name_part(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.isidentifier():
        return value
    if isinstance(value, str):
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in value)
        if sanitized and sanitized[0].isalpha():
            return sanitized
    return fallback


def _meta_resource_for_value(value: Any) -> Any | None:
    if isinstance(value, scalar_wrapper):
        return value.scalar.buffer
    if isinstance(value, BufferLoad):
        return value.buffer
    if isinstance(value, Buffer):
        return value
    return None


def _resource_in(resource: Any, resources: list[Any]) -> bool:
    return any(_same_meta_resource(resource, other) for other in resources)


def _name_meta_value(
    prefix: str,
    value: Any,
    visited: set[int] | None = None,
    owned_resources: list[Any] | None = None,
    named_resources: list[Any] | None = None,
) -> None:
    if visited is None:
        visited = set()
    if named_resources is None:
        named_resources = []
    obj_id = id(value)
    if obj_id in visited:
        return
    visited.add(obj_id)

    resource = _meta_resource_for_value(value)
    if resource is not None:
        if owned_resources is not None and not _resource_in(resource, owned_resources):
            return
        if _resource_in(resource, named_resources):
            return
        IRBuilder.name(prefix, resource)
        named_resources.append(resource)
        return
    if isinstance(value, Var | IterVar):
        if owned_resources is not None:
            return
        IRBuilder.name(prefix, value)
        return
    if _is_meta_class_instance(value):
        existing_prefix = getattr(value, "_tirx_meta_name", None)
        if existing_prefix is not None and existing_prefix != prefix:
            return
        object.__setattr__(value, "_tirx_meta_name", prefix)
        instance_owned_resources = getattr(value, "_tirx_meta_owned_resources", [])
        for field_name, field_value in vars(value).items():
            if field_name.startswith("_tirx_"):
                continue
            _name_meta_value(
                f"{prefix}_{field_name}",
                field_value,
                visited,
                instance_owned_resources,
                named_resources,
            )
        return
    if isinstance(value, list | tuple):
        for i, item in enumerate(value):
            _name_meta_value(f"{prefix}_{i}", item, visited, owned_resources, named_resources)
        return
    if isinstance(value, dict):
        for i, (key, item) in enumerate(value.items()):
            part = _sanitize_meta_name_part(key, f"item{i}")
            _name_meta_value(f"{prefix}_{part}", item, visited, owned_resources, named_resources)


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


def name_meta_class_value(prefix: str, value: Any) -> None:
    """Name all TIR resources owned by a meta_class instance."""
    _name_meta_value(prefix, value)


def launch_thread(
    thread: IterVar | str,  # pylint: disable=redefined-outer-name
    extent: PrimExpr,
) -> frame.LaunchThreadFrame:
    """Launch a thread.

    Parameters
    ----------
    thread : Union[IterVar, str]
        The iteration variable.

    extent : PrimExpr
        The extent of environment thread.

    Returns
    -------
    res : frame.LaunchThreadFrame
        The result LaunchThreadFrame.

    Examples
    --------

    .. code-block:: python

    from tvm.script.ir_builder import tirx as T
    brow = T.env_thread("blockIdx.y")
    T.launch_thread(brow, 1)

    """

    if isinstance(thread, str):
        thread = String(thread)
    return _ffi_api.LaunchThread(thread, extent)  # type: ignore[attr-defined] # pylint: disable=no-member


def env_thread(thread_tag: str, dtype: str = "int32") -> IterVar:
    """Bind a var to thread env

    Parameters
    ----------
    thread_tag : str
        The thread type tag.

    dtype : str
        The data type of the thread env.

    Returns
    -------
    res : IterVar
        The result iteration variable gets bound to the thread env.

    """
    return _ffi_api.EnvThread(thread_tag, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member


def buffer_store(
    buffer: Buffer,  # pylint: disable=redefined-outer-name
    value: PrimExpr,
    indices: list[PrimExpr | slice],
    predicate: PrimExpr | None = None,
) -> None:
    """Buffer store node.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    value : PrimExpr
        The value to be stored.

    indices : List[Union[PrimExpr, slice]]
        The indices location to be stored.

    predicate : Optional[PrimExpr]
        A vector mask of boolean values indicating which lanes of a vector are to be
        stored. The number lanes of the mask must be equal to the number of lanes in
        value.
    """
    from tvm.arith import Analyzer  # pylint: disable=import-outside-toplevel

    if not isinstance(indices, list | tuple | ir.Array):
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
                expr_indices.append(ramp(index.start, step, lanes))
        else:
            expr_indices.append(index)
    if isinstance(value, bool) and buffer.dtype == "bool":
        value = IntImm("bool", value)
    return _ffi_api.BufferStore(  # type: ignore[attr-defined] # pylint: disable=no-member
        buffer, value, expr_indices, predicate
    )


def evaluate(value: PrimExpr) -> None:
    """Evaluate the input expression.

    Parameters
    ----------
    value: PrimExpr
        The input expression to evaluate.
    """
    if isinstance(value, str):
        value = StringImm(value)
    if isinstance(value, bool):
        value = cast(value, "bool")
    return _ffi_api.Evaluate(value)  # type: ignore[attr-defined] # pylint: disable=no-member


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
    """Generate a DtypeConstructor for each PrimExpr dtype.

    Parameters
    ----------
    name: str
        The ffi function name to call, e.g. "Float32", "Int32".
    """
    return DtypeConstructor(name, _ffi_name_to_dtype(name))


def static_assert(x: Any, message: str = ""):
    assert x, message


def add_to_parent(stmt: tir.Stmt) -> None:
    """Add a statement to the parent frame."""
    _ffi_api.AddToParent(stmt)  # type: ignore[attr-defined] # pylint: disable=no-member


# pylint: disable=invalid-name
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

# Float8 variants
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

# Float6 variants
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

# Float4 variants
float4_e2m1fn = func_gen("Float4E2M1FN")
float4_e2m1fnx2 = func_gen("Float4E2M1FNx2")
float4_e2m1fnx4 = func_gen("Float4E2M1FNx4")
float4_e2m1fnx8 = func_gen("Float4E2M1FNx8")
float4_e2m1fnx16 = func_gen("Float4E2M1FNx16")
float4_e2m1fnx32 = func_gen("Float4E2M1FNx32")
float4_e2m1fnx64 = func_gen("Float4E2M1FNx64")

bfloat16 = func_gen("BFloat16")

# Shorthand aliases
f16 = float16
f32 = float32
f64 = float64
bf16 = bfloat16
i8 = int8
i16 = int16
i32 = int32
i64 = int64
u8 = uint8
u16 = uint16
u32 = uint32
u64 = uint64
# pylint: enable=invalid-name


def boolean(expr: PrimExpr | None = None, is_size_var: bool = False) -> PrimExpr:
    """Construct a new tirx.Var with type boolean or cast expression to type boolean.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    is_size_var: bool
        Whether or not to return a SizeVar instead of Var.

    Returns
    -------
    res : PrimExpr
        The new tirx.Var with type boolean or casted expression with type boolean.
    """
    return _ffi_api.Boolean(expr, is_size_var)  # type: ignore[attr-defined] # pylint: disable=no-member


def handle(
    dtype: str | None = None,
    storage_scope: str = "global",
    *,
    is_size_var: bool = False,
) -> Var:
    """Create a TIR var that represents a pointer.

    Parameters
    ----------
    dtype: str
        The data type of the pointer.

    storage_scope: str
        The storage scope of the pointer.

    is_size_var: bool
        Whether or not to return a SizeVar instead of Var.

    Returns
    -------
    res : PrimExpr
        The new tirx.Var with type handle or casted expression with type handle.
    """
    if dtype in ("TensorMap", "tensormap", "CUtensorMap", "cuTensorMap"):
        return _ffi_api.TensorMap()  # type: ignore[attr-defined] # pylint: disable=no-member
    is_unknown_type = dtype is None
    if dtype is None:
        dtype = "void"
    return _ffi_api.Handle(  # type: ignore[attr-defined] # pylint: disable=no-member
        dtype,
        storage_scope,
        is_size_var,
        is_unknown_type,
    )


def TensorMap() -> Var:  # pylint: disable=invalid-name
    """Create a TIRx var that represents a CUDA tensor-map descriptor.

    The host/runtime ABI passes a handle to descriptor storage. CUDA kernel
    codegen lowers this type to ``const __grid_constant__ CUtensorMap`` when it
    appears as a kernel parameter.
    """
    return _ffi_api.TensorMap()  # type: ignore[attr-defined] # pylint: disable=no-member


def void(expr: PrimExpr | None = None, *, is_size_var: bool = False) -> PrimExpr:
    """Construct a new tirx.Var with type void or cast expression to type void.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tirx.Var with type void or casted expression with type void.
    """
    return _ffi_api.Void(expr, is_size_var)  # type: ignore[attr-defined] # pylint: disable=no-member


@deprecated("T.var", "T.{dtype}")
def var(dtype: str, name: str = "") -> Var:
    """Construct a new tirx.Var.

    Parameters
    ----------
    dtype: str
        The dtype of the Var.

    name: str
        The name of the Var.

    Returns
    -------
    res : Var
        The result tirx.Var.
    """
    return Var(name, dtype)  # pylint: disable=no-member


def ptr(dtype: str, storage_scope: str = "global", is_size_var: bool = False) -> Var:
    """The pointer declaration function.

    Parameters
    ----------
    dtype : str
        The data type of the pointer.

    storage_scope : str
        The storage scope of the pointer.

    is_size_var: bool
        Whether or not to return a SizeVar instead of Var.

    Returns
    -------
    res : Var
        The pointer.
    """
    return _ffi_api.Ptr(dtype, storage_scope, is_size_var)  # type: ignore[attr-defined] # pylint: disable=no-member


@deprecated("T.buffer_var", "T.handle")
def buffer_var(dtype: str, storage_scope: str = "global") -> Var:
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


def min(a: PrimExpr, b: PrimExpr) -> PrimExpr:  # pylint: disable=redefined-builtin
    """Compute the minimum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api.min(a, b)  # type: ignore[attr-defined] # pylint: disable=no-member


def max(a: PrimExpr, b: PrimExpr) -> PrimExpr:  # pylint: disable=redefined-builtin
    """Compute the maximum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.
    """
    return _ffi_api.max(a, b)  # type: ignore[attr-defined] # pylint: disable=no-member


def iter_var(v: Var | str, dom: ir.Range, iter_type: str, thread_tag: str) -> IterVar:
    """The iteration variable.

    Parameters
    ----------
    var : Union[Var, str]
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


def comm_reducer(combiner: Callable, identity: list[PrimExpr]) -> CommReducer:
    """
    Create a CommReducer from lambda inputs/outputs and the identities

    Parameters
    ----------
    combiner : Callable
        A binary function which takes two PrimExpr as input to return a PrimExpr.

    identity : List[PrimExpr]
        A list of types of output PrimExpr.

    Returns
    -------
    res : CommReducer
        The CommReducer.
    """
    params = inspect.signature(combiner).parameters
    num_args = len(params)
    args = []
    for name, i in zip(params.keys(), identity + identity):
        if isinstance(i, int):
            args.append(Var(name, "int32"))
        else:
            args.append(Var(name, i.dtype))
    res = combiner(*args)
    if not isinstance(res, tuple):
        res = (res,)
    return CommReducer(args[: num_args // 2], args[num_args // 2 :], res, identity)


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


def Range(begin: PrimExpr, end: PrimExpr) -> ir.Range:  # pylint: disable=invalid-name
    """
    Create a Range object.

    Parameters
    ----------
    begin : PrimExpr
        The begin value of the range.

    end : Optional[PrimExpr]
        The end value of the range.
    """
    return ir.Range(begin, end)


if TYPE_CHECKING:
    T = TypeVar("T")
    C = TypeVar("C")

    # When type checking (and by extension, for linters like Pylint), treat
    # meta_var as an identity function.
    def meta_var(x: T) -> T:
        return x

    def meta_class(cls: C) -> C:
        return cls

else:

    def _install_meta_class(cls):
        if cls.__dict__.get("_tirx_meta_class_installed", False):
            cls._is_meta_class = True
            return cls

        original_init = getattr(cls, "__init__", object.__init__)
        original_setattr = getattr(cls, "__setattr__", object.__setattr__)
        original_init_subclass = getattr(cls, "__init_subclass__", None)

        def __init__(self, *args, **kwargs):
            with _with_meta_construction_scope(self, type(self)) as scope:
                original_init(self, *args, **kwargs)
                _validate_meta_construction_scope(scope)

        def __setattr__(self, name, value):
            if isinstance(value, scalar_wrapper):
                value = value.scalar
            original_setattr(self, name, value)

        @classmethod
        def __init_subclass__(subcls, **kwargs):
            if original_init_subclass is not None:
                original_init_subclass(**kwargs)
            _install_meta_class(subcls)

        cls.__init__ = __init__
        cls.__setattr__ = __setattr__
        cls.__init_subclass__ = __init_subclass__
        cls._is_meta_class = True
        cls._tirx_meta_class_installed = True
        return cls

    def meta_class(cls):
        """Decorator for utility classes used inside @T.prim_func.

        Instances of decorated classes are treated as parser meta values.
        """
        return _install_meta_class(cls)

    class meta_var:
        """A meta variable used in TVMScript metaprogramming.

        The value does not appear in the final TIR and only exists in the parser.

        Parameters
        ----------
        value: Any
            The meta variable.
        """

        def __init__(self, value: Any) -> None:
            self.value = value

        def __iter__(self):
            # Return a generator that yields wrapped items.
            return (meta_var(i) for i in self.value)


# pylint: disable=invalid-name


T = TypeVar("T")
P = ParamSpec("P")


def _op_wrapper(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapped(*args, **kwargs) -> T:
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        return func(*args, **kwargs)

    # Expose underlying tir op name for printer registration
    try:
        wrapped.__tir_op_name__ = getattr(func, "__name__", None)
    except Exception:  # pragma: no cover
        pass
    return wrapped


def _dtype_forward(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"), *args)
        return func(*args, **kwargs)

    # Expose underlying tir op name for printer registration
    try:
        wrapped.__tir_op_name__ = getattr(func, "__name__", None)
    except Exception:  # pragma: no cover
        pass
    return wrapped


def _ptx_ldg32(reg, guard, addr, local_addr):
    if isinstance(addr, Buffer):
        addr = addr[0]
    return _tir_op.call_intrin(reg.dtype, "tirx.ptx.ldg32", reg, guard, addr, local_addr)


_ptx_ldg32.__tir_op_name__ = "ptx.ldg32"


class PTXNamespace:
    """The PTX instruction submodule."""

    def __init__(self):
        self.ldg32 = _ptx_ldg32
        self.ldmatrix = _dtype_forward(_tir_op.ptx_ldmatrix)
        # Apache-compatible variant. Same lowered intrinsic as
        # ``ldmatrix`` but accepts the historical ``(trans, num, dtype,
        # local_ptr, local_offset, smem_ptr, smem_offset)`` form. Coexists
        # with the fork-native version so upstream-derived tests keep
        # working without rewriting their tirx code.
        self.ldmatrix_legacy = _dtype_forward(_tir_op.ptx_ldmatrix_legacy)
        self.stmatrix = _op_wrapper(_tir_op.ptx_stmatrix)
        self.setmaxnreg: Callable[..., Any] = _op_wrapper(_tir_op.ptx_setmaxnreg)
        self.elect_sync: Callable[..., Any] = _op_wrapper(_tir_op.ptx_elect_sync)
        self.fetch_register: Callable[..., Any] = _op_wrapper(_tir_op.ptx_fetch_register)
        self.ld = _op_wrapper(_tir_op.ptx_ld)
        self.ld_acquire = _op_wrapper(_tir_op.ptx_ld_acquire)
        self.ld_volatile = _op_wrapper(_tir_op.ptx_ld_volatile)
        self.ld_global_acquire = _op_wrapper(_tir_op.ptx_ld_global_acquire)
        self.red_scalar = _op_wrapper(_tir_op.ptx_red_scalar)
        self.atom_scalar = _op_wrapper(_tir_op.ptx_atom_scalar)
        self.prefetch_tensormap = _op_wrapper(_tir_op.ptx_prefetch_tensormap)
        self.mbarrier_test_wait_parity = _op_wrapper(_tir_op.ptx_mbarrier_test_wait_parity)
        self.cp_async_bulk_g2s_cta = _op_wrapper(_tir_op.ptx_cp_async_bulk_g2s_cta)
        self.cp_async_bulk_g2s_cluster = _op_wrapper(_tir_op.ptx_cp_async_bulk_g2s_cluster)
        self.cp_async_bulk_s2s_cluster = _op_wrapper(_tir_op.ptx_cp_async_bulk_s2s_cluster)
        self.cp_async_bulk_s2g = _op_wrapper(_tir_op.ptx_cp_async_bulk_s2g)
        self.st = _op_wrapper(_tir_op.ptx_st)
        self.st_bulk = _op_wrapper(_tir_op.ptx_st_bulk)
        self.fns_b32 = _op_wrapper(_tir_op.ptx_fns_b32)
        self.add_rn_f32_bf16 = _op_wrapper(_tir_op.ptx_add_rn_f32_bf16)
        self.mapa = _op_wrapper(_tir_op.ptx_mapa)
        self.map_shared_rank = _op_wrapper(_tir_op.ptx_map_shared_rank)
        self.any_sync = _op_wrapper(_tir_op.ptx_any_sync)
        # Math operations
        self.exp2 = _op_wrapper(_tir_op.ptx_exp2)
        self.rcp = _op_wrapper(_tir_op.ptx_rcp)
        self.reduce3_min_f32 = _op_wrapper(_tir_op.ptx_reduce3_min_f32)
        self.reduce3_max_f32 = _op_wrapper(_tir_op.ptx_reduce3_max_f32)
        # add/sub/mul/fma DPS form: (d_addr, a, b[, c], *, rounding, ftz[, sat])
        self.add_f32 = _op_wrapper(_tir_op.ptx_add_f32)
        self.add_f32x2 = _op_wrapper(_tir_op.ptx_add_f32x2)
        self.add_f64 = _op_wrapper(_tir_op.ptx_add_f64)
        self.sub_f32 = _op_wrapper(_tir_op.ptx_sub_f32)
        self.sub_f32x2 = _op_wrapper(_tir_op.ptx_sub_f32x2)
        self.sub_f64 = _op_wrapper(_tir_op.ptx_sub_f64)
        self.mul_f32 = _op_wrapper(_tir_op.ptx_mul_f32)
        self.mul_f32x2 = _op_wrapper(_tir_op.ptx_mul_f32x2)
        self.mul_f64 = _op_wrapper(_tir_op.ptx_mul_f64)
        self.fma_f32 = _op_wrapper(_tir_op.ptx_fma_f32)
        self.fma_f32x2 = _op_wrapper(_tir_op.ptx_fma_f32x2)
        self.fma_f64 = _op_wrapper(_tir_op.ptx_fma_f64)
        self.max_f32 = _op_wrapper(_tir_op.ptx_max_f32)
        self.mma = MmaNamespace()
        self.cp_async = CpAsyncNamespace()
        self.wgmma = WgmmaNamespace()
        self.mbarrier = MbarrierNamespace()
        self.tcgen05 = Tcgen05Namespace()
        self.bar = BarNamespace()
        self.barrier = BarrierNamespace()
        self.fence = FenceNamespace()
        self.griddepcontrol = GriddepcontrolNamespace()


class MmaNamespace:
    """The MMA instruction submodule."""

    def __init__(self):
        self.sp = _dtype_forward(_tir_op.ptx_mma_sp)
        # Apache-compatible variant of ptx_mma. Coexists with the
        # fork-native ``__call__`` form (``T.ptx.mma(...)``).
        self.legacy = _dtype_forward(_tir_op.ptx_mma_legacy)
        # __call__ corresponds to ptx_mma
        self.__tir_call_op_name__ = "ptx_mma"

    def __call__(self, *args, **kwds):
        return _dtype_forward(_tir_op.ptx_mma)(*args, **kwds)


class CpAsyncNamespace:
    """The CpAsync instruction submodule."""

    def __init__(self):
        self.commit_group = _op_wrapper(_tir_op.ptx_cp_async_commit_group)
        self.wait_group = _op_wrapper(_tir_op.ptx_cp_async_wait_group)
        # Legacy variant: takes (dst_ptr, dst_offset, src_ptr, src_offset,
        # cp_size). Offsets are folded into the pointers; coexists with
        # the fork-native ``__call__`` form.
        self.legacy = _dtype_forward(_tir_op.ptx_cp_async_legacy)
        self.bulk = CpAsyncBulkNamespace()
        self.mbarrier = CpAsyncMbarrierNamespace()

    def __call__(self, *args, **kwds):
        # Accept the legacy 6-arg form ``(elem_dtype, dst, dst_off, src,
        # src_off, cp_size)`` that the printer round-trips for the raw
        # ``tirx.ptx_cp_async`` Call emitted by ``s_tir/transform/
        # InjectPTXAsyncCopy``. The pass-emitted Call has 5 args (no
        # ``tvm_access_ptr`` fold) and a per-element-dtype Call.dtype,
        # so build it directly.
        if len(args) == 6 and isinstance(args[0], str) and "dtype" not in kwds:
            import tvm

            elem_dtype, dst, dst_off, src, src_off, cp_size = args
            return tvm.tirx.Call(
                tvm.DataType(elem_dtype),
                tvm.ir.Op.get("tirx.ptx_cp_async"),
                [dst, dst_off, src, src_off, cp_size],
            )
        return _dtype_forward(_tir_op.ptx_cp_async)(*args, **kwds)

    # __call__ corresponds to ptx_cp_async
    __tir_call_op_name__ = "ptx_cp_async"


class CpAsyncBulkNamespace:
    """The CpAsyncBulk instruction submodule."""

    def __init__(self):
        self.commit_group = _op_wrapper(_tir_op.ptx_cp_async_bulk_commit_group)
        self.wait_group = _op_wrapper(_tir_op.ptx_cp_async_bulk_wait_group)
        self.tensor = CpAsyncBulkTensorNamespace()
        self.s2c = _op_wrapper(_tir_op.ptx_cp_async_bulk_shared_to_cluster)

    def __call__(self, *args, **kwds):
        return _dtype_forward(_tir_op.ptx_cp_async_bulk)(*args, **kwds)

    # __call__ corresponds to ptx_cp_async_bulk
    __tir_call_op_name__ = "ptx_cp_async_bulk"


class CpAsyncBulkTensorNamespace:
    """The CpAsyncBulkTensor instruction submodule."""

    def __init__(self):
        self.g2c = _op_wrapper(_tir_op.ptx_cp_async_bulk_tensor_global_to_cluster)
        self.g2c_tile_gather4 = _op_wrapper(
            _tir_op.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster
        )
        self.s2g = _op_wrapper(_tir_op.ptx_cp_async_bulk_tensor_shared_to_global)
        self.s2g_reduce = _op_wrapper(_tir_op.ptx_cp_async_bulk_tensor_shared_to_global_reduce)
        self.g2c_prefetch = _op_wrapper(_tir_op.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch)

    @staticmethod
    def g2c_bar_addr(
        dim,
        dst_ptr,
        bar_addr,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_hint,
        *coords,
        cache_policy=None,
    ):
        _tir_op._choice("cta_group", cta_group, _tir_op._TCGEN05_CTA_GROUP)
        cache_policy, has_cache_policy = _tir_op._resolve_cache_policy(cache_hint, cache_policy)
        return _tir_op.call_intrin(
            "",
            "tirx.ptx_cp_async_bulk_tensor_global_to_cluster",
            dim,
            dst_ptr,
            bar_addr,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_policy,
            int(has_cache_policy),
            1,
            *coords,
        )

    @staticmethod
    def g2c_tile_gather4_bar_addr(
        dim,
        dst_ptr,
        bar_addr,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_hint,
        *coords,
        cache_policy=None,
    ):
        _tir_op._choice("cta_group", cta_group, _tir_op._TCGEN05_CTA_GROUP)
        cache_policy, has_cache_policy = _tir_op._resolve_cache_policy(cache_hint, cache_policy)
        return _tir_op.call_intrin(
            "",
            "tirx.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster",
            dim,
            dst_ptr,
            bar_addr,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_policy,
            int(has_cache_policy),
            1,
            *coords,
        )


class CpAsyncMbarrierNamespace:
    """The CpAsyncMbarrier instruction submodule."""

    def __init__(self):
        self.arrive = _op_wrapper(_tir_op.ptx_cp_async_mbarrier_arrive)


class WgmmaNamespace:
    """The WGMMA instruction submodule."""

    def __init__(self):
        self.fence: Callable[..., Any] = _op_wrapper(_tir_op.ptx_wgmma_fence)
        self.commit_group = _op_wrapper(_tir_op.ptx_wgmma_commit_group)
        self.wait_group = _op_wrapper(_tir_op.ptx_wgmma_wait_group)
        self.noop_barrier = _op_wrapper(_tir_op.ptx_wgmma_noop_barrier)
        self.mma_async = WgmmaMmaAsyncNamespace()
        self.encode_matrix_descriptor = _op_wrapper(_tir_op.ptx_wgmma_encode_matrix_descriptor)


class WgmmaMmaAsyncNamespace:
    """The WGMMA MMAAsync instruction submodule."""

    def __init__(self):
        self.ss = _op_wrapper(_tir_op.ptx_wgmma_mma_async_ss)
        self.rs = _op_wrapper(_tir_op.ptx_wgmma_mma_async_rs)


class MbarrierNamespace:
    """The Mbarrier instruction submodule."""

    def __init__(self):
        self.init = _op_wrapper(_tir_op.ptx_mbarrier_init)
        self.try_wait = _op_wrapper(_tir_op.ptx_mbarrier_try_wait)
        self.try_wait_once = _op_wrapper(_tir_op.ptx_mbarrier_try_wait_once)
        self.arrive = MbarrierArriveNamespace()


class MbarrierArriveNamespace:
    """The Mbarrier Arrive instruction submodule."""

    def __init__(self):
        self.expect_tx = _op_wrapper(_tir_op.ptx_mbarrier_arrive_expect_tx)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_tir_op.ptx_mbarrier_arrive)(*args, **kwds)

    # __call__ corresponds to ptx_mbarrier_arrive
    __tir_call_op_name__ = "ptx_mbarrier_arrive"


class Tcgen05Namespace:
    """The Tcgen05 instruction submodule."""

    def __init__(self):
        self.alloc = _op_wrapper(_tir_op.ptx_tcgen05_alloc)
        self.dealloc = _op_wrapper(_tir_op.ptx_tcgen05_dealloc)
        self.relinquish_alloc_permit = _op_wrapper(_tir_op.ptx_tcgen05_relinquish_alloc_permit)
        self.encode_matrix_descriptor = _op_wrapper(_tir_op.ptx_tcgen05_encode_matrix_descriptor)
        self.encode_instr_descriptor = _op_wrapper(_tir_op.ptx_tcgen05_encode_instr_descriptor)
        self.encode_instr_descriptor_block_scaled = _op_wrapper(
            _tir_op.ptx_tcgen05_encode_instr_descriptor_block_scaled
        )
        self.ld = _op_wrapper(_tir_op.ptx_tcgen05_ld)
        self.st = _op_wrapper(_tir_op.ptx_tcgen05_st)
        self.cp = _op_wrapper(_tir_op.ptx_tcgen05_cp)
        self.shift = _op_wrapper(_tir_op.ptx_tcgen05_shift)
        self.commit = _op_wrapper(_tir_op.ptx_tcgen05_commit)
        self.wait = Tcgen05WaitNamespace()
        self.mma = Tcgen05MmaNamespace()
        self.fence = Tcgen05FenceNamespace()


class Tcgen05FenceNamespace:
    """The Tcgen05 Fence instruction submodule."""

    def __init__(self):
        self.before_thread_sync = _op_wrapper(_tir_op.ptx_tcgen05_fence_before_thread_sync)
        self.after_thread_sync = _op_wrapper(_tir_op.ptx_tcgen05_fence_after_thread_sync)


class Tcgen05MmaNamespace:
    """The Tcgen05 MMA instruction submodule."""

    def __init__(self):
        self.block_scale = _op_wrapper(_tir_op.ptx_tcgen05_mma_block_scale)
        self.sp = Tcgen05MmaSpNamespace()

    def __call__(self, *args, **kwds):
        return _op_wrapper(_tir_op.ptx_tcgen05_mma)(*args, **kwds)

    # __call__ corresponds to ptx_tcgen05_mma
    __tir_call_op_name__ = "ptx_tcgen05_mma"


class Tcgen05MmaSpNamespace:
    """Tcgen05 Sparse MMA instruction submodule."""

    def __init__(self):
        self.block_scale = _op_wrapper(_tir_op.ptx_tcgen05_mma_sp_block_scale)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_tir_op.ptx_tcgen05_mma_sp)(*args, **kwds)

    # __call__ corresponds to ptx_tcgen05_mma_sp
    __tir_call_op_name__ = "ptx_tcgen05_mma_sp"


class Tcgen05WaitNamespace:
    """The Tcgen05 Wait instruction submodule."""

    def __init__(self):
        self.ld = _op_wrapper(_tir_op.ptx_tcgen05_wait_ld)
        self.st = _op_wrapper(_tir_op.ptx_tcgen05_wait_st)


class BarNamespace:
    """The Bar instruction submodule."""

    def __init__(self):
        self.arrive = _op_wrapper(_tir_op.ptx_bar_arrive)
        self.sync = _op_wrapper(_tir_op.ptx_bar_sync)


class BarrierNamespace:
    """The Barrier instruction submodule."""

    def __init__(self):
        self.cluster = BarrierClusterNamespace()


class BarrierClusterNamespace:
    """The BarrierCluster instruction submodule."""

    def __init__(self):
        self.arrive = _op_wrapper(_tir_op.ptx_barrier_cluster_arrive)
        self.wait = _op_wrapper(_tir_op.ptx_barrier_cluster_wait)


class FenceNamespace:
    """PTX fence instruction submodule."""

    def __init__(self):
        self.proxy_async = _op_wrapper(_tir_op.ptx_fence_proxy_async)
        self.mbarrier_init = _op_wrapper(_tir_op.ptx_fence_mbarrier_init)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_tir_op.ptx_fence)(*args, **kwds)

    __tir_call_op_name__ = "ptx_fence"


class GriddepcontrolNamespace:
    """PTX griddepcontrol instruction submodule (sm_90+)."""

    def __init__(self):
        self.wait = _op_wrapper(_tir_op.ptx_griddepcontrol_wait)
        self.launch_dependents = _op_wrapper(_tir_op.ptx_griddepcontrol_launch_dependents)


class CUDANamespace:
    """The CUDA intrinsics submodule."""

    def __init__(self):
        self.atomic_add = _op_wrapper(_tir_op.cuda_atomic_add)
        self.thread_fence = _op_wrapper(_tir_op.cuda_thread_fence)
        self.warpgroup_sync = _op_wrapper(_tir_op.cuda_warpgroup_sync)
        self.warp_sync = _op_wrapper(_tir_op.cuda_warp_sync)
        self.warp_reduce = _op_wrapper(_tir_op.cuda_warp_reduce)
        self.warp_sum = _op_wrapper(_tir_op.cuda_warp_sum)
        self.warp_max = _op_wrapper(_tir_op.cuda_warp_max)
        self.warp_min = _op_wrapper(_tir_op.cuda_warp_min)
        self.cta_reduce = _op_wrapper(_tir_op.cuda_cta_reduce)
        self.cta_sum = _op_wrapper(_tir_op.cuda_cta_sum)
        self.cta_max = _op_wrapper(_tir_op.cuda_cta_max)
        self.cta_min = _op_wrapper(_tir_op.cuda_cta_min)
        self.copy_bytes = _op_wrapper(_tir_op.cuda_copy_bytes)
        self.copy_128b = _op_wrapper(_tir_op.cuda_copy_128b)
        self.copy_64b = _op_wrapper(_tir_op.cuda_copy_64b)
        self.copy_32b = _op_wrapper(_tir_op.cuda_copy_32b)
        self.copy_16b = _op_wrapper(_tir_op.cuda_copy_16b)
        self.copy_8b = _op_wrapper(_tir_op.cuda_copy_8b)
        self.cta_sync = _op_wrapper(_tir_op.cuda_cta_sync)
        self.grid_sync = _op_wrapper(_tir_op.cuda_grid_sync)
        self.cluster_sync = _op_wrapper(_tir_op.cuda_cluster_sync)
        self.thread_rank = _op_wrapper(_tir_op.cuda_thread_rank)
        self.trap_when_assert_failed = _op_wrapper(_tir_op.cuda_trap_when_assert_failed)
        self.runtime_instr_desc = _op_wrapper(_tir_op.cuda_runtime_instr_desc)
        self.half2float = _op_wrapper(_tir_op.cuda_half2float)
        self.bfloat162float = _op_wrapper(_tir_op.cuda_bfloat162float)
        self.float22half2 = _op_wrapper(_tir_op.cuda_float22half2)
        self.half8tofloat8 = _op_wrapper(_tir_op.cuda_half8tofloat8)
        self.float8tohalf8 = _op_wrapper(_tir_op.cuda_float8tohalf8)
        self.syncthreads_and = _op_wrapper(_tir_op.cuda_syncthreads_and)
        self.syncthreads_or = _op_wrapper(_tir_op.cuda_syncthreads_or)
        self.nano_sleep = _op_wrapper(_tir_op.cuda_nano_sleep)
        self.atomic_cas = _op_wrapper(_tir_op.cuda_atomic_cas)
        self.func_call = _op_wrapper(_tir_op.cuda_func_call)
        self.printf = _op_wrapper(_tir_op.cuda_printf)
        self.ldg = _op_wrapper(_tir_op.cuda_ldg)
        self.get_tmem_addr = _op_wrapper(_tir_op.cuda_get_tmem_addr)
        self.cvta_generic_to_shared = _op_wrapper(_tir_op.cuda_cvta_generic_to_shared)
        self.smem_addr_from_uint64 = _op_wrapper(_tir_op.cuda_smem_addr_from_uint64)
        self.sm100_tma_2sm_mbarrier_addr = _op_wrapper(_tir_op.cuda_sm100_tma_2sm_mbarrier_addr)
        self.uint_as_float = _op_wrapper(_tir_op.cuda_uint_as_float)
        self.float_as_uint = _op_wrapper(_tir_op.cuda_float_as_uint)
        self.ballot_sync = _op_wrapper(_tir_op.cuda_ballot_sync)
        self.ffs_u32 = _op_wrapper(_tir_op.cuda_ffs_u32)
        self.reduce_add_sync_u32 = _op_wrapper(_tir_op.cuda_reduce_add_sync_u32)
        self.reduce_min_sync_u32 = _op_wrapper(_tir_op.cuda_reduce_min_sync_u32)
        self.clock64 = _op_wrapper(_tir_op.cuda_clock64)
        self.make_float2 = _op_wrapper(_tir_op.cuda_make_float2)
        self.float2_x = _op_wrapper(_tir_op.cuda_float2_x)
        self.float2_y = _op_wrapper(_tir_op.cuda_float2_y)
        self.fmul2_rn = _op_wrapper(_tir_op.cuda_fmul2_rn)
        self.fadd2_rn = _op_wrapper(_tir_op.cuda_fadd2_rn)
        self.float22bfloat162_rn = _op_wrapper(_tir_op.cuda_float22bfloat162_rn)
        self.float22bfloat162_rn_from_float2 = _op_wrapper(
            _tir_op.cuda_float22bfloat162_rn_from_float2
        )
        self.bfloat1622float2 = _op_wrapper(_tir_op.cuda_bfloat1622float2)
        self.hmin2 = _op_wrapper(_tir_op.cuda_hmin2)
        self.hmax2 = _op_wrapper(_tir_op.cuda_hmax2)
        self.fp8x4_e4m3_from_float4 = _op_wrapper(_tir_op.cuda_fp8x4_e4m3_from_float4)
        setattr(self, "__shfl_sync", self._shfl_sync)
        setattr(self, "__shfl_up_sync", self._shfl_up_sync)
        setattr(self, "__shfl_down_sync", self._shfl_down_sync)
        setattr(self, "__shfl_xor_sync", self._shfl_xor_sync)
        setattr(self, "__activemask", self._activemask)

    @staticmethod
    def _shfl_sync(mask, var, lane, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.cuda.__shfl_sync", mask, var, lane, width)

    @staticmethod
    def _shfl_up_sync(mask, var, delta, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.cuda.__shfl_up_sync", mask, var, delta, width)

    @staticmethod
    def _shfl_down_sync(mask, var, delta, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.cuda.__shfl_down_sync", mask, var, delta, width)

    @staticmethod
    def _shfl_xor_sync(mask, var, lane_mask, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(
            var.dtype, "tirx.cuda.__shfl_xor_sync", mask, var, lane_mask, width
        )

    @staticmethod
    def _activemask():
        return _tir_op.call_intrin("uint32", "tirx.cuda.__activemask")


class MetalNamespace:
    """The Metal intrinsics submodule."""

    @staticmethod
    def simd_shuffle(var, lane):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.metal.simd_shuffle", var, lane)

    @staticmethod
    def simd_shuffle_up(var, delta):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.metal.simd_shuffle_up", var, delta)

    @staticmethod
    def simd_shuffle_down(var, delta):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.metal.simd_shuffle_down", var, delta)


class WebGPUNamespace:
    """The WebGPU intrinsics submodule."""

    @staticmethod
    def subgroup_shuffle(var, lane):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.webgpu.subgroup_shuffle", var, lane)

    @staticmethod
    def subgroup_shuffle_up(var, delta):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.webgpu.subgroup_shuffle_up", var, delta)

    @staticmethod
    def subgroup_shuffle_down(var, delta):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.webgpu.subgroup_shuffle_down", var, delta)


class NVSHMEMNamespace:
    """The NVSHMEM intrinsics submodule."""

    def __init__(self):
        self.my_pe = _op_wrapper(_tir_op.nvshmem_my_pe)
        self.n_pes = _op_wrapper(_tir_op.nvshmem_n_pes)
        self.signal_op = _op_wrapper(_tir_op.nvshmem_signal_op)
        self.wait_until = _op_wrapper(_tir_op.nvshmem_wait_until)
        self.quiet = _op_wrapper(_tir_op.nvshmem_quiet)
        self.fence = _op_wrapper(_tir_op.nvshmem_fence)
        self.barrier_all = _op_wrapper(_tir_op.nvshmem_barrier_all)
        self.getmem_nbi = NVSHMEMGetMemNBINamespace()
        self.putmem_nbi = NVSHMEMPutMemNBINamespace()
        self.putmem_signal_nbi = NVSHMEMPutMemSignalNBINamespace()


class NVSHMEMGetMemNBINamespace:
    """The NVSHMEM GetMemNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _op_wrapper(_tir_op.nvshmem_getmem_nbi_warp)
        self.block = _op_wrapper(_tir_op.nvshmem_getmem_nbi_block)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_tir_op.nvshmem_getmem_nbi)(*args, **kwds)

    # __call__ corresponds to nvshmem_getmem_nbi
    __tir_call_op_name__ = "nvshmem_getmem_nbi"


class NVSHMEMPutMemNBINamespace:
    """The NVSHMEM PutMemNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _op_wrapper(_tir_op.nvshmem_putmem_nbi_warp)
        self.block = _op_wrapper(_tir_op.nvshmem_putmem_nbi_block)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_tir_op.nvshmem_putmem_nbi)(*args, **kwds)

    # __call__ corresponds to nvshmem_putmem_nbi
    __tir_call_op_name__ = "nvshmem_putmem_nbi"


class NVSHMEMPutMemSignalNBINamespace:
    """The NVSHMEM PutMemSignalNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _op_wrapper(_tir_op.nvshmem_putmem_signal_nbi_warp)
        self.block = _op_wrapper(_tir_op.nvshmem_putmem_signal_nbi_block)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_tir_op.nvshmem_putmem_signal_nbi)(*args, **kwds)

    # __call__ corresponds to nvshmem_putmem_signal_nbi
    __tir_call_op_name__ = "nvshmem_putmem_signal_nbi"


class NKINamespace:
    """The NKI instructions submodule."""

    def __init__(self):
        self.load = _op_wrapper(_tir_op.nki_load)
        self.store = _op_wrapper(_tir_op.nki_store)
        self.tensor_copy = _op_wrapper(_tir_op.nki_tensor_copy)
        self.matmul = _op_wrapper(_tir_op.nki_matmul)
        self.activation = _op_wrapper(_tir_op.nki_activation)
        self.activation_reduce = _op_wrapper(_tir_op.nki_activation_reduce)
        self.reciprocal = _op_wrapper(_tir_op.nki_reciprocal)
        self.tensorreduce = _op_wrapper(_tir_op.nki_tensorreduce)
        self.tensortensor = _op_wrapper(_tir_op.nki_tensortensor)
        self.tensorscalar = _op_wrapper(_tir_op.nki_tensorscalar)
        self.tensorscalar_reduce = _op_wrapper(_tir_op.nki_tensorscalar_reduce)
        self.scalar_tensor_tensor = _op_wrapper(_tir_op.nki_scalar_tensor_tensor)
        self.scalar_tensor_scalar = _op_wrapper(_tir_op.nki_scalar_tensor_scalar)
        self.memset = _op_wrapper(_tir_op.nki_memset)
        self.identity = _op_wrapper(_tir_op.nki_identity)
        self.affine_select = _op_wrapper(_tir_op.nki_affine_select)


ptx = PTXNamespace()
cuda = CUDANamespace()
metal = MetalNamespace()
webgpu = WebGPUNamespace()
nvshmem = NVSHMEMNamespace()
nki = NKINamespace()


#
# Register printer namespace mapping from the builder namespaces
# so that the TVMScript printer emits T.cuda/T.ptx/T.nvshmem/T.nki dotted names.
# This keeps parser and printer consistent using a single registration source.
#
def _register_tir_namespace_printer_names():
    def register_printer_name(op_name, script_name):
        try:
            ir.Op.get(op_name)
        except Exception:
            return
        try:
            _register_op_attr(op_name, "TScriptPrinterName", script_name, level=20)
        except Exception:
            pass

    def visit(ns_obj, dotted_prefix):
        # If the namespace object itself maps to an op via __call__
        call_op = getattr(ns_obj, "__tir_call_op_name__", None)
        if call_op:
            flat_name = f"tirx.{call_op}"
            for op_name in {flat_name, _tir_op._canonical_device_intrin_name(flat_name)}:
                register_printer_name(op_name, dotted_prefix)
        # Walk attributes to find wrapped ops and sub-namespaces
        for name in dir(ns_obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(ns_obj, name)
            except Exception:
                continue
            # Sub-namespace: recurse
            if hasattr(val, "__dict__") and val.__class__.__name__.endswith("Namespace"):
                visit(val, f"{dotted_prefix}.{name}")
                continue
            # Wrapped op (callable with attached __tir_op_name__)
            op_name = getattr(val, "__tir_op_name__", None)
            if callable(val) and op_name:
                flat_name = f"tirx.{op_name}"
                script_name = f"{dotted_prefix}.{name}"
                for full_op_name in {flat_name, _tir_op._canonical_device_intrin_name(flat_name)}:
                    register_printer_name(full_op_name, script_name)

    try:
        visit(ptx, "ptx")
        visit(cuda, "cuda")
        visit(metal, "metal")
        visit(webgpu, "webgpu")
        visit(nvshmem, "nvshmem")
        visit(nki, "nki")
    except Exception:
        # Best-effort registration; avoid import-time hard failure
        pass


# Execute registration on import so printer picks up dotted names
_register_tir_namespace_printer_names()


abs = _op_wrapper(_tir_op.abs)  # pylint: disable=redefined-builtin
acos = _op_wrapper(_tir_op.acos)
acosh = _op_wrapper(_tir_op.acosh)
address_of = _op_wrapper(_tir_op.address_of)
asin = _op_wrapper(_tir_op.asin)
asinh = _op_wrapper(_tir_op.asinh)
atan = _op_wrapper(_tir_op.atan)
atan2 = _op_wrapper(_tir_op.atan2)
atanh = _op_wrapper(_tir_op.atanh)
bitwise_and = _op_wrapper(_tir_op.bitwise_and)
bitwise_not = _op_wrapper(_tir_op.bitwise_not)
bitwise_or = _op_wrapper(_tir_op.bitwise_or)
bitwise_xor = _op_wrapper(_tir_op.bitwise_xor)
ceil = _op_wrapper(_tir_op.ceil)
clz = _op_wrapper(_tir_op.clz)
copysign = _op_wrapper(_tir_op.copysign)
cos = _op_wrapper(_tir_op.cos)
cosh = _op_wrapper(_tir_op.cosh)
erf = _op_wrapper(_tir_op.erf)
exp = _op_wrapper(_tir_op.exp)
exp2 = _op_wrapper(_tir_op.exp2)
exp10 = _op_wrapper(_tir_op.exp10)
filter = _op_wrapper(_tir_op.filter)  # pylint: disable=redefined-builtin
selector = _op_wrapper(_tir_op.selector)
floor = _op_wrapper(_tir_op.floor)
ceildiv = _op_wrapper(_tir_op.ceildiv)
floordiv = _op_wrapper(_tir_op.floordiv)
floormod = _op_wrapper(_tir_op.floormod)
fmod = _op_wrapper(_tir_op.fmod)
fma = _op_wrapper(_tir_op.fma)
hypot = _op_wrapper(_tir_op.hypot)
if_then_else = _op_wrapper(_tir_op.if_then_else)
infinity = _op_wrapper(_tir_op.infinity)
isfinite = _op_wrapper(_tir_op.isfinite)
isinf = _op_wrapper(_tir_op.isinf)
isnan = _op_wrapper(_tir_op.isnan)
isnullptr = _op_wrapper(_tir_op.isnullptr)
ldexp = _op_wrapper(_tir_op.ldexp)
likely = _op_wrapper(_tir_op.likely)
log = _op_wrapper(_tir_op.log)
log1p = _op_wrapper(_tir_op.log1p)
log2 = _op_wrapper(_tir_op.log2)
log10 = _op_wrapper(_tir_op.log10)
lookup_param = _op_wrapper(_tir_op.lookup_param)
max_value = _op_wrapper(_tir_op.max_value)
min_value = _op_wrapper(_tir_op.min_value)
nearbyint = _op_wrapper(_tir_op.nearbyint)
nextafter = _op_wrapper(_tir_op.nextafter)
popcount = _op_wrapper(_tir_op.popcount)
pow = _op_wrapper(_tir_op.pow)  # pylint: disable=redefined-builtin
q_multiply_shift = _op_wrapper(_tir_op.q_multiply_shift)
q_multiply_shift_per_axis = _op_wrapper(_tir_op.q_multiply_shift_per_axis)
ret = _op_wrapper(_tir_op.ret)
continue_loop = _op_wrapper(_tir_op.continue_loop)
break_loop = _op_wrapper(_tir_op.break_loop)
round = _op_wrapper(_tir_op.round)  # pylint: disable=redefined-builtin
rsqrt = _op_wrapper(_tir_op.rsqrt)
shift_left = _op_wrapper(_tir_op.shift_left)
shift_right = _op_wrapper(_tir_op.shift_right)
sigmoid = _op_wrapper(_tir_op.sigmoid)
sin = _op_wrapper(_tir_op.sin)
sinh = _op_wrapper(_tir_op.sinh)
sqrt = _op_wrapper(_tir_op.sqrt)
tan = _op_wrapper(_tir_op.tan)
tanh = _op_wrapper(_tir_op.tanh)
thread_return = _op_wrapper(_tir_op.thread_return)
trunc = _op_wrapper(_tir_op.trunc)
truncdiv = _op_wrapper(_tir_op.truncdiv)
truncmod = _op_wrapper(_tir_op.truncmod)
tvm_access_ptr = _op_wrapper(_tir_op.tvm_access_ptr)
ptr_byte_offset = _op_wrapper(_tir_op.ptr_byte_offset)
tvm_throw_last_error = _op_wrapper(_tir_op.tvm_throw_last_error)
tvm_stack_alloca = _op_wrapper(_tir_op.tvm_stack_alloca)
tvm_stack_make_shape = _op_wrapper(_tir_op.tvm_stack_make_shape)
tvm_stack_make_array = _op_wrapper(_tir_op.tvm_stack_make_array)
call_packed = _op_wrapper(_tir_op.call_packed)
call_cpacked = _op_wrapper(_tir_op.call_cpacked)
call_packed_lowered = _op_wrapper(_tir_op.call_packed_lowered)
call_cpacked_lowered = _op_wrapper(_tir_op.call_cpacked_lowered)
tvm_tuple = _op_wrapper(_tir_op.tvm_tuple)
handle_add_byte_offset = _op_wrapper(_tir_op.handle_add_byte_offset)
tvm_struct_set = _op_wrapper(_tir_op.tvm_struct_set)
tvm_struct_get = _tir_op.tvm_struct_get
tvm_thread_invariant = _op_wrapper(_tir_op.tvm_thread_invariant)
tvm_thread_allreduce = _op_wrapper(_tir_op.tvm_thread_allreduce)
tvm_load_matrix_sync = _op_wrapper(_tir_op.tvm_load_matrix_sync)
tvm_mma_sync = _op_wrapper(_tir_op.tvm_mma_sync)
tvm_bmma_sync = _op_wrapper(_tir_op.tvm_bmma_sync)
tvm_fill_fragment = _op_wrapper(_tir_op.tvm_fill_fragment)
tvm_store_matrix_sync = _op_wrapper(_tir_op.tvm_store_matrix_sync)
tvm_storage_sync = _tir_op.tvm_storage_sync
tvm_global_barrier_kinit = _tir_op.tvm_global_barrier_kinit
tvm_warp_shuffle = _tir_op.tvm_warp_shuffle
tvm_warp_shuffle_up = _tir_op.tvm_warp_shuffle_up
tvm_warp_shuffle_down = _tir_op.tvm_warp_shuffle_down
tvm_warp_shuffle_xor = _tir_op.tvm_warp_shuffle_xor
tvm_warp_activemask = _tir_op.tvm_warp_activemask
make_filled_simdgroup_matrix = _op_wrapper(_tir_op.make_filled_simdgroup_matrix)
simdgroup_load = _op_wrapper(_tir_op.simdgroup_load)
simdgroup_store = _op_wrapper(_tir_op.simdgroup_store)
simdgroup_multiply_accumulate = _op_wrapper(_tir_op.simdgroup_multiply_accumulate)
cooperative_tensor_fill = _op_wrapper(_tir_op.cooperative_tensor_fill)
cooperative_tensor_load = _op_wrapper(_tir_op.cooperative_tensor_load)
cooperative_tensor_store = _op_wrapper(_tir_op.cooperative_tensor_store)
cooperative_tensor_multiply_accumulate = _op_wrapper(_tir_op.cooperative_tensor_multiply_accumulate)
assume = _op_wrapper(_tir_op.assume)
undef = _op_wrapper(_tir_op.undef)
TVMBackendAllocWorkspace = _op_wrapper(_tir_op.TVMBackendAllocWorkspace)
TVMBackendFreeWorkspace = _op_wrapper(_tir_op.TVMBackendFreeWorkspace)
start_profile_intrinsic = _op_wrapper(_tir_op.start_profile_intrinsic)
end_profile_intrinsic = _op_wrapper(_tir_op.end_profile_intrinsic)
anylist_getitem = _op_wrapper(_tir_op.anylist_getitem)
anylist_resetitem = _op_wrapper(_tir_op.anylist_resetitem)
anylist_setitem_call_packed = _op_wrapper(_tir_op.anylist_setitem_call_packed)
anylist_setitem_call_cpacked = _op_wrapper(_tir_op.anylist_setitem_call_cpacked)
vscale = _op_wrapper(_tir_op.vscale)
ignore_loop_partition = _op_wrapper(_tir_op.ignore_loop_partition)
print_buffer = _op_wrapper(_tir_op.print_buffer)
timer_init_cuda = _op_wrapper(_tir_op.timer_init_cuda)
timer_start_cuda = _op_wrapper(_tir_op.timer_start_cuda)
timer_end_cuda = _op_wrapper(_tir_op.timer_end_cuda)
timer_finalize_cuda = _op_wrapper(_tir_op.timer_finalize_cuda)

reinterpret = _dtype_forward(_tir_op.reinterpret)
call_extern = _dtype_forward(_tir_op.call_extern)
call_intrin = _dtype_forward(_tir_op.call_intrin)
call_llvm_intrin = _dtype_forward(_tir_op.call_llvm_intrin)
call_llvm_pure_intrin = _dtype_forward(_tir_op.call_llvm_pure_intrin)
call_pure_extern = _dtype_forward(_tir_op.call_pure_extern)
mma_store = _dtype_forward(_tir_op.mma_store)
mma_fill = _dtype_forward(_tir_op.mma_fill)
mma_store_legacy = _dtype_forward(_tir_op.mma_store_legacy)
mma_fill_legacy = _dtype_forward(_tir_op.mma_fill_legacy)
vectorlow = _dtype_forward(_tir_op.vectorlow)
vectorhigh = _dtype_forward(_tir_op.vectorhigh)
vectorcombine = _dtype_forward(_tir_op.vectorcombine)
get_active_lane_mask = _dtype_forward(_tir_op.get_active_lane_mask)
dp4a = _dtype_forward(_tir_op.dp4a)


broadcast = Broadcast
ramp = Ramp
fabs = abs
tvm_call_packed = call_packed
tvm_call_cpacked = call_cpacked
tvm_call_packed_lowered = call_packed_lowered
tvm_call_cpacked_lowered = call_cpacked_lowered

# pylint: enable=invalid-name

bases = [
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float4_e2m1fn",
    "float16",
    "float32",
    "float64",
]
lanes = [1, 2, 4, 8, 16, 32, 64]

float_types = []
for base in bases:
    for lane in lanes:
        suffix = f"x{lane}" if lane != 1 else ""
        float_types.append(f"{base}{suffix}")

__all__ = [
    *float_types,
    "int8",
    "int16",
    "int32",
    "int64",
    "int8x2",
    "int16x2",
    "int32x2",
    "int64x2",
    "int8x4",
    "int16x4",
    "int32x4",
    "int64x4",
    "int8x8",
    "int16x8",
    "int32x8",
    "int64x8",
    "int8x16",
    "int16x16",
    "int32x16",
    "int64x16",
    "int8x32",
    "int16x32",
    "int32x32",
    "int64x32",
    "int8x64",
    "int16x64",
    "int32x64",
    "int64x64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uint8x2",
    "uint16x2",
    "uint32x2",
    "uint64x2",
    "uint8x4",
    "uint16x4",
    "uint32x4",
    "uint64x4",
    "uint8x8",
    "uint16x8",
    "uint32x8",
    "uint64x8",
    "uint8x16",
    "uint16x16",
    "uint32x16",
    "uint64x16",
    "uint8x32",
    "uint16x32",
    "uint32x32",
    "uint64x32",
    "uint8x64",
    "uint16x64",
    "uint32x64",
    "uint64x64",
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1fn",
    "float16",
    "float32",
    "float64",
    "float4_e2m1fnx2",
    "float8_e4m3fnx4",
    "float8_e5m2x4",
    "float4_e2m1fnx4",
    "float16x2",
    "float32x2",
    "float64x2",
    "float16x4",
    "float32x4",
    "float64x4",
    "float8_e4m3fnx8",
    "float8_e5m2x8",
    "float4_e2m1fnx8",
    "float16x8",
    "float32x8",
    "float64x8",
    "float8_e4m3fnx16",
    "float8_e5m2x16",
    "float4_e2m1fnx16",
    "float16x16",
    "float32x16",
    "float64x16",
    "float8_e4m3fnx32",
    "float8_e5m2x32",
    "float4_e2m1fnx32",
    "float16x32",
    "float32x32",
    "float64x32",
    "float8_e4m3fnx64",
    "float8_e5m2x64",
    "float4_e2m1fnx64",
    "float16x64",
    "float32x64",
    "float64x64",
    "bfloat16",
    "buffer",
    "buffer_decl",
    "prim_func",
    "arg",
    "func_name",
    "func_attr",
    "func_ret",
    "match_buffer",
    "sblock",
    "block_name_suffix_context",
    "init",
    "where",
    "reads",
    "writes",
    "sblock_attr",
    "alloc_buffer",
    "sblock_alloc_buffer",
    "wg_reg_tile",
    "axis",
    "serial",
    "parallel",
    "vectorized",
    "unroll",
    "thread_binding",
    "grid",
    "Assert",
    "attr",
    "hint",
    "While",
    "Break",
    "Continue",
    "If",
    "Then",
    "Else",
    "decl_buffer",
    "launch_thread",
    "env_thread",
    "buffer_store",
    "evaluate",
    "boolean",
    "handle",
    "void",
    "var",
    "ptr",
    "min",
    "max",
    "iter_var",
    "comm_reducer",
    "index_map",
    "target",
    "buffer_var",
    "abs",
    "fabs",
    "acos",
    "acosh",
    "address_of",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "ceil",
    "clz",
    "copysign",
    "cos",
    "cosh",
    "erf",
    "exp",
    "exp2",
    "exp10",
    "floor",
    "ceildiv",
    "floordiv",
    "floormod",
    "fmod",
    "fma",
    "filter",
    "selector",
    "hypot",
    "if_then_else",
    "infinity",
    "isfinite",
    "isinf",
    "isnan",
    "isnullptr",
    "ldexp",
    "likely",
    "log",
    "log1p",
    "log2",
    "log10",
    "lookup_param",
    "max_value",
    "min_value",
    "nearbyint",
    "nextafter",
    "popcount",
    "pow",
    "q_multiply_shift",
    "q_multiply_shift_per_axis",
    "ret",
    "continue_loop",
    "break_loop",
    "reinterpret",
    "round",
    "rsqrt",
    "shift_left",
    "shift_right",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "thread_return",
    "trunc",
    "truncdiv",
    "truncmod",
    "tvm_access_ptr",
    "ptr_byte_offset",
    "tvm_throw_last_error",
    "tvm_stack_alloca",
    "tvm_stack_make_shape",
    "tvm_stack_make_array",
    "call_packed",
    "call_cpacked",
    "call_packed_lowered",
    "call_cpacked_lowered",
    "call_extern",
    "call_intrin",
    "call_llvm_intrin",
    "call_llvm_pure_intrin",
    "call_pure_extern",
    "tvm_tuple",
    "handle_add_byte_offset",
    "tvm_struct_set",
    "tvm_struct_get",
    "tvm_thread_invariant",
    "tvm_thread_allreduce",
    "tvm_load_matrix_sync",
    "tvm_mma_sync",
    "tvm_bmma_sync",
    "tvm_fill_fragment",
    "tvm_store_matrix_sync",
    "tvm_storage_sync",
    "tvm_global_barrier_kinit",
    "tvm_warp_shuffle",
    "tvm_warp_shuffle_up",
    "tvm_warp_shuffle_down",
    "tvm_warp_shuffle_xor",
    "tvm_warp_activemask",
    "make_filled_simdgroup_matrix",
    "simdgroup_load",
    "simdgroup_store",
    "simdgroup_multiply_accumulate",
    "cooperative_tensor_fill",
    "cooperative_tensor_load",
    "cooperative_tensor_store",
    "cooperative_tensor_multiply_accumulate",
    "mma_store",
    "mma_fill",
    "mma_store_legacy",
    "mma_fill_legacy",
    "vectorlow",
    "vectorhigh",
    "vectorcombine",
    "dp4a",
    "assume",
    "undef",
    "tvm_call_packed",
    "tvm_call_cpacked",
    "tvm_call_packed_lowered",
    "tvm_call_cpacked_lowered",
    "TVMBackendAllocWorkspace",
    "TVMBackendFreeWorkspace",
    "start_profile_intrinsic",
    "end_profile_intrinsic",
    "meta_var",
    "anylist_getitem",
    "anylist_resetitem",
    "anylist_setitem_call_packed",
    "anylist_setitem_call_cpacked",
    "llvm_lookup_intrinsic_id",
    "type_annotation",
    "broadcast",
    "ramp",
    "cast",
    # tvm.tirx.expr
    "Var",
    "SizeVar",
    "Reduce",
    "FloatImm",
    "IntImm",
    "StringImm",
    "Cast",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "FloorDiv",
    "FloorMod",
    "Min",
    "Max",
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
    "And",
    "Or",
    "Not",
    "Select",
    "BufferLoad",
    "ProducerLoad",
    "Ramp",
    "Broadcast",
    "Shuffle",
    "Call",
    "CallEffectKind",
    "let",
    "Bind",
    "bind",
    "LetAnnotation",
    "LocalVectorAnnotation",
    "DtypeConstructor",
    "Let",
    "IterVar",
    "CommReducer",
    "Range",
    "vscale",
    "get_active_lane_mask",
    "call_kernel",
    "ignore_loop_partition",
    "print_buffer",
    "timer_init_cuda",
    "timer_start_cuda",
    "timer_end_cuda",
    "timer_finalize_cuda",
]

__all__ += [
    "ComposeLayout",
    "ExecScope",
    "Iter",
    "Layout",
    "R",
    "S",
    "ScopeIdDef",
    "SwizzleLayout",
    "TensorMap",
    "TileLayout",
    "Var",
    "add_to_parent",
    "alloc_cast_frag",
    "alloc_local",
    "alloc_scalar",
    "alloc_shared",
    "alloc_tcgen05_ldst_frag",
    "cluster_id",
    "cta_id",
    "cta_id_in_cluster",
    "cta_id_in_pair",
    "cuda",
    "decl_scalar",
    "device_entry",
    "lane_id",
    "local_scalar",
    "meta_class",
    "metal",
    "nki",
    "nvshmem",
    "ptx",
    "scalar_wrapper",
    "scope_id",
    "shared_scalar",
    "smem",
    "static_assert",
    "thread_id",
    "thread_id_in_wg",
    "tmem",
    "warp_id",
    "warp_id_in_wg",
    "warpgroup_id",
    "webgpu",
]

# Shorthand dtype aliases
__all__ += ["bf16", "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"]
