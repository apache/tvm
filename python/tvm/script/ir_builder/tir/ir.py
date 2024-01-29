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

import functools
import inspect
from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# isort: off
from typing_extensions import Literal

# isort: on

import numpy as np  # type: ignore

from tvm import tir
from tvm import ir
from tvm.ir import Type
from tvm.ir.base import deprecated
from tvm.runtime import String, convert, ndarray
from tvm.target import Target

# pylint: disable=unused-import
from tvm.target.codegen import llvm_lookup_intrinsic_id
from tvm.tir import Buffer, BufferRegion, IndexMap, PrimExpr
from tvm.tir import op as _tir_op
from tvm.tir import type_annotation

# import tir.expr for direct ir construction to pass structural_equal comparison
from tvm.tir.expr import (
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
    Var,
)
from tvm.tir.generic import cast

from . import _ffi_api, frame

# pylint: enable=unused-import


def buffer(
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral],
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = 0,
    offset_factor: int = 0,
    buffer_type: str = "",
    axis_separators: List[int] = None,
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

    Returns
    -------
    res : Buffer
        The declared buffer.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is not None:
        strides = [Var(s, "int32") if isinstance(s, str) else s for s in strides]
    else:
        strides = []
    return _ffi_api.Buffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        dtype,
        "",
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


@deprecated("T.buffer_decl(...)", "T.Buffer(...)")
def buffer_decl(*args, **kwargs):
    return buffer(*args, **kwargs)


def prim_func(is_private: bool = False) -> frame.PrimFuncFrame:
    """The primitive function statement.

    Parameters
    ----------
    is_private : bool
        Whether the PrimFunc is annotated as private
        (if yes, it does not have a global symbol assigned;
        otherwise, the global symbol is the PrimFunc's name)

    Returns
    -------
    res : frame.PrimFuncFrame
        The PrimFuncFrame.
    """
    return _ffi_api.PrimFunc(is_private)  # type: ignore[attr-defined] # pylint: disable=no-member


def arg(name: str, obj: Union[Var, Buffer]) -> Union[Var, Buffer]:
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


def func_attr(attrs: Dict[str, Any]) -> None:
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
    param: Union[Var, BufferLoad, BufferRegion],
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral] = None,
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: List[int] = None,
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
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
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
    )


def block(name: str = "", no_realize: bool = False) -> frame.BlockFrame:
    """The block declaration statement.

    Parameters
    ----------
    name : str
        The name of the block.

    no_realize : bool
        The flag whether to construct BlockRealize or Block.

    Returns
    -------
    res : frame.BlockFrame
        The BlockFrame.
    """
    return _ffi_api.Block(name, no_realize)  # type: ignore[attr-defined] # pylint: disable=no-member


def init() -> frame.BlockInitFrame:
    """The block initialization statement.

    Returns
    -------
    res : frame.BlockInitFrame
        The BlockInitFrame.
    """
    return _ffi_api.Init()  # type: ignore[attr-defined] # pylint: disable=no-member


def where(predicate: Union[PrimExpr, int]) -> None:
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


def reads(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
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


def writes(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
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


def block_attr(attrs: Dict[str, Any]) -> None:
    """The block annotation statement.

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotation of the block.
    """
    return _ffi_api.BlockAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def alloc_buffer(
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral],
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: List[int] = None,
) -> Buffer:
    """The buffer alllocation function.

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

    Returns
    -------
    res : Buffer
        The allocated buffer.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is not None:
        strides = [Var(s, "int32") if isinstance(s, str) else s for s in strides]
    else:
        strides = []
    return _ffi_api.AllocBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
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
    )


def _as_range(dom: Union[ir.Range, List[PrimExpr]]) -> ir.Range:
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
    if isinstance(dom, (list, tuple)):
        return ir.Range(dom[0], dom[1])
    if hasattr(dom, "dtype"):
        return ir.Range(IntImm(dom.dtype, 0), dom)
    return ir.Range(0, dom)


class axis:  # pylint: disable=invalid-name
    """The axis class"""

    @staticmethod
    def spatial(
        dom: Union[ir.Range, List[PrimExpr], Tuple[PrimExpr]],
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
        dom: Union[ir.Range, List[PrimExpr], Tuple[PrimExpr]],
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
        dom: Union[ir.Range, List[PrimExpr], Tuple[PrimExpr]],
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
        dom: Union[ir.Range, List[PrimExpr], Tuple[PrimExpr]],
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
    def remap(kinds: str, bindings: List[PrimExpr], dtype: str = "int32") -> Union[List[Var], Var]:
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
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
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
    return _ffi_api.Serial(start, stop, annotations)  # type: ignore[attr-defined] # pylint: disable=no-member


def parallel(
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
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
    return _ffi_api.Parallel(start, stop, annotations)  # type: ignore[attr-defined] # pylint: disable=no-member


def vectorized(
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
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
    return _ffi_api.Vectorized(start, stop, annotations)  # type: ignore[attr-defined] # pylint: disable=no-member


def unroll(
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
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
    return _ffi_api.Unroll(start, stop, annotations)  # type: ignore[attr-defined] # pylint: disable=no-member


def thread_binding(
    start: PrimExpr,
    stop: PrimExpr = None,
    thread: str = None,
    *,
    annotations: Dict[str, Any] = None,
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


def grid(*extents: PrimExpr) -> frame.ForFrame:
    """The grid For statement.

    Parameters
    ----------
    extents : PrimExpr
        The extents of the iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ffi_api.Grid(extents)  # type: ignore[attr-defined] # pylint: disable=no-member


def Assert(condition: PrimExpr, message: str) -> frame.AssertFrame:  # pylint: disable=invalid-name
    """Create an assertion statement.

    Parameters
    ----------
    condition : PrimExpr
        The PrimExpr to test.

    message : str
        The output error message when the assertion fails.

    Returns
    -------
    res : frame.AssertFrame
        The result AssertFrame.
    """
    if isinstance(condition, bool):
        condition = IntImm("bool", condition)
    return _ffi_api.Assert(condition, message)  # type: ignore[attr-defined] # pylint: disable=no-member


def LetStmt(  # pylint: disable=invalid-name
    value: PrimExpr,
    type_annotation: Optional[Type] = None,  # pylint: disable=redefined-outer-name
    *,
    var: Optional[Var] = None,  # pylint: disable=redefined-outer-name
) -> frame.LetFrame:
    """Create a LetStmt binding

    Parameters
    ----------
    value : PrimExpr
        The value to be bound.
    type_annotation : Optional[Type] = None
        The type annotation of the let binding. Usually it is used for fine-grained var typing,
        particularly, PointerType.
    var : Optional[Var] = None
        The variable to bind. If not specified, a new variable will be created.

    Returns
    -------
    let_frame : frame.LetFrame
        The result LetFrame.
    """
    if type_annotation is not None:
        if callable(type_annotation):
            type_annotation = type_annotation()
        if isinstance(type_annotation, Var):
            type_annotation = type_annotation.type_annotation
    return _ffi_api.LetStmt(value, type_annotation, var)  # type: ignore[attr-defined] # pylint: disable=no-member


def Let(  # pylint: disable=invalid-name
    expr: PrimExpr,
    where: Dict[Var, PrimExpr],  # pylint: disable=redefined-outer-name
) -> PrimExpr:
    """Create a Let expression binding"""
    assert len(where) == 1, "T.Let only allows `where` to have exactly one element"
    var, value = list(where.items())[0]  # pylint: disable=redefined-outer-name
    return tir.Let(var, value, expr)


def let(
    v: Var,
    value: PrimExpr,
    body: PrimExpr = None,
) -> frame.LetFrame:
    """Create a new let binding.

    Parameters
    ----------
    v : Var
        The variable to bind.

    value : PrimExpr
        The value to be bound.

    body : PrimExpr
        The body expression, None will be used if it was not specified.

    Returns
    -------
    res : frame.LetFrame
        The result LetFrame.
    """

    @deprecated("T.let", "T.Let")
    def let_expr(v: Var, value: PrimExpr, body: PrimExpr) -> PrimExpr:
        return tir.Let(v, value, body)

    @deprecated("T.let", "T.LetStmt")
    def let_stmt(v: Var, value: PrimExpr) -> frame.LetFrame:
        return _ffi_api.LegacyLetStmt(v, value)  # type: ignore[attr-defined] # pylint: disable=no-member

    if body is None:
        return let_stmt(v, value)
    else:
        return let_expr(v, value, body)


def realize(
    buffer_slice: BufferRegion,
    storage_scope: str,
    condition: PrimExpr = True,
) -> frame.RealizeFrame:
    """Create a realization.

    Parameters
    ----------
    buffer_slice : BufferRegion
        The region of buffer access.

    storage_scope : str
        The storage scope associated with this realization.

    condition: PrimExpr
        The condition expression, the default is True.

    Returns
    -------
    res : frame.RealizeFrame
        The result RealizeFrame.
    """
    return _ffi_api.Realize(  # type: ignore[attr-defined] # pylint: disable=no-member
        buffer_slice, storage_scope, condition
    )


def allocate(
    extents: List[PrimExpr],
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


def allocate_const(
    data: List[PrimExpr],
    dtype: str,
    extents: List[PrimExpr],
    annotations=None,
) -> frame.AllocateConstFrame:
    """Allocate constant node.

    Parameters
    ----------
    data : List[PrimExpr]
        The data associated with the constant.

    dtype : str
        The data type of the buffer.

    extents : List[PrimExpr]
        The extents of the allocate.

    annotations : Optional[Map]
        Additional annotations about the allocation.
    """
    np_data = np.asarray(data, dtype=dtype)
    prod_extent = 1
    for extent in extents:
        prod_extent *= extent
    prod_shape = 1
    for shape in np_data.shape:
        prod_shape *= shape
    if prod_extent == prod_shape:
        np_data = np_data.reshape(extents)

    return _ffi_api.AllocateConst(  # type: ignore[attr-defined] # pylint: disable=no-member
        ndarray.array(np_data), dtype, extents, annotations
    )


def attr(node: Any, attr_key: str, value: Union[PrimExpr, str]) -> frame.AttrFrame:
    """Create an attribute node.

    Parameters
    ----------
    node : Any
        The node to annotate the attribute.

    attr_key : str
        Attribute type key.

    value : Union[PrimExpr, str]
        The value of the attribute.

    Returns
    -------
    res : frame.AttrFrame
        The result AttrFrame.
    """
    node = convert(node)
    value = convert(value)
    return _ffi_api.Attr(node, attr_key, value)  # type: ignore[attr-defined] # pylint: disable=no-member


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
    scope="global",
    align=0,
    offset_factor=0,
    buffer_type="",
    axis_separators=None,
) -> frame.DeclBufferFrame:
    """Create a buffer declaration node.

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

    Returns
    -------
    res : frame.DeclBufferFrame
        The result DeclBufferFrame.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is not None:
        strides = [Var(s, "int32") if isinstance(s, str) else s for s in strides]
    else:
        strides = []
    return _ffi_api.DeclBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        dtype,
        "",
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def launch_thread(
    thread: Union[IterVar, str],  # pylint: disable=redefined-outer-name
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

    from tvm.script.ir_builder import tir as T
    brow = T.env_thread("blockIdx.y")
    T.launch_thread(brow, 1)

    """

    if isinstance(thread, str):
        thread = String(thread)
    return _ffi_api.LaunchThread(thread, extent)  # type: ignore[attr-defined] # pylint: disable=no-member


def env_thread(thread_tag: str) -> IterVar:
    """Bind a var to thread env

    Parameters
    ----------
    thread_tag : str
        The thread type tag.

    Returns
    -------
    res : IterVar
        The result iteration variable gets bound to the thread env.

    """
    return _ffi_api.EnvThread(thread_tag)  # type: ignore[attr-defined] # pylint: disable=no-member


def buffer_store(
    buffer: Buffer,  # pylint: disable=redefined-outer-name
    value: PrimExpr,
    indices: List[Union[PrimExpr, slice]],
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
    """
    from tvm.arith import Analyzer  # pylint: disable=import-outside-toplevel

    if not isinstance(indices, (list, tuple, ir.Array)):
        indices = [indices]

    expr_indices = []
    for index in indices:
        if isinstance(index, slice):
            step = 1 if index.step is None else index.step
            lanes = Analyzer().simplify((index.stop - index.start + step - 1) // step)
            if lanes == 1:
                expr_indices.append(index.start)
            else:
                expr_indices.append(ramp(index.start, step, int(lanes)))
        else:
            expr_indices.append(index)
    if isinstance(value, bool) and buffer.dtype == "bool":
        value = IntImm("bool", value)
    return _ffi_api.BufferStore(  # type: ignore[attr-defined] # pylint: disable=no-member
        buffer, value, expr_indices
    )


def prefetch(
    buffer: Buffer,  # pylint: disable=redefined-outer-name
    bounds: List[ir.Range],
) -> None:
    """The prefetch hint for a buffer.

    Parameters
    ----------
    buffer : Buffer
        The buffer to be prefetched.
    bounds : List[Range]
        The range to be prefetched.
    """
    return _ffi_api.Prefetch(buffer, bounds)  # type: ignore[attr-defined] # pylint: disable=no-member


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


def func_gen(name: str):
    """Generate a function for each PrimExpr dtype.

    Parameters
    ----------
    name: str
        The ffi function name to call.
    """

    def func(
        expr: Union[
            None,
            PrimExpr,
            Literal["inf", "-inf", "nan"],
            int,
            float,
        ] = None,
        *,
        is_size_var: bool = False,
    ) -> PrimExpr:
        if isinstance(expr, str):
            expr = float(expr)
        return getattr(_ffi_api, name)(expr, is_size_var)

    return func


# pylint: disable=invalid-name
int8 = func_gen(("Int8"))
int16 = func_gen(("Int16"))
int32 = func_gen(("Int32"))
int64 = func_gen(("Int64"))
int8x4 = func_gen(("Int8x4"))
int16x4 = func_gen(("Int16x4"))
int32x4 = func_gen(("Int32x4"))
int64x4 = func_gen(("Int64x4"))
int8x8 = func_gen(("Int8x8"))
int16x8 = func_gen(("Int16x8"))
int32x8 = func_gen(("Int32x8"))
int64x8 = func_gen(("Int64x8"))
int8x16 = func_gen(("Int8x16"))
int16x16 = func_gen(("Int16x16"))
int32x16 = func_gen(("Int32x16"))
int64x16 = func_gen(("Int64x16"))
int8x32 = func_gen(("Int8x32"))
int16x32 = func_gen(("Int16x32"))
int32x32 = func_gen(("Int32x32"))
int64x32 = func_gen(("Int64x32"))
int8x64 = func_gen(("Int8x64"))
int16x64 = func_gen(("Int16x64"))
int32x64 = func_gen(("Int32x64"))
int64x64 = func_gen(("Int64x64"))

uint8 = func_gen(("UInt8"))
uint16 = func_gen(("UInt16"))
uint32 = func_gen(("UInt32"))
uint64 = func_gen(("UInt64"))
uint8x4 = func_gen(("UInt8x4"))
uint16x4 = func_gen(("UInt16x4"))
uint32x4 = func_gen(("UInt32x4"))
uint64x4 = func_gen(("UInt64x4"))
uint8x8 = func_gen(("UInt8x8"))
uint16x8 = func_gen(("UInt16x8"))
uint32x8 = func_gen(("UInt32x8"))
uint64x8 = func_gen(("UInt64x8"))
uint8x16 = func_gen(("UInt8x16"))
uint16x16 = func_gen(("UInt16x16"))
uint32x16 = func_gen(("UInt32x16"))
uint64x16 = func_gen(("UInt64x16"))
uint8x32 = func_gen(("UInt8x32"))
uint16x32 = func_gen(("UInt16x32"))
uint32x32 = func_gen(("UInt32x32"))
uint64x32 = func_gen(("UInt64x32"))
uint8x64 = func_gen(("UInt8x64"))
uint16x64 = func_gen(("UInt16x64"))
uint32x64 = func_gen(("UInt32x64"))
uint64x64 = func_gen(("UInt64x64"))

float8 = func_gen(("Float8"))
float16 = func_gen(("Float16"))
float32 = func_gen(("Float32"))
float64 = func_gen(("Float64"))
float8x4 = func_gen(("Float8x4"))
float16x4 = func_gen(("Float16x4"))
float32x4 = func_gen(("Float32x4"))
float64x4 = func_gen(("Float64x4"))
float8x8 = func_gen(("Float8x8"))
float16x8 = func_gen(("Float16x8"))
float32x8 = func_gen(("Float32x8"))
float64x8 = func_gen(("Float64x8"))
float8x16 = func_gen(("Float8x16"))
float16x16 = func_gen(("Float16x16"))
float32x16 = func_gen(("Float32x16"))
float64x16 = func_gen(("Float64x16"))
float8x32 = func_gen(("Float8x32"))
float16x32 = func_gen(("Float16x32"))
float32x32 = func_gen(("Float32x32"))
float64x32 = func_gen(("Float64x32"))
float8x64 = func_gen(("Float8x64"))
float16x64 = func_gen(("Float16x64"))
float32x64 = func_gen(("Float32x64"))
float64x64 = func_gen(("Float64x64"))
# pylint: enable=invalid-name


def boolean(expr: Optional[PrimExpr] = None, is_size_var: bool = False) -> PrimExpr:
    """Construct a new tir.Var with type boolean or cast expression to type boolean.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    is_size_var: bool
        Whether or not to return a SizeVar instead of Var.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type boolean or casted expression with type boolean.
    """
    return _ffi_api.Boolean(expr, is_size_var)  # type: ignore[attr-defined] # pylint: disable=no-member


def handle(
    dtype: Optional[str] = None, storage_scope: str = "global", *, is_size_var: bool = False
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
        The new tir.Var with type handle or casted expression with type handle.
    """
    is_unknown_type = dtype is None
    if dtype is None:
        dtype = "void"
    return _ffi_api.Handle(  # type: ignore[attr-defined] # pylint: disable=no-member
        dtype,
        storage_scope,
        is_size_var,
        is_unknown_type,
    )


def void(expr: Optional[PrimExpr] = None, *, is_size_var: bool = False) -> PrimExpr:
    """Construct a new tir.Var with type void or cast expression to type void.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type void or casted expression with type void.
    """
    return _ffi_api.Void(expr, is_size_var)  # type: ignore[attr-defined] # pylint: disable=no-member


@deprecated("T.var", "T.{dtype}")
def var(dtype: str, name: str = "") -> Var:
    """Construct a new tir.Var.

    Parameters
    ----------
    dtype: str
        The dtype of the Var.

    name: str
        The name of the Var.

    Returns
    -------
    res : Var
        The result tir.Var.
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


def iter_var(v: Union[Var, str], dom: ir.Range, iter_type: str, thread_tag: str) -> IterVar:
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


def comm_reducer(combiner: Callable, identity: List[PrimExpr]) -> CommReducer:
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
    inverse_index_map: Optional[Callable] = None,
) -> IndexMap:
    """Create a TIR Index mapping"""
    return IndexMap.from_func(mapping, inverse_index_map=inverse_index_map)


def target(
    target_config: Union[Dict, str],
    host: Optional[Union[Dict, str, Target]] = None,
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
    if not isinstance(target_config, (str, dict)):
        raise ValueError(
            f"T.target expected a config dict or string, but got {type(target_config)}"
        )
    if host is not None and not isinstance(host, (str, dict, Target)):
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


class meta_var:  # pylint: disable=invalid-name
    """A meta variable used in TVMScript metaprogramming. It means that the value of the variable
    does not appear in the final TIR, but only stays in the parser.

    Parameters
    ----------
    value: Any
        The meta variable.
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def __iter__(self):
        def f():
            for i in self.value:
                yield meta_var(i)

        return f()


# pylint: disable=invalid-name


def _op_wrapper(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        return func(*args, **kwargs)

    return wrapped


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
floor = _op_wrapper(_tir_op.floor)
ceildiv = _op_wrapper(_tir_op.ceildiv)
floordiv = _op_wrapper(_tir_op.floordiv)
floormod = _op_wrapper(_tir_op.floormod)
fmod = _op_wrapper(_tir_op.fmod)
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
trunc = _op_wrapper(_tir_op.trunc)
truncdiv = _op_wrapper(_tir_op.truncdiv)
truncmod = _op_wrapper(_tir_op.truncmod)
tvm_access_ptr = _op_wrapper(_tir_op.tvm_access_ptr)
tvm_throw_last_error = _op_wrapper(_tir_op.tvm_throw_last_error)
tvm_stack_alloca = _op_wrapper(_tir_op.tvm_stack_alloca)
tvm_stack_make_shape = _op_wrapper(_tir_op.tvm_stack_make_shape)
tvm_stack_make_array = _op_wrapper(_tir_op.tvm_stack_make_array)
tvm_check_return = _op_wrapper(_tir_op.tvm_check_return)
call_packed = _op_wrapper(_tir_op.call_packed)
call_cpacked = _op_wrapper(_tir_op.call_cpacked)
call_packed_lowered = _op_wrapper(_tir_op.call_packed_lowered)
call_cpacked_lowered = _op_wrapper(_tir_op.call_cpacked_lowered)
tvm_tuple = _op_wrapper(_tir_op.tvm_tuple)
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
tvm_warp_shuffle = _tir_op.tvm_warp_shuffle
tvm_warp_shuffle_up = _tir_op.tvm_warp_shuffle_up
tvm_warp_shuffle_down = _tir_op.tvm_warp_shuffle_down
tvm_warp_activemask = _tir_op.tvm_warp_activemask
ptx_wait_group = _op_wrapper(_tir_op.ptx_wait_group)
ptx_commit_group = _op_wrapper(_tir_op.ptx_commit_group)
ptx_cp_async_barrier = _op_wrapper(_tir_op.ptx_cp_async_barrier)
ptx_init_barrier_thread_count = _op_wrapper(_tir_op.ptx_init_barrier_thread_count)
ptx_arrive_barrier = _op_wrapper(_tir_op.ptx_arrive_barrier)
ptx_arrive_barrier_expect_tx = _op_wrapper(_tir_op.ptx_arrive_barrier_expect_tx)
ptx_wait_barrier = _op_wrapper(_tir_op.ptx_wait_barrier)
create_barriers = _op_wrapper(_tir_op.create_barriers)
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


def _dtype_forward(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "dtype" in kwargs:
            args = (kwargs.pop("dtype"),) + args
        return func(*args, **kwargs)

    return wrapped


reinterpret = _dtype_forward(_tir_op.reinterpret)
call_extern = _dtype_forward(_tir_op.call_extern)
call_intrin = _dtype_forward(_tir_op.call_intrin)
call_llvm_intrin = _dtype_forward(_tir_op.call_llvm_intrin)
call_llvm_pure_intrin = _dtype_forward(_tir_op.call_llvm_pure_intrin)
call_pure_extern = _dtype_forward(_tir_op.call_pure_extern)
ptx_mma = _dtype_forward(_tir_op.ptx_mma)
ptx_mma_sp = _dtype_forward(_tir_op.ptx_mma_sp)
ptx_ldmatrix = _dtype_forward(_tir_op.ptx_ldmatrix)
ptx_cp_async = _dtype_forward(_tir_op.ptx_cp_async)
ptx_cp_async_bulk = _dtype_forward(_tir_op.ptx_cp_async_bulk)
mma_store = _dtype_forward(_tir_op.mma_store)
mma_fill = _dtype_forward(_tir_op.mma_fill)
vectorlow = _dtype_forward(_tir_op.vectorlow)
vectorhigh = _dtype_forward(_tir_op.vectorhigh)
vectorcombine = _dtype_forward(_tir_op.vectorcombine)


broadcast = Broadcast
ramp = Ramp
fabs = abs
tvm_call_packed = call_packed
tvm_call_cpacked = call_cpacked
tvm_call_packed_lowered = call_packed_lowered
tvm_call_cpacked_lowered = call_cpacked_lowered


# pylint: enable=invalid-name


__all__ = [
    "int8",
    "int16",
    "int32",
    "int64",
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
    "float8",
    "float16",
    "float32",
    "float64",
    "float8x4",
    "float16x4",
    "float32x4",
    "float64x4",
    "float8x8",
    "float16x8",
    "float32x8",
    "float64x8",
    "float8x16",
    "float16x16",
    "float32x16",
    "float64x16",
    "float8x32",
    "float16x32",
    "float32x32",
    "float64x32",
    "float8x64",
    "float16x64",
    "float32x64",
    "float64x64",
    "buffer",
    "buffer_decl",
    "prim_func",
    "arg",
    "func_name",
    "func_attr",
    "func_ret",
    "match_buffer",
    "block",
    "init",
    "where",
    "reads",
    "writes",
    "block_attr",
    "alloc_buffer",
    "axis",
    "serial",
    "parallel",
    "vectorized",
    "unroll",
    "thread_binding",
    "grid",
    "Assert",
    "realize",
    "allocate",
    "allocate_const",
    "attr",
    "While",
    "If",
    "Then",
    "Else",
    "decl_buffer",
    "launch_thread",
    "env_thread",
    "buffer_store",
    "prefetch",
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
    "trunc",
    "truncdiv",
    "truncmod",
    "tvm_access_ptr",
    "tvm_throw_last_error",
    "tvm_stack_alloca",
    "tvm_stack_make_shape",
    "tvm_stack_make_array",
    "tvm_check_return",
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
    "tvm_warp_shuffle",
    "tvm_warp_shuffle_up",
    "tvm_warp_shuffle_down",
    "tvm_warp_activemask",
    "ptx_mma",
    "ptx_mma_sp",
    "ptx_ldmatrix",
    "ptx_cp_async",
    "ptx_cp_async_bulk",
    "ptx_wait_group",
    "ptx_commit_group",
    "ptx_cp_async_barrier",
    "ptx_init_barrier_thread_count",
    "ptx_arrive_barrier",
    "ptx_arrive_barrier_expect_tx",
    "ptx_wait_barrier",
    "create_barriers",
    "mma_store",
    "mma_fill",
    "vectorlow",
    "vectorhigh",
    "vectorcombine",
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
    # tvm.tir.expr
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
    "LetStmt",
    "Let",
    "IterVar",
    "CommReducer",
    "Range",
    "vscale",
]
