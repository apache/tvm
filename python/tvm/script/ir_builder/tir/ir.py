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
# pylint: disable=missing-docstring
"""IRBuilder for TIR"""

from numbers import Integral
from typing import Any, Dict, List, Optional, Union, Tuple

from tvm.ir import Range, Type
from tvm.tir import (
    Buffer,
    BufferLoad,
    BufferRegion,
    IntImm,
    PrimExpr,
    StringImm,
    Var,
)

from . import _ffi_api, frame


def buffer_decl(
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral],
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "",
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
    return _ffi_api.BufferDecl(  # type: ignore[attr-defined] # pylint: disable=no-member
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


def prim_func() -> frame.PrimFuncFrame:
    """The primitive function statement.

    Returns
    -------
    res : frame.PrimFuncFrame
        The PrimFuncFrame.
    """
    return _ffi_api.PrimFunc()  # type: ignore[attr-defined] # pylint: disable=no-member


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
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
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


def preflattened_buffer(
    postflattened: Buffer,
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
) -> None:
    """The pre-flattened buffer statement.

    Parameters
    ----------
    postflattened : Buffer
        The original buffer to be flattened.

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
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    _ffi_api.PreflattenedBuffer(  # type: ignore[attr-defined] # pylint: disable=no-member
        postflattened,
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
    scope: str = "",
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
    if strides is None:
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


def _as_range(dom: Union[Range, List[PrimExpr]]) -> Range:
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
    if isinstance(dom, Range):
        return dom
    if isinstance(dom, (list, tuple)):
        return Range(dom[0], dom[1])
    return Range(0, dom)


class axis:  # pylint: disable=invalid-name
    @staticmethod
    def spatial(
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
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
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
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
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
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
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
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
        start = 0
    elif stop is None:
        stop = start
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


def evaluate(value: PrimExpr) -> None:
    """Evaluate the input expression.

    Parameters
    ----------
    value: PrimExpr
        The input expression to evaluate.
    """
    if isinstance(value, str):
        value = StringImm(value)
    return _ffi_api.Evaluate(value)  # type: ignore[attr-defined] # pylint: disable=no-member


def int8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type int8 or cast expression to type int8.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type int8 or casted expression with type int8.
    """
    return _ffi_api.Int8(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def int16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type int16 or cast expression to type int16.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type int16 or casted expression with type int16.
    """
    return _ffi_api.Int16(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def int32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type int32 or cast expression to type int32.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type int32 or casted expression with type int32.
    """
    return _ffi_api.Int32(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def int64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type int64 or cast expression to type int64.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type int64 or casted expression with type int64.
    """
    return _ffi_api.Int64(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def uint8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type uint8 or cast expression to type uint8.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type uint8 or casted expression with type uint8.
    """
    return _ffi_api.UInt8(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def uint16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type uint16 or cast expression to type uint16.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type uint16 or casted expression with type uint16.
    """
    return _ffi_api.UInt16(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def uint32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type uint32 or cast expression to type uint32.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type uint32 or casted expression with type uint32.
    """
    return _ffi_api.UInt32(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def uint64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type uint64 or cast expression to type uint64.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type uint64 or casted expression with type uint64.
    """
    return _ffi_api.UInt64(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def float8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type float8 or cast expression to type float8.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type float8 or casted expression with type float8.
    """
    return _ffi_api.Float8(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def float16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type float16 or cast expression to type float16.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type float16 or casted expression with type float16.
    """
    return _ffi_api.Float16(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def float32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type float32 or cast expression to type float32.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type float32 or casted expression with type float32.
    """
    return _ffi_api.Float32(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def float64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type float64 or cast expression to type float64.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type float64 or casted expression with type float64.
    """
    return _ffi_api.Float64(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def int32x4(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type int32x4 or cast expression to type int32x4.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type int32x4 or casted expression with type int32x4.
    """
    return _ffi_api.Int32x4(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def int32x8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type int32x8 or cast expression to type int32x8.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type int32x8 or casted expression with type int32x8.
    """
    return _ffi_api.Int32x8(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def int32x16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type int32x16 or cast expression to type int32x16.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type int32x16 or casted expression with type int32x16.
    """
    return _ffi_api.Int32x16(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def boolean(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type boolean or cast expression to type boolean.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type boolean or casted expression with type boolean.
    """
    return _ffi_api.Boolean(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def handle(expr: Optional[PrimExpr] = None) -> PrimExpr:
    """Construct a new tir.Var with type handle or cast expression to type handle.

    Parameters
    ----------
    expr: PrimExpr
        The expression to be cast.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type handle or casted expression with type handle.
    """
    return _ffi_api.Handle(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def void(expr: Optional[PrimExpr] = None) -> PrimExpr:
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
    return _ffi_api.Void(expr)  # type: ignore[attr-defined] # pylint: disable=no-member


def var(dtype, name="") -> Var:
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


# pylint: enable=invalid-name


__all__ = [
    "buffer_decl",
    "prim_func",
    "arg",
    "func_name",
    "func_attr",
    "func_ret",
    "match_buffer",
    "preflattened_buffer",
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
    "evaluate",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float8",
    "float16",
    "float32",
    "float64",
    "int32x4",
    "int32x8",
    "int32x16",
    "boolean",
    "handle",
    "void",
    "var",
]
