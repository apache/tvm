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

from tvm.ir import Type
from tvm.tir import (
    Buffer,
    BufferLoad,
    BufferRegion,
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
    return _ffi_api.BufferDecl(  # pylint: disable=no-member # type: ignore
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
    return _ffi_api.PrimFunc()  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Arg(name, obj)  # pylint: disable=no-member # type: ignore


def func_name(name: str) -> None:
    """The PrimFunc naming statement.

    Parameters
    ----------
    name : str
        The name of the PrimFunc.
    """
    _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


def func_attr(attrs: Dict[str, Any]) -> None:
    """The PrimFunc annotation statement.

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotations of the PrimFunc.
    """
    _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.FuncRet(ret_type)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.MatchBuffer(  # pylint: disable=no-member # type: ignore
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
    _ffi_api.PreflattenedBuffer(  # pylint: disable=no-member # type: ignore
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
    return _ffi_api.Block(name, no_realize)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Serial(start, stop, annotations)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Parallel(start, stop, annotations)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Vectorized(start, stop, annotations)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Unroll(start, stop, annotations)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.ThreadBinding(  # pylint: disable=no-member # type: ignore
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
    return _ffi_api.Grid(extents)  # pylint: disable=no-member # type: ignore


def evaluate(value: PrimExpr) -> None:
    """Evaluate the input expression.

    Parameters
    ----------
    value: PrimExpr
        The input expression to evaluate.
    """
    if isinstance(value, str):
        value = StringImm(value)
    return _ffi_api.Evaluate(value)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Int8(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Int16(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Int32(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Int64(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.UInt8(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.UInt16(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.UInt32(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.UInt64(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Float8(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Float16(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Float32(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Float64(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Int32x4(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Int32x8(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Int32x16(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Boolean(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Handle(expr)  # pylint: disable=no-member # type: ignore


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
    return _ffi_api.Void(expr)  # pylint: disable=no-member # type: ignore


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
    return Var(name, dtype)  # pylint: disable=no-member # type: ignore


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
