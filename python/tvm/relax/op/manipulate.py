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
"""Manipulation operators."""
from typing import List, Optional, Tuple, Union, Callable

from tvm.ir.expr import PrimExpr
from tvm.tir import IntImm, FloatImm, IndexMap

from . import _ffi_api
from ..expr import Expr, PrimValue, ShapeExpr, Tuple as RxTuple


PrimExprLike = Union[int, PrimExpr]


def broadcast_to(x: Expr, shape: Union[Tuple[PrimExprLike], Expr]) -> Expr:
    """Broadcasts a tensor to a specified shape.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    shape : Union[Tuple[PrimExprLike], Expr]
        The target shape.

    Returns
    -------
    result : relax.Expr
        The broadcasted tensor.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.broadcast_to(x, shape)  # type: ignore


def concat(tensors: Union[Expr, List[Expr]], axis: Optional[int] = 0) -> Expr:
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    tensors : Union[relax.Expr, List[relax.Expr]]
        An Expr in Tuple type, containing the tensors to be concatenated,
        or a list of Tensors.

    axis : Optional[int]
        The axis along which the tensors are concatenated.
        If `axis` is `None`, the input tensor is required to be flattened before concatenation.

    Returns
    -------
    result: relax.Expr
        The concatenated tensor.
    """
    if isinstance(tensors, (list, tuple)):
        tensors = RxTuple(tensors)
    return _ffi_api.concat(tensors, axis)  # type: ignore


def expand_dims(x: Expr, axis: Union[int, List[int]]) -> Expr:
    """Insert new axes at the positions given by `axis`.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axis : Union[int, List[int]]
        The axes at which the input array are expanded.
        All values are required to lie in range `[-data.ndim - 1, data.ndim]`, with the convention
        of negative indexing.

    Returns
    -------
    result : relax.Expr
        The transformed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.expand_dims(x, axis)  # type: ignore


def flatten(x: Expr) -> Expr:
    """Flatten all the tensor dimensions into one.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    Returns
    -------
    result : relax.Expr
        The flattened result.
    """
    return _ffi_api.flatten(x)  # type: ignore


def layout_transform(
    x: Expr,
    index_map: Union[Callable, IndexMap],
    pad_value: Optional[Union[int, float, PrimValue]] = None,
):
    """Modifies the layout of a tensor.

    Parameters
    ----------
    x : relax.Expr
        The input tensor to the operator.

    index_map : Union[Callable, IndexMap]
        The transformation to apply.

    pad_value : Optional[Union[int, float, PrimValue]]
        The value used for padding if the transformation results in implicit padding.
        If not specified, any value can be used.

    Returns
    -------
    result : relax.Expr
        The transformed tensor.
    """
    if callable(index_map):
        index_map = IndexMap.from_func(index_map)
    x_dtype = x.checked_type.dtype

    # Explicitly convert python int/float pad_value to the x's type.  If the default behavior
    # is applied, it would be converted to int32/float32, which may not match the x's type.
    if pad_value is None:
        pass
    elif not isinstance(pad_value, PrimValue):
        if "int" in x_dtype and isinstance(pad_value, int):
            pad_value = IntImm(x_dtype, pad_value)
        elif "float" in x_dtype and (isinstance(pad_value, (int, float))):
            pad_value = FloatImm(x_dtype, float(pad_value))
        pad_value = PrimValue(pad_value)
    return _ffi_api.layout_transform(x, index_map, pad_value)  # type: ignore


def permute_dims(x: Expr, axes: Optional[List[int]] = None) -> Expr:
    """Permutes the dimensions of an array.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axes : Optional[List[int]]
        The target axes order, reverse order if not specified.

    Returns
    -------
    result : relax.Expr
        The transposed result.
    """
    return _ffi_api.permute_dims(x, axes)  # type: ignore


def reshape(x: Expr, shape: Union[Tuple[PrimExprLike], Expr]) -> Expr:
    """Reshape the input array.

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            x.shape = (2, 3, 4), shape = (6, 1, -1), result.shape = (6, 1, 4)
            x.shape = (2, 3, 4), shape = (3, -1, 8), result.shape = (3, 1, 8)
            x.shape = (2, 3, 4), shape = (-1,), result.shape = (24,)

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    shape : Union[Tuple[PrimExprLike], Expr]
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    result : relax.Expr
        The reshaped result.

    Note
    ----
    The ``-1`` inference is only performed at compile-time.
    That is to say, in any case the dimension length of ``-1`` cannot be inferred in
    compile-time, an error will be thrown.
    """
    return _ffi_api.reshape(x, shape)  # type: ignore


def split(
    x: Expr,
    indices_or_sections: Union[int, List[PrimExprLike]],
    axis: int = 0,
) -> Expr:
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis (if possible). Last section will be smaller if the tensor
    size along the given dimension is not divisible by the integer.

    If indices_or_sections is a tuple of mixture of int or PrimExpr,
    the entries indicate the indices where along axis the array is split.

    Parameters
    ----------
    x : relax.Expr
        The tensor to be split.

    indices_or_sections : Union[int, List[PrimExprLike]]
        Indices or sections to split into. Accepts an int or a list.

    axis : int
        The axis over which to split.

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """
    if isinstance(indices_or_sections, int):
        indices_or_sections = IntImm("int64", indices_or_sections)
    return _ffi_api.split(x, indices_or_sections, axis)  # type: ignore


def squeeze(x: Expr, axis: Optional[Union[int, List[int]]] = None) -> Expr:
    """Squeeze axes in the array.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axis : Optional[Union[int, List[int]]
        The set of axes to remove.
        If axis = None, remove all axis of dimensions 1.
        If any specified axis has dimension that does not equal 1, it is an error.

    Returns
    -------
    result : relax.Expr
        The squeezed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _ffi_api.squeeze(x, axis)  # type: ignore
