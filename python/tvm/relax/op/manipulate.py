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
    axis_separators: Optional[Union[int, IndexMap.AXIS_SEPARATOR]] = None,
    input_axis_separators: Optional[Union[int, IndexMap.AXIS_SEPARATOR]] = None,
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

    axis_separators : Optional[Union[int, IndexMap.AXIS_SEPARATOR]]
        The axis_separators for index_map to create non flat buffers.

    Returns
    -------
    result : relax.Expr
        The transformed tensor.
    """
    default_index_dtype = "int64"

    if callable(index_map):
        index_map = IndexMap.from_func(index_map, index_dtype=default_index_dtype)
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

    if axis_separators is None:
        axis_separators = []

    if input_axis_separators is None:
        input_axis_separators = []

    return _ffi_api.layout_transform(
        x, index_map, pad_value, axis_separators, input_axis_separators
    )


def permute_dims(x: Expr, axes: Optional[List[int]] = None) -> Expr:
    """Permutes the dimensions of an array.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axes : Optional[List[int]]
        The target axes order. If not specified, permute_dims will reverse the order of all axes.

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


def collapse_sum_like(data: Expr, collapse_target: Expr) -> Expr:
    """Return a summation of data to the shape of collapse_target.

    For details, please see relax.op.collapse_sum_to.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    collapse_target : relax.Expr
        The tensor whose shape is the shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor after summation.
    """
    return _ffi_api.collapse_sum_like(data, collapse_target)  # type: ignore


def collapse_sum_to(data: Expr, shape: Union[Tuple[PrimExprLike], Expr]) -> Expr:
    """Return a summation of data to the given shape.

    collapse_sum_to is intended as the backward operator of tvm.relax.op.broadcast_to and
    other broadcast operators in the automatic differentiation process.

    We expect that data is the result of broadcasting some tensor of the given shape in some
    broadcast operation. Thus the given `shape` and `data.shape` must follow broadcast rules.

    During computation, all axes of `data.shape` and `shape` are checked from right to left.
    For an axis, if it follows these rules, `data` will be summed over this axis:
    - the axis exists in `data.shape` but not in `shape`, or
    - the axis exists in `data.shape` and equals to 1 in `shape`.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    shape : Union[Tuple[PrimExprLike], relax.Expr]
        The shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor of the given shape after summation.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.collapse_sum_to(data, shape)  # type: ignore


def repeat(data: Expr, repeats: int, axis: Optional[int] = None) -> Expr:
    """Repeats elements of an array.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    repeats : int
        The number of repetitions.

    axis: Optional[int]
        The axis along which to repeat values. The negative numbers are interpreted
        counting from the backward. By default, use the flattened input array, and
        return a flat output array.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = R.const([[1, 2], [3, 4]])
        lv1 = R.repeat(x, repeats=2) # lv1 == [1, 1, 2, 2, 3, 3, 4, 4]
        lv2 = R.repeat(x, repeats=2, axis=1) # lv2 == [[1., 1., 2., 2.],
                                             #         [3., 3., 4., 4.]]
    """
    return _ffi_api.repeat(data, repeats, axis)  # type: ignore


def tile(data: Expr, repeats: Union[int, Tuple[int], List[int]]) -> Expr:
    """Construct an array by repeating data the number of times given by repeats.

    If repeats has length l, and data has dimension d, the result will have dimension of max(l, d).

    If d < l, data is promoted to be l-dimensional by prepending new axes. So a shape (3,) Tensor is
    promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not
    the desired behavior, promote data to d-dimensions manually before calling this function.

    If d > l, reps is promoted to length d by pre-pending 1's to it. Thus for a data of shape
    (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    repeats : Union[int, Tuple[int], List[int]]
        The number of repetitions of data along each axis.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = R.const([[1, 2], [3, 4]])
        lv1 = R.tile(x, reps=(2, 3)) # lv1 = [[1., 2., 1., 2., 1., 2.],
                                     #        [3., 4., 3., 4., 3., 4.],
                                     #        [1., 2., 1., 2., 1., 2.],
                                     #        [3., 4., 3., 4., 3., 4.]]
        lv2 = R.tile(x, reps=2) # lv2 = [[1., 2., 1., 2.],
                                #        [3., 4., 3., 4.]]
    """
    if isinstance(repeats, int):
        repeats = [repeats]
    return _ffi_api.tile(data, repeats)  # type: ignore


def flip(data, axis):
    """Reverses the order of elements along given axis while preserving array shape.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis: int
        axis to flip on

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [[1., 2.], [3., 4.]]
        relax.flip(x, axis=0) = [[3., 4.], [1., 2.]]

        relax.flip(x, axis=1) = [[2., 1.], [4., 3.]]
    """
    return _ffi_api.flip(data, axis)  # type: ignore


def gather_elements(data: Expr, indices: Expr, axis: int = 0) -> Expr:
    """Gather elements from data according to indices along the specified axis.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    indices : relax.Expr
        The indices tensor, must have integer type.

    axis : int
        The axis along which to index. Default is 0.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        data = [[1, 2], [3, 4]]
        indices = [[0, 0], [1, 0]]
        axis = 1
        output = [[1, 1], [4, 3]]

        data = [[1, 2, 3], [4, 5, 6]]
        indices = [[1, 1, 1]]
        axis = 0
        output = [[4, 5, 6]]
    """
    return _ffi_api.gather_elements(data, indices, axis)  # type: ignore


def gather_nd(data: Expr, indices: Expr, batch_dims: int = 0) -> Expr:
    """Update data at positions defined by indices with values in updates.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    indices : relax.Expr
        The indices tensor, must have integer type.

    batch_dims : int
        The number of batch dimensions. Default is 0.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        batch_dims = 0
        data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
        indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
        output  = [0,3]           # output_shape  = [2]

        batch_dims = 1
        data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
        indices = [[1],[0]]                     # indices_shape = [2, 1]
        output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]

    """
    return _ffi_api.gather_nd(data, indices, batch_dims)  # type: ignore


def scatter_elements(
    data: Expr, indices: Expr, updates: Expr, axis: int = 0, reduction: str = "update"
):
    """ONNX style scatter elements. This operation updates its value in `data` to values
    specified by `updates` at specific index positions specified by `indices`.
    For example, in 2D tensor, the update corresponding to the [i][j] entry is performed
    as below:

    .. code-block::

        output[indices[i][j]][j] = updates[i][j] if axis = 0
        output[i][indices[i][j]] = updates[i][j] if axis = 1

    When the `reduction` is set to some reduction function `f`, the update corresponding to
    [i][j] entry is performed as below:

    .. code-block::

        output[indices[i][j]][j] += f(output[indices[i][j]][j], updates[i][j]) if axis = 0
        output[i][indices[i][j]] += f(output[i][indices[i][j]], updates[i][j]) if axis = 1

    Where `f` is update, add, mul, mean, max, min.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    indices: relax.Expr
        The index positions to update in `data`.

    updates: relax.Expr
        Values to replace to.

    axis: int
        Axis to scatter on.

    reduction: str
        Type of reduction to apply: update, add, mul, mean, max, min.
        It is "update" by default.

    Returns
    -------
    result : relax.Expr
        The result has the same size as data, and the same shape as data

    Examples
    --------
    .. code-block:: python

       # inputs
       data = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        indices = [
            [1, 0, 2],
            [0, 2, 1],
        ]
        updates = [
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
        ]
        axis = 0
        reduction = "update"

        # output P
        output = [
            [2.0, 1.1, 0.0]
            [1.0, 0.0, 2.2]
            [0.0, 2.1, 1.2]
        ]

    """
    return _ffi_api.scatter_elements(data, indices, updates, axis, reduction)  # type: ignore


def scatter_nd(data: Expr, indices: Expr, updates: Expr, reduction: str = "update") -> Expr:
    """Scatter updates into an array according to indices.

    Parameters
    ----------
    data: relax.Expr
        The input data to be updated.

    indices: relax.Expr
        The index positions to update in `data`.

    updates: relax.Expr
        Values to replace to.

    reduction: str
        Type of reduction to apply: update, add, mul, max, min.
        It is "update" by default.

    Returns
    -------
    result : relax.Expr
        The result has the same shape as data.

    Examples
    --------
    .. code-block:: python

       # inputs
       data = [1, 2, 3, 4, 5, 6, 7, 8]
       indices = [[4], [3], [1], [7]]
       updates = [9, 10, 11, 12]

       # output
       output = [1, 11, 3, 10, 9, 6, 7, 12]

    """
    return _ffi_api.scatter_nd(data, indices, updates, reduction)  # type: ignore


def one_hot(
    indices: Expr, on_value: PrimValue, off_value: PrimValue, depth: int, axis: int = -1
) -> Expr:
    """Returns a one-hot tensor.

    Parameters
    ----------
    indices : relax.Expr
        The indices to set to `on_value`.

    on_value : relax.PrimValue
        The value to fill at `indices`.

    off_value : relax.PrimValue
        The value to fill at other locations.

    depth : int
        The depth of the one-hot dimension.

    axis : int, optional
        The axis to fill. Default is -1 which adds a new dimension at the end.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        indices = [0, 1, 2]
        depth = 3
        on_value = 1
        off_value = 0

        one_hot(indices, on_value, off_value, depth) =
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    """
    return _ffi_api.one_hot(indices, on_value, off_value, depth, axis)  # type: ignore
