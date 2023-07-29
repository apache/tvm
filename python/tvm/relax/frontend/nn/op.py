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
# pylint: disable=too-many-lines,invalid-name,protected-access
"""nn.Tensor operators."""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from tvm import tir as _tir

from ... import expr as rx
from ... import op as _op
from ...block_builder import BlockBuilder
from ...struct_info import TensorStructInfo, TupleStructInfo
from .core import Tensor

IntExpr = Union[int, _tir.PrimExpr]


def _wrap_nested(expr: rx.Expr, name: str) -> Union[Tensor, Tuple[Tensor]]:
    """Wrap the given relax.Expr, emit it using the current BlockBuilder,
    and automatically handle nested cases if the expr represents a Tuple.

    Parameters
    ----------
    expr : relax.Expr
        The Expr to be wrapped.

    name : str
        Name hint.

    Returns
    -------
    result : Union[Tensor, Tuple[Tensor]]
        The computed result.
    """
    expr = BlockBuilder.current().emit(expr, name)
    if isinstance(expr.struct_info_, TensorStructInfo):
        return Tensor(_expr=expr)
    if isinstance(expr.struct_info_, TupleStructInfo):
        return tuple(
            _wrap_nested(
                rx.TupleGetItem(expr, i),
                name=f"{name}.{i}",
            )
            for i in range(expr.struct_info_.fields)
        )
    raise TypeError(f"Unsupported return type: {expr.struct_info_}")


def add(a: Tensor, b: Tensor, name: str = "add") -> Tensor:
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = add(a, b)
    """
    return _wrap_nested(_op.add(a._expr, b._expr), name)


def multiply(a: Tensor, b: Tensor, name: str = "mul") -> Tensor:
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = multiply(a, b)
    """
    return _wrap_nested(_op.multiply(a._expr, b._expr), name)


def divide(a: Tensor, b: Tensor, name: str = "divide") -> Tensor:
    """Division with numpy-style broadcasting.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = divide(a, b)
    """
    return _wrap_nested(_op.divide(a._expr, b._expr), name)


def matmul(a: Tensor, b: Tensor, out_dtype: Optional[str] = None, name: str = "matmul") -> Tensor:
    """General matrix multiplication of two tensors, with broadcasting on batched dimensions.

    The semantics and output shape deduction rule is specified as
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html.

    Parameters
    ----------
    a : Tensor
        The first input tensor.

    b : Tensor
        The second input tensor.

    out_dtype: Optional[Union[str, DataType]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the the same as input dtype.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = matmul(a, b)
    """
    return _wrap_nested(_op.matmul(a._expr, b._expr, out_dtype=out_dtype), name)


def maximum(x1: Tensor, x2: Tensor, name: str = "maximum"):
    """Element-wise maximum

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.

    x2 : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = maximum(a, b)
    """
    return _wrap_nested(_op.maximum(x1._expr, x2._expr), name)


def minimum(x1: Tensor, x2: Tensor, name: str = "minimum"):
    """Element-wise minimum

    Parameters
    ----------
    x1 : Tensor
        The first input tensor.

    x2 : Tensor
        The second input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Examples
    --------
    .. code:: python

        c = minimum(a, b)
    """
    return _wrap_nested(_op.minimum(x1._expr, x2._expr), name)


def broadcast_to(x: Tensor, shape: Sequence[IntExpr], name: str = "broadcast_to") -> Tensor:
    """Broadcasts a tensor to a specified shape.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    shape : Sequence[IntExpr]
        The target shape.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The broadcasted tensor.
    """
    return _wrap_nested(_op.broadcast_to(x._expr, shape), name)


def permute_dims(x: Tensor, axes: Optional[List[int]] = None, name: str = "permute_dims") -> Tensor:
    """Permutes the dimensions of an array.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    axes : Optional[List[int]]
        The target axes order, reverse order if not specified.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The transposed result.
    """
    return _wrap_nested(_op.permute_dims(x._expr, axes=axes), name)


def reshape(x: Tensor, shape: Sequence[IntExpr], name="reshape") -> Tensor:
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
    x : Tensor
        The input data to the operator.

    shape : Sequence[IntExpr]
        The new shape. Should be compatible with the original shape.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The reshaped result.

    Note
    ----
    The ``-1`` inference is only performed at compile-time.
    That is to say, in any case the dimension length of ``-1`` cannot be inferred in
    compile-time, an error will be thrown.
    """
    return _wrap_nested(_op.reshape(x._expr, shape), name)


def repeat(x: Tensor, repeats: int, axis: Optional[int] = None, name="repeat") -> Tensor:
    """Repeats elements of an array.

    Parameters
    ----------
    data : Tensor
        The input tensor.

    repeats : int
        The number of repetitions.

    axis: Optional[int]
        The axis along which to repeat values. The negative numbers are interpreted
        counting from the backward. By default, use the flattened input array, and
        return a flat output array.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The computed result.

    Examples
    --------
    .. code-block:: python
        np_x = numpy.array([[1, 2], [3, 4]])
        x = Tensor.from_const(np_x)
        lv1 = repeat(x, repeats=2) # lv1 == [1, 1, 2, 2, 3, 3, 4, 4]
        lv2 = repeat(x, repeats=2, axis=1)   # lv2 == [[1., 1., 2., 2.],
                                             #         [3., 3., 4., 4.]]
    """
    return _wrap_nested(_op.repeat(x._expr, repeats, axis), name)


def squeeze(x: Tensor, axis: int = -1, name: str = "squeeze") -> Tensor:
    """Squeeze axes in the array.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    axis : Optional[Union[int, List[int]]
        The set of axes to remove.
        If axis = None, remove all axis of dimensions 1.
        If any specified axis has dimension that does not equal 1, it is an error.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The squeezed result.
    """
    return _wrap_nested(_op.squeeze(x._expr, axis), name)


def take(x: Tensor, indices: Tensor, axis: Optional[int] = None, name="take") -> Tensor:
    """Take elements from a tensor along an axis.
    Its semantic is mostly similar to `numpy.take`
    (https://numpy.org/doc/stable/reference/generated/numpy.take.html),
    which can cover `torch.take` (https://pytorch.org/docs/stable/generated/torch.take.html) and
    `onnx.gather` (https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13).

    Parameters
    ----------
    x : Tensor
        The source tensor.

    indices : Tensor
        The indices of the values to extract.

    axis : Optional[int]
        The axis over which to select values.
        If it is none, the input tensor is required to be one-dimensional.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The taken result.
    """
    return _wrap_nested(_op.take(x._expr, indices._expr, axis), name)


def astype(x: Tensor, dtype: str, name: str = "astype") -> Tensor:
    """Cast input tensor to the given data type.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    dtype: str
        The target data type

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The casted result.
    """
    return _wrap_nested(_op.astype(x._expr, dtype), name)


def silu(x: Tensor, name: str = "silu") -> Tensor:
    r"""Sigmoid Linear Unit function

    .. math::
        \text{SiLU}(x) = x * \text{sigmoid}(x)

    Parameters
    ----------
    data : Tensor
        The input data

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _wrap_nested(_op.nn.silu(x._expr), name)


def softmax(x: Tensor, axis: int = -1, name: str = "softmax") -> Tensor:
    r"""Computes softmax.

    .. math:: \text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Parameters
    ----------
    data: Tensor
        The input data to the operator.

    axis: int
        The axis to sum over when computing softmax.
        If not specified, it is by default the last axis of the input tensor.
        Supports negative indexing.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _wrap_nested(_op.nn.softmax(x._expr, axis), name)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    axes: Union[int, List[int]],
    epsilon: float = 1e-5,
    name: str = "rms_norm",
) -> Tensor:
    r"""
    Root mean square normalization (Biao Zhang and et al., 2019).
    Applies root mean square normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    .. math::

        out = \frac{data}{\sqrt{mean(data, axis)+\epsilon}} * weight

    Parameters
    ----------
    data : Tensor
        Input to which rms_norm will be applied.

    weight : Tensor
        The scale factor.

    axes : Union[int, List[int]]
        The axes that along which the normalization is applied.

    epsilon : float
        Small float added to square mean to avoid dividing by zero.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return _wrap_nested(_op.nn.rms_norm(x._expr, weight._expr, axes, epsilon), name)


def triu(x: Tensor, diagonal: int = 0, name: str = "triu") -> Tensor:
    """Return the upper triangular part of a matrix or a batch of matrices.

    Parameters
    ----------
    x : Tensor
        The tensor that triu will be applied to.
        It is required to have at least two dimensions.

    k : int
        The index indicating the diagonal below which to zero elements.
        If k = 0, the diagonal is the main diagonal.
        If k < 0, the diagonal is below the main diagonal.
        If k > 0, the diagonal is above the main diagonal.

    name : str
        Name hint.

    Returns
    -------
    ret : Tensor
        The result tensor.
    """
    return _wrap_nested(_op.triu(x._expr, diagonal), name)


def full(
    shape: Sequence[IntExpr],
    fill_value: Tensor,
    dtype: str = "float32",
    name: str = "full",
) -> Tensor:
    """Fill array with scalar value.

    Parameters
    ----------
    shape : Sequence[IntExpr]
        The shape of the created tensor.

    fill_value : Tensor
        The value to fill. Must be a scalar tensor.

    dtype : str
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of fill_value.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result tensor.
    """
    from tvm import relax  # pylint: disable=import-outside-toplevel

    if isinstance(fill_value, (_tir.FloatImm, _tir.IntImm)):
        fill_value = relax.const(fill_value.value, dtype=dtype)
    elif isinstance(fill_value, (int, float)):
        fill_value = relax.const(fill_value, dtype=dtype)
    else:
        fill_value = fill_value._expr
    return _wrap_nested(_op.full(shape, fill_value, dtype), name)


def zeros(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "zeros",
) -> Tensor:
    """Construct a tensor of all zeros, with the input shape and dtype.

    Parameters
    ----------
    shape : Sequence[IntExpr]
        The shape of the created tensor.

    dtype : str
        The data type of the created tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result tensor.
    """
    return _wrap_nested(_op.zeros(shape, dtype), name)


def tensor_expr_op(
    tensor_expr_func: Callable,
    name_hint: str,
    args: List[Union[Tensor, _tir.Var]],
    *,
    attrs: Optional[Dict[str, Any]] = None,
):
    """Build the given tensor_expr_func with te.

    Parameters
    ----------
    tensor_expr_func : Callable
        A function that returns a te tensor or a list of tensors.

    name_hint : str
        Name hint.

    args: List[Union[Tensor, _tir.Var]]
        Arguments passed to the function.

    attrs: Optional[Dict[str, Any]]
        A dict of attributes to apply to the function.

    Returns
    -------
    result : Tensor
        The result tensor.
    """

    def _convert(arg):
        if isinstance(arg, Tensor):
            return arg._expr  # pylint: disable=protected-access
        return arg

    return _wrap_nested(
        BlockBuilder.current().emit_te(
            tensor_expr_func,
            *[_convert(arg) for arg in args],
            primfunc_name_hint=name_hint,
            primfunc_attrs=attrs,
        ),
        name=name_hint,
    )
