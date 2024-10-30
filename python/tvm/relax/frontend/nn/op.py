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
# pylint: disable=too-many-lines,invalid-name,protected-access,redefined-outer-name
# pylint: disable=redefined-builtin
"""nn.Tensor operators."""
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from tvm import te
from tvm import tir as _tir
from tvm.script import tir as T

from ... import expr as rx
from ... import op as _op
from ...block_builder import BlockBuilder
from .core import Tensor, get_default_dtype, wrap_nested

IntExpr = Union[int, _tir.PrimExpr]


def unsqueeze(x: Tensor, dim: int, name: str = "unsqueeze") -> Tensor:
    """Add a new axis to a tensor

    Parameters
    ----------
    x : Tensor
        Input tensor to expand.
    dim : int
        Dimension to expand.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Expanded result.
    """
    return wrap_nested(_op.expand_dims(x._expr, dim), name)


def concat(x: List[Tensor], dim: int, name: str = "concat") -> Tensor:
    """Concatenate a list of tensors along an axis.

    Parameters
    ----------
    x : List[Tensor]
        List of tensors to concatenate.
    dim : int
        Dimension to concatenate upon.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Expanded result.
    """
    # Convert tensors to expressions.
    x = [t._expr for t in x]
    return wrap_nested(_op.concat(x, dim), name)


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
    return wrap_nested(_op.add(a._expr, b._expr), name)


def subtract(a: Tensor, b: Tensor, name: str = "subtract") -> Tensor:
    """Subtraction with numpy-style broadcasting.

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

        c = subtract(a, b)
    """
    return wrap_nested(_op.subtract(a._expr, b._expr), name)


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
    return wrap_nested(_op.multiply(a._expr, b._expr), name)


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
    return wrap_nested(_op.divide(a._expr, b._expr), name)


def chunk(x: Tensor, chunks: int, dim: int = 0, name: str = "chunk") -> Tensor:
    """Split a tensor along dim into the specified number of chunks.

    Parameters
    ----------
    x : Tensor
        Input tensor to be split.
    chunks : int
        Number of pieces to slice x into.
    dim : int
        Which dimension to split x.
    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tuple[Tensor]
        A tuple with chunks elements containing slices of x.
    """
    return wrap_nested(_op.split(x._expr, chunks, dim), name)


def sum(
    x: Tensor,
    axis: Optional[Union[int, List[int]]] = None,
    keepdims: bool = False,
    name: str = "sum",
) -> Tensor:
    """Computes the sum of tensor elements over given axes.

    Parameters
    ----------
    x : Tensor
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all of the elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return wrap_nested(_op.sum(x._expr, axis, keepdims), name)


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
        When it is not specified, the output dtype will be the same as input dtype.

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
    return wrap_nested(_op.matmul(a._expr, b._expr, out_dtype=out_dtype), name)


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Optional[Union[int, Tuple]] = 1,
    padding: Optional[Union[int, Tuple, str]] = 0,
    dilation: Optional[Union[int, Tuple]] = 1,
    groups: Optional[int] = 1,
    name: str = "conv1d",
) -> Tensor:
    r"""1D convolution.

    This operator takes the weight as the 1D convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCW`
    and kernel_layout is `OIW`, conv1d takes in
    a data Tensor with shape `(batch_size, in_channels, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_w)`,
    where `kernel_w` is the length of the `W` kernel dimension,
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, x] = \sum_{dx, k}
           \mbox{data}[b, k, \mbox{strides} * x + dx] *
           \mbox{weight}[c, k, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCW` for data and `OIW` for weight), perform the computation,
    then convert to the out_layout.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    weight : Tensor
        The weight expressions.

    bias : Optional[Tensor]
        Optional bias tensor of shape [O].

    strides : Optional[Union[int, Tuple]]
        The strides of convolution. It is required to have length 1.

    padding : Optional[Union[int, Tuple, str]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1 or 2.

    dilation : Optional[Union[int, Tuple]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length 1.

    groups : Optional[int]
        Number of groups to split the input into for grouped convolution.
        The number of input and output channels should be divisible by the number of groups.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    conv_out = _op.nn.conv1d(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if bias is not None:
        conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1]))

    return wrap_nested(conv_out, name)


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Optional[Union[int, Tuple]] = 1,
    padding: Optional[Union[int, Tuple, str]] = 0,
    dilation: Optional[Union[int, Tuple]] = 1,
    groups: Optional[int] = 1,
    data_layout: Optional[str] = "NCHW",
    name: str = "conv2d",
) -> Tensor:
    """Applies a 2D convolution over an input image composed of sevaral input planes

    Parameters
    ----------
    x : Tensor
        Input tensor of shape [B, N, H, W]

    weight : Tensor
        Filters of shape [O, N/groups, kH, kW]

    bias : Optional[Tensor]
        Optional bias tensor of shape [O].

    stride : Optional[Union[int, Tuple]]
        The stride of the convolving kernel. Can be a single number
        or tuple of (sH, sW).

    padding : Optional[[Union[int, Tuple]]]
        Implicit paddings on both sides of the input.

    dilation : Optional[Union[int, Tuple]]
        The spacing between kernel elements. Can be a single number of tuple (dH, dW).

    groups : Optional[int]
        Split input into a number of groups.

    data_layout : Optional[str]
        Layout of input and output data.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result with shape [B, O, oH, oW].
    """
    conv_out = _op.nn.conv2d(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        dilation=dilation,
        data_layout=data_layout,
        groups=groups,
    )
    if bias is not None:
        if data_layout == "NCHW":
            conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1, 1]))
        elif data_layout == "NHWC":
            conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, 1, 1, -1]))
        else:
            raise NotImplementedError(f"Dont know how to handle layout {data_layout}.")

    return wrap_nested(conv_out, name)


def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Optional[Union[int, Tuple]] = 1,
    padding: Optional[Union[int, Tuple, str]] = 0,
    dilation: Optional[Union[int, Tuple]] = 1,
    groups: Optional[int] = 1,
    data_layout: Optional[str] = "NCDHW",
    name: str = "conv3d",
) -> Tensor:
    """Applies a 3D convolution over an input image composed of sevaral input planes

    Parameters
    ----------
    x : Tensor
        Input tensor of shape [B, N, D, H, W]

    weight : Tensor
        Filters of shape [O, N/groups, kD, kH, kW]

    bias : Optional[Tensor]
        Optional bias tensor of shape [O].

    stride : Optional[Union[int, Tuple]]
        The stride of the convolving kernel. Can be a single number
        or tuple of (sD, sH, sW).

    padding : Optional[[Union[int, Tuple]]]
        Implicit paddings on both sides of the input.

    dilation : Optional[Union[int, Tuple]]
        The spacing between kernel elements. Can be a single number of tuple (dD, dH, dW).

    groups : Optional[int]
        Split input into a number of groups.

    data_layout : Optional[str]
        Optional layout of the input and output data.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result with shape [B, O, oD, oH, oW].
    """
    conv_out = _op.nn.conv3d(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_layout=data_layout,
    )
    if bias is not None:
        if data_layout == "NCDHW":
            conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1, 1, 1]))
        elif data_layout == "NDHWC":
            conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, 1, 1, 1, -1]))
        else:
            raise NotImplementedError(f"Dont know how to handle layout {data_layout}.")

    return wrap_nested(conv_out, name)


def conv1d_transpose(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Optional[Union[int, Tuple[int]]] = 1,
    padding: Optional[Union[int, Tuple[int, ...]]] = 0,
    output_padding: Optional[Union[int, Tuple[int]]] = 0,
    dilation: Optional[Union[int, Tuple]] = 1,
    groups: Optional[int] = 1,
    name: str = "conv1d_transpose",
) -> Tensor:
    """1D transposed convolution operator.

    This operator can be seen as the gradient operator of conv1d.

    The output shape can be explained in the simple case when `data_layout == "NCW"` and
    `kernel_layout == "IOW"`. Suppose `data` has shape `(N, in_channel, in_w)`, `weight` has
    shape `(in_channel, out_channel, weight_w)`, we need to assure that `in_channel % groups == 0`.
    The shape of the output will be `(N, out_channel * groups, out_w)`, where

    - `out_w = ((in_w - 1) * strides[0] + weight_w - 2 * padding[0] + output_padding[0])`

    Parameters
    ----------
    data : Tensor
        The input data to the operator.

    weight : Tensor
        The weight tensor.

    strides : Union[int, Tuple[int]]
        The strides of convolution. It is required to have length 1.

    padding : Union[int, Tuple[int, ...]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1 or 2.

    output_padding : Union[int, Tuple[int, ...]], optional
        Used to disambiguate the output shape.

    dilation : Union[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length either 1.

    groups : int
        Number of groups to split the input into for grouped convolution.
        The number of input and output channels should be divisible by the number of groups.

    data_layout : str
        Layout of the input.

    kernel_layout : str
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    out_dtype : Optional[Union[str, DataType]]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    conv_out = _op.nn.conv1d_transpose(
        data=x._expr,
        weight=weight._expr,
        strides=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    if bias is not None:
        conv_out = _op.add(conv_out, _op.reshape(bias._expr, [1, -1, 1]))

    return wrap_nested(conv_out, name)


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
    return wrap_nested(_op.maximum(x1._expr, x2._expr), name)


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
    return wrap_nested(_op.minimum(x1._expr, x2._expr), name)


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
    return wrap_nested(_op.broadcast_to(x._expr, shape), name)


def permute_dims(x: Tensor, axes: Optional[List[int]] = None, name: str = None) -> Tensor:
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
    if name is None:
        x_name = getattr(getattr(x, "_expr", None), "name_hint", None)
        if x_name is not None and "linear" in x_name:
            name = x_name.replace("linear", "matmul")
        else:
            name = "permute_dims"

    return wrap_nested(_op.permute_dims(x._expr, axes=axes), name)


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
    return wrap_nested(_op.reshape(x._expr, shape), name)


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
    return wrap_nested(_op.repeat(x._expr, repeats, axis), name)


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
    return wrap_nested(_op.squeeze(x._expr, axis), name)


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
    return wrap_nested(_op.take(x._expr, indices._expr, axis), name)


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
    # If trying to cast to same dtype as x, skip casting.
    if x.dtype == dtype:
        return x
    return wrap_nested(_op.astype(x._expr, dtype), name)


def relu(x: Tensor, name: str = "relu") -> Tensor:
    """Rectified Linear Unit (ReLU) activation function.

    .. math::
        \text{ReLU}(x) = \text{max}(x, 0)

    Parameters
    ----------
    x : Tensor
        The input data.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return wrap_nested(_op.nn.relu(x._expr), name)


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
    return wrap_nested(_op.nn.silu(x._expr), name)


def gelu(x: Tensor, approximate: Optional[str] = None, name: str = "gelu") -> Tensor:
    r"""Applies the Gaussian Error Linear Units function

    .. math::
        \text{GeLU}(x) = 0.5 * x * (1 + \text{erf}(x * 0.5**0.5))

    where :math:`erf` is the Gauss Error function.

    Parameters
    ----------
    x : Tensor
        The input data

    approximate : Optional[str]
        If set to tanh, use an approximation when calculating CDF.

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
    if approximate == "tanh":
        gelu_out = _op.nn.gelu_tanh(x._expr)
    else:
        gelu_out = _op.nn.gelu(x._expr)
    return wrap_nested(gelu_out, name)


def sigmoid(x: Tensor, name: str = "sigmoid") -> Tensor:
    r"""Computes sigmoid.

    .. math:: \text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Parameters
    ----------
    data: Tensor
        The input data to the operator.

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
    return wrap_nested(_op.sigmoid(x._expr), name)


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
    return wrap_nested(_op.nn.softmax(x._expr, axis), name)


def tanh(x: Tensor, name: str = "tanh") -> Tensor:
    r"""Applies the hyperbolic tangent function.

    .. math::
        \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

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
    return wrap_nested(_op.tanh(x._expr), name)


def exp(x: Tensor, name: str = "exp") -> Tensor:
    r"""Applies the exponential function.

    .. math::
        \text{Exp}(x) = e^x

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

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
    return wrap_nested(_op.exp(x._expr), name)


def permute(x: Tensor, axes: Optional[List[int]], name: str = "permute") -> Tensor:
    """Permutes the dimensions of the input tensor.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    axes : Optional[List[int]]
        The target axes order.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The transposed result.
    """

    return wrap_nested(_op.permute_dims(x._expr, axes=axes), name)


def negative(x: Tensor, name: str = "neg") -> Tensor:
    """Numerical negative of the input tensor.

    Parameters
    ----------
    x : Tensor
        The input data to the operator.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return wrap_nested(_op.negative(x._expr), name)


def layer_norm(
    x: Tensor,
    normalized_shape: Union[int, List[int]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    name: str = "layer_norm",
) -> Tensor:
    r"""
    Layer normalization (Lei Ba and et al., 2016).
    Applies layer normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    .. math::

        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}}
            * gamma + beta

    Unlike batch normalization, the mean and var are computed along the channel dimension.

    Assume the input has size k on axis 1, then both gamma and beta have shape (k,).

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    x : Tensor
        Input to which layer_norm will be applied.

    normalized_shape: Union[int, List[int]]
        The shape of axes to normalize. If a single integer
        is used, it is treated as a singleton list and this
        module will normalize over the last dimension.

    weight: Tensor
        The gamma scale factor.

    bias: Tensor
        The beta offset factor.

    eps: float
        Small float added to variance to avoid dividing by zero.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    dim_num = len(normalized_shape)
    axes = list(range(-dim_num, 0))
    dtype = x._expr.struct_info.dtype

    if weight is not None:
        weight = weight._expr
    else:
        weight = rx.const(np.ones(normalized_shape), dtype=dtype)
    if bias is not None:
        bias = bias._expr
    else:
        bias = rx.const(np.zeros(normalized_shape), dtype=dtype)

    return wrap_nested(
        _op.nn.layer_norm(
            x._expr,
            gamma=weight,
            beta=bias,
            axes=axes,
            epsilon=eps,
        ),
        name=name,
    )


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
    return wrap_nested(_op.nn.rms_norm(x._expr, weight._expr, axes, epsilon), name)


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float = 1e-5,
    channel_axis: int = 1,
    axes: Optional[List[int]] = None,
    name: str = "group_norm",
) -> Tensor:
    r"""
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Parameters
    ----------
    x : Tensor
        Input to which rms_norm will be applied.

    num_groups : int
        Number of groups to separate the channels into.

    weight : Tensor
        The gamma scale factor.

    bias : Tensor
        The beta offset factor.

    epsilon : float
        Small float added to square mean to avoid dividing by zero.

    channel_axis: int
        The channel axis of the data.

    axes : Optional[int]
        Which axes to compute the groupnorm over. If None, assumes first
        two channels should be ignored.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    if weight is not None:
        weight = weight._expr
    if bias is not None:
        bias = bias._expr
    dim = len(x._expr.struct_info.shape)
    if axes is None:
        axes = list(range(2, dim))
    return wrap_nested(
        _op.nn.group_norm(
            x._expr, weight, bias, num_groups, channel_axis=channel_axis, axes=axes, epsilon=eps
        ),
        name,
    )


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
    return wrap_nested(_op.triu(x._expr, diagonal), name)


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
    if isinstance(fill_value, (_tir.FloatImm, _tir.IntImm)):
        fill_value = rx.const(fill_value.value, dtype=dtype)
    elif isinstance(fill_value, (int, float)):
        fill_value = rx.const(fill_value, dtype=dtype)
    else:
        fill_value = fill_value._expr
    return wrap_nested(_op.full(shape, fill_value, dtype), name)


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
    return wrap_nested(_op.zeros(shape, dtype), name)


def ones(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "ones",
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
    return wrap_nested(_op.ones(shape, dtype), name)


def empty(
    shape: Sequence[IntExpr],
    dtype: str = "float32",
    name: str = "empty",
) -> Tensor:
    """Construct an uninitialized tensor, with the input shape and dtype.

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
    return wrap_nested(  # type: ignore
        _op.builtin.alloc_tensor(
            rx.ShapeExpr(shape),  # type: ignore
            dtype,
            runtime_device_index=0,
        ),
        name,
    )


def split(
    ary: Tensor,
    indices_or_sections: Union[int, Sequence[int]],
    axis: int = 0,
    name: str = "split",
) -> Tuple[Tensor, ...]:
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : Tensor
        Input tensor to be split.
    indices_or_sections : Union[int, Sequence[int]]
        Indices or sections to split into.
    axis : int = 0
        The axis along which to split, default is 0.
    name : str
        Name hint.

    Returns
    -------
    result : Tuple[Tensor, ...]
        A list of sub-arrays as the outcome of splitting.
    """
    return wrap_nested(_op.split(ary._expr, indices_or_sections, axis), name)


def pad(
    x: Tensor,
    pad: List[int],
    mode: str = "constant",
    value: int = 0,
    name: str = "pad",
) -> Tensor:
    """
    Apply spatial padding to the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor to be padded.
    pad : List[int]
        List in the format of [before_0, after_0, before_1, after_1, ...]
        indicating how much to pad each axis of x.
    mod : str
        Padding mode to use, constant implies padded elements will use
        value argument.
    value : int
        What to pad with in constant mode.
    name : str
        Name hint for this operator.

    Returns
    -------
    result : Tensor
        Padded output tensor.
    """
    return wrap_nested(_op.nn.pad(x._expr, pad_width=pad, pad_value=value, pad_mode=mode), name)


def square(x: Tensor, name: str = "square") -> Tensor:
    """Computes the element-wise square of the input tensor.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The computed result.
    """
    return wrap_nested(_op.square(x._expr), name)


def sqrt(x: Tensor, name: str = "sqrt") -> Tensor:
    """Computes the element-wise sqrt of the input tensor.

    Parameters
    ----------
    x : Tensor
        The input tensor.

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
    return wrap_nested(_op.sqrt(x._expr), name)


def get_timestep_embedding(
    x: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    name: str = "get_timestep_embedding",
) -> Tensor:
    """
    Timestep calculation as described in Denoising Diffusion Probabilistic Models.

    Parameters
    ----------
    x : Tensor
        A 1-D Tensor of N indices.
    embedding_dim : int
        The dimension of the output.
    flip_sin_to_cos : bool
        If True, change the order of sine and cosine embeddings.
    downscale_freq_shift : float
        Adjusts the frequency of the sinusoidal sampling.
    scale : float
        Weight adjustment for embedding magnitude.
    max_period : int
        Controls the minimum frequency of the embeddings.
    name : str
        The name to label this operator with.

    Returns
    -------
    result : Tensor
        [N x dim] Tensor of positional embeddings.
    """
    dtype = get_default_dtype()

    # Arithmetic should be done in float for precision.
    timesteps = _op.astype(x._expr, "float32")

    half_dim = embedding_dim // 2
    exponent = rx.const(-math.log(max_period), "float32") * _op.arange(
        start=0, end=half_dim, dtype="float32"
    )
    exponent = exponent / (rx.const(half_dim - downscale_freq_shift, "float32"))

    emb = _op.exp(exponent)
    emb = _op.expand_dims(timesteps, 1) * _op.expand_dims(emb, 0)
    # Scale embeddings
    if scale != 1:
        emb = rx.const(scale, "float32") * emb

    # Concat sine and cosine embeddings.
    if flip_sin_to_cos:
        emb = _op.concat([_op.cos(emb), _op.sin(emb)], axis=-1)
    else:
        emb = _op.concat([_op.sin(emb), _op.cos(emb)], axis=-1)

    # Zero pad
    if embedding_dim % 2 == 1:
        emb = _op.nn.pad(emb, (0, 1, 0, 0))

    # Cast to proper output type
    emb = _op.astype(emb, dtype)
    return wrap_nested(emb, name)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = False,
    scale: Optional[float] = None,
    name: str = "scaled_dot_product_attention",
):
    """
    Computes a scaled dot product attention on provided attention
    query, key, and values. Compliant with the functional torch implementation.

    Parameters
    ----------
    query : Tensor
        Tensor representing current attention lookup of shape
        [batch, seq_len, num_heads, head_size].
    key : Tensor
        Tensor representing cross attention mapping of shape
        [batch, seq_len_kv, num_heads_kv, head_size].
    value : Tensor
        Tensor representing embedded attention values of shape
        [batch, seq_len_kv, num_heads_kv, head_size_value].
    attn_mask : Optional[Tensor]
        Optional mask for attention, not yet supported.
    is_causal : Optional[bool]
        If set, uses a causal attention mask.
    scale : Optional[float]
        Optional extra scaling argument applied to attention.
    name : str
        Name hint for this function.
    """
    assert attn_mask is None, "attn_mask not yet supported."
    causal_mask = "TopLeft" if is_causal else None

    attn = _op.nn.attention(
        query._expr, key._expr, value._expr, causal_mask=causal_mask, scale=scale
    )
    return wrap_nested(attn, name)


def interpolate(
    x: Tensor,
    size: Optional[Union[int, Tuple[int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: Optional[bool] = None,
    data_layout: Optional[str] = "NCHW",
    name: str = "interpolate",
):
    """Resize a tensor using the specified mode.

    Parameters
    ----------
    x : Tensor
        Input tensor to be resized.
    size : Optional[Union[int, Tuple[int]]]
        Requested output size, only one of size and scale_factor may
        be specified.
    scale_factor : Optional[Union[float, Tuple[float]]]
        Multiplier for spatial size.
    mode : str
        Algorithm used for sampling.
    align_corners : Optional[bool]
        How to map pixels before and after sampling.
    recompute_scale_factor : Optional[bool]
        Recompute the scale_factor for use in interpolation.
    antialias : Optional[bool]
        Apply antialiasing to output.
    data_layout : Optional[str]
        Layout of the input and output data.
    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
        Output tensor with requested shape.
    """
    assert recompute_scale_factor is None, "recompute_scale_factor is not supported."
    assert antialias is None, "antialias is not supported."

    if size is None:
        size = []
        for i, dim in enumerate(data_layout):
            # Only upscale spatial dimensions.
            if dim not in ["N", "C"]:
                if isinstance(scale_factor, (list, tuple)):
                    size.append(int(x.shape[i] * scale_factor[len(size)]))
                else:
                    size.append(int(x.shape[i] * scale_factor))

    if mode.startswith("nearest"):
        mode = "nearest_neighbor"
    elif mode[0:2] == "bi":
        mode = mode[2:]

    if mode == "nearest_neighbor":
        coord_trans = "asymmetric"
    elif align_corners:
        coord_trans = "align_corners"
    else:
        coord_trans = "half_pixel"

    return wrap_nested(
        _op.image.resize2d(
            x._expr,
            size,
            layout=data_layout,
            method=mode,
            coordinate_transformation_mode=coord_trans,
        ),
        name,
    )


def ccl_allreduce(x: Tensor, op_type: str = "sum", in_group: bool = True, name="ccl_allreduce"):
    """CCL Allreduce operator

    Parameters
    ----------
    x : relax.Expr
      The input tensor.

    op_type : str
      The type of reduction operation to be applied to the input data.
      Now "sum", "prod", "min", "max" and "avg" are supported.

    in_group : bool
      Whether the reduction operation performs globally or in group as default.

    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
      The result tensor of allreduce.
    """
    return wrap_nested(_op.ccl.allreduce(x._expr, op_type, in_group), name)


def ccl_allgather(x: Tensor, num_workers: int, name="ccl_allgather"):
    """CCL Allgather operator

    Parameters
    ----------
    x : relax.Expr
      The input tensor.

    num_workers : int
      Number of workers.

    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
      The result tensor of allgather.
    """
    return wrap_nested(_op.ccl.allgather(x._expr, num_workers), name)


def ccl_broadcast_from_worker0(x: Tensor, name="broadcast_from_worker"):
    """Broadcast data from worker-0 to all other workers.

    Parameters
    ----------
    x : Tensor
      The tensor to be broadcast.
    name : str
        Name hint for this operation.

    Returns
    -------
    result : Tensor
      The same tensor, which has been broadcast to all other workers.
    """
    return wrap_nested(_op.ccl.broadcast_from_worker0(x._expr), name)


def tensor_expr_op(
    tensor_expr_func: Callable,
    name_hint: str,
    args: List[Union[Tensor, _tir.Var, int]],
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

    return wrap_nested(
        BlockBuilder.current().emit_te(
            tensor_expr_func,
            *[_convert(arg) for arg in args],
            primfunc_name_hint=name_hint,
            primfunc_attrs=attrs,
        ),
        name=name_hint,
    )


OutType = TypeVar("OutType", bound=Union[Tensor, Sequence[Tensor]])


def tensor_ir_op(
    func: _tir.PrimFunc,
    name_hint: str,
    args: Union[Tensor, Sequence[Union[Tensor, rx.ShapeExpr, _tir.PrimExpr]]],
    out: OutType,
) -> OutType:
    """Create a `call_tir` binding with given PrimFunc

    Parameters
    ----------
    func : _tir.PrimFunc
        The PrimFunc to call.

    name_hint : str
        Name hint.

    args : Union[Tensor, Sequence[Union[Tensor, rx.ShapeExpr, _tir.PrimExpr]]]
        The arguments to pass to the PrimFunc.

    out : Union[Tensor, List[Tensor]]
        The output tensors.

    Returns
    -------
    result : Tensor
        The result tensor
    """
    from tvm import relax as rx  # pylint: disable=import-outside-toplevel

    call_tir_args, tir_vars = [], []
    if not isinstance(args, (tuple, list)):
        args = [args]

    for arg in args:
        if isinstance(arg, Tensor):
            call_tir_args.append(arg._expr)
        elif isinstance(arg, (rx.ShapeExpr, _tir.PrimExpr)):
            tir_vars.append(arg)
        else:
            raise TypeError(
                "Unsupported type: tensor_ir_op args expect Tensor or ShapeExpr or PrimExpr,"
                f"but got {type(arg)}"
            )

    if isinstance(out, Tensor):
        out_sinfo = [out._expr.struct_info]
    else:
        out_sinfo = [x._expr.struct_info for x in out]

    bb = BlockBuilder.current()
    global_var = bb.add_func(func, name_hint)

    if len(tir_vars) == 0:
        tir_vars = None

    return wrap_nested(
        bb.emit(rx.call_tir(global_var, call_tir_args, out_sinfo, tir_vars=tir_vars)),
        name=name_hint,
    )


def tensor_ir_inplace_op(
    func: _tir.PrimFunc,
    name_hint: str,
    args: Union[Tensor, Sequence[Union[Tensor, rx.ShapeExpr, _tir.PrimExpr]]],
    inplace_indices: Union[int, List[int]],
    out: OutType,
) -> OutType:
    """Create a `call_tir_inplace` binding with given PrimFunc

    Parameters
    ----------
    func : _tir.PrimFunc
        The PrimFunc to call.

    name_hint : str
        Name hint.

    args : Union[Tensor, Sequence[Union[Tensor, rx.ShapeExpr, _tir.PrimExpr]]]
        The arguments to pass to the PrimFunc.

    inplace_indices : Union[int, List[int]]
        Specify which arguments should be used for in-place computations.
        If `inplace_indices` is a single integer, it will be made into a singleton list.
        Suppose `inplace_indices[i] = j`, where `j >= 0`. Then the `i`th output
        will be an alias of `args[j]`.
        If `inplace_indices[i] = -1`, then the `i`th output will be a freshly allocated tensor.
        At least one member of `inplace_indices` must not be -1.

    out : Union[Tensor, List[Tensor]]
        The output tensors.

    Returns
    -------
    result : Tensor
        The result tensor
    """
    from tvm import relax as rx  # pylint: disable=import-outside-toplevel

    call_tir_args, tir_vars = [], []
    if not isinstance(args, (tuple, list)):
        args = [args]

    for arg in args:
        if isinstance(arg, Tensor):
            call_tir_args.append(arg._expr)
        elif isinstance(arg, (rx.ShapeExpr, _tir.PrimExpr)):
            tir_vars.append(arg)
        else:
            raise TypeError(
                "Unsupported type: tensor_ir_inplace_op args expect Tensor or ShapeExpr or"
                f" PrimExpr, but got {type(arg)}"
            )

    if isinstance(out, Tensor):
        out_sinfo = [out._expr.struct_info]
    else:
        out_sinfo = [x._expr.struct_info for x in out]

    bb = BlockBuilder.current()
    global_var = bb.add_func(func, name_hint)

    return wrap_nested(
        bb.emit(
            rx.call_tir_inplace(global_var, call_tir_args, inplace_indices, out_sinfo, tir_vars)
        ),
        name=name_hint,
    )


def extern(
    name: str,
    args: Sequence[Union[Tensor, _tir.PrimExpr, int, float, str]],
    out: OutType,
) -> OutType:
    """Invoke an extern function during runtime. The extern function must be registered with the "
    TVM runtime using `TVM_REGISTER_GLOBAL` (C++), or `tvm.register_func` (Python).

    Parameters
    ----------
    name : str
        The name of the extern function to call.

    args : Sequence[Union[Tensor, _tir.PrimExpr, int, float, str]]
        The arguments to pass to the extern function.

    out : Union[Tensor, List[Tensor]]
        The output tensors, only

    Returns
    -------
    result : Tensor
        The result
    """
    from tvm import relax as rx  # pylint: disable=import-outside-toplevel

    def _convert(arg, name: str):
        if isinstance(arg, Tensor):
            return arg._expr  # pylint: disable=protected-access
        if isinstance(arg, int):
            return rx.PrimValue(_tir.IntImm("int64", arg))
        if isinstance(arg, float):
            return rx.PrimValue(_tir.FloatImm("float64", arg))
        if isinstance(arg, str):
            return rx.StringImm(arg)
        if isinstance(arg, _tir.PrimExpr):
            return rx.PrimValue(arg)
        if isinstance(arg, (tuple, list)):
            return rx.Tuple([_convert(e, f"{name}_{i}") for i, e in enumerate(arg)])
        raise TypeError(f"Unsupported input type: {type(arg)}")

    rx_inputs = _convert(args, "input")
    rx_outputs_sinfo = _convert(out, "dummy").struct_info
    return wrap_nested(
        _op.call_dps_packed(
            name,
            args=rx_inputs,
            out_sinfo=rx_outputs_sinfo,
        ),
        name,
    )  # type: ignore


def debug_func(
    name: str,
    *args: Union[Tensor, _tir.PrimExpr, int, float, str],
    _line_info: Optional[str] = None,
):
    """Call a debug function during runtime. The debug function must be registered with the
    following type signature:

    .. code-block:: python

        @tvm.register_func(name_of_debug_func)
        def debug_func(lineno: str, arg_0, arg_1, ...) -> None:
            ...

    Parameters
    ----------
    name : str
        The name of the debug function to call.

    *args : Union[Tensor, _tir.PrimExpr, int, float, str]
        The arguments to pass to the debug function.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relax as rx

    from .exporter import Exporter
    from .modules import IOEffect

    # pylint: enable=import-outside-toplevel

    if Exporter.current().io_effect is None:
        raise RuntimeError("Debugging is only supported when debug mode is on.")
    io: IOEffect = Exporter.current().io_effect  # type: ignore

    if _line_info is None:
        filename, line_number = inspect.getframeinfo(inspect.currentframe().f_back)[:2]
        _line_info = f"{filename}:{line_number}"

    converted_args = []
    for arg in args:
        if isinstance(arg, Tensor):
            converted_args.append(arg._expr)  # pylint: disable=protected-access
        elif isinstance(arg, int):
            converted_args.append(rx.PrimValue(_tir.IntImm("int64", arg)))
        elif isinstance(arg, float):
            converted_args.append(rx.PrimValue(_tir.FloatImm("float32", arg)))
        elif isinstance(arg, _tir.PrimExpr):
            converted_args.append(rx.PrimValue(arg))
        elif isinstance(arg, str):
            converted_args.append(rx.StringImm(arg))
        else:
            raise TypeError(f"Unsupported type {type(arg)}")

    io.effect = BlockBuilder.current().emit(
        rx.call_pure_packed(
            "vm.builtin.invoke_debug_func",
            io.effect,
            rx.StringImm(name),
            rx.StringImm(_line_info),
            *converted_args,
            sinfo_args=[rx.ObjectStructInfo()],
        ),
        name_hint=io.effect.name_hint,
    )


def print_(tensor: Tensor):
    """Debug printing a Tensor during runtime."""
    filename, line_number = inspect.getframeinfo(inspect.currentframe().f_back)[:2]
    line_info = f"{filename}:{line_number}"
    debug_func("vm.builtin.debug_print", tensor, _line_info=line_info)


def less(a: Tensor, b: Tensor, name: str = "less") -> Tensor:
    """Broadcasted element-wise comparison for (lhs < rhs).

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
    """
    return wrap_nested(_op.less(a._expr, b._expr), name)


def less_equal(a: Tensor, b: Tensor, name: str = "less_equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs <= rhs).

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
    """
    return wrap_nested(_op.less_equal(a._expr, b._expr), name)


def greater(a: Tensor, b: Tensor, name: str = "greater") -> Tensor:
    """Broadcasted element-wise comparison for (lhs > rhs).

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
    """
    return wrap_nested(_op.greater(a._expr, b._expr), name)


def greater_equal(a: Tensor, b: Tensor, name: str = "greater_equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs >= rhs).

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
    """
    return wrap_nested(_op.greater_equal(a._expr, b._expr), name)


def equal(a: Tensor, b: Tensor, name: str = "equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs == rhs).

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
    """
    return wrap_nested(_op.equal(a._expr, b._expr), name)


def not_equal(a: Tensor, b: Tensor, name: str = "not_equal") -> Tensor:
    """Broadcasted element-wise comparison for (lhs != rhs).

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
    """
    return wrap_nested(_op.not_equal(a._expr, b._expr), name)


def where(condition: Tensor, x1: Tensor, x2: Tensor, name: str = "where") -> Tensor:
    """Selecting elements from either the input tensors depending on the value of the
    condition.

    For a given position, return the corresponding value in `x1` if `condition` is True,
    and return the corresponding value in `x2` otherwise.

    Parameters
    ----------
    condition : Tensor
        When True, yield `x1`; otherwise, yield `x2`.
        Must be broadcasting compatible with `x1` and `x2`.
        Must have boolean dtype.

    x1 : Tensor
        The first input tensor.
        Must be broadcasting compatible with `condition` and `x2`.

    x2 : Tensor
        The second input tensor.
        Must be broadcasting compatible with `condition` and `x1`.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result tensor.
    """
    # Cast condition to boolean.
    condition = astype(condition, "bool")
    return wrap_nested(_op.where(condition._expr, x1._expr, x2._expr), name)


def cumsum(
    data: Tensor,
    axis: Optional[int] = None,
    dtype: Optional[str] = None,
    exclusive: Optional[bool] = None,
    name: str = "cumsum",
) -> Tensor:
    """Numpy style cumsum op. Return the cumulative inclusive sum of the elements along
    a given axis.

    Parameters
    ----------
    data : Tensor
        The input data to the operator.

    axis : Optional[int]
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    dtype : Optional[str]
        Type of the returned array and of the accumulator in which the elements are summed.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : Optional[bool]
        If true will return exclusive sum in which the first element is not
        included.

    name : str
        Name hint.

    Returns
    -------
    result : Tensor
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1, 2, 3], [4, 5, 6]]

        cumsum(a)  # if axis is not provided, cumsum is done over the flattened input.
        -> [ 1,  3,  6, 10, 15, 21]

        cumsum(a, dtype="float32")
        -> [  1.,   3.,   6.,  10.,  15.,  21.]

        cumsum(a, axis=0)  # sum over rows for each of the 3 columns
        -> [[1, 2, 3],
            [5, 7, 9]]

        cumsum(a, axis=1)
        -> [[ 1,  3,  6],
            [ 4,  9, 15]]

        a = [1, 0, 1, 0, 1, 1, 0]  # a is a boolean array
        cumsum(a, dtype=int32)  # dtype should be provided to get the expected results
        -> [1, 1, 2, 2, 3, 4, 4]
    """
    return wrap_nested(_op.cumsum(data._expr, axis, dtype, exclusive), name)


def sort(x: Tensor, axis: int = -1, descending: bool = False, name="sort"):
    """Performs sorting along the given axis and returns an array
    in sorted order.

    Parameters
    ----------
    x : Tensor
        The input tensor.

    axis : int
        Axis along which to sort the input tensor.
        By default the last axis of the input is used.

    descending : bool
        Whether to sort in descending order, the default is False

    name : str
        Name hint.

    Returns
    -------
    out : Tensor
        The sorted tensor.
    """
    return wrap_nested(_op.sort(x._expr, axis, descending), name=name)


def argsort(
    data: Tensor, axis: int = -1, descending: bool = False, dtype: str = "int32", name="argsort"
):
    """Performs sorting along the given axis and returns an array of indices
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data : Tensor
        The input data tensor.

    axis : int
        Axis long which to sort the input tensor.

    descending : bool
        Whether to sort in descending order, the default is False

    dtype : str
        The data type of the output indices.

    name : str
        Name hint.

    Returns
    -------
    out : Tensor
        The indices of the sorted tensor.
    """
    return wrap_nested(_op.argsort(data._expr, axis, descending, dtype), name=name)


def topk(
    data: Tensor,
    k: int = 1,
    axis: int = -1,
    ret_type: str = "both",
    largest: bool = True,
    dtype: str = "int32",
    name: str = "topk",
):
    """Get the top k elements in an input tensor along the given axis.

    ret_type specifies the return type, can be one of ("both", "values", "indices").

    Parameters
    ----------
    data : Tensor
        The input data tensor.

    k : int
        Number of top elements to select. Return all elements if k < 1.

    axis : int
        Axis long which to sort the input tensor.

    ret_type: str
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    largest : bool
        Whether to return largest or smallest elements.
        The k smallest elements are returned if largest is False.

    dtype : str
        The data type of the indices output.

    name : str
        Name hint.

    Returns
    -------
    out : Tensor or Tuple[Tensor, Tensor]
        The computed result.
    """
    return wrap_nested(_op.topk(data._expr, k, axis, ret_type, largest, dtype), name=name)


def multinomial_from_uniform(
    prob: Tensor,
    uniform_sample: Tensor,
    sample_indices: Optional[Tensor] = None,
    dtype: str = "int64",
    name: str = "multinomial_from_uniform",
):
    """Returns a tensor where each row contains the index sampled from the multinomial
    probability distribution located in the corresponding row of tensor prob.

    Notes
    -----
    For better cpu performance, use 'vm.builtin.multinomial_from_uniform'.
    For accurate results, ensure probabilities are between 0 and 1 and sum to 1.

    Parameters
    ----------
    prob : Tensor
        A 2-D tensor of shape (batch, vocab_size) representing probability distributions.
        Each row is a distribution across vocabulary for a batch, where:
        Values range from [0, 1], indicating the probability of each vocabulary item.
        The sum of values in each row is 1, forming a valid distribution.

    uniform_sample : Tensor
        The uniformly sampled 2-D tensor with the shape (n, 1).
        Values range from 0 to 1, indicating probabilities sampled uniformly.

    sample_indices : Optional[Tensor]
        The 2-D tensor with the shape [n, 1], which indicates the specific
        probability distribution to sample from. The value of sample_indices[i]
        determines that the ith token should be sampled from the sample_indices[i]th
        probability distribution. For instance, if there are 3 distinct probability
        distributions and the requirement is to sample 2, 3, and 4 tokens from each,
        then sample_indices would be [0, 0, 1, 1, 1, 2, 2, 2, 2].

    dtype : str
        The data type of output tensor.


    Returns
    -------
    result : Tensor
        The computed tensor with shape (n, 1).

    Examples
    --------
    .. code-block:: python

        prob = [[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]]
        usample = [[0.4], [0.9]]
        sample_indices = [[0], [1]]

        multinomial_from_uniform(prob, usample)
        -> [[1], [2]]
        multinomial_from_uniform(prob, usample, sample_indices)
        -> [[1], [2]]
    """
    out_batch = uniform_sample.shape[0]

    if sample_indices is not None:
        assert (
            sample_indices.shape == uniform_sample.shape
        ), "The shape of sample_indices must match the shape of uniform_sample."
    else:
        assert (
            prob.shape[0] == uniform_sample.shape[0]
        ), "Number of samples must match the number of probability distributions."
        sample_indices = Tensor.from_const(np.arange(out_batch).reshape(out_batch, 1))

    return wrap_nested(
        _op.multinomial_from_uniform(prob._expr, uniform_sample._expr, sample_indices._expr, dtype),
        name,
    )


def sample_top_p_top_k_from_sorted_prob(
    sorted_prob: Tensor,
    sorted_index: Tensor,
    top_p: Tensor,
    top_k: Tensor,
    uniform_sample: Tensor,
    sample_indices: Optional[Tensor] = None,
):
    """Samples indices from a sorted probability tensor based on top_p and top_k criteria.

    Notes
    -----
    For accurate results, ensure probabilities are between 0 and 1 and sum to 1.

    Parameters
    ----------
    sorted_prob : Tensor
        A 2-D tensor, with shape (batch, vocab_size), contains probabilities
        sorted in descending order.

    sorted_index: Tensor
        The indices tensor with shape (batch, vocab_size), corresponding to the
        sorted_prob. Potentially from applying argsort on the original probability
        tensor in descending order.

    top_p : Tensor
        The cumulative probability threshold with shape (batch, 1) for nucleus sampling.

    top_k :Tensor
        A tensor with shape (batch, 1), representing the number of top probabilities
        to consider for top-k sampling.

    uniform_sample : Tensor
        Uniformly sampled values with shape (n, 1) are used to select the output indices.

    sample_indices : Optional[Tensor]
        The 2-D tensor with the shape [n, 1], which indicates the specific
        probability distribution to sample from. The value of sample_indices[i]
        determines that the ith token should be sampled from the sample_indices[i]th
        probability distribution. For instance, if there are 3 distinct probability
        distributions and the requirement is to sample 2, 3, and 4 tokens from each,
        then sample_indices would be [0, 0, 1, 1, 1, 2, 2, 2, 2].

    Returns
    -------
    result : Tensor
        The selected indices with shape (n, 1).

    Examples
    --------
    .. code-block:: python

        prob = [[0.1 , 0.4, 0.5],
                [0.3, 0.3, 0.4]]
        sorted_prob = [[0.5, 0.4, 0.1],
                       [0.4, 0.3, 0.3]]
        sorted_index = [[2, 1, 0],
                        [2, 0, 1]]
        top_p = [[0.6],[0.9]]
        top_k = [[3],[2]]
        uniform_sample = [[0.5], [0.6]]
        sample_indices = [[0], [1]]

        sample_top_p_top_k_from_sorted_prob(
            sorted_prob, sorted_index,top_p, top_k, uniform_sample, sample_indices)
        -> [2, 0]

    """
    prob_dtype = sorted_prob.dtype
    index_dtype = sorted_index.dtype
    prob_batch = sorted_prob.shape[0]
    out_batch = uniform_sample.shape[0]

    if sample_indices is not None:
        assert (
            sample_indices.shape == uniform_sample.shape
        ), "The shape of sample_indices must match the shape of uniform_sample."
    else:
        assert (
            sorted_prob.shape[0] == uniform_sample.shape[0]
        ), "Number of samples must match the number of probability distributions."
        sample_indices = Tensor.from_const(
            np.arange(out_batch).reshape(out_batch, 1).astype(np.int64)
        )
        print("sample_indices: ", sample_indices)
    sample_indices_dtype = sample_indices.dtype

    def _cumsum_mask(cumsum_sorted, top_p, top_k, i, j):
        return _tir.all(cumsum_sorted[i, j] < top_p[i, 0], j + 1 < top_k[i, 0])

    @T.prim_func(private=True)
    def _get_renorm_prob(A: T.handle, B: T.handle, C: T.handle, D: T.handle):
        batch, vocab_size = T.int64(is_size_var=True), T.int64(is_size_var=True)
        cumsum_sorted = T.match_buffer(A, (batch, vocab_size), prob_dtype)
        top_p = T.match_buffer(B, (batch, 1), prob_dtype)
        top_k = T.match_buffer(C, (batch, 1), index_dtype)
        renorm_prob = T.match_buffer(D, (batch, 1), prob_dtype)
        for ax0, ax1 in T.grid(batch, vocab_size):
            with T.block("T_get_renorm_prob"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                if not _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, 0):
                    renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, 0]
                elif _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, v_ax1):
                    if v_ax1 + 1 == vocab_size:
                        renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, v_ax1]
                    elif not _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, v_ax1 + 1):
                        renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, v_ax1 + 1]

    @T.prim_func(private=True)
    def _get_index_from_sorted(
        A: T.handle, B: T.handle, C: T.handle, D: T.handle, E: T.handle, F: T.handle
    ):
        batch, vocab_size = T.int64(is_size_var=True), T.int64(is_size_var=True)
        out_batch = T.int64(is_size_var=True)
        cumsum_sorted = T.match_buffer(A, (batch, vocab_size), prob_dtype)
        indices = T.match_buffer(B, (batch, vocab_size), index_dtype)
        renorm_prob = T.match_buffer(C, (batch, 1), prob_dtype)
        usample = T.match_buffer(D, (out_batch, 1), prob_dtype)
        sample_indices = T.match_buffer(E, (out_batch, 1), sample_indices_dtype)
        output_index = T.match_buffer(F, (out_batch, 1), index_dtype)

        for ax0, ax1 in T.grid(out_batch, vocab_size):
            with T.block("T_get_index_from_sorted"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.writes(output_index[v_ax0, 0])
                if (
                    usample[v_ax0, T.int64(0)]
                    < cumsum_sorted[sample_indices[v_ax0, T.int64(0)], v_ax1]
                    / renorm_prob[sample_indices[v_ax0, T.int64(0)], 0]
                    or v_ax1 + 1 == vocab_size
                ):
                    if v_ax1 == 0:
                        output_index[v_ax0, 0] = indices[sample_indices[v_ax0, T.int64(0)], 0]
                    elif (
                        usample[v_ax0, T.int64(0)]
                        >= cumsum_sorted[sample_indices[v_ax0, T.int64(0)], v_ax1 - 1]
                        / renorm_prob[sample_indices[v_ax0, T.int64(0)], 0]
                    ):
                        output_index[v_ax0, 0] = indices[sample_indices[v_ax0, T.int64(0)], v_ax1]

    cumsum_sorted = cumsum(sorted_prob, axis=1)

    renorm_prob = tensor_ir_op(
        _get_renorm_prob,
        "get_renorm_prob",
        args=[cumsum_sorted, top_p, top_k],
        out=Tensor.placeholder(
            [prob_batch, 1],
            prob_dtype,
        ),
    )

    out_index_in_sorted = tensor_ir_op(
        _get_index_from_sorted,
        "get_index_from_sorted",
        args=[cumsum_sorted, sorted_index, renorm_prob, uniform_sample, sample_indices],
        out=Tensor.placeholder([out_batch, 1], index_dtype),
    )
    return out_index_in_sorted


def renormalize_top_p_top_k_prob(prob, sorted_prob, top_p, top_k):
    """Renormalizes probabilities after filtering with top_p and top_k, ensuring
    they sum up to 1.

    Notes
    -----
    For accurate results, ensure probabilities are between 0 and 1 and sum to 1.

    Parameters
    ----------
    prob : Tensor
        A 2-D tensor of shape (batch, vocab_size) representing probability distributions.

    sorted_prob : Tensor
        Probabilities sorted in descending order.

    top_p : Tensor
        The cumulative probability threshold with shape (batch, 1) for nucleus sampling.

    top_k :Tensor
        A tensor with shape (batch, 1), representing the number of top probabilities
        to consider for top-k sampling.

    Returns
    -------
    result : Tensor
        The filtered and nomalized tensor with the sampe shape as input prob.
    """
    prob_dtype = prob.dtype
    top_k_dtype = top_k.dtype
    batch = sorted_prob.shape[0]

    def _cumsum_mask(cumsum_sorted, top_p, top_k, i, j):
        return _tir.all(cumsum_sorted[i, j] < top_p[i, 0], j + 1 < top_k[i, 0])

    @T.prim_func(private=True)
    def _get_renorm_cutoff(A: T.handle, B: T.handle, C: T.handle, D: T.handle, E: T.handle):
        batch, vocab_size = T.int64(), T.int64()
        sorted_prob = T.match_buffer(A, (batch, vocab_size), prob_dtype)
        cumsum_sorted = T.match_buffer(B, (batch, vocab_size), prob_dtype)
        top_p = T.match_buffer(C, (batch, 1), prob_dtype)
        top_k = T.match_buffer(D, (batch, 1), top_k_dtype)
        cutoff = T.match_buffer(E, (batch, 1), prob_dtype)
        for ax0, ax1 in T.grid(batch, vocab_size):
            with T.block("T_get_renorm_cutoff"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                if _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, 0) == 0:
                    cutoff[v_ax0, 0] = sorted_prob[v_ax0, 0]
                elif _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, v_ax1) == 1:
                    if v_ax1 + 1 == vocab_size:
                        cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1]
                    elif _cumsum_mask(cumsum_sorted, top_p, top_k, v_ax0, v_ax1 + 1) == 0:
                        cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1 + 1]

    cumsum_sorted = cumsum(sorted_prob, axis=1)

    renorm_cutoff = tensor_ir_op(
        _get_renorm_cutoff,
        "get_renorm_cutoff",
        args=[sorted_prob, cumsum_sorted, top_p, top_k],
        out=Tensor.placeholder(
            [batch, 1],
            prob_dtype,
        ),
    )

    filtered_prob = tensor_expr_op(
        lambda prob, renorm_cutoff: te.compute(
            prob.shape,
            lambda i, j: _tir.Select(prob[i, j] >= renorm_cutoff[i, 0], prob[i, j], 0.0),
            name="filter_with_top_p_top_k",
        ),
        "filter_with_top_p_top_k",
        args=[prob, renorm_cutoff],
    )
    renorm_prob = filtered_prob / sum(filtered_prob, axis=1, keepdims=True)
    return renorm_prob
