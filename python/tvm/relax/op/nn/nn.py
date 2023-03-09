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
"""Relax Neural Network (NN) operators"""
from typing import List, Optional, Tuple, Union

from tvm import DataType

from . import _ffi_api
from ...expr import Expr


def conv2d(
    data: Expr,
    weight: Expr,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    out_layout: Optional[str] = None,
    out_dtype: Optional[Union[str, DataType]] = None,
) -> Expr:
    r"""2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, conv2d takes in
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_h, kernel_w)`,
    where `kernel_h` and `kernel_w` is the lengths of the `H` and `W` kernel dimensions,
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, y, x] = \sum_{dy, dx, k}
           \mbox{data}[b, k, \mbox{strides}[0] * y  + dy, \mbox{strides}[1] * x + dx] *
           \mbox{weight}[c, k, dy, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCHW` for data and `OIHW` for weight), perform the computation,
    then convert to the out_layout.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    weight : relax.Expr
        The weight expressions.

    strides : Union[int, Tuple[int, int]]
        The strides of convolution. It is required to have length either 1 or 2.

    padding : Union[int, Tuple[int, ...]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1, 2 or 4.

    dilation : Union[int, Tuple[int, int]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length either 1 or 2.

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
    result : relax.Expr
        The computed result.
    """
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)

    return _ffi_api.conv2d(  # type: ignore
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def conv2d_transpose(
    data: Expr,
    weight: Expr,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    output_padding: Union[int, Tuple[int, int]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    data_layout: str = "NCHW",
    kernel_layout: str = "IOHW",
    out_layout: Optional[str] = None,
    out_dtype: Optional[Union[str, DataType]] = None,
) -> Expr:
    r"""Two dimensional transposed convolution operator.

    This operator is intended to be the gradient operator of conv2d. That means, if

    `out = conv2d(data, weight, strides, padding, dilation)`,

    The gradient w.r.t. data can be calculated as follows:

    `data_grad = conv2d_transpose(out_grad, weight, strides, padding, output_padding, dilation)`,

    where `output_padding` is a parameter used to determine the output shape.

    The output shape can be explained in the simple case when `data_layout == "NCHW"` and
    `kernel_layout == "IOHW"`. Suppose `data` has shape `(N, in_channel, in_h, in_w)`, `weight` has
    shape `(in_channel, out_channel, weight_h, weight_w)`, we need to assure that
    `in_channel % groups == 0`. The shape of the output will be
    `(N, out_channel * groups, out_h, out_w)`, where

    - `out_h = ((in_h - 1) * strides[0] + weight_h - 2 * padding[0] + output_padding[0])`
    - `out_w = ((in_w - 1) * strides[1] + weight_w - 2 * padding[1] + output_padding[1])`

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    weight : relax.Expr
        The weight expressions.

    strides : Union[int, Tuple[int, int]]
        The strides of convolution. It is required to have length either 1 or 2.

    padding : Union[int, Tuple[int, ...]]
        The padding of convolution on both sides of inputs before convolution.
        It is required to have length either 1, 2 or 4.

    output_padding : Union[int, Tuple[int, ...]], optional
        Used to disambiguate the output shape.

    dilation : Union[int, Tuple[int, int]]
        Specifies the dilation rate to be used for dilated convolution.
        It is required to have length either 1 or 2.

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
    result : relax.Expr
        The computed result.
    """
    # TODO: symbolic shape is not fully supported now
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)

    return _ffi_api.conv2d_transpose(  # type: ignore
        data,
        weight,
        strides,
        padding,
        output_padding,
        dilation,
        groups,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def max_pool2d(
    data: Expr,
    pool_size: Union[int, Tuple[int, int]] = (1, 1),
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    ceil_mode: bool = False,
    layout: str = "NCHW",
    out_layout: Optional[str] = None,
) -> Expr:
    r"""2D maximum pooling operator.

    This operator takes data as input and does 2D max value calculation
    with in pool_size sized window by striding defined by stride.

    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w) and pool_size (kh, kw)

    .. math::

        \mbox{out}(b, c, y, x)  = \max_{m=0, \ldots, kh-1} \max_{n=0, \ldots, kw-1}
             \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)

    Padding is applied to data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    pool_size : Union[int, Tuple[int, int]]
        The size of window for pooling. It is required to have length either 1 or 2.

    strides : Union[int, Tuple[int, int]]
        The strides of pooling. It is required to have length either 1 or 2.

    padding : Union[int, Tuple[int, ...]]
        The padding for pooling. It is required to have length either 1, 2 or 4.

    dilation : Union[int, Tuple[int, int]]
        The dilation of pooling. It is required to have length either 1 or 2.

    ceil_mode : bool
        A boolean indicating if use ceil or floor to compute the output shape.
        By using ceil, every element in the input tensor will be covered by a sliding window.

    layout : str
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    Returns
    -------
    result : Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)

    return _ffi_api.max_pool2d(  # type: ignore
        data, pool_size, strides, padding, dilation, ceil_mode, layout, out_layout
    )


def avg_pool2d(
    data: Expr,
    pool_size: Union[int, Tuple[int, int]] = (1, 1),
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    ceil_mode: bool = False,
    layout: str = "NCHW",
    out_layout: Optional[str] = None,
) -> Expr:
    r"""2D average pooling operator.

    This operator takes data as input and does 2D avarage value calculation
    with in pool_size sized window by striding defined by stride.

    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w) and pool_size (kh, kw)

    .. math::

        \mbox{out}(b, c, y, x)  = \frac{1}{kh * kw} \sum_{m=0, \ldots, kh-1}
            \sum_{n=0, \ldots, kw-1}
            \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)

    Padding is applied to data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    pool_size : Union[int, Tuple[int, int]]
        The size of window for pooling. It is required to have length either 1 or 2.

    strides : Union[int, Tuple[int, int]]
        The strides of pooling. It is required to have length either 1 or 2.

    padding : Union[int, Tuple[int, ...]]
        The padding for pooling. It is required to have length either 1, 2 or 4.

    dilation : Union[int, Tuple[int, int]]
        The dilation of pooling. It is required to have length either 1 or 2.

    ceil_mode : bool
        A boolean indicating if use ceil or floor to compute the output shape.
        By using ceil, every element in the input tensor will be covered by a sliding window.

    layout : str
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    Returns
    -------
    result : Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)

    return _ffi_api.avg_pool2d(  # type: ignore
        data, pool_size, strides, padding, dilation, ceil_mode, layout, out_layout
    )


def adaptive_avg_pool2d(
    data: Expr,
    output_size: Optional[Union[int, Tuple[int, int]]] = None,
    layout: str = "NCHW",
    out_layout: Optional[str] = None,
) -> Expr:
    r"""2D adaptive average pooling operator. This operator is experimental.

    This operator takes data as input and does 2D average value calculation
    across each window represented by WxH.


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_height, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input height and width will be used
        as output height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size x output_size) for any input (NCHW).

        If a tuple of integers (height, width) are provided for output_size,
        the output size is (N x C x height x width) for any input (NCHW).

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    output_size : Optional[Union[int, Tuple[int, int]]]
        Output height and width.
        If not specified, it will be the same as the input height and width.
        If specified, it is required to have length either 1 or 2.

    layout : str
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output. If not specified, it is the same as data_layout

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    return _ffi_api.adaptive_avg_pool2d(data, output_size, layout, out_layout)  # type: ignore


def relu(data: Expr) -> Expr:
    """Rectified linear unit.

    .. math::
        text{ReLU}(x) = max(x, 0)

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.relu(data)  # type: ignore


def gelu(data: Expr) -> Expr:
    """Gaussian Error Linear Units function

    .. math::
        text{GeLU}(x) = 0.5 * x * (1 + erf(x * 0.5**0.5))

    where :math:`erf` is the Gauss Error function.

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.gelu(data)  # type: ignore


def silu(data: Expr) -> Expr:
    """Sigmoid Linear Unit function

    .. math::
        text{SiLU}(x) = x * sigmoid(x)

    Parameters
    ----------
    data : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.silu(data)  # type: ignore


def softmax(data: Expr, axis: int = -1) -> Expr:
    r"""Computes softmax.

    .. math:: text{softmax}(x)_i = frac{exp(x_i)}{\sum_j exp(x_j)}

    Parameters
    ----------
    data: relax.Expr
        The input data to the operator.

    axis: int
        The axis to sum over when computing softmax.
        If not specified, it is by default the last axis of the input tensor.
        Supports negative indexing.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _ffi_api.softmax(data, axis)  # type: ignore


def log_softmax(data: Expr, axis: int = -1) -> Expr:
    r"""Computes log softmax.

    .. math::

        \text{log\_softmax}(x_i) = \log\left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}\right)

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: relax.Expr
        The input data to the operator.

    axis: int
        The axis to sum over when computing log softmax.
        If not specified, it is by default the last axis of the input tensor.
        Supports negative indexing.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.log_softmax(data, axis)  # type: ignore


def batch_norm(
    data: Expr,
    gamma: Expr,
    beta: Expr,
    moving_mean: Expr,
    moving_var: Expr,
    axis: int,
    epsilon: float = 1e-5,
    center: bool = True,
    scale: bool = True,
) -> Expr:
    r"""
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    .. math::

        data\_mean[i] = mean(data[:,i,:,...]) \\
        data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

        out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}}
            * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated by

    .. code:: python

        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
        moving_var = moving_var * momentum + data_var * (1 - momentum)

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is 1.
    Specifying -1 sets the channel axis to be the last item in the input shape.

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    gamma : relax.Expr
        The gamma scale factor.

    beta : relax.Expr
        The beta offset factor.

    moving_mean : relax.Expr
        Running mean of input.

    moving_var : relax.Expr
        Running variance of input.

    axis : int
        The axis along which the normalization is applied.

    epsilon : float
        Small float added to variance to avoid dividing by zero.

    center : bool
        Indicating if the beta offset will be added to the normalized tensor.

    scale : bool
        Indicating if the gamma scale will be multiplied.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.batch_norm(  # type: ignore
        data, gamma, beta, moving_mean, moving_var, axis, epsilon, center, scale
    )


def layer_norm(
    data: Expr,
    gamma: Expr,
    beta: Expr,
    axes: Union[int, List[int]],
    epsilon: float = 1e-5,
    center: bool = True,
    scale: bool = True,
) -> Expr:
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
    data : relax.Expr
        Input to which layer_norm will be applied.

    gamma : relax.Expr
        The gamma scale factor.

    beta : relax.Expr
        The beta offset factor.

    axes : Union[int, List[int]]
        The axes that along which the normalization is applied.

    epsilon : float
        Small float added to variance to avoid dividing by zero.

    center : bool
        Indicating if the beta offset will be added to the normalized tensor.

    scale : bool
        Indicating if the gamma scale will be multiplied.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axes, int):
        axes = [axes]
    return _ffi_api.layer_norm(data, gamma, beta, axes, epsilon, center, scale)  # type: ignore


def group_norm(
    data: Expr,
    gamma: Expr,
    beta: Expr,
    num_groups: int,
    channel_axis: int,
    axes: Union[int, List[int]],
    epsilon: float = 1e-5,
    center: bool = True,
    scale: bool = True,
) -> Expr:
    r"""
    Group normalization (Yuxin Wu and et al., 2016).
    Applies group normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array. First separate the input array
    into groups along the channel axis. Then apply layer normalization to each group.

    Parameters
        ----------
    data : relax.Expr
        Input to which group_norm will be applied.

    gamma : relax.Expr
        The gamma scale factor.

    beta : relax.Expr
        The beta offset factor.

    num_groups : int
        Number of groups to separate the channels into.

    channel_axis : int
        The index of the channel axis in the input data.

    axes : Union[int, List[int]]
        The axes that along which the normalization is applied (excluding the group axis)

    epsilon : float
        Small float added to variance to avoid dividing by zero.

    center : bool
        Indicating if the beta offset will be added to the normalized tensor.

    scale : bool
        Indicating if the gamma scale will be multiplied.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axes, int):
        axes = [axes]
    return _ffi_api.group_norm(  # type: ignore
        data, gamma, beta, num_groups, channel_axis, axes, epsilon, center, scale
    )


def dropout(data: Expr, rate: float = 0.5) -> Expr:
    """Applies the dropout operation to the input tensor.

    During training, each element of the input is set to zero with
    probability ``p``. The whole array is scaled by ``1/(1-p)``
    to keep the expected sum of the input unchanged.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    rate : float
        The probability for an element to be reset to 0.

    Returns
    -------
    result : relax.Expr
        The result of dropout, which is a tuple of two tensors.
        The first one is the original tensor and the second one is a
        mask tensor (1.0 where element not dropped, 0.0 where dropped)
    """
    return _ffi_api.dropout(data, rate)  # type: ignore


def cross_entropy_with_logits(predictions: Expr, labels: Expr) -> Expr:
    r"""CrossEntropy with logits between the predictions and labels.

    The shape of predictions and labels must be the same. And when ndim >= 2,
    the first dimension is regarded as the batch_size N. In this case the
    computed result will divide by N to perform a mean reduction.

    .. math::

        \text{cross\_entropy\_with\_logits}(x_i, y_i) = \frac{\sum_i -x_i \cdot y_i}{N}

    Parameters
    ----------
    predictions : relax.Expr
      The predictions.

    labels : relax.Expr
      The labels (the ground truth values).

    Returns
    -------
    result : relax.Expr
      The computed result.
    """
    return _ffi_api.cross_entropy_with_logits(predictions, labels)  # type: ignore


def attention(query: Expr, key: Expr, value: Expr, bias: Optional[Expr] = None) -> Expr:
    r"""Computes fused multi head attention.

    All input tensors are of 4-D tensors with BSNH layout.

    .. math::
        FMA(Q, K, V) = \text{Softmax}(Q @ K^T) @ V

    .. note::
        The input tensor is required to have float16 dtype

    Parameters
    ----------
    query: relax.Expr
        The input query to the operator. The layout of the input query should be
        (batch_size, seq_len, num_head, head_dim).

    key: relax.Expr
        The input key to the operator. The layout of the input key should be
        (batch_size, seq_len_kv, num_head, head_dim).

    value: relax.Expr
        The input value to the operator. The layout of the input value should be
        (batch_size, seq_len_kv, num_head, head_dim_v).

    bias: Optional[Expr]
        The optional attention bias to the operator. The layout of the attention bias should be
        (batch_size, num_head, seq_len, seq_len_kv),
        (batch_size, seq_len, seq_len_kv) or (batch_size, seq_len_kv).

    Returns
    -------
    result : relax.Expr
        The computed result. The layout of the output should be
        (batch_size, seq_len, num_head, head_dim_v).
    """
    return _ffi_api.attention(query, key, value, bias)  # type: ignore
