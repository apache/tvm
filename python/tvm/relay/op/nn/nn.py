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
# pylint: disable=invalid-name, too-many-lines
"""Neural network operations."""
from tvm.relay import expr

from ...expr import Constant, Expr, const
from ..dyn.nn import _make as _dyn_make
from . import _make
from .utils import get_pad_tuple1d, get_pad_tuple2d, get_pad_tuple3d


def conv1d(
    data,
    weight,
    strides=1,
    padding=0,
    dilation=1,
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCW",
    kernel_layout="OIW",
    out_layout="",
    out_dtype="",
):
    r"""1D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCW`
    and kernel_layout is `OIW`, conv1d takes in
    a data Tensor with shape `(batch_size, in_channels, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_size)`
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, w] = \sum_{dw, k}
           \mbox{data}[b, k, \mbox{strides}[0] * w + dw] *
           \mbox{weight}[c, k, dw]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCW` for data and `OIW` for weight), perform the computation,
    then convert to the out_layout.


    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Optional[int, Tuple[int]]
        The strides of convolution.

    padding : Optional[int, Tuple[int]]
        The padding of convolution on both sides of the input before convolution.

    dilation : Optional[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    groups : Optional[int]
        Currently unused for 1D convolution.

    channels : Optional[int]
        Number of output channels of this convolution.

    kernel_size : Optional[int, Tuple[int]]
        The spatial dimension of the convolution kernel.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    if isinstance(strides, int):
        strides = (strides,)
    if isinstance(dilation, int):
        dilation = (dilation,)
    padding = get_pad_tuple1d(padding)
    return _make.conv1d(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def conv2d(
    data,
    weight,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, conv2d takes in
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_size[0], kernel_size[1])`
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
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Optional[int, Tuple[int]]
        The strides of convolution.

    padding : Optional[int, Tuple[int]]
        The padding of convolution on both sides of inputs before convolution.

    dilation : Optional[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    groups : Optional[int]
        Number of groups for grouped convolution.

    channels : Optional[int]
        Number of output channels of this convolution.

    kernel_size : Optional[int, Tuple[int]]
        The spatial of the convolution kernel.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    # TODO enforce 4-way padding in topi/nn/conv2d after #4644 merged
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.conv2d(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def conv3d(
    data,
    weight,
    strides=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCDHW",
    kernel_layout="OIDHW",
    out_layout="",
    out_dtype="",
):
    r"""3D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCDHW`
    and kernel_layout is `OIDHW`, conv3d takes in
    a data Tensor with shape `(batch_size, in_channels, depth, height, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_size[0], kernel_size[1],
    kernel_size[2])` to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, z, y, x] = \sum_{dz, dy, dx, k}
           \mbox{data}[b, k, \mbox{strides}[0] * z  + dz, \mbox{strides}[1] * y  + dy,
           \mbox{strides}[2] * x + dx] * \mbox{weight}[c, k, dz, dy, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCDHW` for data and `OIDHW` for weight), perform the computation,
    then convert to the out_layout.


    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Optional[Tuple[int]]
        The strides of convolution.

    padding : Optional[int, Tuple[int]]
        The padding of convolution on both sides of inputs before convolution.

    dilation : Optional[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    groups : Optional[int]
        Number of groups for grouped convolution.

    channels : Optional[int]
        Number of output channels of this convolution.

    kernel_size : Optional[int, Tuple[int]]
        The spatial of the convolution kernel.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    padding = get_pad_tuple3d(padding)
    return _make.conv3d(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def contrib_conv3d_winograd_without_weight_transform(
    data,
    weight,
    tile_size,
    strides=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCDHW",
    kernel_layout="OIDHW",
    out_layout="",
    out_dtype="",
):
    r"""3D convolution with winograd algorithm.

    The basic parameters are the same as the ones in vanilla conv3d.
    It assumes the weight is pre-transformed by nn.contrib_conv3d_winograd_weight_transform

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    tile_size : int
        The Tile size of winograd. E.g. 2 for F(2x2x2, 3x3x3) and 4 for F(4x4x4, 3x3x3)

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 3-way padding to 6-way padding
    padding = get_pad_tuple3d(padding)
    return _make.contrib_conv3d_winograd_without_weight_transform(
        data,
        weight,
        tile_size,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def conv3d_transpose(
    data,
    weight,
    strides=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCDHW",
    kernel_layout="OIDHW",
    out_layout="",
    output_padding=(0, 0, 0),
    out_dtype="",
):
    r"""3D transpose convolution.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Optional[Tuple[int]]
        The strides of convolution.

    padding : Optional[int, Tuple[int]]
        The padding of convolution on both sides of inputs before convolution.

    dilation : Optional[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    groups : Optional[int]
        Number of groups for grouped convolution.

    channels : Optional[int]
        Number of output channels of this convolution.

    kernel_size : Optional[int, Tuple[int]]
        The spatial of the convolution kernel.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision conv3d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    padding = get_pad_tuple3d(padding)

    return _make.conv3d_transpose(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
    )


def conv2d_transpose(
    data,
    weight,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="IOHW",
    out_layout="",
    output_padding=(0, 0),
    out_dtype="",
):
    """Two dimensional transposed convolution operator.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Tuple[int], optional
        The strides of convolution.

    padding : Tuple[int], optional
        The padding of convolution on both sides of inputs.

    dilation : Tuple[int], optional
        Specifies the dilation rate to be used for dilated convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    output_padding : Tuple[int], optional
        Used to disambiguate the output shape.

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.conv2d_transpose(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
    )


def conv1d_transpose(
    data,
    weight,
    strides=(1,),
    padding=(0,),
    dilation=(1,),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCW",
    kernel_layout="OIW",
    out_layout="",
    output_padding=(0,),
    out_dtype="",
):
    """One dimensional transposed convolution operator.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Tuple[int], optional
        The strides of convolution.

    padding : Tuple[int], optional
        The padding of convolution on both sides of inputs.

    dilation : Tuple[int], optional
        Specifies the dilation rate to be used for dilated convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    output_padding : Tuple[int], optional
        Used to disambiguate the output shape.

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.conv1d_transpose(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
    )


def softmax(data, axis=-1):
    r"""Computes softmax.

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: tvm.relay.Expr
        The input data to the operator.

    axis: int, optional
        The axis to sum over when computing softmax

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.softmax(data, axis)


def fast_softmax(data, axis=-1):
    r"""Computes softmax.
    Use approximation to compute exponent for faster speed.

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: tvm.relay.Expr
        The input data to the operator.
    axis: int, optional
        The axis to sum over when computing softmax

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.fast_softmax(data, axis)


def log_softmax(data, axis=-1):
    r"""Computes log softmax.

    .. math::

        \text{log_softmax}(x)_i = \log \frac{exp(x_i)}{\sum_j exp(x_j)}

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: tvm.relay.Expr
        The input data to the operator.

    axis: int, optional
        The axis to sum over when computing log softmax

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.log_softmax(data, axis)


def max_pool1d(
    data,
    pool_size=(1,),
    strides=(1,),
    dilation=(1,),
    padding=(0,),
    layout="NCW",
    out_layout="",
    ceil_mode=False,
):
    r"""1D maximum pooling operator.

    This operator takes data as input and does 1D max value calculation
    with in pool_size sized window by striding defined by stride.

    In the default case, where the data_layout is `NCW`
    a data Tensor with shape `(batch_size, channels, width)`,
    to produce an output Tensor.

    The ceil_mode is used to take ceil or floor while computing out shape.
    count_include_pad indicates including or excluding padded input values in computation.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : int or tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : int or tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size,)
    if isinstance(strides, int):
        strides = (strides,)
    if isinstance(dilation, int):
        dilation = (dilation,)
    padding = get_pad_tuple1d(padding)
    return _make.max_pool1d(
        data, pool_size, strides, dilation, padding, layout, out_layout, ceil_mode
    )


def max_pool2d(
    data,
    pool_size=(1, 1),
    strides=(1, 1),
    dilation=(1, 1),
    padding=(0, 0),
    layout="NCHW",
    out_layout="",
    ceil_mode=False,
):
    r"""2D maximum pooling operator.

    This operator takes data as input and does 2D max value calculation
    with in pool_size sized window by striding defined by stride


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
    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    padding = get_pad_tuple2d(padding)
    return _make.max_pool2d(
        data, pool_size, strides, dilation, padding, layout, out_layout, ceil_mode
    )


def max_pool3d(
    data,
    pool_size=(1, 1, 1),
    strides=(1, 1, 1),
    dilation=(1, 1, 1),
    padding=(0, 0, 0),
    layout="NCDHW",
    out_layout="",
    ceil_mode=False,
):
    r"""3D maximum pooling operator.

    This operator takes data as input and does 3D max value calculation
    with in pool_size sized window by striding defined by stride.


    In the default case, where the data_layout is `NCDHW`
    a data Tensor with shape `(batch_size, channels, depth, height, width)`,
    to produce an output Tensor.

    The ceil_mode is used to take ceil or floor while computing out shape.
    count_include_pad indicates including or excluding padded input values in computation.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    padding = get_pad_tuple3d(padding)
    return _make.max_pool3d(
        data, pool_size, strides, dilation, padding, layout, out_layout, ceil_mode
    )


def avg_pool1d(
    data,
    pool_size=(1,),
    strides=(1,),
    dilation=(1,),
    padding=(0,),
    layout="NCW",
    out_layout="",
    ceil_mode=False,
    count_include_pad=False,
):
    r"""1D average pooling operator.

    This operator takes data as input and does 1D average value calculation
    with in pool_size sized window by striding defined by stride

    In the default case, where the data_layout is `NCW`
    a data Tensor with shape `(batch_size, channels, width)`,
    to produce an output Tensor.

    The ceil_mode is used to take ceil or floor while computing out shape.
    count_include_pad indicates including or excluding padded input values in computation.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : int or tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : int or tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    count_include_pad : bool, optional
        To include padding to compute the average.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size,)
    if isinstance(strides, int):
        strides = (strides,)
    if isinstance(dilation, int):
        dilation = (dilation,)
    padding = get_pad_tuple1d(padding)
    return _make.avg_pool1d(
        data,
        pool_size,
        strides,
        dilation,
        padding,
        layout,
        out_layout,
        ceil_mode,
        count_include_pad,
    )


def avg_pool2d(
    data,
    pool_size=(1, 1),
    strides=(1, 1),
    dilation=(1, 1),
    padding=(0, 0),
    layout="NCHW",
    out_layout="",
    ceil_mode=False,
    count_include_pad=False,
):
    r"""2D average pooling operator.

    This operator takes data as input and does 2D average value calculation
    with in pool_size sized window by striding defined by stride


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w), pool_size (kh, kw)

    .. math::

        \mbox{out}(b, c, y, x)  = \frac{1}{kh * kw} \sum_{m=0}^{kh-1} \sum_{n=0}^{kw-1}
             \mbox{data}(b, c, \mbox{stride}[0] * y + m, \mbox{stride}[1] * x + n)

    Padding is applied to data before the computation.
    ceil_mode is used to take ceil or floor while computing out shape.
    count_include_pad indicates including or excluding padded input values in computation.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    count_include_pad : bool, optional
        To include padding to compute the average.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    padding = get_pad_tuple2d(padding)
    return _make.avg_pool2d(
        data,
        pool_size,
        strides,
        dilation,
        padding,
        layout,
        out_layout,
        ceil_mode,
        count_include_pad,
    )


def avg_pool3d(
    data,
    pool_size=(1, 1, 1),
    strides=(1, 1, 1),
    dilation=(1, 1, 1),
    padding=(0, 0, 0),
    layout="NCDHW",
    out_layout="",
    ceil_mode=False,
    count_include_pad=False,
):
    r"""3D average pooling operator.

    This operator takes data as input and does 3D average value calculation
    with in pool_size sized window by striding defined by stride


    In the default case, where the data_layout is `NCDHW`
    a data Tensor with shape `(batch_size, channels, depth, height, width)`,
    to produce an output Tensor.

    The ceil_mode is used to take ceil or floor while computing out shape.
    count_include_pad indicates including or excluding padded input values in computation.
    This operator accepts data layout specification.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    dilation : int or tuple of int, optional
        The dilation of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    count_include_pad : bool, optional
        To include padding to compute the average.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size, pool_size)
    if isinstance(strides, int):
        strides = (strides, strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    padding = get_pad_tuple3d(padding)
    return _make.avg_pool3d(
        data,
        pool_size,
        strides,
        dilation,
        padding,
        layout,
        out_layout,
        ceil_mode,
        count_include_pad,
    )


def max_pool2d_grad(
    out_grad,
    data,
    pool_size=(1, 1),
    strides=(1, 1),
    padding=(0, 0),
    layout="NCHW",
    out_layout="",
    ceil_mode=False,
):
    r"""Gradient of 2D maximum pooling operator.

    This operator takes out_grad and data as input and calculates gradient of max_pool2d.

    Parameters
    ----------
    out_grad : tvm.relay.Expr
        The output gradient

    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.max_pool2d_grad(
        out_grad, data, pool_size, strides, padding, layout, out_layout, ceil_mode
    )


def avg_pool2d_grad(
    out_grad,
    data,
    pool_size=(1, 1),
    strides=(1, 1),
    padding=(0, 0),
    layout="NCHW",
    out_layout="",
    ceil_mode=False,
    count_include_pad=False,
):
    r"""Gradient of 2D average pooling operator.

    This operator takes out_grad and data as input and calculates gradient of avg_pool2d.

    Parameters
    ----------
    out_grad : tvm.relay.Expr
        The output gradient

    data : tvm.relay.Expr
        The input data to the operator.

    pool_size : int or tuple of int, optional
        The size of window for pooling.

    strides : tuple of int, optional
        The strides of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    count_include_pad : bool, optional
        To include padding to compute the average.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.avg_pool2d_grad(
        out_grad,
        data,
        pool_size,
        strides,
        padding,
        layout,
        out_layout,
        ceil_mode,
        count_include_pad,
    )


def global_max_pool2d(data, layout="NCHW", out_layout=""):
    r"""2D global maximum pooling operator.

    This operator takes data as input and does 2D max value calculation
    across each window represented by WxH.


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w)

    .. math::

        \mbox{out}(b, c, 1, 1)  = \max_{m=0, \ldots, h} \max_{n=0, \ldots, w}
             \mbox{data}(b, c, m, n)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.global_max_pool2d(data, layout, out_layout)


def global_avg_pool2d(data, layout="NCHW", out_layout=""):
    r"""2D global average pooling operator.

    This operator takes data as input and does 2D average value calculation
    across each window represented by WxH.


    In the default case, where the data_layout is `NCHW`
    a data Tensor with shape `(batch_size, in_channels, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, h, w)

    .. math::

        \mbox{out}(b, c, 1, 1)  = \frac{1}{h * w} \sum_{m=0}^{h-1} \sum_{n=0}^{w-1}
             \mbox{data}(b, c, m, n)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    out_layout : Optional[str]
        Layout of the output

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.global_avg_pool2d(data, layout, out_layout)


def upsampling(
    data, scale_h=1, scale_w=1, layout="NCHW", method="nearest_neighbor", align_corners=False
):
    """Upsampling.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, h*scale_h, w*scale_w)

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("bilinear", "nearest_neighbor", "bicubic")

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    scale_h : tvm.relay.Expr or int or float
        The scale factor for height upsampling.

    scale_w : tvm.relay.Expr or int or float
        The scale factor for width upsampling.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [nearest_neighbor, bilinear, bicubic].

    align_corners : bool, optional
        Whether to keep corners in proper place.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(scale_h, Constant):
        scale_h = scale_h.data.numpy().item()
    if isinstance(scale_w, Constant):
        scale_w = scale_w.data.numpy().item()
    if isinstance(scale_h, Expr) or isinstance(scale_w, Expr):
        if not isinstance(scale_h, Expr):
            scale_h = const(scale_h, "float64")
        if not isinstance(scale_w, Expr):
            scale_w = const(scale_w, "float64")
        return _dyn_make.upsampling(data, scale_h, scale_w, layout, method, align_corners)
    return _make.upsampling(data, scale_h, scale_w, layout, method, align_corners)


def upsampling3d(
    data,
    scale_d=1,
    scale_h=1,
    scale_w=1,
    layout="NCDHW",
    method="nearest_neighbor",
    coordinate_transformation_mode="half_pixel",
):
    """3D Upsampling.

    This operator takes data as input and does 3D scaling to the given scale factor.
    In the default case, where the data_layout is `NCDHW`
    with data of shape (n, c, d, h, w)
    out will have a shape (n, c, d*scale_d, h*scale_h, w*scale_w)

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("trilinear", "nearest_neighbor")

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    scale_d : tvm.relay.Expr
        The scale factor for depth upsampling.

    scale_h : tvm.relay.Expr
        The scale factor for height upsampling.

    scale_w : tvm.relay.Expr
        The scale factor for width upsampling.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [nearest_neighbor, trilinear].

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        Available options are "half_pixel", "align_corners" and "asymmetric".

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(scale_d, Constant):
        scale_d = scale_d.data.numpy().item()
    if isinstance(scale_h, Constant):
        scale_h = scale_h.data.numpy().item()
    if isinstance(scale_w, Constant):
        scale_w = scale_w.data.numpy().item()
    if isinstance(scale_d, Expr) or isinstance(scale_h, Expr) or isinstance(scale_w, Expr):
        if not isinstance(scale_d, Expr):
            scale_d = const(scale_d, "float64")
        if not isinstance(scale_h, Expr):
            scale_h = const(scale_h, "float64")
        if not isinstance(scale_w, Expr):
            scale_w = const(scale_w, "float64")
        return _dyn_make.upsampling3d(
            data, scale_d, scale_h, scale_w, layout, method, coordinate_transformation_mode
        )
    return _make.upsampling3d(
        data, scale_d, scale_h, scale_w, layout, method, coordinate_transformation_mode
    )


def batch_flatten(data):
    """BatchFlatten.

    This operator flattens all the dimensions except for the batch dimension.
    which results a 2D output.

    For data with shape ``(d1, d2, ..., dk)``
    batch_flatten(data) returns reshaped output of shape ``(d1, d2*...*dk)``.


    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    Returns
    -------
    result : tvm.relay.Expr
        The Flattened result.
    """
    return _make.batch_flatten(data)


def bias_add(data, bias, axis=1):
    """add_bias operator.

    Add 1D bias to the axis of data.
    This function is a special case of add which allows
    inference of shape of the bias from data.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    bias : tvm.relay.Expr
        The bias to be added.

    axis : int, optional
        The axis to add the bias.

    Returns
    -------
    result : tvm.relay.Expr
        The final result.
    """
    return _make.bias_add(data, bias, axis)


def matmul(tensor_a, tensor_b, units=None, out_dtype="", transpose_a=False, transpose_b=False):
    """Matmul operator.
    Applies a linear transformation. The A & B can be transposed.

    .. math::

        `C = A * B`

    Parameters
    ----------
    data : tvm.relay.Expr
        The first input of the operator,
        of shape `(d_1, d_2, ..., d_n, units_in)` or `(d_1, d_2, ..., units_in, d_n)`.

    weight : tvm.relay.Expr
        The second input expressions, 2-D matrix,
        of shape `(units_in, units)` or `(units, units_in)`.

    units : Optional[int]
        Number of hidden units of the matmul transformation.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision matmul,
        of shape `(d_1, d_2, ..., d_n, units)`.

    transpose_a : Optional[bool] = False
        Whether the data tensor is in transposed format.

    transpose_b : Optional[bool] = False
        Whether the weight tensor is in transposed format.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # Since currently `nn.dense` has better topi schedule support, will prefer to use `dense`
    # rather than `matmul` for better compatibility
    if not transpose_a and transpose_b:
        # TODO(jcf94): Remove this when `nn.matmul` is finnaly ready
        return dense(tensor_a, tensor_b, units, out_dtype)
    return _make.matmul(tensor_a, tensor_b, units, out_dtype, transpose_a, transpose_b)


def dense(data, weight, units=None, out_dtype=""):
    """Dense operator.
    Applies a linear transformation

    .. math::

    `Y = X * W^T`

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator,
        of shape `(d_1, d_2, ..., d_n, units_in)`.

    weight : tvm.relay.Expr
        The weight expressions, 2-D matrix,
        of shape `(units, units_in)`.

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense,
        of shape `(d_1, d_2, ..., d_n, units)`.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.dense(data, weight, units, out_dtype)


def contrib_dense_pack(data, weight, weight_layout="NC", units=None, out_dtype=""):
    """Dense operator.
    Applies a linear transformation with packed weight

    .. math::

    `Y = X * W^T`

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator,
        of shape `(batch, units_in)`.

    weight : tvm.relay.Expr
        The transformed weight expressions, 3-D matrix,
        of shape `(units // pack_weight_tile, units_in, pack_weight_tile)`.

    weight_layout: str
        The layout of weight, such as "NC" or "NC8n".

    units : int, optional
        Number of hidden units of the dense transformation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.contrib_dense_pack(data, weight, weight_layout, units, out_dtype)


def fifo_buffer(data, buffer, axis):
    """FIFO buffer to enable computation reuse in CNNs with sliding indow input

    Compute equivalent of

    .. code-block:: python

        concat(buffer, data, axis=axis)
        .slice_axis(axis=axis,
                    begin=data.shape[axis],
                    end=data.shape[axis]+buffer.shape[axis])

    Useful for

    * Encoding explicit re-use of computation in convolution ops operated on a sliding window input
    * Implementing a FIFO queue to cache intermediate results, e.g. as in Fast WaveNet.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data
    buffer : tvm.relay.Expr
        Previous value of the FIFO buffer
    axis : int
        Specify which axis should be used for buffering

    Returns
    -------
    result : tvm.relay.Expr
        Updated value for the buffer
    """
    return _make.fifo_buffer(data, buffer, axis)


def relu(data):
    """Rectified linear unit.

    .. math::
       out = max(x, 0)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.relu(data)


def leaky_relu(data, alpha=0.01):
    """This operator takes data as input and does Leaky version
    of a Rectified Linear Unit.

    .. math::

        `y = x > 0 ? x : alpha * x`

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    alpha : float
        Slope coefficient for the negative half axis.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.leaky_relu(data, alpha)


def prelu(data, alpha, axis=1):
    """This operator takes data as input and does Leaky version
    of a Rectified Linear Unit.

    .. math::

        y = x > 0 ? x : alpha * x

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    alpha : tvm.relay.Expr
        Slope coefficient for the negative half axis.

    axis : int, optional
        Specify which shape axis the channel is specified.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.prelu(data, alpha, axis)


def pad(data, pad_width, pad_value=0, pad_mode="constant"):
    r"""Padding

    This operator takes in a tensor and pads each axis by the specified
    widths using the specified value.

    Parameters
    ----------
    data: tvm.relay.Expr
        The input data to the operator
    pad_width: tuple of <tuple of <int>>, or tvm.relay.Expr, required
        Number of values padded to the edges of each axis, in the format
        of ((before_1, after_1), ..., (before_N, after_N))
    pad_value: float, or tvm.relay.Expr, optional, default=0
        The value used for padding
    pad_mode: 'constant', 'edge', 'reflect'
        'constant' pads with constant_value pad_value
        'edge' pads using the edge values of the input array
        'reflect' pads by reflecting values with respect to the edge
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(pad_width, Constant):
        pad_width = [list(i) for i in pad_width.data.numpy()]
    if not isinstance(pad_value, Expr):
        pad_value = const(pad_value)
    if isinstance(pad_width, Expr):
        return _dyn_make.pad(data, pad_width, pad_value, pad_mode)
    return _make.pad(data, pad_width, pad_value, pad_mode)


def dilate(data, strides, dilation_value=0.0):
    """Dilate data with given dilation value (0 by default).

    Parameters
    ----------
    data : tvm.relay.Expr
        n-D, can be any layout.

    strides : tuple of <int>
        Dilation stride on each dimension, 1 means no dilation.

    dilation_value : int/float, optional
        Value used to dilate the input.

    Returns
    -------
    Output : tvm.relay.Expr
        The computed result
    """
    return _make.dilate(data, strides, dilation_value)


def mirror_pad(data, pad_width, mode="SYMMETRIC"):
    r"""MirrorPadding

    This operator takes in a tensor and pads each axis by the specified
    widths using mirroring of the border pixels.

    Parameters
    ----------
    data: tvm.relay.Expr
        The input data to the operator
    pad_width: tuple of <tuple of <int>>, required
        Number of values padded to the edges of each axis, in the format
        of ((before_1, after_1), ..., (before_N, after_N))
    mode: string, optional, default='SYMMETRIC'
        What type of mirroring to use, must be SYMMETRIC or REFLECT.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.mirror_pad(data, pad_width, mode)


def lrn(data, size=5, axis=1, bias=2, alpha=0.00001, beta=0.75):
    """This operator takes data as input and does local response normalization.

    Normalize the input in a local region across or within feature maps.
    Each input value is divided by (data / (bias + (alpha * sum_data ^2 /size))^beta)
    where n is the size of each local region, and the sum is taken over the region
    centered at that value (zero padding is added where necessary).

    .. math::
        (data / (bias + (alpha * sum_data ^2 /size))^beta)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    size : int, optional
        The size of the local region to be considered for normalization.

    axis : int, optional
        Input data layout channel axis. Default value is 1 for NCHW format

    bias : float, optional
        The offset parameter to avoid dividing by 0.

    alpha : float, optional
        The scaling parameter.

    beta : float, optional
        The exponent parameter.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.lrn(data, size, axis, alpha, beta, bias)


def l2_normalize(data, eps, axis=None):
    """Perform L2 normalization on the input data

    .. math::
        y(i, j) = x(i, j) / sqrt(max(sum(x^2), eps))

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    eps : float
        epsilon value

    axis : list of int, optional
        axis over the normalization applied

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.l2_normalize(data, eps, axis)


def dropout(data, rate=0.5):
    """Applies the dropout operation to the input array.

    During training, each element of the input is set to zero with
    probability ``p``. The whole array is rescaled by ``1/(1-p)``
    to keep the expected sum of the input unchanged.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    rate : float, optional (default=0.5)
        The probability for an element to be reset to 0.

    Returns
    -------
    result : tvm.relay.Expr
        The result of dropout
    """
    return expr.TupleWrapper(dropout_raw(data, rate), 2)[0]


def dropout_raw(data, rate=0.5):
    """Applies the dropout operation to the input array.

    During training, each element of the input is set to zero with
    probability ``p``. The whole array is rescaled by ``1/(1-p)``
    to keep the expected sum of the input unchanged.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    rate : float, optional (default=0.5)
        The probability for an element to be reset to 0.

    Returns
    -------
    result : tvm.relay.Expr
        The result of dropout
    """
    return _make.dropout(data, rate)


def batch_norm(
    data, gamma, beta, moving_mean, moving_var, axis=1, epsilon=1e-5, center=True, scale=True
):
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
    data : tvm.relay.Expr
        Input to which batch_norm will be applied.

    gamma : tvm.relay.Expr
        The gamma scale factor.

    beta : tvm.relay.Expr
        The beta offset factor.

    moving_mean : tvm.relay.Expr
        Running mean of input,

    moving_var : tvm.relay.Expr
        Running variance of input.

    axis : int, optional, default=1
        Specify along which shape axis the channel is specified.

    epsilon : double, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, optional, default=True
        If true, multiply by gamma. If False, gamma is not used.
        When the next layer is piecewise linear (also e.g. nn.relu),
        this can be disabled since the scaling will be done by the next layer.

    Returns
    -------
    result : relay.Tuple([tvm.relay.Expr, tvm.relay.Expr, tvm.relay.Expr])
        Tuple of normed data (same shape as input),
        new running mean (k-length vector),
        and new running variance (k-length vector)
    """
    result = _make.batch_norm(
        data, gamma, beta, moving_mean, moving_var, axis, epsilon, center, scale
    )
    return expr.TupleWrapper(result, 3)


def instance_norm(data, gamma, beta, axis=1, epsilon=1e-5, center=True, scale=True):
    r"""
    Instance Normalization (Ulyanov and et al., 2016)
    Applies instance normalization to the n-dimensional input array.

    .. math::

        out = \frac{data - mean(data)}{\sqrt{var(data)+\epsilon}}
            * gamma + beta

    The instance normalization is similar to batch normalization, but unlike
    batch normalization, the mean and var are calculated per-dimension
    separately for each object(instance) in a mini-batch, not over a batch.
    And the same normalization is applied both at test and train time.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*.

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel'.  The default is 1. Specifying -1 sets the channel axis
    to be the last item in the input shape.

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : tvm.relay.Expr
        Input to which instance_norm will be applied.

    gamma : tvm.relay.Expr
        The gamma scale factor.

    beta : tvm.relay.Expr
        The beta offset factor.

    axis : int, optional, default=1
        Specify along which shape axis the channel is specified.

    epsilon : double, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, optional, default=True
        If True, multiply by gamma. If False, gamma is not used.

    Returns
    -------
    result : tvm.relay.Expr
        The normalized data.

    .. _`Instance Normalization: The Missing Ingredient for Fast Stylization`:
        https://arxiv.org/abs/1607.08022
    """
    return _make.instance_norm(data, gamma, beta, axis, epsilon, center, scale)


def layer_norm(data, gamma, beta, axis=-1, epsilon=1e-5, center=True, scale=True):
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
    data : tvm.relay.Expr
        Input to which layer_norm will be applied.

    gamma : tvm.relay.Expr
        The gamma scale factor.

    beta : tvm.relay.Expr
        The beta offset factor.

    axis : int, optional, default=-1
        The axis that should be normalized, typically the axis of the channels.

    epsilon : double, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, optional, default=True
        If True, multiply by gamma. If False, gamma is not used.

    Returns
    -------
    result : tvm.relay.Expr
        The normalized data.
    """
    return _make.layer_norm(data, gamma, beta, axis, epsilon, center, scale)


def group_norm(data, gamma, beta, num_groups, axis=1, epsilon=1e-5, center=True, scale=True):
    r"""
    Group normalization normalizes over group of channels for each training examples.
    We can say that, Group Norm is in between Instance Norm and Layer Norm. When we put
    all the channels into a single group, group normalization becomes Layer normalization.
    And, when we put each channel into different groups it becomes Instance normalization

    https://arxiv.org/pdf/1803.08494.pdf

    Applies group normalization to the n-dimensional input array by seperating the input channels
    into 'num_groups' groups, each containing 'num_channels / num_groups' channels.
    The mean and standard-deviation are calculated separately over the each group. gamma and
    beta are learnable per-channel affine transform parameter vectors of size num_channels.

    .. math::

        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}}
            * gamma + beta

    Unlike batch normalization, the mean and var are computed along a group of channels.

    If the input has size k on axis 1, then both gamma and beta have shape (k,).

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : tvm.relay.Expr
        Input to which group_norm will be applied.

    gamma : tvm.relay.Expr
        The gamma scale factor.

    beta : tvm.relay.Expr
        The beta offset factor.

    num_groups : int
        The number of groups to separate the channels into.

    axis : int, optional, default=1
        The axis of the channels.

    epsilon : double, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, optional, default=True
        If True, multiply by gamma. If False, gamma is not used.

    Returns
    -------
    result : tvm.relay.Expr
        The normalized data.
    """
    return _make.group_norm(data, gamma, beta, num_groups, axis, epsilon, center, scale)


def batch_matmul(tensor_a, tensor_b, out_dtype="", transpose_a=False, transpose_b=True):
    r"""
    Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

    Both `tensor_a` and `tensor_b` can be transposed. For legacy reason, we use NT format
    (transpose_a=False, transpose_b=True) by default.

    .. math::

        \mbox{batch_matmul}(A, B)[i, :, :] = \mbox{matmul}(A[i, :, :], B[i, :, :])

    Parameters
    ----------
    tensor_a : tvm.relay.Expr
        The first input.

    tensor_b : tvm.relay.Expr
        The second input.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.batch_matmul(tensor_a, tensor_b, out_dtype, transpose_a, transpose_b)


# pylint: disable=no-else-return,inconsistent-return-statements
def sparse_dense(dense_mat, sparse_mat, sparse_lhs=False):
    r"""
    Computes the matrix multiplication of `dense_mat` and `sparse_mat`, where `dense_mat` is
    a dense matrix and `sparse_mat` is a sparse (either BSR or CSR) namedtuple with
    fields `data`, `indices`, and `indptr`.

    \if sparse_lhs=False:
        .. math::

            \mbox{sparse_dense}(dense_mat, sparse_mat)[m, n]
            = \mbox{matmul}(D, \mbox{as_dense}(S)^T)[m, n]

    \if sparse_lhs=True:
        .. math::

            \mbox{sparse_dense}(dense_mat, sparse_mat)[m, n]
            = \mbox{matmul}(\mbox{as_dense}(S), (D)^T)[m, n]

    where `as_dense` returns dense equivalent of the given S(sparse matrix)
    while performing matmul with given D(dense matrix).

    See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    and
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.bsr_matrix.html
    for more detail on the sparse matrix representation.

    Parameters
    ----------
    dense_mat : tvm.relay.Expr
        The input dense matrix for the matrix multiplication

    sparse_mat : Union[namedtuple, Tuple[ndarray, ndarray, ndarray]].
        The input sparse matrix for the matrix multiplication.

    sparse_lhs : bool, optional
        Indicates whether lhs or rhs matrix is sparse. Default value is False.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    if hasattr(sparse_mat, "indices"):
        return _make.sparse_dense(
            dense_mat, sparse_mat.data, sparse_mat.indices, sparse_mat.indptr, sparse_lhs
        )
    else:
        return _make.sparse_dense(
            dense_mat, sparse_mat[0], sparse_mat[1], sparse_mat[2], sparse_lhs
        )


def sparse_transpose(x):
    r"""
    Computes the fast matrix transpose of x,
    where x is a sparse tensor in CSR format (represented as a namedtuple
    with fields `data`, `indices`, and `indptr`).

    ** Currently only support Square Matrices **

    .. math::

        \mbox{sparse_transpose}(x)[n, n] = (x^T)[n, n]

    Please refer to https://github.com/scipy/scipy/blob/v1.3.0/scipy/sparse/csr.py
    for the algorithm implemented in this operator.

    Parameters
    ----------
    x : Union[namedtuple, Tuple[ndarray, ndarray, ndarray]].
        The sparse weight matrix for the fast matrix transpose.

    Returns
    -------
    result : relay.Tuple([tvm.relay.Expr, tvm.relay.Expr, tvm.relay.Expr])
        Tuple of output sparse tensor (same shape and format as input),
        i.e. if CSR then output is in ([data, indices, indptr]) form
    """
    if hasattr(x, "indices"):
        return expr.TupleWrapper(_make.sparse_transpose(x.data, x.indices, x.indptr), 3)
    return expr.TupleWrapper(_make.sparse_transpose(x[0], x[1], x[2]), 3)


# pylint: disable=no-else-return,inconsistent-return-statements
def sparse_add(dense_mat, sparse_mat):
    r"""
    Computes the matrix addition of `dense_mat` and `sparse_mat`, where `dense_mat` is
    a dense matrix and `sparse_mat` is a sparse (CSR) namedtuple with
    fields `data`, `indices`, and `indptr`.

    .. math::

        \mbox{sparse_add}(dense_mat, sparse_mat)[m, n] = \mbox{add}(\mbox{as_dense}(S), (D))[m, n]

    where `as_dense` returns dense equivalent of the given S(sparse matrix)
    while performing addition with given D(dense matrix).

    Parameters
    ----------
    dense_mat : tvm.relay.Expr
        The input dense matrix for the matrix addition

    sparse_mat : Union[namedtuple, Tuple[ndarray, ndarray, ndarray]].
        The input sparse matrix(CSR) for the matrix addition.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.

    Examples
    -------
    .. code-block:: python

        dense_data = [[ 3.,   4.,   4. ]
                      [ 4.,  2.,  5. ]]
        sparse_data = [4., 8.]
        sparse_indices =[0, 2]
        sparse_indptr =[0, 1, 2]

        output = relay.sparse_add(dense_data, sparse_data, sparse_indices, sparse_indptr)

        output = [[ 7.,   4.,   4. ]
                  [ 4.,  2.,  13. ]]
    """
    if hasattr(sparse_mat, "indices"):
        return _make.sparse_add(dense_mat, sparse_mat.data, sparse_mat.indices, sparse_mat.indptr)
    else:
        return _make.sparse_add(dense_mat, sparse_mat[0], sparse_mat[1], sparse_mat[2])


def contrib_conv2d_winograd_without_weight_transform(
    data,
    weight,
    tile_size,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""2D convolution with winograd algorithm.

    The basic parameters are the same as the ones in vanilla conv2d.
    It assumes the weight is pre-transformed by nn.contrib_conv2d_winograd_weight_transform

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    tile_size : int
        The Tile size of winograd. E.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.contrib_conv2d_winograd_without_weight_transform(
        data,
        weight,
        tile_size,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def contrib_conv2d_gemm_without_weight_transform(
    data,
    weight,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""2D convolution with gemm algorithm.

    The basic parameters are the same as the ones in vanilla conv2d.
    It assumes the weight is pre-transformed by nn.contrib_conv2d_gemm_weight_transform

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.contrib_conv2d_gemm_without_weight_transform(
        data,
        weight,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def contrib_conv2d_nchwc(
    data,
    kernel,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW8c",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""Variant of 2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output, following a specialized
    NCHWc data layout.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.contrib_conv2d_NCHWc(
        data,
        kernel,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def contrib_depthwise_conv2d_nchwc(
    data,
    kernel,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW8c",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""Variant of 2D depthwise convolution.

    This operator takes the weight as the depthwise convolution kernel
    and depthwise convolves it with data to produce an output, following a specialized
    NCHWc data layout.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.contrib_depthwise_conv2d_NCHWc(
        data,
        kernel,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def contrib_conv2d_winograd_weight_transform(weight, tile_size):
    r"""Weight Transformation part for 2D convolution with winograd algorithm.

    We separate this as a single op to enable pre-compute for inference.
    Use this together with nn.contrib_conv2d_winograd_without_weight_transform

    Parameters
    ----------
    weight : tvm.relay.Expr
        The weight expressions.

    tile_size : int
        The Tile size of winograd. E.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.contrib_conv2d_winograd_weight_transform(weight, tile_size)


def contrib_conv2d_gemm_weight_transform(weights, tile_rows, tile_cols):
    r"""Weight Transformation part for 2D convolution with gemm algorithm.

    We separate this as a single op to enable pre-compute for inference.
    Use this together with nn.contrib_conv2d_gemm_without_weight_transform

    Parameters
    ----------
    weights : tvm.relay.Expr
        The weight expressions.
    tile_rows: int
        Tile rows of the weight transformation for ConvGemm.
    tile_cols: int
       Tile columns of the weight transformation for ConvGemm.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.contrib_conv2d_gemm_weight_transform(weights, tile_rows, tile_cols)


def contrib_conv3d_winograd_weight_transform(weight, tile_size):
    r"""Weight Transformation part for 3D convolution with winograd algorithm.

    We separate this as a single op to enable pre-compute for inference.
    Use this together with nn.contrib_conv3d_winograd_without_weight_transform

    Parameters
    ----------
    weight : tvm.relay.Expr
        The weight expressions.

    tile_size : int
        The Tile size of winograd. E.g. 2 for F(2x2x2, 3x3x3) and 4 for F(4x4x4, 3x3x3)

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.contrib_conv3d_winograd_weight_transform(weight, tile_size)


def contrib_conv2d_winograd_nnpack_weight_transform(weight, convolution_algorithm, out_dtype=""):
    r"""Weight Transformation part for 2D convolution with winograd algorithm.

    We separate this as a single op to enable pre-compute for inference.
    Use this together with nn.contrib_conv2d_winograd_without_weight_transform

    Parameters
    ----------
    weight : tvm.relay.Expr
        The weight expressions.

    convolution_algorithm : int
        The Tile size of winograd. E.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.contrib_conv2d_winograd_nnpack_weight_transform(
        weight, convolution_algorithm, out_dtype
    )


def deformable_conv2d(
    data,
    offset,
    weight,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    deformable_groups=1,
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="",
):
    r"""Deformable 2d convolution.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    offset : tvm.relay.Expr
        The offset expressions.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    deformable_groups : int, optional
        Number of deformable groups.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.

    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.deformable_conv2d(
        data,
        offset,
        weight,
        strides,
        padding,
        dilation,
        deformable_groups,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def bitpack(data, bits=1, pack_axis=1, bit_axis=2, pack_type="uint32", name="BitPack"):
    """Tensor packing for bitserial operations.

    The values along the input tensor's pack_axis are quantized
    and packed together into the specified pack_type in a new bit axis.

    For example, consider bitpacking with data to be a tensor with shape `[1, 64, 128, 128]`,
    pack_axis=1, bit_axis=4, pack_type=uint8, and bits=2. The output in this case will
    be of shape `[1, 8, 128, 128, 2]`. The dimension of axis 1 has been reduced by a factor
    of 8 since each value is packed into an 8-bit uint8. Axis 4 is now two bitplanes
    representing the quantized value of the incoming data. The output tensor is now
    ready to be used in a bitserial operation.

    Parameters
    ----------
    data : tvm.relay.expr
        The incoming tensor to be packed.

    bits : int
        Number of bits that should be packed.

    pack_axis : int
        Axis that should be decomposed and packed.

    bit_axis : int
        New axis containing bitplane.

    pack_type : str
        Datatype to pack bits into.

    name : str, optional
        Name of the operation.

    Returns
    -------
    result : tvm.relay.Expr
        The packed tensor.
    """
    return _make.bitpack(data, bits, pack_axis, bit_axis, pack_type, name)


def bitserial_conv2d(
    data,
    weight,
    strides=(1, 1),
    padding=(0, 0),
    channels=None,
    kernel_size=(3, 3),
    activation_bits=1,
    weight_bits=1,
    data_layout="NCHW",
    kernel_layout="OIHW",
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    r"""2D convolution using bitserial computation.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    activation_bits : int
        Number of bits to pack for activations.

    weight_bits : int
        Number of bits to pack for weights.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel

    pack_dtype: str, optional
        Datatype to pack bits into.

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.bitserial_conv2d(
        data,
        weight,
        strides,
        padding,
        channels,
        kernel_size,
        activation_bits,
        weight_bits,
        data_layout,
        kernel_layout,
        pack_dtype,
        out_dtype,
        unipolar,
    )


def bitserial_dense(
    data,
    weight,
    units=None,
    data_bits=1,
    weight_bits=1,
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    """Bitserial Dense operator.
    Applies matrix multiplication of two quantized matrices
    using a fast bitserial algorithm.

    .. math::

    `Y = X * W`

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    units : int, optional
        Number of hidden units of the dense transformation.

    data_bits : int
        Number of bits incoming tensor should be packed with.

    weight_bits : int
        Number of bits weight tensor should be packed with.

    pack_dtype : str, optional
        Datatype to pack individual bits into before computation.

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense.

    unipolar : bool, optional
        Whether to use unipolar or bipolar quantization for inputs.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.bitserial_dense(
        data, weight, units, data_bits, weight_bits, pack_dtype, out_dtype, unipolar
    )


def cross_entropy(predictions, targets):
    """CrossEntropy without logits.

    Parameters
    ----------
    predictions : tvm.relay.Expr
      The predictions.

    targets : tvm.relay.Expr
      The targets.

    Returns
    -------
    result : tvm.relay.Expr
      The computed result.
    """
    return _make.cross_entropy(predictions, targets)


def cross_entropy_with_logits(predictions, targets):
    """CrossEntropy with logits.

    Parameters
    ----------
    predictions : tvm.relay.Expr
      The predictions.

    targets : tvm.relay.Expr
      The targets.

    Returns
    -------
    result : tvm.relay.Expr
      The computed result.
    """
    return _make.cross_entropy_with_logits(predictions, targets)


def nll_loss(predictions, targets, weights, reduction="mean", ignore_index=-100):
    """Negative log likelihood loss.

    output{n, i_1, i_2, ..., i_k} = -p * w
      where t = target{n, i_1, i_2, ..., i_k}
            p = predictions{n, t, i_1, i_2, i_k}
            w = weights{n, i_1, i_2, ..., i_k} if t != ignore_index else 0

    result = reduction(output)

    Parameters
    ----------
    predictions : tvm.relay.Expr
      The predictions.

    targets : tvm.relay.Expr
      The target value of each prediction.

    weights : tvm.relay.Expr
      The weight of each target value.

    reduction : string
      The reduction method to apply to the output.
      Possible values are "mean", "sum" and "none".

    ignore_index : int
      The target value to ignore.

    Returns
    -------
    result : tvm.relay.Expr
      The computed result.
    """
    return _make.nll_loss(predictions, targets, weights, reduction, ignore_index)


def depth_to_space(data, block_size, layout="NCHW", mode="DCR"):
    """Convert channels into spatial blocks.

    Parameters
    ----------
    data : tvm.relay.Expr
        Input data with channels divisible by block_size**2

    block_size : int
        Size of blocks to convert channels into.

    layout : string
        One of NCHW or NHWC, indicates channel axis.

    mode : string
        One of DCR or CDR, indicates which order channels
        are accessed in.

    Returns
    -------
    result : tvm.relay.Expr
        Tensor with shape [in_batch, in_channel / block_size * block_size,
                           in_height * block_size, in_width * block_size]
    """
    return _make.depth_to_space(data, block_size, layout, mode)


def space_to_depth(data, block_size, layout="NCHW"):
    """Convert spatial blocks into channels.

    Parameters
    ----------
    data : tvm.relay.Expr
        Input data with spatial dimensions divisible by block_size

    block_size : int
        Size of blocks to decompose into channels.

    layout : string
        One of NCHW or NHWC, indicates channel axis.

    Returns
    -------
    result : tvm.relay.Expr
        Tensor with shape [in_batch, in_channel * block_size * block_size,
                           in_height / block_size, in_width / block_size]
    """
    return _make.space_to_depth(data, block_size, layout)


def adaptive_max_pool1d(data, output_size=None, layout="NCW", out_layout=""):
    r"""1D adaptive max pooling operator. This operator is experimental.

    This operator takes data as input and does 1D max value calculation
    across each window represented by W.


    In the default case, where the data_layout is `NCW`
    a data Tensor with shape `(batch_size, in_channels, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input height and width will be used
        as output height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size) for any input (NCW).

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    if isinstance(output_size, int):
        output_size = [output_size]
    return _make.adaptive_max_pool1d(data, output_size, layout, out_layout)


def adaptive_avg_pool1d(data, output_size=None, layout="NCW", out_layout=""):
    r"""1D adaptive average pooling operator. This operator is experimental.

    This operator takes data as input and does 1D average value calculation
    across each window represented by W.


    In the default case, where the data_layout is `NCW`
    a data Tensor with shape `(batch_size, in_channels, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input height and width will be used
        as output width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size) for any input (NCW).

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    if isinstance(output_size, int):
        output_size = [output_size]
    return _make.adaptive_avg_pool1d(data, output_size, layout, out_layout)


def adaptive_max_pool2d(data, output_size=None, layout="NCHW", out_layout=""):
    r"""2D adaptive max pooling operator. This operator is experimental.

    This operator takes data as input and does 2D max value calculation
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
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    return _make.adaptive_max_pool2d(data, output_size, layout, out_layout)


def adaptive_avg_pool2d(data, output_size=None, layout="NCHW", out_layout=""):
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
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    return _make.adaptive_avg_pool2d(data, output_size, layout, out_layout)


def adaptive_max_pool3d(data, output_size=None, layout="NCDHW", out_layout=""):
    r"""3D adaptive max pooling operator. This operator is experimental.

    This operator takes data as input and does 3D max value calculation
    across each window represented by DxWxH.

    In the default case, where the data_layout is `NCDHW`
    a data Tensor with shape `(batch_size, in_channels, depth, height, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_depth, output_height, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input depth, height and width will be used
        as output depth, height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size x output_size x output_size) for any input (NCDHW).

        If a tuple of integers (depth, height, width) are provided for output_size,
        the output size is (N x C x depth x height x width) for any input (NCDHW).

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    return _make.adaptive_max_pool3d(data, output_size, layout, out_layout)


def adaptive_avg_pool3d(data, output_size=None, layout="NCDHW", out_layout=""):
    r"""3D adaptive avg pooling operator. This operator is experimental.

    This operator takes data as input and does 3D avg value calculation
    across each window represented by DxWxH.

    In the default case, where the data_layout is `NCDHW`
    a data Tensor with shape `(batch_size, in_channels, depth, height, width)`,
    to produce an output Tensor with shape
    (batch_size, in_channels, output_depth, output_height, output_width).

    The pooling kernel and stride sizes are automatically chosen for
    desired output sizes.

    For output_size:
        If this argument is not provided, input depth, height and width will be used
        as output depth, height and width.

        If a single integer is provided for output_size, the output size is
        (N x C x output_size x output_size x output_size) for any input (NCDHW).

        If a tuple of integers (depth, height, width) are provided for output_size,
        the output size is (N x C x depth x height x width) for any input (NCDHW).

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    output_size : tuple of int. optional
        Output height and width.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [] or output_size
    return _make.adaptive_avg_pool3d(data, output_size, layout, out_layout)


def global_max_pool1d(data, layout="NCW", out_layout=""):
    r"""1D global maximum pooling operator.

    This operator takes data as input and does 1D max value calculation
    across each window represented by W.

    In the default case, where the data_layout is `NCW`
    a data Tensor with shape `(batch_size, in_channels, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, w)
    .. math::

        \mbox{out}(b, c, 1)  = \max_{n=0, \ldots, w} \mbox{data}(b, c, n)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [1]
    return _make.adaptive_max_pool1d(data, output_size, layout, out_layout)


def global_avg_pool1d(data, layout="NCW", out_layout=""):
    r"""1D global average pooling operator.

    This operator takes data as input and does 1D average value calculation
    across each window represented by W.

    In the default case, where the data_layout is `NCW`
    a data Tensor with shape `(batch_size, in_channels, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, w)

    .. math::

        \mbox{out}(b, c, 1)  = \frac{1}{w} \sum_{n=0}^{w-1} \mbox{data}(b, c, n)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [1]
    return _make.adaptive_avg_pool1d(data, output_size, layout, out_layout)


def global_max_pool3d(data, layout="NCDHW", out_layout=""):
    r"""3D global maximum pooling operator.

    This operator takes data as input and does 3D max value calculation
    across each window represented by DxWxH.

    In the default case, where the data_layout is `NCDHW`
    a data Tensor with shape `(batch_size, in_channels, depth, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, d, h, w)
    .. math::

        \mbox{out}(b, c, 1, 1, 1)  =  \max_{l=0, \ldots, d},  \max_{m=0, \ldots, h},
             \max_{n=0, \ldots, w} \mbox{data}(b, c, l, m, n)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [1, 1, 1]
    return _make.adaptive_max_pool3d(data, output_size, layout, out_layout)


def global_avg_pool3d(data, layout="NCDHW", out_layout=""):
    r"""3D global average pooling operator.

    This operator takes data as input and does 3D average value calculation
    across each window represented by DxWxH.

    In the default case, where the data_layout is `NCDHW`
    a data Tensor with shape `(batch_size, in_channels, depth, height, width)`,
    to produce an output Tensor with the following rule:

    with data of shape (b, c, d, h, w)

    .. math::

        \mbox{out}(b, c, 1, 1, 1)  = \frac{1}{d * h * w} \sum_{l=0}^{d-1}  \sum_{m=0}^{h-1}
             \sum_{n=0}^{w-1} \mbox{data}(b, c, l, m, n)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    out_layout : str, optional
        Layout of the output.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    output_size = [1, 1, 1]
    return _make.adaptive_avg_pool3d(data, output_size, layout, out_layout)


def correlation(
    data1, data2, kernel_size, max_displacement, stride1, stride2, padding, is_multiply, layout
):
    r"""Applies correlation to inputs.

    The correlation layer performs multiplicative patch comparisons between two feature maps.
    Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and
    :math:`c` being their width, height, and number of channels, the correlation layer lets the
    network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.

    For now we consider only a single comparison of two patches. The 'correlation' of two patches
    centered at :math:`x_{1}` in the first map and :math:`x_{2}` in the second map is then defined
    as:

    .. math::

        c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>

    for a square patch of size :math:`K:=2k+1`.

    Note that the equation above is identical to one step of a convolution in neural networks, but
    instead of convolving data with a filter, it convolves data with other    data. For this
    reason, it has no training weights.

    Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all
    patch combinations involves :math:`w^{2}*h^{2}` such computations.

    Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes
    correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,
    by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize
    :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhood
    centered around :math:`x_{1}`.

    The final output is defined by the following expression:

    .. math::

        out[n, q, i, j] = c(x_{i, j}, x_{q})

    where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q`
    denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.

    Parameters
    ----------
    data1 : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]

    data2 : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]

    kernel_size: int
        Kernel size for correlation, must be an odd number

    max_displacement: int
        Max displacement of Correlation

    stride1: int
        Stride for data1

    stride2: int
        Stride for data2 within the neightborhood centered around data1

    padding : int or a list/tuple of 2 or 4 ints
        Padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    is_multiply: bool
        operation type is either multiplication or substraction

    layout: str
        layout of data1, data2 and the output

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if isinstance(padding, int):
        padding = (padding, padding)
    return _make.correlation(
        data1, data2, kernel_size, max_displacement, stride1, stride2, padding, is_multiply, layout
    )


def space_to_batch_nd(data, block_shape, paddings, pad_value=0):
    r"""Divide spatial dimensions of the data into a grid of blocks
    and interleave them into batch dim.

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D with shape [batch, spatial_shape, remaining_shape]

    block_shape : relay.Expr
        1-D of size [M] where M is number of spatial dims, specifies block size
        for each spatial dimension.

    paddings : relay.Expr
        2-D of shape [M, 2] where M is number of spatial dims, specifies
        [before, after] paddings for each spatial dimension.

    pad_value : float, or relay.Expr, optional, default=0
        The value used for padding.

    Returns
    -------
    result : relay.Expr
        N-D Tensor with shape
        [in_batch * prod(block_shape),
        padded_data[1] / block_shape[0], ..., padded_data[M] / block_shape[M-1],
        remaining_shape]
    """

    return _make.space_to_batch_nd(data, block_shape, paddings, pad_value)


def batch_to_space_nd(data, block_shape, crops):
    r"""Reshape the batch dimension into spatial dimensions.

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D with shape [batch, spatial_shape, remaining_shape]

    block_shape : relay.Expr
        1-D of size [M] where M is number of spatial dims, specifies block size
        for each spatial dimension.

    crops : relay.Expr
        2-D of shape [M, 2] where M is number of spatial dims, specifies
        [begin, end] crop size for each spatial dimension.

    Returns
    -------
    result : relay.Expr
        N-D Tensor with shape
        [batch / prod(block_shape),
        in_shape[1] * block_shape[0] - crops[0,0] - crops[0,1], ...,
        in_shape[M] * block_shape[M-1] - crops[M-1, 0] - crops[M-1, 1],
        remaining_shape]
    """

    return _make.batch_to_space_nd(data, block_shape, crops)


def conv2d_backward_weight(
    grad,
    data,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    grad_layout="NCHW",
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_dtype="",
):
    r"""The gradient of conv2d with respect to weight.

    This operator takes the output gradient `grad` and convolves it with `data` as
    the convolution kernel, to produce the gradient with respect to weight.

    Note that the parameter `kernel_size` is the spatial size of the corresponding
    forward convolution kernel, not that of `data`. `grad_layout` and
    `kernel_layout` are the layouts of `grad` and the weight gradient respectively.

    Other parameters are the same as the conv2d op. See its documentation for more
    details.

    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    padding = get_pad_tuple2d(padding)

    return _make.conv2d_backward_weight(
        grad,
        data,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        grad_layout,
        data_layout,
        kernel_layout,
        out_dtype,
    )
