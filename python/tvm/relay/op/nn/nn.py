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
#pylint: disable=invalid-name, too-many-lines
"""Neural network operations."""
from __future__ import absolute_import as _abs
from ...expr import TupleWrapper
from . import _make


def conv2d(data,
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
           out_dtype=""):
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
    return _make.conv2d(data, weight, strides, padding, dilation,
                        groups, channels, kernel_size, data_layout,
                        kernel_layout, out_layout, out_dtype)


def conv2d_transpose(data,
                     weight,
                     strides=(1, 1),
                     padding=(0, 0),
                     dilation=(1, 1),
                     groups=1,
                     channels=None,
                     kernel_size=None,
                     data_layout="NCHW",
                     kernel_layout="OIHW",
                     output_padding=(0, 0),
                     out_dtype=""):
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

    output_padding : Tuple[int], optional
        Additional zero-padding to be added to one side of the output.

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.conv2d_transpose(data, weight, strides, padding, dilation,
                                  groups, channels, kernel_size, data_layout,
                                  kernel_layout, output_padding, out_dtype)


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

    axis: int
        The axis to sum over when computing softmax

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.log_softmax(data, axis)


def max_pool2d(data,
               pool_size=(1, 1),
               strides=(1, 1),
               padding=(0, 0),
               layout="NCHW",
               ceil_mode=False):
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

    strides : tuple of int, optional
        The strides of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.max_pool2d(data, pool_size, strides, padding,
                            layout, ceil_mode)

def avg_pool2d(data,
               pool_size=(1, 1),
               strides=(1, 1),
               padding=(0, 0),
               layout="NCHW",
               ceil_mode=False,
               count_include_pad=False):
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

    strides : tuple of int, optional
        The strides of pooling.

    padding : tuple of int, optional
        The padding for pooling.

    layout : str, optional
        Layout of the input.

    ceil_mode : bool, optional
        To enable or disable ceil while pooling.

    count_include_pad : bool, optional
        To include padding to compute the average.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.avg_pool2d(data, pool_size, strides, padding,
                            layout, ceil_mode, count_include_pad)

def global_max_pool2d(data,
                      layout="NCHW"):
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

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.global_max_pool2d(data, layout)

def global_avg_pool2d(data,
                      layout="NCHW"):
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

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.global_avg_pool2d(data, layout)


def upsampling(data,
               scale=1,
               layout="NCHW",
               method="NEAREST_NEIGHBOR"):
    """Upsampling.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, h*scale, w*scale)

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("BILINEAR", "NEAREST_NEIGHBOR")

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    scale : tvm.relay.Expr
        The scale factor for upsampling.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [NEAREST_NEIGHBOR, BILINEAR].

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.upsampling(data, scale, layout, method)


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


def dense(data, weight, units=None, out_dtype=""):
    """Dense operator.
    Applies a linear transformation

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

    out_dtype : str, optional
        Specifies the output data type for mixed precision dense.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.dense(data, weight, units, out_dtype)


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


def leaky_relu(data, alpha):
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

        `y = x > 0 ? x : alpha * x`

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


def pad(data,
        pad_width,
        pad_value=0.0):
    r"""Padding

    This operator takes in a tensor and pads each axis by the specified
    widths using the specified value.

    Parameters
    ----------
    data: tvm.relay.Expr
        The input data to the operator
    pad_width: tuple of <tuple of <int>>, required
        Number of values padded to the edges of each axis, in the format
        of ((before_1, after_1), ..., (before_N, after_N))
    pad_value: float, optional, default=0.0
        The value used for padding

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.pad(data, pad_width, pad_value)


def lrn(data, size=5, axis=1, bias=2, alpha=.00001, beta=0.75):
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
    result = _make.dropout(data, rate)
    return TupleWrapper(result, 2)[0]


def batch_norm(data,
               gamma,
               beta,
               moving_mean,
               moving_var,
               axis=1,
               epsilon=1e-5,
               center=True,
               scale=True):
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
    vectors. They are global statistics for the whole dataset, which are updated by::

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
        Small float added to variance to avoid diving by zero.

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
    result = _make.batch_norm(data,
                              gamma,
                              beta,
                              moving_mean,
                              moving_var,
                              axis,
                              epsilon,
                              center,
                              scale)
    return TupleWrapper(result, 3)


def batch_matmul(x, y):
    r"""
    Computes batch matrix multiplication of `x` and `y` when `x` and `y` are data
    in batch.

    .. math::

        \mbox{batch_matmul}(x, y)[i, :, :] = \mbox{matmul}(x[i, :, :], y[i, :, :]^T)

    Parameters
    ----------
    x : tvm.relay.Expr
        The first input.

    y : tvm.relay.Expr
        The second input.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.batch_matmul(x, y)


def contrib_conv2d_winograd_without_weight_transform(data,
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
                                                     out_dtype=""):
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
    return _make.contrib_conv2d_winograd_without_weight_transform(
        data, weight, tile_size, strides, padding, dilation,
        groups, channels, kernel_size, data_layout,
        kernel_layout, out_layout, out_dtype)


def contrib_conv2d_winograd_nnpack_without_weight_transform(data,
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
                                                            out_dtype=""):
    r"""2D convolution with the NNPACK implementation of winograd algorithm.

    The basic parameters are the same as the ones in vanilla conv2d.
    It assumes the weight is pre-transformed by nn.contrib_conv2d_winograd_nnpack_weight_transform

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
    return _make.contrib_conv2d_winograd_nnpack_without_weight_transform(
        data, weight, strides, padding, dilation,
        groups, channels, kernel_size, data_layout,
        kernel_layout, out_layout, out_dtype)


def contrib_conv2d_nchwc(data,
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
                         out_dtype=""):
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
    return _make.contrib_conv2d_NCHWc(data, kernel, strides, padding, dilation,
                                      groups, channels, kernel_size, data_layout,
                                      kernel_layout, out_layout, out_dtype)

def contrib_depthwise_conv2d_nchwc(data,
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
                                   out_dtype=""):
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
    return _make.contrib_depthwise_conv2d_NCHWc(data, kernel, strides, padding, dilation,
                                                groups, channels, kernel_size, data_layout,
                                                kernel_layout, out_layout, out_dtype)

def contrib_conv2d_winograd_weight_transform(weight,
                                             tile_size):
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


def contrib_conv2d_winograd_nnpack_weight_transform(weight,
                                                    convolution_algorithm,
                                                    out_dtype=""):
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
        weight, convolution_algorithm, out_dtype)


def deformable_conv2d(data,
                      offset,
                      weight,
                      strides=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      deformable_groups=1,
                      groups=1,
                      channels=None,
                      kernel_size=None,
                      data_layout='NCHW',
                      kernel_layout='OIHW',
                      out_layout='',
                      out_dtype=''):
    r""" Deformable 2d convolution.

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
    return _make.deformable_conv2d(data, offset, weight, strides, padding, dilation,
                                   deformable_groups, groups, channels, kernel_size, data_layout,
                                   kernel_layout, out_layout, out_dtype)
