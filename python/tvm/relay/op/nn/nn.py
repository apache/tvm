"""Neural network operations."""
from __future__ import absolute_import as _abs
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
           weight_layout="OIHW",
           out_layout="",
           out_dtype=""):
    r"""2D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCHW`
    and weight_layout is `OIHW`, conv2d takes in
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
    data : relay.Expr
        The input data to the operator.

    weight : relay.Expr
        The weight expressions.

    strides : tuple of int, optional
        The strides of convoltution.

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

    weight_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.conv2d(data, weight, strides, padding, dilation,
                        groups, channels, kernel_size, data_layout,
                        weight_layout, out_layout, out_dtype)


def conv2d_transpose(data,
                     weight,
                     strides=(1, 1),
                     padding=(0, 0),
                     dilation=(1, 1),
                     groups=1,
                     channels=None,
                     kernel_size=None,
                     data_layout="NCHW",
                     weight_layout="OIHW",
                     output_padding=(0, 0),
                     out_dtype=""):
    """Two dimensional trnasposed convolution operator.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    weight : relay.Expr
        The weight expressions.

    strides : Tuple[int], optional
        The strides of convoltution.

    padding : Tuple[int], optional
        The padding of convolution on both sides of inputs.

    dilation : Tuple[int], optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    weight_layout : str, optional
        Layout of the weight.

    output_padding : Tuple[int], optional
        Additional zero-padding to be added to one side of the output.

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.conv2d_transpose(data, weight, strides, padding, dilation,
                                  groups, channels, kernel_size, data_layout,
                                  weight_layout, output_padding, out_dtype)


def softmax(data, axis):
    r"""Computes softmax.

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: relay.Expr
        The input data to the operator.

    axis: int
        The axis to sum over when computing softmax

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.softmax(data, axis)


def log_softmax(data, axis):
    r"""Computes log softmax.

    .. math::

        \text{log_softmax}(x)_i = \log \frac{exp(x_i)}{\sum_j exp(x_j)}

    .. note::
        This operator can be optimized away for inference.

    Parameters
    ----------
    data: relay.Expr
        The input data to the operator.

    axis: int
        The axis to sum over when computing softmax

    Returns
    -------
    result : relay.Expr
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
    data : relay.Expr
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
    result : relay.Expr
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
    data : relay.Expr
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
    result : relay.Expr
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
    data : relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    Returns
    -------
    result : relay.Expr
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
    data : relay.Expr
        The input data to the operator.

    layout : str, optional
        Layout of the input.

    Returns
    -------
    result : relay.Expr
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

    method indicates the algorithm to be used while calculating ghe out value
    and method can be one of ("BILINEAR", "NEAREST_NEIGHBOR")

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    scale : relay.Expr
        The scale factor for upsampling.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [NEAREST_NEIGHBOR, BILINEAR].

    Returns
    -------
    result : relay.Expr
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
    data : relay.Expr
        The input data to the operator.

    Returns
    -------
    result: relay.Expr
        The Flattened result.
    """
    return _make.batch_flatten(data)

def relu(data):
    """Rectified linear unit.

    .. math::
       out = max(x, 0)

    Parameters
    ----------
    data : relay.Expr
        The input data

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.relu(data)


def pad(data,
        pad_width,
        pad_value=0.0):
    r"""Padding

    This operator takes in a tensor and pads each axis by the specified
    widths using the specified value.

    Parameters
    ----------
    data: relay.Expr
        The input data to the operator
    pad_width: tuple of <tuple of <int>>, required
        Number of values padded to the edges of each axis, in the format
        of ((before_1, after_1), ..., (before_N, after_N))
    pad_value: float, optional, default=0.0
        The value used for padding

    Returns
    -------
    result : relay.Expr
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
    data : relay.Expr
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
    result : relay.Expr
        The computed result.
    """
    return _make.lrn(data, size, axis, alpha, beta, bias)


def l2_normalize(data, eps, axis=None):
    """Perform L2 normalization on the input data

    .. math::
        y(i, j) = x(i, j) / sqrt(max(sum(x^2), eps))

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    eps : float
        epsilon value

    axis : list of int, optional
        axis over the normalization applied

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.l2_normalize(data, eps, axis)
