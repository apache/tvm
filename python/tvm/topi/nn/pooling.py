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
"""TVM operator pooling compute."""

from tvm import te, tirx

from .. import cpp

POOL_TYPE_CODE = {"avg": 0, "max": 1}


def global_pool(data, pool_type, layout="NCHW"):
    """Perform global pooling on height and width dimension of data.
       It decides the height and width dimension according to the layout string,
       in which 'W' and 'H' means width and height respectively.
       Width and height dimension cannot be split.
       For example, NCHW, NCHW16c, etc. are valid for pool,
       while NCHW16w, NCHW16h are not.
       See parameter `layout` for more information of the layout string convention.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D with shape of layout

    pool_type : str
        Pool type, 'max' or 'avg'

    layout : str
        Layout of the input data.
        The layout is supposed to be composed of upper cases, lower cases and numbers,
        where upper case indicates a dimension and
        the corresponding lower case with factor size indicates the split dimension.
        For example, NCHW16c can describe a 5-D tensor of
        [batch_size, channel, height, width, channel_block],
        in which channel_block=16 is a split of dimension channel.

    Returns
    -------
    output : tvm.te.Tensor
        n-D in same layout with height and width dimension size of 1.
        e.g., for NCHW, the output shape will be [batch, channel, 1, 1]
    """
    return cpp.nn.global_pool(data, POOL_TYPE_CODE[pool_type], layout)


def pool_grad(
    grads,
    data,
    kernel,
    stride,
    padding,
    pool_type,
    ceil_mode=False,
    count_include_pad=True,
    layout="NCHW",
):
    """Gradient of pooling on height and width dimension of data.
       It decides the height and width dimension according to the layout string,
       in which 'W' and 'H' means width and height respectively.
       Width and height dimension cannot be split.
       For example, NCHW, NCHW16c, etc. are valid for pool,
       while NCHW16w, NCHW16h are not.
       See parameter `layout` for more information of the layout string convention.

    Parameters
    ----------
    grads : tvm.te.Tensor
        n-D with shape of layout

    data : tvm.te.Tensor
        n-D with shape of layout

    kernel : list/tuple of two ints
        Kernel size, [kernel_height, kernel_width]

    stride : list/tuple of two ints
        Stride size, [stride_height, stride_width]

    padding : list/tuple of four ints
        Pad size, [pad_top, pad_left, pad_bottom, pad_right]]

    pool_type : str
        Pool type, 'max' or 'avg'

    ceil_mode : bool
        Whether to use ceil when calculating output size.

    count_include_pad: bool
        Whether include padding in the calculation when pool_type is 'avg'

    layout: string
        Layout of the input data.
        The layout is supposed to be composed of upper cases, lower cases and numbers,
        where upper case indicates a dimension and
        the corresponding lower case with factor size indicates the split dimension.
        For example, NCHW16c can describe a 5-D tensor of
        [batch_size, channel, height, width, channel_block],
        in which channel_block=16 is a split of dimension channel.


    Returns
    -------
    output : tvm.te.Tensor
        n-D in the same layout
    """
    return cpp.nn.pool_grad(
        grads,
        data,
        kernel,
        stride,
        padding,
        POOL_TYPE_CODE[pool_type],
        ceil_mode,
        layout,
        count_include_pad,
    )


def adaptive_pool(data, output_size, pool_type, layout="NCHW"):
    """Perform pooling on height and width dimension of data.
       The pooling kernel and stride sizes are automatically chosen for desired
       output sizes.
       It decides the height and width dimension according to the layout string,
       in which 'W' and 'H' means width and height respectively.
       Width and height dimension cannot be split.
       For example, NCHW, NCHW16c, etc. are valid for pool,
       while NCHW16w, NCHW16h are not.
       See parameter `layout` for more information of the layout string convention.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D with shape of layout

    output_size : tuple of int
        output height and width.

    pool_type : str
        Pool type, 'max' or 'avg'

    layout: string
        Layout of the input data.
        The layout is supposed to be composed of upper cases, lower cases and numbers,
        where upper case indicates a dimension and
        the corresponding lower case with factor size indicates the split dimension.
        For example, NCHW16c can describe a 5-D tensor of
        [batch_size, channel, height, width, channel_block],
        in which channel_block=16 is a split of dimension channel.

    Returns
    -------
    output : tvm.te.Tensor
        n-D in the same layout
    """
    return cpp.nn.adaptive_pool(data, output_size, POOL_TYPE_CODE[pool_type], layout)


def adaptive_pool1d(data, output_size, pool_type, layout="NCW"):
    """Perform pooling on three dimensional data.
    See the two dimensional version above for details.
    """
    return cpp.nn.adaptive_pool1d(data, output_size, POOL_TYPE_CODE[pool_type], layout)


def adaptive_pool3d(data, output_size, pool_type, layout="NCDHW"):
    """Perform pooling on three dimensional data.
    See the two dimensional version above for details.
    """
    return cpp.nn.adaptive_pool3d(data, output_size, POOL_TYPE_CODE[pool_type], layout)


def pool1d(
    data,
    kernel,
    stride,
    dilation,
    padding,
    pool_type,
    ceil_mode=False,
    layout="NCW",
    count_include_pad=True,
):
    """Perform pooling on width dimension of data.
       Width axis is determined according to the layout string.
       in which 'w' means width.
       Width dimension cannot be split.
       For example, NCW, NCW16c, etc. are valid for pool,
       while NCW16w is not.
       See parameter `layout` for more information of the layout string convention.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D with shape of layout

    kernel : list/tuple of one int or int
        Kernel size, [kernel_width]

    stride : list/tuple of one int or int
        Stride size, [stride_width]

    dilation: list/tuple of two ints
        Dilation size, [dilation_height, dilation_width]

    padding : list/tuple of two ints
        Pad size, [pad_left, pad_right]

    pool_type : str
        Pool type, 'max' or 'avg'

    ceil_mode : bool
        Whether to use ceil when calculating output size.

    layout: string
        Layout of the input data.
        The layout is supposed to be composed of upper cases, lower cases and numbers,
        where upper case indicates a dimension and
        the corresponding lower case with factor size indicates the split dimension.
        For example, NCW16c can describe a 4-D tensor of
        [batch_size, channel, width, channel_block],
        in which channel_block=16 is a split of dimension channel.

    count_include_pad: bool
        Whether include padding in the calculation when pool_type is 'avg'

    Returns
    -------
    output : tvm.te.Tensor
        n-D in the same layout
    """
    if isinstance(kernel, int):
        kernel = [
            kernel,
        ]
    if isinstance(stride, int):
        stride = [
            stride,
        ]
    return cpp.nn.pool1d(
        data,
        kernel,
        stride,
        dilation,
        padding,
        POOL_TYPE_CODE[pool_type],
        ceil_mode,
        layout,
        count_include_pad,
    )


def pool2d(
    data,
    kernel,
    stride,
    dilation,
    padding,
    pool_type,
    ceil_mode=False,
    layout="NCHW",
    count_include_pad=True,
):
    """Perform pooling on height and width dimension of data.
       It decides the height and width dimension according to the layout string,
       in which 'W' and 'H' means width and height respectively.
       Width and height dimension cannot be split.
       For example, NCHW, NCHW16c, etc. are valid for pool,
       while NCHW16w, NCHW16h are not.
       See parameter `layout` for more information of the layout string convention.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D with shape of layout

    kernel : list/tuple of two ints
        Kernel size, [kernel_height, kernel_width]

    stride : list/tuple of two ints
        Stride size, [stride_height, stride_width]

    dilation: list/tuple of two ints
        Dilation size, [dilation_height, dilation_width]

    padding : list/tuple of four ints
        Pad size, [pad_top, pad_left, pad_bottom, pad_right]]

    pool_type : str
        Pool type, 'max' or 'avg'

    ceil_mode : bool
        Whether to use ceil when calculating output size.

    layout: string
        Layout of the input data.
        The layout is supposed to be composed of upper cases, lower cases and numbers,
        where upper case indicates a dimension and
        the corresponding lower case with factor size indicates the split dimension.
        For example, NCHW16c can describe a 5-D tensor of
        [batch_size, channel, height, width, channel_block],
        in which channel_block=16 is a split of dimension channel.

    count_include_pad: bool
        Whether include padding in the calculation when pool_type is 'avg'

    Returns
    -------
    output : tvm.te.Tensor
        n-D in the same layout
    """
    return cpp.nn.pool2d(
        data,
        kernel,
        stride,
        dilation,
        padding,
        POOL_TYPE_CODE[pool_type],
        ceil_mode,
        layout,
        count_include_pad,
    )


def pool3d(
    data,
    kernel,
    stride,
    dilation,
    padding,
    pool_type,
    ceil_mode=False,
    layout="NCDHW",
    count_include_pad=True,
):
    """Perform pooling on depth, height and width dimension of data.
       It decides the depth, height and width dimension according to the layout string,
       in which 'D', 'W' and 'H' means depth, width and height respectively.
       Depth, width and height dimension cannot be split.
       For example, NCDHW, NCDHW16c, etc. are valid for pool,
       while NCDHW16d, NCDHW16w, NCDHW16h are not.
       See parameter `layout` for more information of the layout string convention.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D with shape of layout

    kernel : list/tuple of three ints
        Kernel size, [kernel_depth, kernel_height, kernel_width]

    stride : list/tuple of three ints
        Stride size, [stride_depth, stride_height, stride_width]

    dilation: list/tuple of two ints
        Dilation size, [dilation_height, dilation_width]

    padding : list/tuple of six ints
        Pad size, [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right]

    pool_type : str
        Pool type, 'max' or 'avg'

    ceil_mode : bool
        Whether to use ceil when calculating output size.

    layout: string
        Layout of the input data.
        The layout is supposed to be composed of upper cases, lower cases and numbers,
        where upper case indicates a dimension and
        the corresponding lower case with factor size indicates the split dimension.
        For example, NCDHW16c can describe a 6-D tensor of
        [batch_size, channel, depth, height, width, channel_block],
        in which channel_block=16 is a split of dimension channel.

    count_include_pad: bool
        Whether include padding in the calculation when pool_type is 'avg'

    Returns
    -------
    output : tvm.te.Tensor
        n-D in the same layout
    """
    return cpp.nn.pool3d(
        data,
        kernel,
        stride,
        dilation,
        padding,
        POOL_TYPE_CODE[pool_type],
        ceil_mode,
        layout,
        count_include_pad,
    )


def avg_pool_divisor(data, ndim, kernel, stride, padding, ceil_mode, divisor):
    """Average-pool the final ``ndim`` axes with a fixed, nonzero divisor.

    ``kernel``, ``stride`` and symmetric ``padding`` accept scalars or per-axis
    sequences. An omitted ``stride`` uses ``kernel``. Following PyTorch's
    ``ceil_mode`` semantics, trailing windows must start before the right padding.
    Values outside the input contribute zero; ``divisor`` replaces the element count.
    """

    def expand(value):
        values = (value,) if isinstance(value, int) else tuple(value)
        return values * ndim if len(values) == 1 else values

    kernel = expand(kernel)
    stride = kernel if stride is None or stride == [] else expand(stride)
    padding = expand(padding)
    spatial = list(data.shape)[-ndim:]
    output = []
    for length, k, s, p in zip(spatial, kernel, stride, padding):
        extent = (length + 2 * p - k + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode:
            extent = tirx.min(extent, (length + p + s - 1) // s)
        output.append(extent)
    shape = list(data.shape)[:-ndim] + output
    axes = [te.reduce_axis((0, k), f"r{i}") for i, k in enumerate(kernel)]
    dtype = "float32" if data.dtype in ("float16", "bfloat16") else data.dtype

    def window(*indices):
        positions = [indices[-ndim + i] * stride[i] - padding[i] + axes[i] for i in range(ndim)]
        valid = tirx.all(
            *[tirx.all(pos >= 0, pos < length) for pos, length in zip(positions, spatial)]
        )
        value = tirx.if_then_else(
            valid, data[(*indices[:-ndim], *positions)].astype(dtype), tirx.const(0, dtype)
        )
        return te.sum(value, axis=axes)

    sums = te.compute(shape, window, name="pool_sum")
    return te.compute(
        shape,
        lambda *i: (sums[i] / tirx.const(divisor, dtype)).astype(data.dtype),
        name="pool_divisor",
    )
