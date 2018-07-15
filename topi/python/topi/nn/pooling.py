"""TVM operator pooling compute."""
from __future__ import absolute_import
from .. import cpp

POOL_TYPE_CODE = {
    "avg": 0,
    "max": 1
}

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
    data : tvm.Tensor
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
    output : tvm.Tensor
        n-D in same layout with height and width dimension size of 1.
        e.g., for NCHW, the output shape will be [batch, channel, 1, 1]
    """
    return cpp.nn.global_pool(data, POOL_TYPE_CODE[pool_type], layout)


def pool(data,
         kernel,
         stride,
         padding,
         pool_type,
         ceil_mode=False,
         layout="NCHW",
         count_include_pad=True):
    """Perform pooling on height and width dimension of data.
       It decides the height and width dimension according to the layout string,
       in which 'W' and 'H' means width and height respectively.
       Width and height dimension cannot be split.
       For example, NCHW, NCHW16c, etc. are valid for pool,
       while NCHW16w, NCHW16h are not.
       See parameter `layout` for more information of the layout string convention.

    Parameters
    ----------
    data : tvm.Tensor
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
    output : tvm.Tensor
        n-D in the same layout
    """
    return cpp.nn.pool(data, kernel, stride, padding,
                       POOL_TYPE_CODE[pool_type], ceil_mode, layout, count_include_pad)
