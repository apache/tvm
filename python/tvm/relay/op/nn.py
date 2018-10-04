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
    """Two dimensional convolution operator.

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    weight : relay.Expr
        The weight expressions.

    strides : tuple of int, optional
        The strides of convoltution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    weight_layout : str, optional
        Layout of the weight.

    out_layout : str, optional
        Layout of the output.

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.
    """
    return _make.conv2d(data, weight, strides, padding, dilation,
                        groups, channels, kernel_size, data_layout,
                        weight_layout, out_layout, out_dtype)
