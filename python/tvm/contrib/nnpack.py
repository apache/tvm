"""External function interface to NNPACK libraroes."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api

def config(nthreads):
    """Configure the nnpack library.

    Parameters
    ----------
    nthreads : int
        The threads number of nnpack thread pool, must be a nonnegative.

    """
    _Config(nthreads)

def fully_connected_inference(lhs, rhs, nthreads=1):
    """Create an extern op that compute fully connected of 1D tensor lhs and
    2D tensor rhs with nnpack.

    Parameters
    ----------
    lhs : Tensor
        lhs 1D array input[input_channels] of FP32 elements
    rhs : Tensor
        lhs 2D matrix kernel[output_channels][input_channels] of FP32 elements

    Returns
    -------
    C : Tensor
        lhs 1D array out[output_channels] of FP32 elements.
    """
    m = rhs.shape[0]
    return _api.extern(
        (m, ), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.fully_connected_inference",
            ins[0], ins[1], outs[0], nthreads), name="C")

def fully_connected_output(lhs, rhs, nthreads=1):
    """Create an extern op that compute fully connected of 2D tensor lhs and
    2D tensor rhs with nnpack.

    Parameters
    ----------
    lhs : Tensor
        lhs 2D matrix input[batch_size][input_channels] of FP32 elements
    rhs : Tensor
        lhs 2D matrix kernel[output_channels][input_channels] of FP32 elements

    Returns
    -------
    C : Tensor
        lhs 2D array out[batch_size][output_channels] of FP32 elements.
    """
    n = lhs.shape[0]
    m = rhs.shape[0]
    return _api.extern(
        (n, m), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.fully_connected_output",
            ins[0], ins[1], outs[0], nthreads), name="C")

def convolution_inference(data, kernel, bias, padding, stride, nthreads=1):
    """Create an extern op to do inference convolution of 3D tensor data and
    4D tensor kernel and 1D tensor bias with nnpack.

    Parameters
    ----------
    data : Tensor
        data 3D tensor input[input_channels][input_height][input_width] of
        FP32 elements.
    kernel : Tensor
        kernel 4D tensor kernel[output_channels][input_channels][kernel_height]
        [kernel_width] of FP32 elements.
    bias : Tensor
        bias 1D array bias[output_channels][input_channels][kernel_height]
        [kernel_width] of FP32 elements.
    padding : list
        padding A 4-dim list of [pad_top, pad_bottom, pad_left, pad_right],
        which indicates the padding around the feature map.
    stride : list
        stride A 2-dim list of [stride_height, stride_width], which indicates
        the stride.

    Returns
    -------
    output : Tensor
        output 3D tensor output[output_channels][output_height][output_width]
        of FP32 elements.
    """

    assert isinstance(padding, list) and len(padding) == 4
    assert isinstance(stride, list) and len(stride) == 2
    _, input_height, input_width = data.shape
    output_channels, _, kernel_height, kernel_width = kernel.shape
    output_height = (input_height + padding[0] + padding[1] - kernel_height) / stride[0] + 1
    output_width = (input_width + padding[0] + padding[1] - kernel_width) / stride[1] + 1

    return _api.extern(
        (output_channels, output_height, output_width), [data, kernel, bias],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.convolution_inference", ins[0], ins[1], ins[2],
            outs[0], padding[0], padding[1], padding[2], padding[3],
            stride[0], stride[1], nthreads), name="C")

def convolution_output(data, kernel, bias, padding, nthreads=1):
    """Create an extern op to compute convolution of 4D tensor data and
    4D tensor kernel and 1D tensor bias with nnpack.

    Parameters
    ----------
    data : Tensor
        data 4D tensor input[batch_size][input_channels][input_height]
        [input_width] of FP32 elements.
    kernel : Tensor
        kernel 4D tensor kernel[output_channels][input_channels][kernel_height]
        [kernel_width] of FP32 elements.
    bias : Tensor
        bias 1D array bias[output_channels][input_channels][kernel_height]
        [kernel_width] of FP32 elements.
    padding : list
        padding A 4-dim list of [pad_top, pad_bottom, pad_left, pad_right],
        which indicates the padding around the feature map.

    Returns
    -------
    output : Tensor
        output 4D tensor output[batch_size][output_channels][output_height]
        [output_width] of FP32 elements.
    """

    assert isinstance(padding, list) and len(padding) == 4
    batch, _, input_height, input_width = data.shape
    output_channels, _, kernel_height, kernel_width = kernel.shape
    output_height = (input_height + padding[0] + padding[1] - kernel_height) + 1
    output_width = (input_width + padding[0] + padding[1] - kernel_width) + 1

    return _api.extern(
        (batch, output_channels, output_height, output_width), [data, kernel, bias],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.convolution_output", ins[0], ins[1], ins[2],
            outs[0], padding[0], padding[1], padding[2], padding[3], nthreads), name="C")

_init_api("tvm.contrib.nnpack")
