"""External function interface to NNPACK libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin
from .._ffi.function import _init_api

def is_available():
    """Check whether NNPACK is available, that is, `nnp_initialize()`
    returns `nnp_status_success`.
    """
    return _initialize() == 0

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


class ConvolutionAlgorithm:
    AUTO = 0
    FFT_8x8 = 1
    FFT_16x16 = 2
    WT_8x8 = 3
    IMPLICIT_GEMM = 4
    DIRECT = 5
    WT_8x8_FP16 = 6


class ConvolutionTransformStrategy:
    COMPUTE = 1
    PRECOMPUTE = 2


def convolution_inference(
        data, kernel, bias, padding, stride, nthreads=1,
        algorithm=ConvolutionAlgorithm.AUTO):
    """Create an extern op to do inference convolution of 4D tensor data and
    4D tensor kernel and 1D tensor bias with nnpack.

    Parameters
    ----------
    data : Tensor
        data 4D tensor input[batch][input_channels][input_height][input_width] of
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
        output 4D tensor output[batch][output_channels][output_height][output_width]
        of FP32 elements.
    """

    assert isinstance(padding, list) and len(padding) == 4
    assert isinstance(stride, list) and len(stride) == 2
    batch, _, input_height, input_width = data.shape
    output_channels, _, kernel_height, kernel_width = kernel.shape
    output_height = (input_height + padding[0] + padding[1] - kernel_height) / stride[0] + 1
    output_width = (input_width + padding[0] + padding[1] - kernel_width) / stride[1] + 1

    return _api.extern(
        (batch, output_channels, output_height, output_width),
        [data, kernel, bias] if bias is not None else [data, kernel],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.convolution_inference",
            ins[0],
            ins[1],
            ins[2] if bias is not None else 0,
            outs[0], padding[0], padding[1], padding[2], padding[3],
            stride[0], stride[1], nthreads, algorithm), name="C")

def convolution_inference_without_weight_transform(
        data, transformed_kernel, bias, padding, stride, nthreads=1,
        algorithm=ConvolutionAlgorithm.AUTO):
    """Create an extern op to do inference convolution of 4D tensor data and
    4D pre-transformed tensor kernel and 1D tensor bias with nnpack.

    Parameters
    ----------
    data : Tensor
        data 4D tensor input[batch][input_channels][input_height][input_width] of
        FP32 elements.
    transformed_kernel : Tensor
        transformed_kernel 4D tensor kernel[output_channels][input_channels][tile]
        [tile] of FP32 elements.
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
        output 4D tensor output[batch][output_channels][output_height][output_width]
        of FP32 elements.
    """

    assert algorithm in (ConvolutionAlgorithm.WT_8x8,
                         ConvolutionAlgorithm.WT_8x8_FP16)
    assert isinstance(padding, list) and len(padding) == 4
    assert isinstance(stride, list) and len(stride) == 2
    batch, _, input_height, input_width = data.shape
    output_channels, _, _, _ = transformed_kernel.shape
    kernel_height, kernel_width = (3, 3)
    output_height = (input_height + padding[0] + padding[1] - kernel_height) / stride[0] + 1
    output_width = (input_width + padding[0] + padding[1] - kernel_width) / stride[1] + 1

    return _api.extern(
        (batch, output_channels, output_height, output_width),
        [data, transformed_kernel, bias] if bias is not None else [data, transformed_kernel],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.convolution_inference_without_weight_transform",
            ins[0],
            ins[1],
            ins[2] if bias is not None else 0,
            outs[0], padding[0], padding[1], padding[2], padding[3],
            stride[0], stride[1], nthreads, algorithm), name="C")

def convolution_inference_weight_transform(
        kernel, nthreads=1,
        algorithm=ConvolutionAlgorithm.AUTO):
    """Create an extern op to do inference convolution of 3D tensor data and
    4D tensor kernel and 1D tensor bias with nnpack.

    Parameters
    ----------
    kernel : Tensor
        kernel 4D tensor kernel[output_channels][input_channels][kernel_height]
        [kernel_width] of FP32 elements.

    Returns
    -------
    output : Tensor
        output 4D tensor output[output_channels][input_channels][tile][tile]
        of FP32 elements.
    """
    assert algorithm in (ConvolutionAlgorithm.WT_8x8, ConvolutionAlgorithm.WT_8x8_FP16)
    output_channels, input_channels, _, _ = kernel.shape

    transform_tile_size = 8
    return _api.extern(
        (output_channels, input_channels, transform_tile_size, transform_tile_size),
        [kernel],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.convolution_inference_weight_transform",
            ins[0], outs[0], nthreads, algorithm), name="transform_kernel")

_init_api("tvm.contrib.nnpack")
