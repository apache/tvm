"""External function interface to CuDNN v7 library."""
# pylint: disable-msg=C0103
import ctypes
import numpy as np
from .. import api as _api
from .. import intrin as _intrin
from .. import get_global_func as _get_global_func


# algos can be read from cudnn.h
_FWD_ALGOS = [
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
]

_BWD_FILTER_ALGOS = [
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
    # non-deterministic
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
    # non-deterministic, algo0 with workspaceS
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
    # not implemented
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
]

_BWD_DATA_ALGOS = [
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
    # non-deterministic
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",
]

_ALGO_TYPE = [
    "fwd",
    "bwd_filter",
    "bwd_data"
]

def algo_to_index(algo_type, algo_name):
    """Return a index represents the algorithm, which can be used in
    calling CuDNN function

    Parameters
    ----------
        algo_type : str
            ["fwd", "bwd_filter", "bwd_data]

        algo_name : str
            algorithm name in cudnn definition
            fwd = [
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
                "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
                "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_FWD_ALGO_COUNT",
            ]
            bwd_filter = [
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
                # non-deterministic
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
                # non-deterministic, algo0 with workspaceS
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",
                # not implemented
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",
            ]
            bwd_data = [
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
                # non-deterministic
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
                "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",
            ]

    Returns
    -------
        algo: int
            algorithm index

    """
    idx = -1
    if algo_type == "fwd":
        idx = _FWD_ALGOS.index(algo_name)
    elif algo_type == "bwd_filter":
        idx = _BWD_FILTER_ALGOS.index(algo_name)
    elif algo_type == "bwd_data":
        idx = _BWD_DATA_ALGOS.index(algo_name)
    assert idx >= 0
    return idx


def _get_np_int32_array_handle(arr):
    """Return a void_p handle for a numpy array

    Parameters
    ----------
    arr: numpy.NDArray
        source numpy array

    Returns
    -------
    ptr:  ctypes.c_void_p
        pointer to the data
    """
    assert arr.dtype == np.int32
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    return ctypes.cast(ptr, ctypes.c_void_p)


def conv2d_w_shape(in_channel,
                   out_channel,
                   filter_h,
                   filter_w):
    """Get weight shape for a 2D convolution

    Parameters
    ----------
    in_channel: int
        input channel
    out_channel: int
        output channel
    filter_h: int
        filter height
    filter_w: int
        filter width

    Returns
    -------
    wshape: list
        weight shape
    """
    return [out_channel, in_channel, filter_h, filter_w]

def conv2d_output_shape(tensor_format,
                        pad_h,
                        pad_w,
                        stride_h,
                        stride_w,
                        dilation_h,
                        dilation_w,
                        x_shape,
                        w_shape):
    """Get output shape of 2D convolution

    Paramters
    ---------
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    pad_h: int
        height pad
    pad_w: int
        weight pad
    stride_h: int
        height stride
    stride_w: int
        width stride
    dilation_h: int
        height dilation
    dilation_w: int
        width dilation
    x_shape: list
        input shape
    w_shape: list
        weight shape

    Returns
    -------
    oshape: list
        output shape
    """
    assert isinstance(x_shape, list)
    assert isinstance(w_shape, list)
    assert len(x_shape) == 4
    assert len(w_shape) == 4
    oshape = np.zeros((len(x_shape)), dtype=np.int32)
    func = _get_global_func("tvm.contrib.cudnn.conv2d.output_shape")
    func(tensor_format,
         pad_h,
         pad_w,
         stride_h,
         stride_w,
         dilation_h,
         dilation_w,
         x_shape[0].value,
         x_shape[1].value,
         x_shape[2].value,
         x_shape[3].value,
         w_shape[0].value,
         w_shape[1].value,
         w_shape[2].value,
         w_shape[3].value,
         _get_np_int32_array_handle(oshape))
    return list(oshape)


def conv2d_find_algo(tensor_format,
                     pad_h,
                     pad_w,
                     stride_h,
                     stride_w,
                     dilation_h,
                     dilation_w,
                     x_shape,
                     w_shape,
                     y_shape):
    """Choose the best algo for the given input.

    Paramters
    ---------
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    pad_h: int
        height pad
    pad_w: int
        weight pad
    stride_h: int
        height stride
    stride_w: int
        width stride
    dilation_h: int
        height dilation
    dilation_w: int
        width dilation
    x_shape: list
        input shape
    w_shape: list
        weight shape
    y_shape: list
        output shape

    Returns
    -------
    algo: int
        algo chosen by CUDNN
    """
    func = _get_global_func("tvm.contrib.cudnn.conv2d.find_algo")
    return func(tensor_format,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                x_shape[0].value,
                x_shape[1].value,
                x_shape[2].value,
                x_shape[3].value,
                w_shape[0].value,
                w_shape[1].value,
                w_shape[2].value,
                w_shape[3].value,
                y_shape[0],
                y_shape[1],
                y_shape[2],
                y_shape[3])


def conv2d_forward(x,
                   w,
                   stride_h=1,
                   stride_w=1,
                   pad_h=0,
                   pad_w=0,
                   dilation_h=1,
                   dilation_w=1,
                   conv_mode=1,
                   tensor_format=0,
                   algo=-1):
    """Create an extern op that compute 2D convolution with CuDNN

    Parameters
    ----------
    x: Tensor
        input feature map
    w: Tensor
        convolution weight
    stride_h: int
        height stride
    stride_w: int
        width stride
    pad_h: int
        height pad
    pad_w: int
        weight pad
    dilation_h: int
        height dilation
    dilation_w: int
        width dilation
    conv_mode: int
        0: CUDNN_CONVOLUTION
        1: CUDNN_CROSS_CORRELATION
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    algo: int
        Forward algorithm, get index from ```algo_to_index``` function
        if algo == -1, the best algo will be chosen by CUDNN

    Returns
    -------
    y: Tensor
        The result tensor
    """
    oshape = conv2d_output_shape(tensor_format,
                                 pad_h,
                                 pad_w,
                                 stride_h,
                                 stride_w,
                                 dilation_h,
                                 dilation_w,
                                 list(x.shape),
                                 list(w.shape))
    if algo == -1:
        algo = conv2d_find_algo(tensor_format,
                                pad_h,
                                pad_w,
                                stride_h,
                                stride_w,
                                dilation_h,
                                dilation_w,
                                list(x.shape),
                                list(w.shape),
                                oshape)

    return _api.extern(
        oshape, [x, w],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.cudnn.conv2d.forward",
            conv_mode,
            tensor_format,
            algo,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            ins[0],
            ins[1],
            outs[0]), name="y")
