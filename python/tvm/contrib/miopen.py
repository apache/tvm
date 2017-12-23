"""External function interface to MIOpen library."""
# pylint: disable-msg=C0103
import ctypes
import numpy as np
from .. import api as _api
from .. import intrin as _intrin
from .. import get_global_func as _get_global_func


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


def conv2d_output_shape(conv_mode,
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
    func = _get_global_func("tvm.contrib.miopen.conv2d.output_shape")
    func(conv_mode,
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


def conv2d_forward(x,
                   w,
                   stride_h=1,
                   stride_w=1,
                   pad_h=0,
                   pad_w=0,
                   dilation_h=1,
                   dilation_w=1,
                   conv_mode=1,
                   algo=0):
    """Create an extern op that compute 2D convolution with MIOpen

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

    Returns
    -------
    y: Tensor
        The result tensor
    """
    oshape = conv2d_output_shape(conv_mode,
                                 pad_h,
                                 pad_w,
                                 stride_h,
                                 stride_w,
                                 dilation_h,
                                 dilation_w,
                                 list(x.shape),
                                 list(w.shape))
    return _api.extern(
        oshape, [x, w],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.miopen.conv2d.forward",
            conv_mode,
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
