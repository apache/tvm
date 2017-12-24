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


def conv2d_forward(x,
                   w,
                   stride_h=1,
                   stride_w=1,
                   pad_h=0,
                   pad_w=0,
                   dilation_h=1,
                   dilation_w=1,
                   conv_mode=0):
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
        0: miopenConvolution
        1: miopenTranspose

    Returns
    -------
    y: Tensor
        The result tensor
    """
    assert conv_mode == 0, "Transpose convolutions not supported yet."
    oshape = np.zeros((len(x.shape)), dtype=np.int32)
    xshape = x.shape
    wshape = w.shape
    setup_func = _get_global_func("tvm.contrib.miopen.conv2d.setup")
    algo = setup_func(conv_mode,
                      pad_h,
                      pad_w,
                      stride_h,
                      stride_w,
                      dilation_h,
                      dilation_w,
                      xshape[0].value,
                      xshape[1].value,
                      xshape[2].value,
                      xshape[3].value,
                      wshape[0].value,
                      wshape[1].value,
                      wshape[2].value,
                      wshape[3].value,
                      _get_np_int32_array_handle(oshape))

    return _api.extern(
        list(oshape), [x, w],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.miopen.conv2d.forward",
            conv_mode,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            algo,
            ins[0],
            ins[1],
            outs[0]), name="y")
