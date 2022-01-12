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
"""External function interface to MIOpen library."""
# pylint: disable-msg=C0103
import ctypes
import numpy as np
import tvm
import tvm._ffi

from tvm import te


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


def conv2d_forward(
    x,
    w,
    stride_h=1,
    stride_w=1,
    pad_h=0,
    pad_w=0,
    dilation_h=1,
    dilation_w=1,
    conv_mode=0,
    data_type=1,
    group_count=1,
):
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
    data_type: int
        0: miopenHalf (fp16)
        1: miopenFloat (fp32)
    group_count: int
        number of groups
    Returns
    -------
    y: Tensor
        The result tensor
    """
    assert 0 <= conv_mode <= 2, "0: miopenConvolution / 1: miopenTranspose / 2: miopenGroupConv"
    if group_count > 1:
        conv_mode = 2
    oshape = np.zeros((len(x.shape)), dtype=np.int32)
    xshape = x.shape
    wshape = w.shape
    setup_func = tvm._ffi.get_global_func("tvm.contrib.miopen.conv2d.setup")
    algo = setup_func(
        conv_mode,
        data_type,
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
        group_count,
        _get_np_int32_array_handle(oshape),
    )

    return te.extern(
        list(oshape),
        [x, w],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.miopen.conv2d.forward",
            conv_mode,
            data_type,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            algo,
            ins[0],
            ins[1],
            outs[0],
        ),
        name="y",
    )


def softmax(x, axis=-1):
    """Compute softmax with MIOpen

    Parameters
    ----------
    x : tvm.te.Tensor
        The input tensor

    axis : int
        The axis to compute softmax over

    Returns
    -------
    ret : tvm.te.Tensor
        The result tensor
    """
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.miopen.softmax.forward", ins[0], outs[0], axis
        ),
        name="y",
    )


def log_softmax(x, axis=-1):
    """Compute log softmax with MIOpen

    Parameters
    ----------
    x : tvm.te.Tensor
        The input tensor

    axis : int
        The axis to compute log softmax over

    Returns
    -------
    ret : tvm.te.Tensor
        The result tensor
    """
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.miopen.log_softmax.forward", ins[0], outs[0], axis
        ),
        name="y",
    )
