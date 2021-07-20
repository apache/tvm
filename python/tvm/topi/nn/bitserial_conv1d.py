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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
# pylint: disable=unused-argument, redefined-builtin
"""Bitserial Conv1D operators"""
import tvm
from tvm import te
from .pad import pad
from .bitserial_util import bitpack
from ..utils import get_const_tuple


def bitserial_conv1d_ncw(
    data,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    """Bitserial Conv1D operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [num_filter, in_channel, filter_width]

    stride : int or a list/tuple of one int
        stride size, or [stride_width]

    padding : int or a list/tuple of two ints
        padding size, [pad_left, pad_right]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    unipolar: bool
        if binarization style is in unipolar 1/0 format, instead of bipolar -1/+1 format

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, out_channel, out_width]
    """
    assert isinstance(stride, int) or len(stride) == 1
    Input_q = bitpack(data, activation_bits, pack_axis=1, bit_axis=2, pack_type=pack_dtype)
    if len(kernel.shape) == 3:
        Filter_q = bitpack(kernel, weight_bits, pack_axis=1, bit_axis=3, pack_type=pack_dtype)
    else:
        Filter_q = kernel
    batch, in_channel, activation_bits, in_width = Input_q.shape
    num_filter, _, kernel_w, weight_bits = Filter_q.shape

    if isinstance(padding, int):
        LPAD = RPAD = padding
    else:
        LPAD, RPAD = padding
    pad_before = [0, 0, 0, LPAD]
    pad_after = [0, 0, 0, RPAD]

    PadInput_q = pad(Input_q, pad_before, pad_after, name="pad_temp")
    # compute the output shape
    if not isinstance(stride, int):
        stride = stride[0]
    out_channel = num_filter
    out_width = (in_width - kernel_w + LPAD + RPAD) // stride + 1

    rc = te.reduce_axis((0, in_channel), name="rc")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    b1 = te.reduce_axis((0, activation_bits), name="b1")
    b2 = te.reduce_axis((0, weight_bits), name="b2")

    if unipolar:

        def _conv(nn, ff, xx):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    (
                        tvm.tir.popcount(
                            PadInput_q[nn, rc, b1, xx * stride + rx] & Filter_q[ff, rc, rx, b2]
                        )
                        - tvm.tir.popcount(
                            PadInput_q[nn, rc, b1, xx * stride + rx] & ~Filter_q[ff, rc, rx, b2]
                        )
                    )
                    << (b1b2)
                ).astype(out_dtype),
                axis=[rc, rx, b2, b1],
            ).astype(out_dtype)

    else:

        def _conv(nn, ff, xx):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    tvm.tir.popcount(
                        PadInput_q[nn, rc, b1, xx * stride + rx] & Filter_q[ff, rc, rx, b2]
                    )
                    << (b1b2)
                ).astype(out_dtype),
                axis=[rc, rx, b2, b1],
            ).astype(out_dtype)

    return te.compute(
        (batch, out_channel, out_width),
        _conv,
        name="Conv1dOutput",
        tag="bitserial_conv1d_ncw",
    )


def bitserial_conv1d_nwc(
    data,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    """Bitserial Conv1D operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_width, in_channel]

    kernel : tvm.te.Tensor
        3-D with shape [filter_width, in_channel, num_filter]

    stride : int or a list/tuple of one int
        stride size, or [stride_width]

    padding : int or a list/tuple of two ints
        padding size, [pad_left, pad_right]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    unipolar: bool
        if binarization style is in unipolar 1/0 format, instead of bipolar -1/+1 format

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 1
    Input_q = bitpack(data, activation_bits, pack_axis=2, bit_axis=3, pack_type=pack_dtype)
    if len(kernel.shape) == 3:
        Filter_q = bitpack(kernel, weight_bits, pack_axis=1, bit_axis=3, pack_type=pack_dtype)
        kernel_w, _, num_filter, _ = get_const_tuple(Filter_q.shape)
    else:
        Filter_q = kernel
        kernel_w, _, _, num_filter = get_const_tuple(Filter_q.shape)
    batch, in_width, in_channel_q, _ = get_const_tuple(Input_q.shape)

    if isinstance(padding, int):
        LPAD = RPAD = padding
    else:
        LPAD, RPAD = padding
    pad_before = [0, LPAD, 0, 0]
    pad_after = [0, RPAD, 0, 0]

    # compute the output shape
    if not isinstance(stride, int):
        stride = stride[0]
    out_channel = num_filter
    out_width = (in_width - kernel_w + LPAD + RPAD) // stride + 1
    PadInput_q = pad(Input_q, pad_before, pad_after, name="PaddedInput")

    rc = te.reduce_axis((0, in_channel_q), name="rc")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    b1 = te.reduce_axis((0, activation_bits), name="b1")
    b2 = te.reduce_axis((0, weight_bits), name="b2")

    if unipolar:

        def _conv(nn, xx, ff):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    (
                        tvm.tir.popcount(
                            PadInput_q[nn, xx * stride + rx, rc, b1] & Filter_q[rx, rc, ff, b2]
                        )
                        - tvm.tir.popcount(
                            PadInput_q[nn, xx * stride + rx, rc, b1] & ~Filter_q[rx, rc, ff, b2]
                        )
                    )
                    << b1b2
                ).astype(out_dtype),
                axis=[rc, rx, b2, b1],
            )

    else:

        def _conv(nn, xx, ff):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    tvm.tir.popcount(
                        PadInput_q[nn, xx * stride + rx, rc, b1] & Filter_q[rx, rc, ff, b2]
                    )
                    << b1b2
                ).astype(out_dtype),
                axis=[rc, rx, b2, b1],
            )

    conv = te.compute(
        (batch, out_width, out_channel),
        _conv,
        name="Conv1dOutput",
        tag="bitserial_conv1d_nwc",
    )

    return conv


@tvm.target.generic_func
def bitserial_conv1d_legalize(attrs, inputs, types):
    """Legalizes Bitserial Conv1D op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # not to change by default
    return None
