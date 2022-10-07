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
"""Implementations of several conv2d variations, all tensorized using tensordot and optimized for
Cortex-M DSP. Currently contains a standard conv2d and depthwise conv2d implementation, but could be
extended to add a grouped conv2d operator. Due to the way we tensorize, this schedule ONLY works
when the data and kernel layouts are NCHWxc and OIHWxi respectively, where x is the number of
input channels divided by the number of groups."""

import random
import string
from typing import Union, Tuple

from tvm import te
from tvm.tir import indexdiv, indexmod
from tvm.topi.utils import traverse_inline
from tvm.topi.nn.pad import pad

from .micro_kernel.tensordot import (
    make_intrin_tensordot,
    tensordot_impl,
)


def _unpack_2d_argument(argument: Union[int, Tuple]) -> Tuple:
    if isinstance(argument, int):
        return (argument, argument)
    assert len(argument) == 2
    return argument


def _check_no_dilation(dilation: Union[int, Tuple]) -> None:
    """Takes a dilation argument as an integer or tuple, and makes sure both dimensions are 1.
    Dilation prevents us from using DSP instructions, so this schedule can't work (aside from the
    niche case where dilation_h == stride_h and dilation_w == stride_w, which is rare enough we
    probably don't need to support it)."""

    dilation_h, dilation_w = _unpack_2d_argument(dilation)
    assert dilation_h == dilation_w == 1


def _unpack_padding(padding: Tuple) -> Tuple:
    assert isinstance(padding, tuple)
    if len(padding) == 2:
        (pad_up, pad_down), (pad_left, pad_right) = padding
    else:
        pad_up, pad_left, pad_down, pad_right = padding
    return pad_up, pad_left, pad_down, pad_right


def _pad_if_needed(data: te.tensor.Tensor, layout: str, padding: Tuple) -> te.tensor.Tensor:
    """Performs padding on a te.tensor.Tensor object if necessary. If padding = (0, 0, 0, 0), the
    input tensor is returned unmodified. We only care about tuples here - "VALID" and "SAME" padding
    will be converted by the importer TFLite importer if present."""

    pad_up, pad_left, pad_down, pad_right = padding
    if not any(padding):
        return data

    # We want to pad the "H" and "W" columns, and their position depends on the layout
    pad_before, pad_after = [0, 0, 0, 0], [0, 0, 0, 0]
    pad_before[layout.index("H")] = pad_up
    pad_before[layout.index("W")] = pad_left
    pad_after[layout.index("H")] = pad_down
    pad_after[layout.index("W")] = pad_right
    return pad(data, pad_before, pad_after, name="padded_data")


def _compute_output_dim(
    data_dim: int, kernel_dim: int, pad_before: int, pad_after: int, stride: int
) -> int:
    """Computes an output dimension of a convolution, given the data dimension, kernel dimension,
    padding, and stride along that axis. Note that when stride > 1, this division will often not
    be perfectly even."""
    return (data_dim + pad_before + pad_after - kernel_dim) // stride + 1


def _get_suffix() -> str:
    """Returns a random eight-character string to append to C function names. Prevents accidental
    re-definition of functions if the same operator appears twice in a Relay graph."""
    return "".join(random.choices(string.ascii_uppercase, k=8))


def conv2d_nhwc_ohwi_dsp_compute(_cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Standard conv2d schedule that can be tensorized using tensordot."""

    stride_h, stride_w = _unpack_2d_argument(strides)
    pad_up, pad_left, pad_down, pad_right = _unpack_padding(padding)
    _check_no_dilation(dilation)

    batch_size, data_h, data_w, in_channels = data.shape
    output_channels, kernel_h, kernel_w, _ = kernel.shape
    assert kernel.shape[3] == in_channels

    output_h = _compute_output_dim(data_h, kernel_h, pad_up, pad_down, stride_h)
    output_w = _compute_output_dim(data_w, kernel_w, pad_left, pad_right, stride_w)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")
    kc_i = te.reduce_axis((0, in_channels), name="rc")

    padded_data = _pad_if_needed(data, "NHWC", (pad_up, pad_left, pad_down, pad_right))
    return te.compute(
        (batch_size, output_h, output_w, output_channels),
        lambda n, y, x, c: te.sum(
            padded_data[n, y * stride_h + kh_i, x * stride_w + kw_i, kc_i].astype(out_dtype)
            * kernel[c, kh_i, kw_i, kc_i].astype(out_dtype),
            axis=(kh_i, kw_i, kc_i),
        ),
        name="conv2d",
        tag="conv2d_nhwc_ohwi_dsp",
    )


def _make_conv2d_tensorization(padded_data, kernel):
    _, _, padded_w, in_channels = padded_data.shape
    _, kernel_h, kernel_w, _ = kernel.shape
    in_dtype = padded_data.dtype
    suffix = _get_suffix()
    assert in_dtype == kernel.dtype

    data_slice = te.placeholder((kernel_h, kernel_w, in_channels), name="a", dtype=in_dtype)
    kernel_slice = te.placeholder((kernel_h, kernel_w, in_channels), name="b", dtype=in_dtype)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")
    kc_i = te.reduce_axis((0, in_channels), name="kc_i")

    output_slice = te.compute(
        (1,),
        lambda k: te.sum(
            data_slice[kh_i, kw_i, kc_i].astype("int32")
            * kernel_slice[kh_i, kw_i, kc_i].astype("int32"),
            axis=[kh_i, kw_i, kc_i],
        ),
        name="c",
    )

    # TVM has a really strange bug where the outer reduction axis (kh_i) having length 1 causes the
    # decl_buffer strides check to fail. height_stride is a dark magic workaround for this.
    height_stride = in_channels * padded_w if kernel_h > 1 else in_channels
    jump = (padded_w - kernel_w) * in_channels
    tensordot_params = (in_dtype, kernel_h, jump, kernel_w * in_channels, suffix)
    intrin_tensordot = make_intrin_tensordot(
        (data_slice, kernel_slice, output_slice),
        ([height_stride, in_channels, 1], [kernel_w * in_channels, in_channels, 1]),
        tensordot_params,
    )

    tensordot_code = tensordot_impl(*tensordot_params)
    return (intrin_tensordot, tensordot_code)


def depthwise_conv2d_nchw_oihw_dsp_compute(
    _cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Depthwise conv2d schedule that can be tensorized using tensordot."""

    stride_h, stride_w = _unpack_2d_argument(strides)
    pad_up, pad_left, pad_down, pad_right = _unpack_padding(padding)
    _check_no_dilation(dilation)

    batch_size, in_channels, data_h, data_w = data.shape
    _, c_mul, kernel_h, kernel_w = kernel.shape
    output_channels = in_channels * c_mul
    assert kernel.shape[0] == in_channels

    output_h = _compute_output_dim(data_h, kernel_h, pad_up, pad_down, stride_h)
    output_w = _compute_output_dim(data_w, kernel_w, pad_left, pad_right, stride_w)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")

    padded_data = _pad_if_needed(data, "NCHW", (pad_up, pad_left, pad_down, pad_right))
    return te.compute(
        (batch_size, output_channels, output_h, output_w),
        lambda n, c, y, x: te.sum(
            padded_data[
                n,
                indexdiv(c, c_mul),
                y * stride_h + kh_i,
                x * stride_w + kw_i,
            ].astype(out_dtype)
            * kernel[indexdiv(c, c_mul), indexmod(c, c_mul), kh_i, kw_i].astype(out_dtype),
            axis=(kh_i, kw_i),
        ),
        name="depthwise_conv2d",
        tag="depthwise_conv2d_nchw_oihw_dsp",
    )


def _make_depthwise_conv2d_tensorization(padded_data, kernel):
    _, _, _, padded_w = padded_data.shape
    _, _, kernel_h, kernel_w = kernel.shape

    in_dtype = padded_data.dtype
    suffix = _get_suffix()
    assert in_dtype == kernel.dtype

    data_slice = te.placeholder((kernel_h, kernel_w), name="a", dtype=in_dtype)
    kernel_slice = te.placeholder((kernel_h, kernel_w), name="b", dtype=in_dtype)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")

    output_slice = te.compute(
        (1,),
        lambda k: te.sum(
            data_slice[kh_i, kw_i].astype("int32") * kernel_slice[kh_i, kw_i].astype("int32"),
            axis=[kh_i, kw_i],
        ),
        name="c",
    )

    jump = padded_w - kernel_w
    tensordot_params = (in_dtype, kernel_h, jump, kernel_w, suffix)
    intrin_tensordot = make_intrin_tensordot(
        (data_slice, kernel_slice, output_slice),
        ([padded_w, 1], [kernel_w, 1]),
        tensordot_params,
    )

    tensordot_code = tensordot_impl(*tensordot_params)
    return (intrin_tensordot, tensordot_code)


def tensordot_conv2ds_schedule(_cfg, outs):
    """Schedule function using v7e-m DSP instructions for all the conv2d operators in this file. We
    use one schedule function for them all, because they are tensorized with the same kernel."""

    schedule = te.create_schedule([x.op for x in outs])

    def _callback(operator):
        if "conv2d" in operator.tag:
            output = operator.output(0)
            padded_data = output.op.input_tensors[0]
            kernel = output.op.input_tensors[1]

            if operator.tag == "conv2d_nhwc_ohwi_dsp":
                b_ax, y_ax, x_ax, co_ax = schedule[output].op.axis
                kh_ax, kw_ax, ci_ax = schedule[output].op.reduce_axis
                schedule[output].reorder(b_ax, y_ax, x_ax, co_ax, kh_ax, kw_ax, ci_ax)
                intrin, code = _make_conv2d_tensorization(padded_data, kernel)

            elif operator.tag == "depthwise_conv2d_nchw_oihw_dsp":
                b_ax, y_ax, x_ax, co_ax = schedule[output].op.axis
                kh_ax, kw_ax = schedule[output].op.reduce_axis
                schedule[output].reorder(b_ax, co_ax, y_ax, x_ax, kh_ax, kw_ax)
                intrin, code = _make_depthwise_conv2d_tensorization(padded_data, kernel)

            else:
                raise ValueError(f"Cannot tensorize {operator.tag} with tensordot!")

            schedule[output].tensorize(kh_ax, intrin)
            schedule[output].pragma(b_ax, "import_c", code)

    traverse_inline(schedule, outs[-1].op, _callback)
    return schedule
