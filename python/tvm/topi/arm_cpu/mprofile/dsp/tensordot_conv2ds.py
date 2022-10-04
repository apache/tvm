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
# pylint: disable=invalid-name, no-value-for-parameter
"""Direct implementation of conv2d."""

import random
import string

from tvm import te, tir
from tvm.tir import indexdiv, indexmod
from tvm.topi.utils import traverse_inline
from tvm.topi.nn.pad import pad

from .micro_kernel.tensordot import (
    make_intrin_tensordot,
    tensordot_impl,
)


# Dilation prevents us from using DSP instructions, so this schedule can't work (aside from the
# niche case where dilation_h == stride_h and dilation_w == stride_w, which is rare enough we
# probably don't need to support it).
def _check_no_dilation(dilation):
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert dilation_h == dilation_w == 1


def _unpack_strides(strides):
    assert isinstance(strides, int) or len(strides) == 2
    if isinstance(strides, int):
        return (strides, strides)
    else:
        return strides


def _unpack_padding(padding):
    assert isinstance(padding, tuple)
    if len(padding) == 2:
        (pad_up, pad_down), (pad_left, pad_right) = padding
    else:
        pad_up, pad_left, pad_down, pad_right = padding
    return pad_up, pad_left, pad_down, pad_right


# We only care about tuples here - "VALID" and "SAME" padding will be converted by the importer.
def _pad_if_needed(data, pad_up, pad_left, pad_down, pad_right):
    if pad_up or pad_left or pad_down or pad_right:
        return pad(
            data,
            [0, pad_up, pad_left, 0],
            [0, pad_down, pad_right, 0],
            name="padded_data"
        )
    else:
        return data


def _compute_output_dim(data_dim, kernel_dim, pad_before, pad_after, stride):
    return (data_dim - kernel_dim + pad_before + pad_after) // stride + 1


# Offsets to "prefer" the bottom right corner. This is done to match TensorFlow's convention, but
# does NOT match the other TVM schedules. We violate this convention because it improves accuracy on
# models imported from TensorFlow.
def _compute_offset(data_dim, kernel_dim, pad_before, pad_after, stride):
    return (data_dim - kernel_dim + pad_before + pad_after) % stride


# Prevents re-definition of C functions
def _get_suffix():
    return "".join(random.choices(string.ascii_uppercase, k=8))


def conv2d_nhwc_ohwi_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    stride_h, stride_w = _unpack_strides(strides)
    pad_up, pad_left, pad_down, pad_right = _unpack_padding(padding)
    _check_no_dilation(dilation)

    batch_size, data_h, data_w, in_channels = data.shape
    output_channels, kernel_h, kernel_w, _ = kernel.shape
    assert kernel.shape[3] == in_channels

    output_h = _compute_output_dim(data_h, kernel_h, pad_up, pad_down, stride_h)
    output_w = _compute_output_dim(data_w, kernel_w, pad_left, pad_right, stride_w)
    y_offset = _compute_offset(data_h, kernel_h, pad_up, pad_down, stride_h)
    x_offset = _compute_offset(data_w, kernel_w, pad_left, pad_right, stride_w)

    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    rc = te.reduce_axis((0, in_channels), name="rc")

    padded_data = _pad_if_needed(data, pad_up, pad_left, pad_down, pad_right)
    return te.compute(
        (batch_size, output_h, output_w, output_channels),
        lambda n, y, x, c: te.sum(
            padded_data[
                n, y_offset + y * stride_h + ry, x_offset + x * stride_w + rx, rc
            ].astype(out_dtype)
            * kernel[c, ry, rx, rc].astype(out_dtype),
            axis=(ry, rx, rc),
        ),
        name="conv2d",
        tag="conv2d_nhwc_ohwi_dsp",
    )


def _make_conv2d_tensorization(padded_data, kernel):
    _, padded_h, padded_w, in_channels = padded_data.shape
    _, kernel_h, kernel_w, _ = kernel.shape
    in_dtype = padded_data.dtype
    suffix = _get_suffix()
    assert in_dtype == kernel.dtype

    data_slice = te.placeholder((kernel_h, kernel_w, in_channels), name="a", dtype=in_dtype)
    kernel_slice = te.placeholder((kernel_h, kernel_w, in_channels), name="b", dtype=in_dtype)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")
    kc_i = te.reduce_axis((0, in_channels), name="kc_i")

    output_slice = te.compute((1,),
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
    data_buf = tir.decl_buffer(
        data_slice.shape, data_slice.dtype, name="data", offset_factor=1,
        strides=[height_stride, in_channels, 1],
    )
    kernel_buf = tir.decl_buffer(
        kernel_slice.shape, kernel_slice.dtype, name="kernel", offset_factor=1,
        strides=[kernel_w * in_channels, in_channels, 1]
    )
    output_buf = tir.decl_buffer(
        output_slice.shape, output_slice.dtype, name="output", offset_factor=1, strides=[1]
    )

    jump = (padded_w - kernel_w) * in_channels
    tensordot_params = (in_dtype, kernel_h, jump, kernel_w * in_channels, suffix)

    intrin_tensordot = make_intrin_tensordot(
        output_slice.op,
        {data_slice: data_buf, kernel_slice: kernel_buf, output_slice: output_buf},
        tensordot_params
    )

    tensordot_code = tensordot_impl(*tensordot_params)
    return (intrin_tensordot, tensordot_code)


def depthwise_conv2d_nchw_oihw_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    stride_h, stride_w = _unpack_strides(strides)
    pad_up, pad_left, pad_down, pad_right = _unpack_padding(padding)
    _check_no_dilation(dilation)

    batch_size, in_channels, data_h, data_w = data.shape
    _, c_mul, kernel_h, kernel_w = kernel.shape
    output_channels = in_channels * c_mul
    print(data.shape)
    print(kernel.shape)
    assert kernel.shape[0] == in_channels

    output_h = _compute_output_dim(data_h, kernel_h, pad_up, pad_down, stride_h)
    output_w = _compute_output_dim(data_w, kernel_w, pad_left, pad_right, stride_w)
    y_offset = _compute_offset(data_h, kernel_h, pad_up, pad_down, stride_h)
    x_offset = _compute_offset(data_w, kernel_w, pad_left, pad_right, stride_w)

    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    padded_data = _pad_if_needed(data, pad_up, pad_left, pad_down, pad_right)
    return te.compute(
        (batch_size, output_h, output_w, output_channels),
        lambda n, y, x, c: te.sum(
            padded_data[
                n, indexdiv(c, c_mul), y_offset + y * stride_h + ry, x_offset + x * stride_w + rx,
            ].astype(out_dtype)
            * kernel[indexdiv(c, c_mul), indexmod(c, c_mul), ry, rx].astype(out_dtype),
            axis=(ry, rx),
        ),
        name="depthwise_conv2d",
        tag="depthwise_conv2d_nchw_oihw_dsp",
    )

def _make_depthwise_conv2d_tensorization(padded_data, kernel):
    _, _, padded_h, padded_w = padded_data.shape
    _, _, kernel_h, kernel_w = kernel.shape

    in_dtype = padded_data.dtype
    suffix = _get_suffix()
    assert in_dtype == kernel.dtype

    data_slice = te.placeholder((kernel_h, kernel_w), name="a", dtype=in_dtype)
    kernel_slice = te.placeholder((kernel_h, kernel_w), name="b", dtype=in_dtype)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")

    output_slice = te.compute((1,),
        lambda k: te.sum(
            data_slice[kh_i, kw_i].astype("int32")
            * kernel_slice[kh_i, kw_i].astype("int32"),
            axis=[kh_i, kw_i],
        ),
        name="c",
    )

    data_buf = tir.decl_buffer(
        data_slice.shape, data_slice.dtype, name="data", offset_factor=1,
        strides=[padded_w, 1],
    )
    kernel_buf = tir.decl_buffer(
        kernel_slice.shape, kernel_slice.dtype, name="kernel", offset_factor=1,
        strides=[kernel_w, 1]
    )
    output_buf = tir.decl_buffer(
        output_slice.shape, output_slice.dtype, name="output", offset_factor=1, strides=[1]
    )

    jump = padded_w - kernel_w
    tensordot_params = (in_dtype, kernel_h, jump, kernel_w, suffix)

    intrin_tensordot = make_intrin_tensordot(
        output_slice.op,
        {data_slice: data_buf, kernel_slice: kernel_buf, output_slice: output_buf},
        tensordot_params
    )

    tensordot_code = tensordot_impl(*tensordot_params)
    return (intrin_tensordot, tensordot_code)


def tensordot_conv2ds_schedule(cfg, outs):
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
                print(code)

            schedule[output].tensorize(kh_ax, intrin)
            schedule[output].pragma(b_ax, "import_c", code)

    traverse_inline(schedule, outs[-1].op, _callback)
    return schedule
