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

from tvm import autotvm, te, tir
from tvm.autotvm.task import deserialize_args
from tvm.topi.utils import simplify, traverse_inline
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from tvm.tir.expr import Mul

from .micro_kernel.tensordot import (
    make_intrin_tensordot,
    tensordot_impl,
)


def conv2d_nhwc_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    print("Activating Conv2D NHWC schedule")
    """Compute function for v7e-m DSP instructions of conv2d."""
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch_size, in_height, in_width, in_channels = data.shape
    out_channels, kernel_h, kernel_w, _ = kernel.shape
    assert kernel.shape[3] == in_channels

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    padded_data = pad(data, pad_before, pad_after, name="padded_data")

    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    rc = te.reduce_axis((0, in_channels), name="rc")

    return te.compute(
        (batch_size, out_height, out_width, out_channels),
        lambda nn, yy, xx, ff: te.sum(
            padded_data[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * kernel[ff, ry, rx, rc].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="conv2d",
        tag="conv2d_nhwc",
    )


def _make_tensorization(padded_data, kernel):
    _, padded_h, padded_w, in_channels = padded_data.shape
    _, kernel_h, kernel_w, _ = kernel.shape
    in_dtype = padded_data.dtype
    suffix = "".join(random.choices(string.ascii_uppercase, k=8))
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


def conv2d_nhwc_dsp_schedule(cfg, outs):
    """Schedule function for v7e-m DSP instructions of conv2d."""
    schedule = te.create_schedule([x.op for x in outs])

    def _callback(operator):
        if "conv2d_nhwc" not in operator.tag:
            return

        # extract tensors
        output = operator.output(0)
        padded_data = output.op.input_tensors[0]
        kernel = output.op.input_tensors[1]

        b_ax, y_ax, x_ax, co_ax = schedule[output].op.axis
        kh_ax, kw_ax, ci_ax = schedule[output].op.reduce_axis
        schedule[output].reorder(b_ax, y_ax, x_ax, co_ax, kh_ax, kw_ax, ci_ax)

        intrin, code = _make_tensorization(padded_data, kernel)
        schedule[output].tensorize(kh_ax, intrin)
        schedule[output].pragma(b_ax, "import_c", code)

    traverse_inline(schedule, outs[-1].op, _callback)
    return schedule
