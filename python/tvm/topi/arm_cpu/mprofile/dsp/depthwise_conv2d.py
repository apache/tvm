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
"""ARM Cortex-M DSP schedule for depthwise_conv2d"""

import random
import string

from tvm import te, topi
from tvm.topi.utils import traverse_inline
from tvm.topi.nn.pad import pad

from .micro_kernel.multi_channel_convolve import (
    intrin_multi_channel_convolve,
    multi_channel_convolve_impl,
)
from .micro_kernel.common import num_simd_lanes_per_word


def depthwise_conv2d_nhwc_dsp_compute(_cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute function for v7e-m DSP instructions of DepthwiseConv2D. Has a lot of requirements
    for use - if not all apply, the fallback implementation will be used instead."""
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    # We do not support dilation currently. It would be possible, but it would require
    # modifying the way the kernel is packed. Gnarly.
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert dilation_h == dilation_w == 1

    batch_size, height, width, channels = data.shape
    kernel_h, kernel_w, _, _ = kernel.shape
    simd_lanes = num_simd_lanes_per_word(data.dtype)

    # We don't support different numbers of input and output channels.
    assert channels == kernel.shape[2]
    assert kernel.shape[3] == 1

    # We take in int8 as our dtype, but we spit out int32. This is because we cannot
    # round until we compute activations.
    assert out_dtype == "int32"

    # Padding the data requires COPYING THE ENTIRE INPUT TENSOR, which
    # is slow and bad. We should really implement a strip mining
    # routine to avoid this, but TVM has terrible support for that.

    if padding == "SAME":
        # This assumption makes the logic easier. Could be removed with work.
        assert height % stride_h == width % stride_w == 0

        output_h = height // stride_h
        output_w = width // stride_w

        # This padding behavior is consistent with other TVM depthwise_conv2d schedules. However it
        # differs from the TensorFlow, which only pads the bottom right if stride > 1. This probably
        # brings down accuracy slightly for models imported from TFLite.
        pad_down = 1 if stride_h == 1 else 0
        pad_right = 1 if stride_w == 1 else 0

        padded_data = pad(
            data,
            [0, kernel_h // 2, kernel_w // 2, 0],
            [0, pad_down, pad_right, 0],
            name="padded_data",
        )

    elif padding == "VALID":
        assert height > kernel_h and width > kernel_w
        output_h = (height - kernel_h) // stride_h + 1
        output_w = (width - kernel_w) // stride_w + 1
        padded_data = data

    elif isinstance(padding, tuple):
        if len(padding) == 2:
            pad_up, pad_down = padding[0]
            pad_left, pad_right = padding[1]
        else:
            pad_up, pad_left, pad_down, pad_right = padding

        output_h = (height - kernel_h + pad_up + pad_down) // stride_h + 1
        output_w = (width - kernel_w + pad_left + pad_right) // stride_w + 1
        padded_data = pad(
            data,
            [0, pad_up, pad_left, 0],
            [0, pad_down, pad_right, 0],
            name="padded_data",
        )

    else:
        raise RuntimeError()
    _, padded_h, padded_w, _ = padded_data.shape

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")
    reshaped_kernel = topi.reshape(kernel, (channels // simd_lanes, kernel_h, kernel_w, simd_lanes))
    return te.compute(
        (batch_size, output_h, output_w, channels),
        lambda h, i, j, k: te.sum(
            padded_data[h, (i * stride_h) + kh_i, (j * stride_w) + kw_i, k].astype("int32")
            * reshaped_kernel[k // simd_lanes, kh_i, kw_i, k % simd_lanes].astype("int32"),
            axis=(kh_i, kw_i),
        ),
        name="depthwise_conv2d",
        tag=f"depthwise_conv2d_nhwc_{padded_h}_{padded_w}_dsp",
    )


def depthwise_conv2d_nhwc_dsp_schedule(_cfg, outs):

    """Schedule function for v7e-m DSP instructions of conv2d."""
    schedule = te.create_schedule([x.op for x in outs])

    def _callback(operator):
        if "depthwise_conv2d_nhwc" not in operator.tag:
            return

        # extract tensors
        output = operator.output(0)
        padded_data = output.op.input_tensors[0]
        reshaped_kernel = output.op.input_tensors[1]
        in_dtype = padded_data.dtype

        _, padded_h, padded_w, channels = padded_data.shape
        _, kernel_h, kernel_w, _ = reshaped_kernel.shape
        suffix = "".join(random.choices(string.ascii_uppercase, k=8))

        b_ax, y_ax, x_ax, c_ax = schedule[output].op.axis
        ky_ax, kx_ax = schedule[output].op.reduce_axis
        simd_lanes = num_simd_lanes_per_word(in_dtype)
        c_ax_o, c_ax_i = schedule[output].split(c_ax, factor=simd_lanes)
        schedule[output].reorder(b_ax, c_ax_o, y_ax, x_ax, ky_ax, kx_ax, c_ax_i)

        multi_channel_convolve = intrin_multi_channel_convolve(
            in_dtype, padded_h, padded_w, channels, kernel_h, kernel_w, suffix
        )
        schedule[output].tensorize(ky_ax, multi_channel_convolve)
        schedule[output].pragma(
            b_ax,
            "import_c",
            multi_channel_convolve_impl(
                in_dtype, padded_h, padded_w, channels, kernel_h, kernel_w, suffix
            ),
        )

    traverse_inline(schedule, outs[-1].op, _callback)
    return schedule
