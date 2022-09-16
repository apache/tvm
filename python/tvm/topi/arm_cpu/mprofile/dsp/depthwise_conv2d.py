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

from tvm import te
from tvm.topi.utils import traverse_inline, get_const_tuple
from tvm.topi.nn.pad import pad
from tvm import tir

from .micro_kernel.quad_channel_convolve import (
    intrin_quad_channel_convolve,
    quad_channel_convolve_impl,
)

# For depthwise_conv2d, kernels are normally given in HWOI format,
# which when input_channels = output channels, we will call HWC.
# This is bad, as we want "related" parts of the kernel to be next
# to each other, so we can use __SMLAD later.
#
# Consider a 3x3 int8 kernel with no bias vector, with eight
# channels. Let us specify entries in the kernel as H_W_C - i.e.
# where 0_2_3 represents the rightmost position in the first row
# of channel 4/8 (4 because of zero indexing). Each [ ] represents
# a 32-bit integer. We currently store the kernel as:
#
# 0 ................................31
# [ 0_0_0 || 0_0_1 || 0_0_2 || 0_0_3 ] [ 0_0_4 || 0_0_5 || 0_0_6 || 0_0_7 ]
# [ 0_1_0 || 0_1_1 || 0_1_2 || 0_1_3 ] [ 0_1_4 || 0_1_5 || 0_1_6 || 0_1_7 ]
# [ 0_2_0 || 0_2_1 || 0_2_2 || 0_2_3 ] [ 0_2_4 || 0_2_5 || 0_2_6 || 0_2_7 ]
# [ 1_0_0 || 1_0_1 || 1_0_2 || 1_0_3 ] [ 1_0_4 || 1_0_5 || 1_0_6 || 1_0_7 ]
# [ 1_1_0 || 1_1_1 || 1_1_2 || 1_1_3 ] [ 1_1_4 || 1_1_5 || 1_1_6 || 1_1_7 ]
# [ 1_2_0 || 1_2_1 || 1_2_2 || 1_2_3 ] [ 1_2_4 || 1_2_5 || 1_2_6 || 1_2_7 ]
# [ 2_0_0 || 2_0_1 || 2_0_2 || 2_0_3 ] [ 2_0_4 || 2_0_5 || 2_0_6 || 2_0_7 ]
# [ 2_1_0 || 2_1_1 || 2_1_2 || 2_1_3 ] [ 2_1_4 || 2_1_5 || 2_1_6 || 2_1_7 ]
# [ 2_2_0 || 2_2_1 || 2_2_2 || 2_2_3 ] [ 2_2_4 || 2_2_5 || 2_2_6 || 2_2_7 ]
#
# Let 0x00 be all zeros. We rearrange into:
#
# 0 ................................31
# [ 0_0_0 || 0_0_1 || 0_1_0 || 0_1_1 ] [ 0_0_2 || 0_0_3 || 0_1_2 || 0_1_3 ]
# [ 0_2_0 || 0_2_1 || 1_0_0 || 1_0_1 ] [ 0_2_2 || 0_2_3 || 1_0_2 || 1_0_3 ]
# [ 1_1_0 || 1_1_1 || 1_2_0 || 1_2_1 ] [ 1_1_2 || 1_1_3 || 1_2_2 || 1_2_3 ]
# [ 2_0_0 || 2_0_1 || 2_1_0 || 2_1_1 ] [ 2_0_2 || 2_0_3 || 2_1_2 || 2_1_3 ]
# [ 2_2_0 || 2_2_1 || 0x000 || 0x000 ] [ 2_2_2 || 2_2_3 || 0x000 || 0x000 ]
# [ 0_0_4 || 0_0_5 || 0_1_4 || 0_1_5 ] [ 0_0_6 || 0_0_7 || 0_1_6 || 0_1_7 ]
# [ 0_2_4 || 0_2_5 || 1_0_4 || 1_0_5 ] [ 0_2_6 || 0_2_7 || 1_0_6 || 1_0_7 ]
# [ 1_1_4 || 1_1_5 || 1_2_4 || 1_2_5 ] [ 1_1_6 || 1_1_7 || 1_2_6 || 1_2_7 ]
# [ 2_0_4 || 2_0_5 || 2_1_4 || 2_1_5 ] [ 2_0_6 || 2_0_7 || 2_1_6 || 2_1_7 ]
# [ 2_2_4 || 2_2_5 || 0x000 || 0x000 ] [ 2_2_6 || 2_2_7 || 0x000 || 0x000 ]
#
# This saves us six operations comapred to the original ordering, as we
# do not need halfword packing instructions.
#
# This kernel re-arranging function will be used for 3x3 kernels (as that
# is all this DSP implementation currently supports) but would work with
# any M*N kernel such that M*N is odd.


def _rearrange_kernel(kernel):
    # Kernel must be HWC format.
    kernel_h, kernel_w, channels, _ = get_const_tuple(kernel.shape)
    assert channels % 4 == 0

    # This restriction could be removed by only using tir.if_then_else to add padding
    # zeros if (kernel_w * kernel_h) % 2 == 1, and filling completely otherwise.
    assert (kernel_w * kernel_h) % 2 == 1

    def fcompute(c_o, pos, c_i):
        channel = (2 * (pos % 2)) + (c_i % 2) + (4 * c_o)
        true_pos_index = 2 * (pos // 2) + (c_i // 2)

        return tir.if_then_else(
            true_pos_index < (kernel_h * kernel_w),
            kernel[true_pos_index // kernel_w, true_pos_index % kernel_w, channel, 0],
            tir.const(0, "int8"),
        )

    return te.compute(
        (channels // 4, kernel_h * kernel_w + 1, 4),
        fcompute,
        name="packed_kernel",
    )


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

    # We require that the number of channels be divisible by 4. This restriction could
    # be removed with strip mining if people cared.
    assert channels % 4 == 0

    # We don't support different numbers of input and output channels.
    assert channels == kernel.shape[2]
    assert kernel.shape[3] == 1

    # We take in int8 as our dtype, but we spit out int32. This is because we cannot
    # round until we compute activations.
    assert out_dtype == "int32"

    # This can pretty easily be generalized in the future. Likely worth doing, and this
    # function was written to make doing so easy. Should only require adding more calls
    # to QUAD_CHANNEL_REARRANGE_SUM.
    assert kernel_w == kernel_h == 3

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

    packed_kernel = _rearrange_kernel(kernel)
    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")
    return te.compute(
        (batch_size, output_h, output_w, channels),
        lambda h, i, j, k: te.sum(
            padded_data[h, (i * stride_h) + kh_i, (j * stride_w) + kw_i, k].astype("int32")
            * packed_kernel[
                k // 4,
                (2 * ((3 * kh_i + kw_i) // 2)) + ((k % 4) // 2),
                (2 * ((kh_i + kw_i) % 2)) + (k % 2),
            ].astype("int32"),
            axis=(kh_i, kw_i),
        ),
        name="depthwise_conv2d",
        tag=f"depthwise_conv2d_nhwc_{padded_h}_{padded_w}_dsp",
    )


def depthwise_conv2d_nhwc_dsp_schedule(_cfg, outs):

    """Schedule function for v7e-m DSP instructions of conv2d."""
    schedule = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "depthwise_conv2d_nhwc" not in op.tag:
            return

        # extract tensors
        output = op.output(0)
        padded_data = output.op.input_tensors[0]
        packed_kernel = output.op.input_tensors[1]
        kernel = packed_kernel.op.input_tensors[0]

        _, _, padded_w, channels = padded_data.shape
        kernel_h, kernel_w, _, _ = kernel.shape
        suffix = "".join(random.choices(string.ascii_uppercase, k=8))

        b_ax, y_ax, x_ax, c_ax = schedule[output].op.axis
        ky_ax, kx_ax = schedule[output].op.reduce_axis
        c_ax_o, c_ax_i = schedule[output].split(c_ax, factor=4)
        schedule[output].reorder(b_ax, c_ax_o, y_ax, x_ax, ky_ax, kx_ax, c_ax_i)

        quad_channel_convolve = intrin_quad_channel_convolve(
            padded_w, channels, kernel_h, kernel_w, suffix
        )
        schedule[output].tensorize(ky_ax, quad_channel_convolve)
        schedule[output].pragma(
            b_ax,
            "import_c",
            quad_channel_convolve_impl(padded_w, channels, kernel_h, kernel_w, suffix),
        )

    traverse_inline(schedule, outs[-1].op, _callback)
    return schedule
