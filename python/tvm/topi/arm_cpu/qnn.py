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
"""Contains TVMScript implementations of some QNN operators for Arm.

Currently, the only ops with compute functions are fused regular and depthwise convolutions for
Arm Cortex-M with DSP.
"""

from typing import Tuple

import tvm
from tvm import te
from tvm.tir import const
from tvm.script import tir as T
from ..utils import get_const_tuple
from .mprofile.dsp.micro_kernel import tensordot


def int_ceil_division(x, y):
    return -(x // -y)


def _compute_output_dim(data_length, kernel_length, stride):
    return int_ceil_division(data_length + 1 - kernel_length, stride)


def _pick_tensordot_impl(attrs, inputs, num_outputs=2, is_depthwise=False):
    """Helper function that chooses the right implementation of micro_kernel.tensordot.

    Takes as input the parameters of the conv2d, and returns a tuple of TWO (function_name,
    function_code). The first pair (the aligned one) is for even numbered output channels, and the
    second pair (the offset one) is for odd-numbered output channels. This function is used for
    regular and depthwise convolutions.

    We need different implementations for even vs odd numbered output channels, because the "start"
    of an odd output channel in the data tensor or kernel might or might not be on a word boundary,
    and the tensordot code expects all input pointers to be word-aligned.
    """
    data, kernel = inputs[0:2]
    rq_output_zero_point_const = inputs[10]
    assert len(rq_output_zero_point_const.op.body) == 1
    output_zero_point = rq_output_zero_point_const.op.body[0]

    _, stride_w = get_const_tuple(attrs.strides)

    if is_depthwise:
        assert attrs.data_layout == "NCHW"
        assert attrs.kernel_layout == "IOHW"
        _, _, height, width = get_const_tuple(data.shape)
        _, out_channels, kernel_h, kernel_w = get_const_tuple(kernel.shape)

        dimensions = (width, kernel_h, kernel_w)
        in_stride = stride_w
        data_per_oc_size = height * width
    else:
        assert attrs.data_layout == "NHWC"
        assert attrs.kernel_layout == "OHWI"
        _, height, width, in_channels = get_const_tuple(data.shape)
        out_channels, kernel_h, kernel_w, _ = get_const_tuple(kernel.shape)

        dimensions = (width * in_channels, kernel_h, kernel_w * in_channels)
        in_stride = in_channels * stride_w
        data_per_oc_size = 0

    assert attrs.out_layout is not None
    if attrs.out_layout == "NHWC":
        out_stride = out_channels
    elif attrs.out_layout == "NCHW":
        out_stride = 1
    else:
        raise ValueError(f"Unsupported output layout {attrs.out_layout}!")

    x_strides = (in_stride, out_stride)
    aligned_func = tensordot.tensordot_int16_impl(
        num_outputs,
        dimensions,
        (0, 0, 0),
        x_strides,
        output_zero_point=output_zero_point,
    )

    kernel_per_oc_size = dimensions[1] * dimensions[2]

    offsets = (data_per_oc_size % 2, kernel_per_oc_size % 2, 0)
    offset_func = tensordot.tensordot_int16_impl(
        num_outputs,
        dimensions,
        offsets,
        x_strides,
        output_zero_point=output_zero_point,
    )

    return (aligned_func, offset_func)


def _make_tscript_ptr(buffer, offset, length, dtype="int16"):
    return T.tvm_access_ptr(
        T.type_annotation(dtype=dtype),
        buffer.data,
        offset,
        length,
        1,
        dtype="handle",
    )


def _make_tscript_call(func_name, *args):
    return T.evaluate(T.call_extern(func_name, *args, dtype="int32"))


def _make_conv2d_primfunc(
    call_dimensions: Tuple,
    buffer_shapes: Tuple[Tuple, Tuple, Tuple, Tuple, Tuple],
    aligned_func: Tuple[str, str],
    offset_func: Tuple[str, str],
    ptr_gens: Tuple,
):
    height, width, out_channels = call_dimensions
    data_shape, kernel_shape, bias_shape, scale_shape, output_shape = buffer_shapes
    aligned_func_name, aligned_func_code = aligned_func
    offset_func_name, offset_func_code = offset_func
    output_ptr, data_ptr, kernel_ptr = ptr_gens

    # If the functions are identical, we can skip the second loop
    if aligned_func_name == offset_func_name:
        aligned_channels = out_channels
        offset_channels = tvm.tir.const(0)
        c_step = tvm.tir.const(1)
    else:
        aligned_channels = out_channels // 2
        offset_channels = out_channels // 2
        c_step = tvm.tir.const(2)

    def bias_ptr(bias, c):
        return _make_tscript_ptr(bias, c, 1, dtype="int32")

    def scale_ptr(scale, c):
        return _make_tscript_ptr(scale, c, 1, dtype="int32")

    @T.prim_func
    def biased_quantized_conv2d(
        data_handle: T.handle,
        kernel_handle: T.handle,
        bias_handle: T.handle,
        scale_handle: T.handle,
        output_handle: T.handle,
    ) -> None:

        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        data = T.match_buffer(data_handle, data_shape, dtype="int16")
        kernel = T.match_buffer(kernel_handle, kernel_shape, dtype="int16")
        bias = T.match_buffer(bias_handle, bias_shape, dtype="int32")

        # We don't specify a data type for the requantization scale, even though we will read it as
        # an int32. This is because we must pretend it is a float32, as Relay's requantize op only
        # allows floating point scales.
        scale = T.match_buffer(scale_handle, scale_shape)
        output = T.match_buffer(output_handle, output_shape, dtype="int16")

        # This hack prevents TVM from seeing these variables as "unused". I should be using T.reads
        # and T.writes, but they don't work. I think it's an issue with BufferTouchedDomain.
        # pylint: disable=unused-variable
        output[0, 0, 0, 0] = 0
        __1 = data[0, 0, 0, 0]
        __2 = kernel[0, 0, 0, 0]
        __3 = bias[0, 0, 0, 0]
        __4 = scale[0]
        # pylint: enable=unused-variable

        for c_ax, y_ax, x_ax in T.grid(aligned_channels, height, width):
            with T.block("conv2d_aligned"):
                T.block_attr({"pragma_import_c": aligned_func_code})
                y, x, c = T.axis.remap("SSS", [y_ax, x_ax, c_ax])
                _make_tscript_call(
                    aligned_func_name,
                    output_ptr(output, y, x, c * c_step),
                    data_ptr(data, y, x, c * c_step),
                    kernel_ptr(kernel, c * c_step),
                    bias_ptr(bias, c * c_step),
                    scale_ptr(scale, c * c_step),
                )

        for c_ax, y_ax, x_ax in T.grid(offset_channels, height, width):
            with T.block("conv2d_offset"):
                T.block_attr({"pragma_import_c": offset_func_code})
                y, x, c = T.axis.remap("SSS", [y_ax, x_ax, c_ax])
                _make_tscript_call(
                    offset_func_name,
                    output_ptr(output, y, x, c * c_step + 1),
                    data_ptr(data, y, x, c * c_step + 1, offset=1),
                    kernel_ptr(kernel, c * c_step + 1, offset=1),
                    bias_ptr(bias, c * c_step + 1),
                    scale_ptr(scale, c * c_step + 1),
                )

    return biased_quantized_conv2d


def qnn_conv2d(attrs, inputs, out_type):
    """Compute for qnn.conv2d with NHWC layout.

    Note that this is a DIFFERENT layout from the Hexagon variant, because they have special
    instructions Cortex-M doesn't have. We expect the kernel to have OHWI layout. We also assume
    that padding is not necessary, as it will have been done by another pass.
    """

    # Make a few checks to unpack the function arguments and ensure it was called with the right
    # arguments. Note that unlike most schedules, qnn_conv2d does not use a wrapper.
    assert len(inputs) == 11
    data, kernel, _izp, _kzp, _iscale, _kscale, bias, scale = inputs[0:8]
    output_layout = attrs.out_layout
    assert output_layout == "NHWC"

    _, height, width, in_channels = get_const_tuple(data.shape)
    out_channels, kernel_h, kernel_w, _ = get_const_tuple(kernel.shape)
    y_stride, x_stride = get_const_tuple(attrs.strides)

    out_height = _compute_output_dim(height, kernel_h, y_stride)
    out_width = _compute_output_dim(width, kernel_w, x_stride)

    # Decide how many sums our function should have running at the same time. Doing
    # this lets us do "more work" for each memory load, but doing too many of them causes us to run
    # out of registers. Currently this is set to either 1 or 2, but autotuning this value would
    # improve performance a lot. Tracked by https://github.com/apache/tvm/issues/13528.

    num_outputs = 2

    # Next, decide whether whether we need "parity alternation". For example, if we have an
    # 8x3x3x3 kernel (8 output channels, height 3, width 3, input channels 3) in the OHWI layout,
    # then every output channel kernel slice will be 27 halfwords. This means every other output
    # channel will not be word aligned, which will cause slowness/crashes!

    # We solve this problem by handling the "aligned" and "offset" output channels with different
    # versions of our tensordot function. The "aligned func" assumes that the start positions of the
    # output, data, and kernel are given exactly by their pointer. The "offset" version assumes that
    # the "true" start of the output is the value in the output pointer, plus an offset of 0 or 1.
    # _pick_tensordot_impl decides whether this is the case. If not, we only want to generate one
    # function (to save flash), so offset_func is a tuple of empty strings.

    aligned_func, offset_func = _pick_tensordot_impl(attrs, inputs, num_outputs, False)

    # Helper functions to make pointers
    def output_ptr(buffer, y, x, c):
        return _make_tscript_ptr(
            buffer,
            y * const(out_width * out_channels) + x * const(out_channels * num_outputs) + c,
            1,
        )

    # We need to disable pylint's unused argument checker, as the kwarg offset is unused but must
    # be present for compatibility. We cannot add an underscore as we normally would, as this makes
    # the keyword not match.

    # pylint: disable=unused-argument
    def data_ptr(buffer, y, x, c, offset=0):
        return _make_tscript_ptr(
            buffer,
            y * const(y_stride * width * in_channels)
            + x * const(x_stride * num_outputs * in_channels),
            1,
        )

    # pylint: enable=unused-argument

    def kernel_ptr(buffer, c, offset=0):
        return _make_tscript_ptr(
            buffer,
            c * const(kernel_h * kernel_w * in_channels) - offset,
            1,
        )

    prim_func = _make_conv2d_primfunc(
        (const(out_height), const(out_width // num_outputs), const(out_channels)),
        (data.shape, kernel.shape, bias.shape, scale.shape, out_type.shape),
        aligned_func,
        offset_func,
        (output_ptr, data_ptr, kernel_ptr),
    )

    output = te.extern_primfunc([data, kernel, bias, scale], prim_func, name="tir", dtype="int16")
    return [output]


def schedule_qnn_conv2d(_attrs, _outs, _target):
    """Schedule function for qnn.conv2d."""
    return None


def qnn_depthwise_conv2d(attrs, inputs, out_type):
    """Compute for qnn.depthwise_conv2d with NCHW layout.

    Works basically the same way as regular conv2d - see above.
    """

    assert len(inputs) == 11
    data, kernel, _izp, _kzp, _iscale, _kscale, bias, scale = inputs[0:8]
    output_layout = attrs.out_layout
    assert output_layout == "NHWC"

    _, _, height, width = get_const_tuple(data.shape)
    _, out_channels, kernel_h, kernel_w = get_const_tuple(kernel.shape)
    _, out_height, out_width, _ = get_const_tuple(out_type.shape)
    y_stride, x_stride = get_const_tuple(attrs.strides)

    out_height = _compute_output_dim(height, kernel_h, y_stride)
    out_width = _compute_output_dim(width, kernel_w, x_stride)

    num_outputs = 2

    aligned_func, offset_func = _pick_tensordot_impl(attrs, inputs, num_outputs, True)

    # Helper functions for making pointers.
    def output_ptr(buffer, y, x, c):
        return _make_tscript_ptr(
            buffer,
            y * const(out_width * out_channels) + x * const(out_channels * num_outputs) + c,
            1,
        )

    def data_ptr(buffer, y, x, c, offset=0):
        if height * width % 2 == 1:
            x_ptr_offset = tvm.tir.const(-1)
        else:
            x_ptr_offset = tvm.tir.const(0)

        return _make_tscript_ptr(
            buffer,
            c * const(width * height)
            + y * const(y_stride * width)
            + x * const(x_stride * num_outputs)
            + offset * x_ptr_offset,
            1,
        )

    def kernel_ptr(buffer, c, offset=0):
        return _make_tscript_ptr(
            buffer,
            c * tvm.tir.const(kernel_h * kernel_w) - offset,
            1,
        )

    prim_func = _make_conv2d_primfunc(
        (const(out_height), const(out_width // num_outputs), const(out_channels)),
        (data.shape, kernel.shape, bias.shape, scale.shape, out_type.shape),
        aligned_func,
        offset_func,
        (output_ptr, data_ptr, kernel_ptr),
    )

    output = te.extern_primfunc([data, kernel, bias, scale], prim_func, name="tir", dtype="int16")
    return [output]


def schedule_qnn_depthwise_conv2d(_attrs, _outs, _target):
    """Schedule function for qnn.depthwise_conv2d."""
    return None
