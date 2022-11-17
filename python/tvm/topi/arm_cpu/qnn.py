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
"""Contains the TVMScript implementations of some QNN operators for Arm. Currently, the only ops
with schedules are fused regular and depthwise convolutions for Arm Cortex-M with DSP."""


import tvm
from tvm import te, tir
from tvm.script import tir as T
from ..utils import get_const_tuple
from ..nn.utils import get_pad_tuple
from ..nn.pad import pad
from .. import tag, nn
from tvm.tir import TensorIntrin
from tvm.topi.arm_cpu.mprofile.dsp.micro_kernel import tensordot
import textwrap


def int_ceil_division(x, y):
    return -(x // -y)


def _compute_output_dim(data_length, kernel_length, stride):
    return int_ceil_division(data_length + 1 - kernel_length, stride)


def _pick_tensordot_impl(attrs, data, kernel, num_sums=2, is_depthwise=False):
    """Helper function that chooses the right implementation of micro_kernel.tensordot depending on
    the input parameters of the conv2d. It returns a tuple of TWO function_name, function_code
    pairs - one (the aligned one) for even-numbered output channels, and one (the offset one) for
    odd-numbered output channels. This function is used for regular and depthwise convolutions.

    We need different implementations for even vs odd numbered output channels, because the "start"
    of an odd output channel in the data tensor or kernel might or might not be on a word boundary,
    and the tensordot code expects all input pointers to be word-aligned."""

    stride_h, stride_w = get_const_tuple(attrs.strides)

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
    aligned_func = tensordot.tensordot_int16_impl(num_sums, dimensions, (0, 0, 0), x_strides)

    kernel_per_oc_size = dimensions[1] * dimensions[2]

    offsets = (data_per_oc_size % 2, kernel_per_oc_size % 2, 0)
    offset_func = tensordot.tensordot_int16_impl(num_sums, dimensions, offsets, x_strides)

    return (aligned_func, offset_func)


def _get_align_tuple(output_channels, is_unaligned):
    if is_unaligned:
        alignment_vals = (output_channels // 2, output_channels // 2, 2)
    else:
        alignment_vals = (output_channels, 0, 1)
    return tuple(tvm.tir.const(n) for n in alignment_vals)


def _get_tscript_const_tuple(values):
    return tuple(tvm.tir.const(n) for n in get_const_tuple(values))


def _make_tscript_ptr(buffer, offset, length):
    return T.tvm_access_ptr(
        T.type_annotation(dtype="int16"),
        buffer.data,
        offset,
        length,
        1,
        dtype="handle",
    )


def _make_tscript_call(func_name, *args):
    return T.evaluate(T.call_extern(func_name, *args, dtype="int32"))


def qnn_conv2d(attrs, inputs, out_type):
    """Compute for qnn.conv2d with NHWC layout. Note that this is a DIFFERENT layout from the
    Hexagon variant, because they have special instructions Cortex-M doesn't have. We also expect
    the kernel to have OHWI layout. We also assume that padding is not necessary, as it will have
    been done by another pass."""

    """Step one: Make a few checks to unpack the function arguments and ensure it was called with
    the right arguments. Note that this function does not use a wrapper."""
    assert len(inputs) == 11
    data, kernel, _izp, _kzp, _iscale, _kscale, bias, rq_scale = inputs[0:8]
    output_layout = attrs.out_layout
    assert output_layout == "NHWC"
    assert rq_scale.dtype == "int32"

    _, height, width, in_channels = get_const_tuple(data.shape)
    out_channels, kernel_h, kernel_w, _ = get_const_tuple(kernel.shape)
    y_stride, x_stride = get_const_tuple(attrs.strides)

    out_height = _compute_output_dim(height, kernel_h, y_stride)
    out_width = _compute_output_dim(width, kernel_w, x_stride)

    """Step two: we decide how many sums our function should have running at the same time. Doing
    this lets us do "more work" for each memory load, but doing too many of them causes us to run
    out of registers. Currently this is set to either 1 or 2, but autotuning this value would
    improve performance a lot."""
    num_sums = 2

    """Step three: decide whether whether we need "parity alternation". For example, if we have an
    8x3x3x3 kernel (8 output channels, height 3, width 3, input channels 3) in the OHWI layout, then
    every output channel kernel slice will be 27 halfwords. This means every other output channel
    will not be word aligned, which will cause slowness/crashes!

    We solve this problem by handling the "aligned" and "offset" output channels with different
    versions of our tensordot function. The "aligned func" assumes that the start positions of the
    output, data, and kernel are given exactly by their pointer. The "offset" version assumes that
    the "true" start of the output is the value in the output pointer, plus an offset of 0 or 1.

    _pick_tensordot_impl decides whether this is the case. If not, we only want to generate one
    function (to save flash), so offset_func is a tuple of empty strings."""
    aligned_func, offset_func = _pick_tensordot_impl(attrs, data, kernel, num_sums, False)
    aligned_func_name, aligned_func_code = aligned_func
    offset_func_name, offset_func_code = offset_func

    """These constants decide how much the aligned and offset functions will be called. They are
    chosen so that the offset function will be deleted if not used."""
    if aligned_func_name != offset_func_name:
        aligned_calls = tvm.tir.const(out_channels // 2)
        offset_calls = tvm.tir.const(out_channels // 2)
        tc_oc_step = tvm.tir.const(2)

    else:
        aligned_calls = tvm.tir.const(out_channels)
        offset_calls = tvm.tir.const(0)
        tc_oc_step = tvm.tir.const(1)

    # Helper functions for making pointers.

    tc_out_y_stride = tvm.tir.const(out_width * out_channels)
    tc_out_x_stride = tvm.tir.const(out_channels * num_sums)
    tc_out_calls_per_row = tvm.tir.const(out_width // num_sums)
    tc_out_num_rows = tvm.tir.const(out_height)

    def output_ptr(buffer, y, x, c):
        return _make_tscript_ptr(
            buffer,
            y * tc_out_y_stride + x * tc_out_x_stride + c,
            1,
        )

    tc_data_y_stride = tvm.tir.const(y_stride * width * in_channels)
    tc_data_x_stride = tvm.tir.const(x_stride * num_sums * in_channels)

    def data_ptr(buffer, y, x):
        return _make_tscript_ptr(
            buffer,
            y * tc_data_y_stride + x * tc_data_x_stride,
            tc_data_x_stride * num_sums,
        )

    tc_kernel_oc_stride = tvm.tir.const(kernel_h * kernel_w * in_channels)

    def kernel_ptr(buffer, c, offset=0):
        return _make_tscript_ptr(
            buffer,
            c * tc_kernel_oc_stride - offset,
            tc_kernel_oc_stride + offset,
        )

    @T.prim_func
    def biased_quantized_conv2d(
        data_handle: T.handle,
        kernel_handle: T.handle,
        bias_handle: T.handle,
        requantize_handle: T.handle,
        output_handle: T.handle,
    ) -> None:

        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        DATA = T.match_buffer(data_handle, data.shape, dtype="int16")
        KERNEL = T.match_buffer(kernel_handle, kernel.shape, dtype="int16")
        BIAS = T.match_buffer(bias_handle, bias.shape, dtype="int32")
        REQUANTIZE_SCALE = T.match_buffer(requantize_handle, rq_scale.shape, dtype="int32")
        OUTPUT = T.match_buffer(output_handle, out_type.shape, dtype=out_type.dtype)

        # This hack prevents TVM from seeing these variables as "unused". I should be using T.reads
        # and T.writes, but they don't work. I think it's an issue with BufferTouchedDomain.
        OUTPUT[0, 0, 0, 0] = 0
        x = DATA[0, 0, 0, 0]
        y = KERNEL[0, 0, 0, 0]

        for oc, oh, ow in T.grid(aligned_calls, tc_out_num_rows, tc_out_calls_per_row):
            with T.block("conv2d_aligned"):
                T.block_attr({"pragma_import_c": aligned_func_code})
                voh, vow, voc = T.axis.remap("SSS", [oh, ow, oc])
                _make_tscript_call(
                    aligned_func_name,
                    output_ptr(OUTPUT, voh, vow, voc * tc_oc_step),
                    data_ptr(DATA, voh, vow),
                    kernel_ptr(KERNEL, voc * tc_oc_step),
                    BIAS[0, 0, 0, voc * tc_oc_step],
                    REQUANTIZE_SCALE[voc * tc_oc_step],
                )

        for oc, oh, ow in T.grid(offset_calls, tc_out_num_rows, tc_out_calls_per_row):
            with T.block("conv2d_offset"):
                T.block_attr({"pragma_import_c": offset_func_code})
                voh, vow, voc = T.axis.remap("SSS", [oh, ow, oc])
                _make_tscript_call(
                    offset_func_name,
                    output_ptr(OUTPUT, voh, vow, voc * tc_oc_step + 1),
                    data_ptr(DATA, voh, vow),
                    kernel_ptr(KERNEL, voc * tc_oc_step + 1, offset=1),
                    BIAS[0, 0, 0, voc * tc_oc_step + 1],
                    REQUANTIZE_SCALE[voc * tc_oc_step + 1],
                )

    output = te.extern_primfunc(
        [data, kernel, bias, rq_scale], biased_quantized_conv2d, name="tir", dtype="int16"
    )
    return [output]


def schedule_qnn_conv2d(attrs, outs, target):
    """Schedule function for qnn.conv2d."""
    return None


def qnn_depthwise_conv2d(attrs, inputs, out_type):
    """Compute for qnn.depthwise_conv2d with NCHW layout. Works basically the same way as regular
    conv2d - see above."""

    assert len(inputs) == 11
    data, kernel, _izp, _kzp, _iscale, _kscale, bias, rq_scale = inputs[0:8]
    output_layout = attrs.out_layout
    assert output_layout == "NHWC"
    assert rq_scale.dtype == "int32"

    _, in_channels, height, width = get_const_tuple(data.shape)
    _, out_channels, kernel_h, kernel_w = get_const_tuple(kernel.shape)
    _, out_height, out_width, _ = get_const_tuple(out_type.shape)
    y_stride, x_stride = get_const_tuple(attrs.strides)

    out_height = _compute_output_dim(height, kernel_h, y_stride)
    out_width = _compute_output_dim(width, kernel_w, x_stride)

    num_sums = 2

    aligned_func, offset_func = _pick_tensordot_impl(attrs, data, kernel, num_sums, True)
    aligned_func_name, aligned_func_code = aligned_func
    offset_func_name, offset_func_code = offset_func

    if offset_func_name:
        aligned_calls = tvm.tir.const(out_channels // 2)
        offset_calls = tvm.tir.const(out_channels // 2)
        tc_oc_step = tvm.tir.const(2)
        if height * width % 2 == 1:
            x_ptr_offset = tvm.tir.const(-1)
        else:
            x_ptr_offset = tvm.tir.const(0)

    else:
        aligned_calls = tvm.tir.const(out_channels)
        offset_calls = tvm.tir.const(0)
        tc_oc_step = tvm.tir.const(1)
        x_ptr_offset = tvm.tir.const(0)

    # Helper functions for making pointers.

    # NHWC
    tc_out_y_stride = tvm.tir.const(out_width * out_channels)
    tc_out_x_stride = tvm.tir.const(out_channels * num_sums)
    tc_out_calls_per_row = tvm.tir.const(out_width // num_sums)
    tc_out_num_rows = tvm.tir.const(out_height)

    def output_ptr(buffer, y, x, c):
        return _make_tscript_ptr(
            buffer,
            y * tc_out_y_stride + x * tc_out_x_stride + c,
            1,
        )

    # NCHW
    tc_data_oc_stride = tvm.tir.const(width * height)
    tc_data_y_stride = tvm.tir.const(y_stride * width)
    tc_data_x_stride = tvm.tir.const(x_stride * num_sums)

    def data_ptr(buffer, c, y, x, offset=0):
        return _make_tscript_ptr(
            buffer,
            c * tc_data_oc_stride + y * tc_data_y_stride + x * tc_data_x_stride + offset,
            tc_data_x_stride * num_sums,
        )

    # OIHW
    tc_kernel_ic_stride = tvm.tir.const(kernel_h * kernel_w)

    def kernel_ptr(buffer, ic, offset=0):
        return _make_tscript_ptr(
            buffer,
            ic * tc_kernel_ic_stride - offset,
            tc_kernel_ic_stride + offset,
        )

    @T.prim_func
    def biased_quantized_conv2d(
        data_handle: T.handle,
        kernel_handle: T.handle,
        bias_handle: T.handle,
        requantize_handle: T.handle,
        output_handle: T.handle,
    ) -> None:

        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        DATA = T.match_buffer(data_handle, data.shape, dtype="int16")
        KERNEL = T.match_buffer(kernel_handle, kernel.shape, dtype="int16")
        BIAS = T.match_buffer(bias_handle, bias.shape, dtype="int32")
        REQUANTIZE_SCALE = T.match_buffer(requantize_handle, rq_scale.shape, dtype="int32")
        OUTPUT = T.match_buffer(output_handle, out_type.shape, dtype=out_type.dtype)

        # This hack prevents TVM from seeing these variables as "unused". I should be using T.reads
        # and T.writes, but they don't work. I think it's an issue with BufferTouchedDomain.
        OUTPUT[0, 0, 0, 0] = 0
        x = DATA[0, 0, 0, 0]
        y = KERNEL[0, 0, 0, 0]

        for oc, oh, ow in T.grid(offset_calls, tc_out_num_rows, tc_out_calls_per_row):
            with T.block("depthwise_conv2d_aligned"):
                T.block_attr({"pragma_import_c": aligned_func_code})
                voh, vow, voc = T.axis.remap("SSS", [oh, ow, oc])

                _make_tscript_call(
                    aligned_func_name,
                    output_ptr(OUTPUT, voh, vow, voc * tc_oc_step),
                    data_ptr(DATA, voc * tc_oc_step, voh, vow),
                    kernel_ptr(KERNEL, voc * tc_oc_step),
                    BIAS[0, 0, 0, voc * tc_oc_step],
                    REQUANTIZE_SCALE[voc * tc_oc_step],
                )

        for oc, oh, ow in T.grid(offset_calls, tc_out_num_rows, tc_out_calls_per_row):
            with T.block("depthwise_conv2d_offset"):
                T.block_attr({"pragma_import_c": offset_func_code})
                voh, vow, voc = T.axis.remap("SSS", [oh, ow, oc])
                _make_tscript_call(
                    offset_func_name,
                    output_ptr(OUTPUT, voh, vow, voc * tc_oc_step + 1),
                    data_ptr(DATA, voc * tc_oc_step + 1, voh, vow, offset=x_ptr_offset),
                    kernel_ptr(KERNEL, voc * tc_oc_step + 1, offset=1),
                    BIAS[0, 0, 0, voc * tc_oc_step + 1],
                    REQUANTIZE_SCALE[voc * tc_oc_step + 1],
                )

    output = te.extern_primfunc(
        [data, kernel, bias, rq_scale], biased_quantized_conv2d, name="tir", dtype="int16"
    )
    return [output]


def schedule_qnn_depthwise_conv2d(attrs, outs, target):
    """Schedule function for qnn.depthwise_conv2d."""
    return None
