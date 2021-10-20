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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D int8 schedule on ARM"""
from tvm import te
from tvm import autotvm
from .. import tag
from ..utils import traverse_inline, get_const_tuple
from ..generic import conv2d as conv2d_generic
from .. import nn
from ..nn.conv2d import _get_workload as _get_conv2d_workload
from .tensor_intrin import dot_int8_int8_int32
from .conv2d_gemm import (
    compute_conv2d_gemm_without_weight_transform,
    schedule_conv2d_gemm_interleaved,
    schedule_conv2d_gemm_native,
)
from .arm_utils import get_tiling_B_interleaved_t


def _get_default_config(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """
    Get default int8 schedule config for the workload
    """
    wkl = _get_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype)
    is_kernel_1x1 = wkl.kernel_h == 1 and wkl.kernel_w == 1
    if is_kernel_1x1:
        conv2d_generic.fallback_schedule_cpu_1x1_int8(cfg, wkl, int32_lanes=2, num_int8_elements=4)
    else:
        conv2d_generic.fallback_schedule_cpu_common_int8(
            cfg, wkl, int32_lanes=2, num_int8_elements=4
        )


@autotvm.register_topi_compute("conv2d_NCHWc_int8.arm_cpu")
def conv2d_NCHWc_int8(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d int8 with NCHWc layout"""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn

    oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn, n_elems = get_const_tuple(kernel.shape)
    num_filter = oc_chunk * oc_bn

    # If no config was set, we can fallback to NCHW config.
    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder((num_filter, in_channel, kh, kw), dtype=kernel.dtype),
            strides,
            padding,
            dilation,
            out_dtype,
        )
    return nn.conv2d_NCHWc_int8_compute(
        data, kernel, strides, padding, dilation, layout, out_layout, out_dtype
    )


@autotvm.register_topi_schedule("conv2d_NCHWc_int8.arm_cpu")
def schedule_conv2d_NCHWc_int8(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if "conv2d_NCHWc_int8" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = (
                data_vec.op.input_tensors[0]
                if isinstance(data_vec.op, te.tensor.ComputeOp) and "pad" not in data_vec.op.tag
                else data_vec
            )
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, cfg, data_vec, kernel_vec, conv_out, outs[0]]
            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, _ = get_const_tuple(kernel_vec.shape)
            dtype = "uint" if data.dtype == "uint8" else "int"
            if kh == 1 and kw == 1:
                conv2d_generic.schedule_conv_NCHWc_cpu_1x1_int8(
                    *args, int32_lanes=4, intrin=dot_int8_int8_int32(int32_lanes=4, dtype=dtype)
                )
            else:
                conv2d_generic.schedule_conv_NCHWc_cpu_common_int8(
                    *args, int32_lanes=4, intrin=dot_int8_int8_int32(int32_lanes=4, dtype=dtype)
                )

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


def _compute_conv2d_NHWC_quantized(
    cfg, data, kernel, strides, padding, dilation, out_dtype, interleave_A
):
    N, IH, IW, IC = get_const_tuple(data.shape)
    KH, KW, _, OC = get_const_tuple(kernel.shape)
    tile_rows_B, tile_cols_B = get_tiling_B_interleaved_t(interleave_A)

    kernel = nn.conv2d_gemm_weight_transform(kernel, tile_rows_B, tile_cols_B)
    return compute_conv2d_gemm_without_weight_transform(
        cfg, data, kernel, strides, padding, dilation, out_dtype, (KH, KW), OC, interleave_A
    )


def _compute_conv2d_NHWC_quantized_without_transform(
    cfg,
    data,
    B,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size=None,
    output_channels=None,
    interleave_A=False,
):
    return compute_conv2d_gemm_without_weight_transform(
        cfg,
        data,
        B,
        strides,
        padding,
        dilation,
        out_dtype,
        kernel_size,
        output_channels,
        interleave_A,
    )


def _schedule_conv2d_NHWC_quantized(cfg, outs, interleave_A):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    # Vectorize the output and then inline all the rest
    out = outs[0]
    n, h, w, c = out.op.axis
    n_h_fused = s[out].fuse(n, h)
    outer, inner = s[out].split(c, 4)
    s[out].vectorize(inner)
    s[out].parallel(n_h_fused)

    def _callback(op):
        """Traverse operators from computation graph"""
        if op.name == "conv2d_gemm_output":
            conv_out = op.output(0)
            if interleave_A:
                schedule_conv2d_gemm_interleaved(cfg, s, conv_out, out)
            else:
                schedule_conv2d_gemm_native(cfg, s, conv_out, out)
            if out != conv_out:
                s[conv_out].compute_at(s[out], inner)
            else:
                C = conv_out.op.input_tensors[0]
                if interleave_A:
                    s[C].compute_at(s[out], inner)

    traverse_inline(s, outs[0].op, _callback)
    return s


# Interleaved schedules: those schedule will interleave the input data. The
# weights are interleaved and transposed
@autotvm.register_topi_compute("conv2d_NHWC_quantized_interleaved.arm_cpu")
def compute_conv2d_NHWC_quantized_interleaved(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Interface for interleaved compute_conv2d_NHWC_quantized_interleaved"""
    return _compute_conv2d_NHWC_quantized(
        cfg, data, kernel, strides, padding, dilation, out_dtype, True
    )


@autotvm.register_topi_compute("conv2d_NHWC_quantized_interleaved_without_transform.arm_cpu")
def compute_conv2d_NHWC_quantized_interleaved_without_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    """Interface for interleaved compute_conv2d_NHWC_quantized_interleaved_without_transform"""
    return _compute_conv2d_NHWC_quantized_without_transform(
        cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels, True
    )


@autotvm.register_topi_schedule("conv2d_NHWC_quantized_interleaved.arm_cpu")
def schedule_conv2d_NHWC_quantized_interleaved(cfg, outs):
    """Interface for interleaved schedule_conv2d_NHWC_quantized_interleaved"""
    return _schedule_conv2d_NHWC_quantized(cfg, outs, True)


# Native schedules: those schedule won't interleave A (which is left in its native form).
# The weights are interleaved and transposed
@autotvm.register_topi_compute("conv2d_NHWC_quantized_native.arm_cpu")
def compute_conv2d_NHWC_quantized_native(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Interface for native compute_conv2d_NHWC_quantized"""
    return _compute_conv2d_NHWC_quantized(
        cfg, data, kernel, strides, padding, dilation, out_dtype, False
    )


@autotvm.register_topi_compute("conv2d_NHWC_quantized_native_without_transform.arm_cpu")
def compute_conv2d_NHWC_quantized_native_without_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    """Interface for compute_conv2d_NHWC_quantized_native_without_transform"""
    return _compute_conv2d_NHWC_quantized_without_transform(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        kernel_size,
        output_channels,
        False,
    )


@autotvm.register_topi_schedule("conv2d_NHWC_quantized_native.arm_cpu")
def schedule_conv2d_NHWC_quantized_native(cfg, outs):
    """Interface for native schedule_conv2d_NHWC_quantized"""
    return _schedule_conv2d_NHWC_quantized(cfg, outs, False)
