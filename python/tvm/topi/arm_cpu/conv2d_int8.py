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
from tvm import te, target, autotvm
from ..utils import traverse_inline, get_const_tuple
from ..generic import conv2d as conv2d_generic
from .. import nn
from ...target import codegen
from ..nn.conv2d import _get_workload as _get_conv2d_workload, unpack_NCHWc_to_nchw
from ..x86.conv2d_int8 import _pack_data
from ..nn.utils import get_pad_tuple
from .tensor_intrin import dot_int8_int8_int32_neon_82, dot_int8_int8_int32_neon
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
        conv2d_generic.fallback_schedule_cpu_1x1_int8(cfg, wkl, int32_lanes=4, num_int8_elements=4)
    else:
        conv2d_generic.fallback_schedule_cpu_common_int8(
            cfg, wkl, int32_lanes=4, num_int8_elements=4
        )


@autotvm.register_topi_compute("conv2d_NCHWc_int8.arm_cpu")
def conv2d_NCHWc_int8(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d int8 with NCHWc layout"""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload

    if len(data.shape) == 5:  # data is in nchwc
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        in_channel = ic_chunk * ic_bn

        oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn, _ = get_const_tuple(kernel.shape)
        num_filter = oc_chunk * oc_bn
    else:
        # data is nchw, implicitly treat it as nchw1c
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kh, kw = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kh == 1 and kw == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kh, kw))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    dilated_kernel_h = (kh - 1) * dh + 1
    dilated_kernel_w = (kw - 1) * dw + 1
    oh = (ih - dilated_kernel_h + pt + pb) // sh + 1
    ow = (iw - dilated_kernel_w + pl + pr) // sw + 1

    # input and output should be a multiple of 8 (intrinsics are 8 lanes)
    cfg.define_split(
        "tile_ic", in_channel, num_outputs=2, filter=lambda y: y.size[-1] % min(8, in_channel) == 0
    )
    cfg.define_split(
        "tile_oc", num_filter, num_outputs=2, filter=lambda y: y.size[-1] % min(8, num_filter) == 0
    )
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

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
    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        data, kernel = _pack_data(cfg, data, kernel)

    n_elems = int(kernel.shape[-1])

    return nn.conv2d_NCHWc_int8(
        data, kernel, strides, padding, dilation, layout, out_layout, out_dtype, n_elems=n_elems
    )


def is_int8_hw_support(data_dtype, kernel_dtype):
    """
    Checks to ensure that we can use int8 on arm
    1) The datatypes are correct.
    2) LLVM version has support for the instructions.
    """
    # 1) Check datatypes
    is_dtype_support = data_dtype == kernel_dtype and "int8" in data_dtype

    # 2) Check LLVM support
    llvm_version = codegen.llvm_version_major()
    is_llvm_support = llvm_version >= 8

    # 3) Check target
    current_target = target.Target.current(allow_none=False)
    is_target_support = bool(
        current_target.features.has_asimd or current_target.features.has_dotprod
    )

    return is_dtype_support and is_llvm_support and is_target_support


@autotvm.register_topi_schedule("conv2d_NCHWc_int8.arm_cpu")
def schedule_conv2d_NCHWc_int8(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _callback(op):
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
            _, _, kh, kw, _, _, n_elems = get_const_tuple(kernel_vec.shape)
            assert n_elems == 4
            dtype = "uint" if data.dtype == "uint8" else "int"
            current_target = target.Target.current(allow_none=False)
            if current_target.features.has_dotprod:
                intrin = dot_int8_int8_int32_neon_82(int32_lanes=4, dtype=dtype)
            elif current_target.features.has_asimd:
                assert dtype == "int", "uint8 not supported if dot product is not available"
                intrin = dot_int8_int8_int32_neon()
            else:
                raise RuntimeError(
                    "Cannot schedule schedule_NCHWc_int8 without neon or arm v8.2 neon support"
                )
            # On raspberry pi 4s, we see poor performance when the fused
            # operations are inlined into the main computation body. These
            # fused ops dominated the runtime on small convolutions repeatedly
            # blow the cache. Using workloads from resnet50, inceptionv3, and
            # mobilenetv3, we empirically determine the size at which inline is
            # not worth it to be kernel heigh * kernel width < 500. These tests
            # were only run on raspberry pi 4, other arm cpus may have larger
            # caches where inlining has good performance.
            if target.Target.current().mcpu == "cortex-a72" and kh * kw < 500:
                inline_fused = False
            else:
                inline_fused = True
            if kh == 1 and kw == 1:
                conv2d_generic.schedule_conv_NCHWc_cpu_1x1_int8(
                    *args, int32_lanes=4, int8_elems=4, intrin=intrin, inline_fused=inline_fused
                )
            else:
                conv2d_generic.schedule_conv_NCHWc_cpu_common_int8(
                    *args, int32_lanes=4, int8_elems=4, intrin=intrin, inline_fused=inline_fused
                )

    traverse_inline(s, outs[0].op, _callback)
    return s


def conv2d_nchw_int8(data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NCHW layout and int8 dtype"""
    layout = "NCHW"
    # pylint: disable=no-value-for-parameter
    packed_out = conv2d_NCHWc_int8(
        data, kernel, strides, padding, dilation, layout, layout, out_dtype
    )
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_conv2d_nchw_int8(outs):
    """Create the schedule for conv2d_nchw_int8"""
    # pylint: disable=no-value-for-parameter
    return schedule_conv2d_NCHWc_int8(outs)


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


@autotvm.register_topi_schedule("conv2d_NHWC_quantized_interleaved_without_transform.arm_cpu")
def schedule_conv2d_NHWC_quantized_interleaved_without_transform(cfg, outs):
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


@autotvm.register_topi_schedule("conv2d_NHWC_quantized_native_without_transform.arm_cpu")
def schedule_conv2d_NHWC_quantized_native_without_transform(cfg, outs):
    """Interface for native schedule_conv2d_NHWC_quantized"""
    return _schedule_conv2d_NHWC_quantized(cfg, outs, False)
