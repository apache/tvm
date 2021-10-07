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
"""Definition of ARM CPU operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re
import logging

from tvm import topi
from ....target import arm_isa
from ....topi.generic import conv2d as conv2d_generic
from .generic import *
from .. import op as _op

logger = logging.getLogger("strategy")


@schedule_reduce.register("arm_cpu")
def schedule_reduce_cpu(attrs, outs, target):
    """schedule reduction ops for arm_cpu"""
    with target:
        return topi.x86.schedule_reduce(outs)


@schedule_injective.register(["arm_cpu", "micro_dev"])
def schedule_injective_arm_cpu(_, outs, target):
    """schedule injective ops for arm cpu"""
    with target:
        return topi.arm_cpu.schedule_injective(outs)


@schedule_concatenate.register(["arm_cpu", "micro_dev"])
def schedule_concatenate_arm_cpu(_, outs, target):
    """schedule concatenate for arm cpu"""
    with target:
        return topi.arm_cpu.schedule_concatenate(outs)


@conv2d_strategy.register(["arm_cpu", "micro_dev"])
def conv2d_strategy_arm_cpu(attrs, inputs, out_type, target):
    """conv2d arm cpu strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    isa = arm_isa.IsaAnalyzer(target)

    if groups == 1:
        if layout == "NCHW":
            if kernel_layout == "OIHW":
                # ARM conv2d spatial pack schedule.
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.arm_cpu.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.arm_cpu",
                )

                # Intel x86 conv2d schedule.
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.x86.conv2d_nchw),
                    wrap_topi_schedule(topi.x86.schedule_conv2d_nchw),
                    name="conv2d_nchw.x86",
                )

                # check if winograd algorithm is applicable
                _, _, kh, kw = get_const_tuple(kernel.shape)
                pt, pl, pb, pr = topi.nn.get_pad_tuple(padding, (kh, kw))
                is_winograd_applicable = (
                    "float" in data.dtype
                    and "float" in kernel.dtype
                    and kh == 3
                    and kw == 3
                    and stride_h == 1
                    and stride_w == 1
                    and dilation_h == 1
                    and dilation_w == 1
                )
                if is_winograd_applicable:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.arm_cpu.conv2d_nchw_winograd),
                        wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nchw_winograd),
                        name="conv2d_nchw_winograd.arm_cpu",
                        plevel=5,
                    )
                    if "nnpack" in target.libs and pt == 1 and pb == 1 and pl == 1 and pr == 1:
                        strategy.add_implementation(
                            wrap_compute_conv2d(topi.arm_cpu.conv2d_nchw_winograd_nnpack),
                            wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nchw_winograd_nnpack),
                            name="conv2d_nchw_winograd_nnpack.arm_cpu",
                            plevel=15,
                        )
            elif re.match(r"OIHW\d*o", kernel_layout):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.arm_cpu.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.arm_cpu",
                )
            else:
                raise RuntimeError(
                    "Unsupported weight layout {} for conv2d NCHW".format(kernel_layout)
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            logger.warning("conv2d_hwcn is not optimized for arm cpu.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_hwcn),
                wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
                name="conv2d_hwcn.generic",
            )
        elif layout == "NHWC":
            if "SMLAD" in isa and kernel_layout == "HWOI":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.arm_cpu.conv2d_nhwc_direct_simd),
                    wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nhwc_direct_simd),
                    name="conv2d_nhwc_direct_simd.micro_dev",
                )
            elif kernel_layout == "HWIO":
                is_aarch64 = topi.arm_cpu.arm_utils.is_aarch64_arm()
                has_dot_prod = topi.arm_cpu.arm_utils.is_dotprod_available()
                if has_dot_prod and data.dtype in ["int8", "uint8"]:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.arm_cpu.compute_conv2d_NHWC_quantized_native),
                        wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_NHWC_quantized_native),
                        name="conv2d_NHWC_quantized_native.arm_cpu",
                    )
                if is_aarch64 and data.dtype in ["int8", "uint8"]:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.arm_cpu.compute_conv2d_NHWC_quantized_interleaved),
                        wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_NHWC_quantized_interleaved),
                        name="conv2d_NHWC_quantized_interleaved.arm_cpu",
                    )
                if (not is_aarch64) or (data.dtype not in ["int8", "uint8"]):
                    # TODO(@giuseros)
                    # This strategy errors out for quantized data types when tuning.
                    # Let's use this only for non-aarch64 or non-quantized cases
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.arm_cpu.conv2d_nhwc_spatial_pack),
                        wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nhwc_spatial_pack),
                        name="conv2d_nhwc_spatial_pack.arm_cpu",
                    )
            else:
                raise RuntimeError(
                    "Unsupported kernel layout {} for conv2d NHWC".format(kernel_layout)
                )

        else:
            raise RuntimeError("Unsupported conv2d layout {} for arm cpu".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW" or re.match(r"OIHW\d*o", kernel_layout)
            # ARM conv2d depthwise schedule
            if kernel_layout == "OIHW":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.arm_cpu.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.arm_cpu.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.arm_cpu",
                )

            # TODO:
            # This schedule has incorrect result on some hardware platforms (like NV Jetson TX2)
            # Let us comment it out but not remove.
            # see discussion:
            # https://discuss.tvm.apache.org/t/autotuner-incorrect-result-after-tuning-mobilenetv2-on-arm-cpu/6088
            # strategy.add_implementation(
            #     wrap_compute_conv2d(topi.arm_cpu.depthwise_conv2d_nchw_spatial_pack),
            #     wrap_topi_schedule(topi.arm_cpu.schedule_depthwise_conv2d_nchw_spatial_pack),
            #     name="depthwise_conv2d_nchw_spatial_pack.arm_cpu",
            #     plevel=15)

            # Intel x86 depthwise conv2d schedule.
            channel_multiplier = get_const_tuple(inputs[1].shape)[1]
            if channel_multiplier == 1 and dilation_h == 1 and dilation_w == 1:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.x86.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.x86",
                )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            is_aarch64 = topi.arm_cpu.arm_utils.is_aarch64_arm()
            if is_aarch64 or "+neon" in target.mattr:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.arm_cpu.compute_depthwise_conv2d_nhwc),
                    wrap_topi_schedule(topi.arm_cpu.schedule_depthwise_conv2d_nhwc),
                    name="depthwise_conv2d_nhwc.arm_cpu",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                    wrap_topi_schedule(conv2d_generic.schedule_depthwise_conv2d_nhwc),
                    name="depthwise_conv2d_nhwc.generic",
                )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {} for arm cpu".format(layout))
    else:  # group_conv2d
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.arm_cpu.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.arm_cpu.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.arm_cpu",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            logger.warning("group_conv2d with layout NHWC is not optimized for arm cpu.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
                name="group_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported group_conv2d layout {} for arm cpu".format(layout))
    return strategy


@conv2d_NCHWc_strategy.register("arm_cpu")
def conv2d_NCHWc_strategy_arm_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc adopted from x86"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.x86.conv2d_NCHWc, True, True),
        wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc),
        name="conv2d_NCHWc.x86",
    )
    return strategy


@depthwise_conv2d_NCHWc_strategy.register("arm_cpu")
def depthwise_conv2d_NCHWc_strategy_arm_cpu(attrs, inputs, out_type, target):
    """depthwise_conv2d_NCHWc adopted from x86"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.x86.depthwise_conv2d_NCHWc, True, True),
        wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_NCHWc),
        name="depthwise_conv2d_NCHWc.x86",
    )
    return strategy


def wrap_compute_conv2d_winograd_nnpack(topi_compute):
    """wrap topi compute for conv2d_winograd NNPack"""

    def _compute_conv2d_nnpack(attrs, inputs, out_type):
        padding = attrs.get_int_tuple("padding")
        strides = attrs.get_int_tuple("strides")
        dilation = attrs.get_int_tuple("dilation")
        out_dtype = attrs.get_str("out_dtype")
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        return [topi_compute(inputs[0], inputs[1], None, strides, padding, dilation, out_dtype)]

    return _compute_conv2d_nnpack


@conv2d_winograd_without_weight_transfrom_strategy.register("arm_cpu")
def conv2d_winograd_without_weight_transfrom_strategy_arm_cpu(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom arm cpu strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    strides = attrs.get_int_tuple("strides")
    kernel = inputs[1]
    assert dilation == (1, 1), "Do not support dilate now"
    assert strides == (1, 1), "Do not support strides now"
    assert groups == 1, "Do not supoort arbitrary group number"
    strategy = _op.OpStrategy()
    if layout == "NCHW":
        if len(kernel.shape) == 5:
            pad_kh, pad_kw, _, _, _ = get_const_tuple(inputs[1].shape)
            tile_size = attrs.get_int("tile_size")
            kh = pad_kh - tile_size + 1
            kw = pad_kw - tile_size + 1
            assert kh == 3 and kw == 3
            strategy.add_implementation(
                wrap_compute_conv2d(topi.arm_cpu.conv2d_nchw_winograd),
                wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nchw_winograd),
                name="conv2d_nchw_winograd.arm_cpu",
            )
        elif len(kernel.shape) == 4:
            # kernel must be packed by winograd nnpack
            assert "nnpack" in target.libs
            strategy.add_implementation(
                wrap_compute_conv2d_winograd_nnpack(
                    topi.arm_cpu.conv2d_nchw_winograd_nnpack_without_weight_transform
                ),
                wrap_topi_schedule(
                    topi.arm_cpu.schedule_conv2d_nchw_winograd_nnpack_without_weight_transform
                ),
                name="conv2d_nchw_winograd_nnpack_withou_weight_transform.arm_cpu",
                plevel=15,
            )
        else:
            raise RuntimeError("Unsupported kernel shape: {}".format(kernel.shape))
    else:
        raise RuntimeError(
            "Unsupported conv2d_winograd_without_weight_transfrom layout {}".format(layout)
        )
    return strategy


def wrap_compute_conv2d_gemm(topi_compute):
    """wrap topi compute for conv2d_gemm"""

    def _compute_conv2d_gemm(attrs, inputs, out_type):
        padding = attrs.get_int_tuple("padding")
        strides = attrs.get_int_tuple("strides")
        dilation = attrs.get_int_tuple("dilation")
        out_dtype = attrs.get_str("out_dtype")
        channels = attrs["channels"]
        kernel_size = attrs["kernel_size"]
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        return [
            topi_compute(
                inputs[0], inputs[1], strides, padding, dilation, out_dtype, kernel_size, channels
            )
        ]

    return _compute_conv2d_gemm


@conv2d_gemm_without_weight_transform_strategy.register("arm_cpu")
def conv2d_gemm_without_weight_transform_strategy_arm_cpu(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom arm cpu strategy"""
    layout = attrs.data_layout
    data = inputs[0]
    strategy = _op.OpStrategy()

    interleaved_compute = topi.arm_cpu.compute_conv2d_NHWC_quantized_interleaved_without_transform
    native_compute = topi.arm_cpu.compute_conv2d_NHWC_quantized_native_without_transform
    if layout == "NHWC" and data.dtype in ["int8", "uint8"]:
        strategy.add_implementation(
            wrap_compute_conv2d_gemm(native_compute),
            wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_NHWC_quantized_native),
            name="conv2d_NHWC_quantized_native_without_transform.arm_cpu",
        )
        strategy.add_implementation(
            wrap_compute_conv2d_gemm(interleaved_compute),
            wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_NHWC_quantized_interleaved),
            name="conv2d_NHWC_quantized_interleaved_without_transform.arm_cpu",
        )
    else:
        raise RuntimeError(
            "Unsupported conv2d_NHWC_quantized_without_transform layout {0}"
            "with datatype {1}".format(layout, data.dtype)
        )
    return strategy


@conv2d_transpose_strategy.register(["arm_cpu", "micro_dev"])
def conv2d_transpose_strategy_arm_cpu(attrs, inputs, out_type, target):
    """conv2d_transpose arm cpu strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d_transpose(topi.arm_cpu.conv2d_transpose_nchw),
        wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_transpose_nchw),
        name="conv2d_tranpose_nchw.arm_cpu",
    )
    return strategy


@bitserial_conv2d_strategy.register("arm_cpu")
def bitserial_conv2d_strategy_arm_cpu(attrs, inputs, out_type, target):
    """bitserial_conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.x86.bitserial_conv2d_nchw),
            wrap_topi_schedule(topi.x86.schedule_bitserial_conv2d_nchw),
            name="bitserial_conv2d_nchw.arm_cpu",
        )
    elif layout == "NHWC":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.arm_cpu.bitserial_conv2d_nhwc),
            wrap_topi_schedule(topi.arm_cpu.schedule_bitserial_conv2d_nhwc),
            name="bitserial_conv2d_nhwc.arm_cpu",
        )
    else:
        raise ValueError("Data layout {} not supported.".format(layout))
    return strategy


@bitserial_dense_strategy.register("arm_cpu")
def schedule_bitserial_dense_arm_cpu(attrs, inputs, out_type, target):
    """bitserial_dense arm cpu strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_bitserial_dense(topi.arm_cpu.bitserial_dense),
        wrap_topi_schedule(topi.arm_cpu.schedule_bitserial_dense),
        name="bitserial_dense.arm_cpu",
    )
    return strategy
