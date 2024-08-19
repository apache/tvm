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
"""Definition of adreno operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from .generic import *
from .. import op as _op


@conv2d_NCHWc_strategy.register("adreno")
@conv2d_strategy.register("adreno")
def conv2d_strategy_adreno(attrs, inputs, out_type, target):
    """conv2d adreno strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    groups = attrs.groups
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if (
            (data_layout == "NCHW" and kernel_layout == "OIHW")
            or (data_layout == "NCHW4c" and kernel_layout == "OIHW4o")
            or (data_layout == "NCHW" and kernel_layout == "OIHW4o")
        ):
            if len(kernel.shape) == 4:
                oc, _, kh, kw = get_const_tuple(kernel.shape)
            else:
                oc, _, kh, kw, _ = get_const_tuple(kernel.shape)
            # We cannot use textures for case than number of channels is less than 4.
            # So, we use compute functions from cuda.
            if len(kernel.shape) == 4 and oc < 4:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
                    name="conv2d_nchw.cuda",
                )
                return strategy
            if (
                (2 < kh < 8 and 2 < kw < 8 and kh == kw)
                and (stride_h == 1 and stride_w == 1)
                and (dilation_h == 1 and dilation_w == 1)
                and not (data_layout == "NCHW" and kernel_layout == "OIHW4o")
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.adreno.conv2d_nchw_winograd),
                    wrap_topi_schedule(topi.adreno.schedule_conv2d_nchw_winograd),
                    name="conv2d_nchw_winograd.image2d",
                    plevel=5,
                )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.conv2d_nchwc),
                wrap_topi_schedule(topi.adreno.schedule_conv2d_nchwc),
                name="conv2d_nchwc.image2d",
                plevel=10,
            )
        elif (
            (data_layout == "NHWC" and kernel_layout == "HWIO")
            or (data_layout == "NHWC4c" and kernel_layout == "HWIO4o")
            or (data_layout == "NHWC" and kernel_layout == "HWIO4o")
        ):
            if len(kernel.shape) == 4:
                kh, kw, _, oc = get_const_tuple(kernel.shape)
            else:
                kh, kw, _, oc, _ = get_const_tuple(kernel.shape)
            # We cannot use textures for case than number of channels is less than 4.
            # So, we use compute functions from cuda.
            if len(kernel.shape) == 4 and oc < 4:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.gpu.conv2d_nhwc),
                    wrap_topi_schedule(topi.gpu.schedule_conv2d_nhwc),
                    name="conv2d_nhwc.gpu",
                )
                return strategy
            if (
                (2 < kh < 8 and 2 < kw < 8 and kh == kw)
                and (stride_h == 1 and stride_w == 1)
                and (dilation_h == 1 and dilation_w == 1)
                and not (data_layout == "NHWC" and kernel_layout == "HWIO4o")
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.adreno.conv2d_nhwc_winograd),
                    wrap_topi_schedule(topi.adreno.schedule_conv2d_nhwc_winograd),
                    name="conv2d_nhwc_winograd.image2d",
                    plevel=5,
                )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.conv2d_nhwc),
                wrap_topi_schedule(topi.adreno.schedule_conv2d_nhwc),
                name="conv2d_nhwc.image2d",
                plevel=10,
            )
        else:
            raise RuntimeError(
                "Layout not supported: ("
                + data_layout
                + ", "
                + kernel_layout
                + ") - only support NCHW4c / OIHW4o and NHWC / HWOI layouts for conv2d"
            )
    else:
        # cannot use is_depthwise_conv2d because it does not know about NHWC4c/HWOI4o layouts
        if data_layout == "NCHW":
            ic = data.shape[1]
        elif data_layout == "NCHW4c":
            ic = data.shape[1] * data.shape[4]
        elif data_layout == "NHWC":
            ic = data.shape[3]
        elif data_layout == "NHWC4c":
            ic = data.shape[3] * data.shape[4]
        else:
            raise RuntimeError(f"Unsupported depthwise_conv2d data layout {data_layout}")
        if kernel_layout == "OIHW":
            oc = kernel.shape[0]
        elif kernel_layout == "OIHW4o":
            oc = kernel.shape[0] * kernel.shape[4]
        elif kernel_layout == "HWOI":
            oc = kernel.shape[2]
        elif kernel_layout == "HWOI4o":
            oc = kernel.shape[2] * kernel.shape[4]
        else:
            raise RuntimeError(f"Unsupported depthwise_conv2d kernel layout {kernel_layout}")

        if ic == oc == groups:
            if (data_layout == "NCHW" and kernel_layout == "OIHW") or (
                data_layout == "NCHW4c" and kernel_layout == "OIHW4o"
            ):
                # We cannot use textures for case than number of channels is less than 4.
                # So, we use compute functions from cuda.
                if len(kernel.shape) == 4 and oc < 4:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.depthwise_conv2d_nchw),
                        wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nchw),
                        name="depthwise_conv2d_nchw.cuda",
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nchwc),
                        wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nchwc),
                        name="depthwise_conv2d_nchwc.image2d",
                        plevel=10,
                    )
            elif (data_layout == "NHWC" and kernel_layout == "HWOI") or (
                data_layout == "NHWC4c" and kernel_layout == "HWOI4o"
            ):
                if data.shape[-1] >= 4:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nhwc),
                        wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nhwc),
                        name="depthwise_conv2d_nhwc.image2d",
                        plevel=10,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                        wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                        name="depthwise_conv2d_nhwc.cuda",
                    )
            else:
                raise RuntimeError(
                    "Layout not supported: ("
                    + data_layout
                    + ", "
                    + kernel_layout
                    + ") - only support NCHW4c / OIHW4o and NHWC / HWOI layouts for conv2d"
                )
        elif (data_layout == "NCHW4c" or data_layout == "NCHW") and (
            kernel_layout == "OIHW" or kernel_layout == "OIHW4o"
        ):
            pad_in_chunks = (len(data.shape) == 5 and data.shape[1] % groups != 0) or (
                len(data.shape) == 4 and data.shape[1] % (groups * 4) != 0
            )
            pad_out_chunks = (len(kernel.shape) == 5 and kernel.shape[0] % groups != 0) or (
                len(kernel.shape) == 4 and kernel.shape[0] % (groups * 4) != 0
            )

            if not (pad_in_chunks or pad_out_chunks):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.adreno.group_conv2d_nchwc),
                    wrap_topi_schedule(topi.adreno.schedule_group_conv2d_nchwc),
                    name="group_conv2d_nchwc.image2d",
                    plevel=10,
                )
            elif len(data.shape) == 4 and len(kernel.shape) == 4:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.group_conv2d_nchw, has_groups=True),
                    wrap_topi_schedule(topi.cuda.schedule_group_conv2d_nchw),
                    name="group_conv2d_nchw.cuda",
                )
            else:
                raise RuntimeError(
                    "General group convolution is not currently supported for NCHWc layouts"
                )
        else:
            raise RuntimeError(
                "General group convolution has limited support for NCHW(4c) layouts..."
            )
    return strategy


@conv2d_winograd_without_weight_transform_strategy.register("adreno")
def conv2d_winograd_without_weight_transform_strategy_adreno(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transform adreno strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    assert dilation == (1, 1), "Do not support dilate now"
    assert groups == 1, "Do not support arbitrary group number"
    strategy = _op.OpStrategy()
    if layout in ("NCHW", "NCHW4c"):
        strategy.add_implementation(
            wrap_compute_conv2d(topi.adreno.conv2d_nchw_winograd_without_weight_transform),
            wrap_topi_schedule(topi.adreno.schedule_conv2d_nchw_winograd_without_weight_transform),
            name="conv2d_nchw_winograd_without_weight_transform.image2d",
            plevel=5,
        )
    elif layout in ("NHWC", "NHWC4c"):
        strategy.add_implementation(
            wrap_compute_conv2d(topi.adreno.conv2d_nhwc_winograd_without_weight_transform),
            wrap_topi_schedule(topi.adreno.schedule_conv2d_nhwc_winograd_without_weight_transform),
            name="conv2d_nhwc_winograd_without_weight_transform.image2d",
            plevel=5,
        )
    else:
        raise RuntimeError(f"Unsupported conv2d_winograd_without_weight_transform layout {layout}")
    return strategy


@conv2d_transpose_strategy.register("adreno")
def conv2d_transpose_strategy_adreno(attrs, inputs, out_type, target):
    """conv2d_transpose adreno strategy"""
    strategy = _op.OpStrategy()
    _, kernel = inputs
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.groups
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    assert dilation == (1, 1), "not support dilate now"

    if (groups == 1) and (
        (data_layout == "NCHW" and kernel_layout == "IOHW")
        or (data_layout == "NCHW4c" and kernel_layout == "IOHW4o")
        or (data_layout == "NCHW" and kernel_layout == "IOHW4o")
    ):
        if len(kernel.shape) == 4:
            _, oc, _, _ = get_const_tuple(kernel.shape)
        else:
            _, oc, _, _, _ = get_const_tuple(kernel.shape)
        # We cannot use textures for case than number of channels is less than 4.
        # So, we use compute functions from cuda.
        if len(kernel.shape) == 4 and oc < 4:
            strategy.add_implementation(
                wrap_compute_conv2d_transpose(topi.cuda.conv2d_transpose_nchw),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_transpose_nchw),
                name="conv2d_transpose_nchw.cuda",
            )
            return strategy
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.adreno.conv2d_transpose_nchwc),
            wrap_topi_schedule(topi.adreno.schedule_conv2d_transpose_nchwc),
            name="conv2d_transpose_nchwc.image2d",
            plevel=10,
        )
    elif data_layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.cuda.conv2d_transpose_nchw, has_groups=True),
            wrap_topi_schedule(topi.cuda.schedule_conv2d_transpose_nchw),
            name="conv2d_transpose_nchw.cuda",
        )
    else:
        raise RuntimeError(
            "Layout not supported: ("
            + data_layout
            + ", "
            + kernel_layout
            + ") - only support NCHW, NCHW4c / IOHW4o layouts for conv2d_transpose"
        )
    return strategy


@schedule_pool.register("adreno")
def schedule_pool_adreno(attrs, outs, target):
    """schedule pooling ops for adreno"""
    with target:
        if attrs.layout == "NCHW4c":
            return topi.adreno.schedule_pool(outs, attrs.layout)
        return topi.cuda.schedule_pool(outs, attrs.layout)


@schedule_injective.register(["adreno"])
def schedule_injective_adreno(attrs, outs, target):
    """schedule injective ops for adreno"""
    with target:
        return topi.adreno.schedule_injective(outs)


@schedule_reduce.register(["adreno"])
def schedule_reduce_adreno(attrs, outs, target):
    """schedule reduction ops for adreno GPU"""
    with target:
        return topi.adreno.schedule_reduce(outs)


@schedule_adaptive_pool.register(["adreno"])
def schedule_adaptive_pool_adreno(attrs, outs, target):
    """schedule adaptive pooling ops for adreno"""
    with target:
        return topi.adreno.schedule_adaptive_pool(outs, attrs.layout)


@concatenate_strategy.register(["adreno"])
def concatenate_strategy_adreno(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_concat(topi.transform.concatenate),
        wrap_topi_schedule(topi.adreno.schedule_injective),
        name="concatenate.adreno",
    )
    return strategy
