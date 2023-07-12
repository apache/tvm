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
"""Definition of bifrost operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re
from tvm import topi
from .generic import *
from .. import op as _op


@conv2d_strategy.register("bifrost")
def conv2d_strategy_bifrost(attrs, inputs, out_type, target):
    """conv2d mali(bifrost) strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            if kernel_layout == "OIHW":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.bifrost.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.bifrost.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.bifrost",
                )

                _, _, kh, kw = get_const_tuple(kernel.shape)
                if (
                    kh == 3
                    and kw == 3
                    and stride_h == 1
                    and stride_w == 1
                    and dilation_h == 1
                    and dilation_w == 1
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.bifrost.conv2d_nchw_winograd),
                        wrap_topi_schedule(topi.bifrost.schedule_conv2d_nchw_winograd),
                        name="conv2d_nchw_winograd.bifrost",
                        plevel=5,
                    )
            elif re.match(r"OIHW\d*o", kernel_layout):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.bifrost.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.bifrost.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.bifrost",
                )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            # For now just reuse general Mali strategy.
            strategy.add_implementation(
                wrap_compute_conv2d(topi.mali.conv2d_nhwc_spatial_pack),
                wrap_topi_schedule(topi.mali.schedule_conv2d_nhwc_spatial_pack),
                name="conv2d_nhwc_spatial_pack.bifrost",
            )
        else:
            raise RuntimeError(f"Unsupported conv2d layout {layout} for Mali(Bifrost)")
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.bifrost.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.bifrost",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            # For now just reuse general Mali strategy.
            strategy.add_implementation(
                wrap_compute_conv2d(topi.mali.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.mali.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nchw.bifrost",
            )
        else:
            raise RuntimeError(f"Unsupported depthwise_conv2d layout {layout} for Mali(Bifrost)")
    else:  # group_conv2d
        raise RuntimeError("group_conv2d is not supported for Mali(Bifrost)")
    return strategy


@conv2d_winograd_without_weight_transform_strategy.register("bifrost")
def conv2d_winograd_without_weight_transform_strategy_bifrost(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transform mali(bifrost) strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    strides = attrs.get_int_tuple("strides")
    assert dilation == (1, 1), "Do not support dilate now"
    assert strides == (1, 1), "Do not support strides now"
    assert groups == 1, "Do not support arbitrary group number"
    strategy = _op.OpStrategy()
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_conv2d(topi.bifrost.conv2d_nchw_winograd),
            wrap_topi_schedule(topi.bifrost.schedule_conv2d_nchw_winograd),
            name="conv2d_nchw_winograd.bifrost",
        )
    else:
        raise RuntimeError(f"Unsupported conv2d_winograd_without_weight_transform layout {layout}")
    return strategy


@dense_strategy.register("bifrost")
def dense_strategy_bifrost(attrs, inputs, out_type, target):
    """dense mali(bifrost) strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(topi.bifrost.dense),
        wrap_topi_schedule(topi.bifrost.schedule_dense),
        name="dense.bifrost",
    )
    return strategy
