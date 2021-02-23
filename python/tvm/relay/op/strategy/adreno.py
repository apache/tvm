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
import re
from tvm import topi
from .generic import *
from .. import op as _op

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
    assert out_type.dtype == "float16", "No float32 input/output tensor support is currently provided for Adreno GPU"
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.mali.conv2d_nchw_spatial_pack),
                wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_spatial_pack),
                name="conv2d_nchw_spatial_pack.mali",
            )
        elif data_layout == "NCHW4c" and kernel_layout == "OIHW4o":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.conv2d_nchwc),
                wrap_topi_schedule(topi.adreno.schedule_conv2d_nchwc),
                name="conv2d_nchwc.image2d",
                plevel=10
            )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.conv2d_nchwc_acc32),
                wrap_topi_schedule(topi.adreno.schedule_conv2d_nchwc_acc32),
                name="conv2d_nchwc_acc32.image2d",
                plevel=20
            )
        else:
            raise RuntimeError("Layout not supported: ("+data_layout+", "+kernel_layout+") - only support NCHW4c / OIHW4o layouts for conv2d")
    elif is_depthwise_conv2d(data.shape, data_layout, kernel.shape, kernel_layout, groups):
        if data_layout == "NCHW4c" and kernel_layout == "OIHW4o":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nchwc),
                wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nchwc),
                name="depthwise_conv2d_nchwc.image2d",
                plevel=10
            )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.adreno.depthwise_conv2d_nchwc_acc32),
                wrap_topi_schedule(topi.adreno.schedule_depthwise_conv2d_nchwc_acc32),
                name="depthwise_conv2d_nchwc_acc32.image2d",
                plevel=20
            )
        else:
            raise RuntimeError("Layout not supported: ("+data_layout+", "+kernel_layout+") - only support NCHW4c / OIHW4o layouts for conv2d")
    else:
        raise RuntimeError("General group convolution is not currently supported")
    return strategy

