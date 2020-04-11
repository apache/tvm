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
"""Definition of mali operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re
import topi
from .generic import *
from .. import op as _op

@conv2d_strategy.register("mali")
def conv2d_strategy_mali(attrs, inputs, out_type, target):
    """conv2d mali strategy"""
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
                    wrap_compute_conv2d(topi.mali.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.mali")
                # check if winograd algorithm is applicable
                _, _, kh, kw = get_const_tuple(kernel.shape)
                if kh == 3 and kw == 3 and stride_h == 1 and stride_w == 1 and \
                   dilation_h == 1 and dilation_w == 1:
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.mali.conv2d_nchw_winograd),
                        wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_winograd),
                        name="conv2d_nchw_winograd.mali",
                        plevel=5)
            elif re.match(r"OIHW\d*o", kernel_layout):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.mali.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.mali")
            else:
                raise RuntimeError("Unsupported weight layout {} for conv2d NCHW".
                                   format(kernel_layout))
        else:
            raise RuntimeError("Unsupported conv2d layout {} for mali".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.mali.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.mali.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.mali")
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {} for mali".format(layout))
    else: # group_conv2d
        raise RuntimeError("group_conv2d is not supported for mali")
    return strategy

@conv2d_winograd_without_weight_transfrom_strategy.register("mali")
def conv2d_winograd_without_weight_transfrom_strategy_mali(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom mali strategy"""
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
        assert len(kernel.shape) == 5, "Kernel must be packed into 5-dim"
        strategy.add_implementation(
            wrap_compute_conv2d(topi.mali.conv2d_nchw_winograd),
            wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_winograd),
            name="conv2d_nchw_winograd.mali")
    else:
        raise RuntimeError("Unsupported conv2d_winograd_without_weight_transfrom layout {}".
                           format(layout))
    return strategy

@dense_strategy.register("mali")
def dense_strategy_mali(attrs, inputs, out_type, target):
    """dense mali strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(wrap_compute_dense(topi.mali.dense),
                                wrap_topi_schedule(topi.mali.schedule_dense),
                                name="dense.mali")
    return strategy
