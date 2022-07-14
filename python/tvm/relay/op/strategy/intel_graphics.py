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
"""Definition of x86 operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from .generic import *
from .. import op as _op


@conv2d_strategy.register("intel_graphics")
def conv2d_strategy_intel_graphics(attrs, inputs, out_type, target):
    """conv2d intel graphics strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.intel_graphics.conv2d_nchw),
                wrap_topi_schedule(topi.intel_graphics.schedule_conv2d_nchw),
                name="conv2d_nchw.intel_graphics",
            )
            # conv2d_NCHWc won't work without alter op layout pass
            # TODO(@Laurawly): fix this
            strategy.add_implementation(
                wrap_compute_conv2d(
                    topi.intel_graphics.conv2d_NCHWc, need_data_layout=True, need_out_layout=True
                ),
                wrap_topi_schedule(topi.intel_graphics.schedule_conv2d_NCHWc),
                name="conv2d_NCHWc.intel_graphics",
                plevel=5,
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {} for intel graphics".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.intel_graphics.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.intel_graphics.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.intel_graphics",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        raise RuntimeError("group_conv2d is not supported for intel graphics")
    return strategy


@conv2d_NCHWc_strategy.register("intel_graphics")
def conv2d_NCHWc_strategy_intel_graphics(attrs, inputs, out_type, target):
    """conv2d_NCHWc intel_graphics strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(
            topi.intel_graphics.conv2d_NCHWc, need_data_layout=True, need_out_layout=True
        ),
        wrap_topi_schedule(topi.intel_graphics.schedule_conv2d_NCHWc),
        name="conv2d_NCHWc.intel_graphics",
    )
    return strategy
