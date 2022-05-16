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
"""Conv2D alter op for Qualcomm Adreno GPU"""

import logging

import re
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from ..utils import get_const_tuple
from ..nn import conv2d_alter_layout

logger = logging.getLogger("topi")

# Number of wildcards for matching of supported layouts to be transformed
_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_OIHWo_matcher = re.compile("^OIHW[0-9]+o$")
_NHWCc_matcher = re.compile("^NHWC[0-9]+c$")
_HWIOo_matcher = re.compile("^HWIO[0-9]+o$")
_HWOIo_matcher = re.compile("^HWOI[0-9]+o$")


@conv2d_alter_layout.register("adreno")
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    """
    Prepare of the new conv2d with proper target blocked layout attributes
    OpenCL Textures supports 1d/2d/3d/4d tetures but read happens always only for 4 elements
    in a line. Thus way we are supporting for now only 4d conversions on the end
    NCHW -> NCHW4c & OIHW ->OIHW4o
    NHWC -> NHWC4c & HWIO -> HWIO4o & HWOI -> HWOI4o
    """
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    # Parse the attributes.
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data_tensor, kernel_tensor = tinfos
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype
    out_dtype = out_type.dtype

    if isinstance(dispatch_ctx, autotvm.task.ApplyGraphBest):
        cfg = dispatch_ctx.query(target, None)
        workload = cfg.workload
    else:
        impl, outs = relay.backend.te_compiler.select_implementation(
            relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target
        )
        workload = autotvm.task.get_workload(outs)
        if workload is None:
            return None

        cfg = dispatch_ctx.query(target, workload)

    topi_tmpl = workload[0]

    if "conv2d_nchwc" in topi_tmpl:  # covers both conv2d_nchwc and depthwise_conv2d_nchwc
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            batch, in_channels, in_height, in_width = data_tensor.shape
            out_channles, _, kernel_h, kernel_w = kernel_tensor.shape
            in_channel_block = in_channels % 4
            if in_channel_block == 0:
                in_channel_block = 4
            num_filter_block = out_channles % 4
            if num_filter_block == 0:
                num_filter_block = 4

            # no support yet for tensors that cannot be divisible by factor 4
            if in_channel_block != 4 or num_filter_block != 4:
                return None

            batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
            out_channel, in_filter_channel, kh, kw = get_const_tuple(kernel_tensor.shape)

            # update new attrs
            new_attrs["channels"] = out_channel
            new_attrs["data_layout"] = "NCHW%dc" % in_channel_block
            # (oc, ic, h, w) -> (OC, ic, h, w, oc)
            new_attrs["kernel_layout"] = "OIHW%do" % num_filter_block
            new_attrs["out_layout"] = "NCHW%dc" % num_filter_block

            # Store altered operator's config for applying of tuned AutoTVM statistics
            new_data = te.placeholder(
                (batch_size, in_channel // in_channel_block, height, width, in_channel_block),
                dtype=data_dtype,
            )
            new_kernel = te.placeholder(
                (out_channel // num_filter_block, in_filter_channel, kh, kw, num_filter_block),
                dtype=kernel_tensor.dtype,
            )
            new_workload = autotvm.task.args_to_workload(
                [
                    new_data,
                    new_kernel,
                    strides,
                    padding,
                    dilation,
                    out_dtype,
                ],
                topi_tmpl,  # "conv2d_nchwc.image2d",
            )
            dispatch_ctx.update(target, new_workload, cfg)
        else:
            assert _NCHWc_matcher.match(data_layout)
            assert _OIHWo_matcher.match(kernel_layout)
        return relay.nn.conv2d(*inputs, **new_attrs)

    if "conv2d_nhwc" in topi_tmpl:  # covers both conv2d_nhwcc and depthwise_conv2d_nhwcc
        if (data_layout == "NHWC" and kernel_layout == "HWIO") or (
            data_layout == "NHWC" and kernel_layout == "HWOI"
        ):
            if kernel_layout == "HWIO":
                batch_size, in_height, in_width, in_channels = data_tensor.shape
                kernel_h, kernel_w, in_filter_channel, out_channles = kernel_tensor.shape
            else:
                batch_size, in_height, in_width, in_channels = data_tensor.shape
                kernel_h, kernel_w, out_channles, in_filter_channel = kernel_tensor.shape
            in_channel_block = in_channels % 4
            if in_channel_block == 0:
                in_channel_block = 4
            num_filter_block = out_channles % 4
            if num_filter_block == 0:
                num_filter_block = 4

            # no support yet for tensors cannot be divisible by factor 4
            if in_channel_block != 4 or num_filter_block != 4:
                return None

            # update new attrs
            new_attrs["channels"] = out_channles
            new_attrs["data_layout"] = "NHWC%dc" % in_channel_block
            # (h, w, ic, oc) -> (h, w, ic, OC, oc)
            if kernel_layout == "HWIO":
                new_attrs["kernel_layout"] = "HWIO%do" % num_filter_block
            else:
                new_attrs["kernel_layout"] = "HWOI%do" % num_filter_block
            new_attrs["out_layout"] = "NHWC%dc" % num_filter_block

            # Store altered operator's config for applying of tuned AutoTVM statistics
            new_data = te.placeholder(
                (
                    batch_size,
                    in_height,
                    in_width,
                    in_channels // in_channel_block,
                    in_channel_block,
                ),
                dtype=data_dtype,
            )
            if kernel_layout == "HWIO":
                new_kernel = te.placeholder(
                    (
                        kernel_h,
                        kernel_w,
                        in_filter_channel,
                        out_channles // num_filter_block,
                        num_filter_block,
                    ),
                    dtype=kernel_tensor.dtype,
                )
            else:
                new_kernel = te.placeholder(
                    (
                        kernel_h,
                        kernel_w,
                        out_channles // num_filter_block,
                        in_filter_channel,
                        num_filter_block,
                    ),
                    dtype=kernel_tensor.dtype,
                )
            new_workload = autotvm.task.args_to_workload(
                [
                    new_data,
                    new_kernel,
                    strides,
                    padding,
                    dilation,
                    out_dtype,
                ],
                topi_tmpl,
            )
            dispatch_ctx.update(target, new_workload, cfg)
        else:
            assert _NHWCc_matcher.match(data_layout)
            assert _HWIOo_matcher.match(kernel_layout) or _HWOIo_matcher.match(kernel_layout)
        return relay.nn.conv2d(*inputs, **new_attrs)

    return None
