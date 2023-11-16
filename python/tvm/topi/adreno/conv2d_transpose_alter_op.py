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
"""Conv2D Transpose alter op for Qualcomm Adreno GPU"""

import logging

import re
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from ..utils import get_const_tuple
from ..nn import conv2d_transpose_alter_layout

logger = logging.getLogger("topi")

# Number of wildcards for matching of supported layouts to be transformed
_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_IOHWo_matcher = re.compile("^IOHW[0-9]+o$")


@conv2d_transpose_alter_layout.register("adreno")
def _alter_conv2d_transpose_layout(attrs, inputs, tinfos, out_type):
    """
    Prepare of the new conv2d_transpose with proper target blocked layout attributes
    OpenCL Textures supports 1d/2d/3d/4d tetures but read happens always only for 4 elements
    in a line. Thus way we are supporting for now only 4d conversions on the end
    NCHW -> NCHW4c & IOHW ->IOHW4o
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
    out_dtype = out_type.dtype

    if isinstance(dispatch_ctx, autotvm.task.ApplyGraphBest):
        cfg = dispatch_ctx.query(target, None)
        workload = cfg.workload
    else:
        impl, outs = relay.backend.te_compiler.select_implementation(
            relay.op.get("nn.conv2d_transpose"), attrs, tinfos, out_type, target
        )
        workload = autotvm.task.get_workload(outs)
        cfg = dispatch_ctx.query(target, workload)

    topi_tmpl = workload[0]

    if "conv2d_transpose_nchwc" in topi_tmpl:  # covers conv2d_transpose_nchwc
        if data_layout == "NCHW" and kernel_layout == "IOHW":
            batch, in_channels, in_height, in_width = data_tensor.shape
            _, out_channles, kernel_h, kernel_w = kernel_tensor.shape
            in_channel_block = in_channels % 4
            if in_channel_block == 0:
                in_channel_block = 4
            num_filter_block = out_channles % 4
            if num_filter_block == 0:
                num_filter_block = 4

            # no support yet for tensors that cannot be divisible by factor 4
            if num_filter_block != 4:
                return None

            batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
            in_filter_channel, out_channel, kh, kw = get_const_tuple(kernel_tensor.shape)

            # update new attrs
            new_attrs["channels"] = out_channel
            if in_channel_block == 4:
                new_attrs["data_layout"] = f"NCHW{in_channel_block}c"
            else:
                new_attrs["data_layout"] = "NCHW"
            # (oc, ic, h, w) -> (ic, OC, h, w, oc)
            new_attrs["kernel_layout"] = f"IOHW{num_filter_block}o"
            new_attrs["out_layout"] = f"NCHW{num_filter_block}c"

            # Store altered operator's config for applying of tuned AutoTVM statistics
            if in_channel_block == 4:
                new_data = te.placeholder(
                    (batch_size, in_channel // in_channel_block, height, width, in_channel_block),
                    dtype=data_dtype,
                )
            else:
                new_data = data_tensor
            new_kernel = te.placeholder(
                (in_filter_channel, out_channel // num_filter_block, kh, kw, num_filter_block),
                dtype=kernel_tensor.dtype,
            )
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, dilation, out_dtype],
                topi_tmpl,  # "conv2d_transpose_nchwc.image2d",
            )
            dispatch_ctx.update(target, new_workload, cfg)
        else:
            assert _NCHWc_matcher.match(data_layout)
            assert _IOHWo_matcher.match(kernel_layout)
        return relay.nn.conv2d_transpose(*inputs, **new_attrs)

    return None
