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
"""Conv2D alter op and legalize functions for x86"""

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm

from ..utils import get_const_tuple
from ..nn import conv2d_alter_layout, conv2d_infer_layout
from .conv2d import _get_default_config


@conv2d_alter_layout.register(["intel_graphics"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    if isinstance(dispatch_ctx, autotvm.task.ApplyGraphBest):
        cfg = dispatch_ctx.query(target, None)
        workload = cfg.workload
    else:
        _, outs = relay.backend.te_compiler.select_implementation(
            relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target
        )
        workload = autotvm.task.get_workload(outs)
        if workload is None:
            # The best implementation is not an AutoTVM template,
            # we then assume it's not necessary to alter this op.
            return None
        cfg = dispatch_ctx.query(target, workload)

    topi_tmpl = workload[0]
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data_tensor, kernel_tensor = tinfos
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype
    out_dtype = out_type.dtype

    if topi_tmpl == "conv2d_NCHWc.intel_graphics":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        if cfg.is_fallback:
            _get_default_config(cfg, data_tensor, kernel_tensor, strides, padding, out_dtype, False)
        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, _, kh, kw = get_const_tuple(kernel_tensor.shape)
        ic_bn = cfg["tile_ic"].val if hasattr(cfg["tile_ic"], "val") else cfg["tile_ic"].size[-1]
        oc_bn = cfg["tile_oc"].val if hasattr(cfg["tile_oc"], "val") else cfg["tile_oc"].size[-1]

        # update new attrs
        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = f"NCHW{ic_bn}c"
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs["kernel_layout"] = f"OIHW{ic_bn}i{oc_bn}o"
        new_attrs["out_layout"] = f"NCHW{oc_bn}c"

        # Store altered operator's config
        new_data = te.placeholder(
            (batch_size, in_channel // ic_bn, height, width, ic_bn), dtype=data_dtype
        )
        new_kernel = te.placeholder(
            (out_channel // oc_bn, in_channel // ic_bn, kh, kw, ic_bn, oc_bn), dtype=kernel_dtype
        )
        new_workload = autotvm.task.args_to_workload(
            [
                new_data,
                new_kernel,
                strides,
                padding,
                dilation,
                new_attrs["data_layout"],
                new_attrs["out_layout"],
                out_dtype,
            ],
            "conv2d_NCHWc.intel_graphics",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)

    return None


@conv2d_infer_layout.register("intel_graphics")
def _conv2d_infer_layout(workload, cfg):
    _, data, kernel, strides, padding, dilation, layout, dtype = workload
    batch_size, in_channel, in_height, in_width = data[1]
    out_channel, _, k_height, k_width = kernel[1]
    out_height = (in_height + 2 * padding[0] - k_height) // strides[0] + 1
    out_width = (in_width + 2 * padding[1] - k_width) // strides[1] + 1
    tile_ic, tile_oc = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    in_shape = (batch_size, in_channel // tile_ic, in_height, in_width, tile_ic)
    in_layout = f"NCHW{tile_ic}c"
    out_shape = (batch_size, out_channel // tile_oc, out_height, out_width, tile_oc)
    out_layout = f"NCHW{tile_oc}c"
    return ((in_shape, in_layout),), ((out_shape, out_layout),)
