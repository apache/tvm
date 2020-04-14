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
"""Conv2D alter op and legalize functions for arm cpu"""

import logging

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm

from ..nn import conv2d_alter_layout
from ..util import get_const_tuple
from ..x86.conv2d import _get_default_config as _get_x86_default_config

logger = logging.getLogger('topi')


@conv2d_alter_layout.register(["arm_cpu"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current

    _, outs = relay.backend.compile_engine.select_implementation(
        relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target)
    workload = autotvm.task.get_workload(outs)
    if workload is None:
        # The best implementation is not an AutoTVM template,
        # we then assume it's not necessary to alter this op.
        return None
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:  # if is fallback, clear query cache and return None
        autotvm.task.clear_fallback_cache(target, workload)
        return None

    topi_tmpl = workload[0]
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data, kernel = tinfos
    out_dtype = out_type.dtype

    # Extract data types
    data_tensor, kernel_tensor = tinfos
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype

    idxd = tvm.tir.indexdiv

    if topi_tmpl == "conv2d_nchw_spatial_pack.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg['tile_co'].size[-1]

        new_attrs['kernel_layout'] = 'OIHW%do' % VC

        new_data = data
        new_kernel = te.placeholder((idxd(CO, VC), CI, KH, KW, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_spatial_pack.arm_cpu")
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_nhwc_spatial_pack.arm_cpu":
        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        N, H, W, CI = get_const_tuple(data.shape)
        KH, KW, _, CO = get_const_tuple(kernel.shape)
        VC = cfg['tile_co'].size[-1]

        new_attrs['kernel_layout'] = 'OHWI%do' % VC

        new_data = data
        new_kernel = te.placeholder((idxd(CO, VC), KH, KW, CI, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nhwc_spatial_pack.arm_cpu")
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_nchw_winograd.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg['tile_k'].size[-1]
        tile_size = 4

        weight_expr = inputs[1]
        weight_expr = relay.nn.contrib_conv2d_winograd_weight_transform(
            weight_expr, tile_size=tile_size)
        weight_expr = relay.reshape(weight_expr,
                                    newshape=(KH + tile_size - 1,
                                              KW + tile_size - 1,
                                              idxd(CO, VC), VC, CI))
        weight_expr = relay.transpose(weight_expr, axes=[0, 1, 2, 4, 3])

        new_attrs['tile_size'] = tile_size

        new_data = data
        new_kernel = te.placeholder((KH + tile_size - 1,
                                     KW + tile_size -1,
                                     idxd(CO, VC), CI, VC),
                                    kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            'conv2d_nchw_winograd.arm_cpu')
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight_expr, **new_attrs)

    if topi_tmpl == "conv2d_nchw_winograd_nnpack.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        # pre-compute winograd_nnpack transform
        # for winograd_nnpack_fp16, the the precompute prune pass must run on device,
        # where float16 is supported
        weight_dtype = 'float32'
        weight_expr = inputs[1]
        transformed_weight = relay.nn.contrib_conv2d_winograd_nnpack_weight_transform(
            weight_expr,
            convolution_algorithm=cfg['winograd_nnpack_algorithm'].val,
            out_dtype=weight_dtype)

        new_data = data
        new_kernel = te.placeholder((CO, CI, 8, 8), "float32")

        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, None, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu")
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], transformed_weight, **new_attrs)

    if topi_tmpl == "depthwise_conv2d_nchw_spatial_pack.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, M, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg['tile_co'].size[-1]

        new_attrs['kernel_layout'] = 'OIHW%do' % (cfg['tile_co'].size[-1])

        # Store the same config for the altered operator (workload)
        new_data = data
        new_kernel = te.placeholder((idxd(CO, VC), M, KH, KW, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "depthwise_conv2d_nchw_spatial_pack.arm_cpu")
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_NCHWc.x86":
        # Converting NCHW to NCHWc.
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        if cfg.is_fallback:
            _get_x86_default_config(cfg, data_tensor, kernel_tensor, strides, padding,
                                    out_dtype, False, data_layout)
        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, _, kh, kw = get_const_tuple(kernel_tensor.shape)
        ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

        # update new attrs
        new_attrs['channels'] = out_channel
        new_attrs['data_layout'] = 'NCHW%dc' % ic_bn
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)
        new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

        # Store altered operator's config
        new_data = te.placeholder((batch_size, in_channel//ic_bn, height, width, ic_bn),
                                  dtype=data_dtype)
        new_kernel = te.placeholder((out_channel//oc_bn, in_channel//ic_bn,
                                     kh, kw, ic_bn, oc_bn), dtype=kernel_tensor.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_attrs["data_layout"],
             new_attrs["out_layout"], out_dtype], topi_tmpl)
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)

    if topi_tmpl == "depthwise_conv2d_NCHWc.x86":
        # Converting NCHW to NCHWc.
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        if cfg.is_fallback:
            _get_x86_default_config(cfg, data_tensor, kernel_tensor, strides, padding,
                                    out_dtype, True, data_layout)

        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, channel_multiplier, kh, kw = get_const_tuple(kernel_tensor.shape)
        ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
        assert channel_multiplier == 1

        # update new attrs
        new_attrs['channels'] = out_channel
        new_attrs['data_layout'] = 'NCHW%dc' % ic_bn
        new_attrs['kernel_layout'] = 'OIHW1i%do' % oc_bn
        new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

        # Store altered operator's config.
        new_data = te.placeholder((batch_size, in_channel//ic_bn, height, width, ic_bn),
                                  dtype=data_dtype)
        new_kernel = te.placeholder((out_channel//oc_bn, 1, kh, kw, 1, oc_bn), dtype=kernel_dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_attrs['data_layout'],
             new_attrs['out_layout'], out_dtype], topi_tmpl)
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_depthwise_conv2d_nchwc(*inputs, **new_attrs)

    return None
