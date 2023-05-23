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

import logging

import re
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from .conv2d import _get_default_config
from .conv2d_int8 import is_int8_hw_support, _get_default_config_int8
from ..utils import get_const_tuple
from ..nn import conv2d_legalize, conv2d_alter_layout
from ..generic.conv2d import conv2d_alter_int8_common

logger = logging.getLogger("topi")

_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_OIHWio_matcher = re.compile("^OIHW[0-9]+i[0-9]+o$")


@conv2d_alter_layout.register("cpu")
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
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
            # The best implementation is not an AutoTVM template.
            # It may be from the auto-scheduler
            if impl.name.find("winograd") != -1:
                if dilation != (1, 1):
                    logger.warning("Does not support weight pre-transform for dilated convolution.")
                    return None

                assert data_layout == "NHWC" and kernel_layout == "HWIO"
                N, H, W, CI = get_const_tuple(data_tensor.shape)
                KH, KW, _, CO = get_const_tuple(kernel_tensor.shape)

                # Pre-compute weight transformation in winograd
                tile_size = 4
                # HWIO -> OIHW
                kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
                # alpha, alpha, CO, CI
                weight = relay.nn.contrib_conv2d_winograd_weight_transform(
                    kernel_transform, tile_size=tile_size
                )
                new_attrs["tile_size"] = tile_size
                new_attrs["channels"] = CO
                return relay.nn.contrib_conv2d_winograd_without_weight_transform(
                    inputs[0], weight, **new_attrs
                )
            return None

        cfg = dispatch_ctx.query(target, workload)

    topi_tmpl = workload[0]

    if topi_tmpl == "conv2d_NCHWc.x86":
        # we only convert conv2d_NCHW to conv2d_NCHWc for x86
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            if cfg.is_fallback:
                _get_default_config(
                    cfg,
                    data_tensor,
                    kernel_tensor,
                    strides,
                    padding,
                    dilation,
                    out_dtype,
                    False,
                    data_layout,
                )
            batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
            out_channel, _, kh, kw = get_const_tuple(kernel_tensor.shape)
            ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

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
                (out_channel // oc_bn, in_channel // ic_bn, kh, kw, ic_bn, oc_bn),
                dtype=kernel_tensor.dtype,
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
                topi_tmpl,
            )
            dispatch_ctx.update(target, new_workload, cfg)
        else:
            assert _NCHWc_matcher.match(data_layout)
            assert _OIHWio_matcher.match(kernel_layout)
        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_NCHWc_int8.x86":
        # TODO(@icemelon9, @anijain2305): Need to support data layout NHWC with kernel layout HWIO
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        if cfg.is_fallback:
            _get_default_config_int8(
                cfg,
                data_tensor,
                kernel_tensor,
                strides,
                padding,
                dilation,
                out_dtype,
                False,
                data_layout,
                int32_lanes=16,
            )

        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, channel_multiplier, kh, kw = get_const_tuple(kernel_tensor.shape)
        ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

        # update new attrs
        n_elems = 4
        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = f"NCHW{ic_bn}c"
        new_attrs["kernel_layout"] = f"OIHW{ic_bn // n_elems:n}i{oc_bn:n}o{n_elems:n}i"
        new_attrs["out_layout"] = f"NCHW{oc_bn}c"

        # Store altered operator's config.
        new_data = te.placeholder(
            (batch_size, in_channel // ic_bn, height, width, ic_bn), dtype=data_dtype
        )
        new_kernel = te.placeholder(
            (out_channel // oc_bn, in_channel // ic_bn, kh, kw, ic_bn // n_elems, oc_bn, n_elems),
            dtype=kernel_dtype,
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
            topi_tmpl,
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)

    if topi_tmpl == "depthwise_conv2d_NCHWc.x86":
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            if cfg.is_fallback:
                _get_default_config(
                    cfg,
                    data_tensor,
                    kernel_tensor,
                    strides,
                    padding,
                    dilation,
                    out_dtype,
                    True,
                    data_layout,
                )

            batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
            out_channel, channel_multiplier, kh, kw = get_const_tuple(kernel_tensor.shape)
            ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
            assert channel_multiplier == 1

            # update new attrs
            new_attrs["channels"] = out_channel
            new_attrs["data_layout"] = f"NCHW{ic_bn}c"
            new_attrs["kernel_layout"] = f"OIHW1i{oc_bn}o"
            new_attrs["out_layout"] = f"NCHW{oc_bn}c"

            # Store altered operator's config.
            new_data = te.placeholder(
                (batch_size, in_channel // ic_bn, height, width, ic_bn), dtype=data_dtype
            )
            new_kernel = te.placeholder(
                (out_channel // oc_bn, 1, kh, kw, 1, oc_bn), dtype=kernel_dtype
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
                topi_tmpl,
            )
            dispatch_ctx.update(target, new_workload, cfg)
        else:
            assert _NCHWc_matcher.match(data_layout)
            assert _OIHWio_matcher.match(kernel_layout)
        return relay.nn.contrib_depthwise_conv2d_nchwc(*inputs, **new_attrs)

    return None


@conv2d_legalize.register("cpu")
def _conv2d_legalize(attrs, inputs, arg_types):
    """Legalizes Conv2D op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    # Intel vector intructions require data and kernel to have different dtypes.
    if data_tensor.dtype == "int8" and kernel_tensor.dtype == "int8":
        data_dtype = "uint8"
    if is_int8_hw_support(data_dtype, kernel_dtype):
        return conv2d_alter_int8_common(
            data, data_tensor, kernel, kernel_tensor, output_tensor, attrs, data_dtype, 4, 16
        )
    return None
