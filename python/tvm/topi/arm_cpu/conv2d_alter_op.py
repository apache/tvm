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

from ..nn import conv2d_alter_layout, conv2d_legalize
from ..utils import get_const_tuple
from ..x86.conv2d import _get_default_config as _get_x86_default_config
from .conv2d_int8 import is_int8_hw_support
from .arm_utils import get_tiling_B_interleaved_t
from ..nn.utils import get_pad_tuple

logger = logging.getLogger("topi")


def interleave_transpose_weights(inputs, data, kernel, interleave_A):
    """Transform the weight matrix by reshaping, interleaving and transposing it

    Parameters
    ----------
    inputs : tvm.relay.Expr
        Grouped input symbols
    data :
        Input shape and dtype
    kernel :
        Input shape and dtype
    interleave_A: indicates if we expect matrix A to be interleaved

    Returns
    ----------
    new_kernel : tvm.te.placeholder
                 A placeholder with the new shape
    new_kernel_expr : tvm.relay.Expr
                The relay expression of the weights
    """
    assert (
        data.dtype == "int8"
        and kernel.dtype == "int8"
        or data.dtype == "uint8"
        and kernel.dtype == "uint8"
    )

    KH, KW, IC, OC = get_const_tuple(kernel.shape)
    K = KH * KW * IC
    N = OC

    # Get tiling information for the interleaved transposed version of B
    tile_rows_B, tile_cols_B = get_tiling_B_interleaved_t(interleave_A)

    pad_K = 0
    pad_N = 0

    if N % tile_rows_B != 0:
        pad_N = tile_rows_B - (N % tile_rows_B)
    if K % tile_cols_B != 0:
        pad_K = tile_cols_B - (K % tile_cols_B)

    N_padded = N + pad_N
    K_padded = K + pad_K
    new_kernel_expr = relay.nn.contrib_conv2d_gemm_weight_transform(
        inputs[1], tile_rows_B, tile_cols_B
    )
    new_kernel = te.placeholder(
        (N_padded // tile_rows_B, K_padded // tile_cols_B, tile_rows_B, tile_cols_B), kernel.dtype
    )
    return new_kernel, new_kernel_expr


@conv2d_alter_layout.register(["arm_cpu"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current

    _, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target
    )
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

    # We don't perform layout alteration for NHWC layout with real data types
    if data_layout == "NHWC" and data_dtype not in ["uint8", "int8"]:
        return None

    if topi_tmpl == "conv2d_nchw_spatial_pack.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg["tile_co"].size[-1]

        new_attrs["kernel_layout"] = "OIHW%do" % VC

        new_data = data
        new_kernel = te.placeholder((idxd(CO, VC), CI, KH, KW, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_spatial_pack.arm_cpu",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_nhwc_spatial_pack.arm_cpu":
        assert (
            data.dtype == "int8"
            and kernel.dtype == "int8"
            or data.dtype == "uint8"
            and kernel.dtype == "uint8"
        )

        assert data_layout == "NHWC" and kernel_layout == "HWIO"

        data_expr, kernel_expr = inputs

        data_int16 = relay.cast(data_expr, dtype="int16")
        kernel_int16 = relay.cast(kernel_expr, dtype="int16")

        new_attrs = {k: attrs[k] for k in attrs.keys()}

        new_data = te.placeholder(data.shape, "int16")
        new_kernel = te.placeholder(kernel.shape, "int16")

        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nhwc_spatial_pack.arm_cpu",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(data_int16, kernel_int16, **new_attrs)

    if topi_tmpl == "conv2d_nchw_winograd.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg["tile_k"].size[-1]
        tile_size = 4

        weight_expr = inputs[1]
        weight_expr = relay.nn.contrib_conv2d_winograd_weight_transform(
            weight_expr, tile_size=tile_size
        )
        weight_expr = relay.reshape(
            weight_expr, newshape=(KH + tile_size - 1, KW + tile_size - 1, CO // VC, VC, CI)
        )
        weight_expr = relay.transpose(weight_expr, axes=[0, 1, 2, 4, 3])

        new_attrs["tile_size"] = tile_size
        new_attrs["channels"] = CO

        new_data = data
        new_kernel = te.placeholder(
            (KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), CI, VC), kernel.dtype
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd.arm_cpu",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight_expr, **new_attrs
        )

    if topi_tmpl == "conv2d_nchw_winograd_nnpack.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        new_attrs["channels"] = CO

        # pre-compute winograd_nnpack transform
        # for winograd_nnpack_fp16, the the precompute prune pass must run on device,
        # where float16 is supported
        weight_dtype = "float32"
        weight_expr = inputs[1]
        transformed_weight = relay.nn.contrib_conv2d_winograd_nnpack_weight_transform(
            weight_expr,
            convolution_algorithm=cfg["winograd_nnpack_algorithm"].val,
            out_dtype=weight_dtype,
        )

        new_data = data
        new_kernel = te.placeholder((CO, CI, 8, 8), "float32")

        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, None, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd_nnpack_without_weight_transform.arm_cpu",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], transformed_weight, **new_attrs
        )

    if topi_tmpl == "depthwise_conv2d_nchw_spatial_pack.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, M, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg["tile_co"].size[-1]

        new_attrs["kernel_layout"] = "OIHW%do" % (cfg["tile_co"].size[-1])

        # Store the same config for the altered operator (workload)
        new_data = data
        new_kernel = te.placeholder((idxd(CO, VC), M, KH, KW, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "depthwise_conv2d_nchw_spatial_pack.arm_cpu",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_NCHWc.x86":
        # Converting NCHW to NCHWc.
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        if cfg.is_fallback:
            _get_x86_default_config(
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
        new_attrs["data_layout"] = "NCHW%dc" % ic_bn
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs["kernel_layout"] = "OIHW%di%do" % (ic_bn, oc_bn)
        new_attrs["out_layout"] = "NCHW%dc" % oc_bn

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
        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)

    if topi_tmpl == "depthwise_conv2d_NCHWc.x86":
        # Converting NCHW to NCHWc.
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        if cfg.is_fallback:
            _get_x86_default_config(
                cfg, data_tensor, kernel_tensor, strides, padding, out_dtype, True, data_layout
            )

        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, channel_multiplier, kh, kw = get_const_tuple(kernel_tensor.shape)
        ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
        assert channel_multiplier == 1

        # update new attrs
        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = "NCHW%dc" % ic_bn
        new_attrs["kernel_layout"] = "OIHW1i%do" % oc_bn
        new_attrs["out_layout"] = "NCHW%dc" % oc_bn

        # Store altered operator's config.
        new_data = te.placeholder(
            (batch_size, in_channel // ic_bn, height, width, ic_bn), dtype=data_dtype
        )
        new_kernel = te.placeholder((out_channel // oc_bn, 1, kh, kw, 1, oc_bn), dtype=kernel_dtype)
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
        return relay.nn.contrib_depthwise_conv2d_nchwc(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_NCHWc_int8.arm_cpu":
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
            )

        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, channel_multiplier, kh, kw = get_const_tuple(kernel_tensor.shape)
        ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
        n_elems = 8

        # convert kernel data layout from 4D to 7D
        data_expr, kernel_expr = inputs
        kernel_IHWO = relay.transpose(kernel_expr, axes=(1, 2, 3, 0))
        kernel_IHWOo = relay.reshape(kernel_IHWO, (in_channel, kh, kw, out_channel // oc_bn, oc_bn))
        kernel_OHWoI = relay.transpose(kernel_IHWOo, axes=(3, 1, 2, 4, 0))
        kernel_OHWoIi = relay.reshape(
            kernel_OHWoI, (out_channel // oc_bn, kh, kw, oc_bn, in_channel // ic_bn, ic_bn)
        )
        kernel_OHWoIie = relay.reshape(
            kernel_OHWoIi,
            (out_channel // oc_bn, kh, kw, oc_bn, in_channel // ic_bn, ic_bn // n_elems, n_elems),
        )
        kernel_OIHWioe = relay.transpose(kernel_OHWoIie, axes=(0, 4, 1, 2, 5, 3, 6))

        # update new attrs
        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = "NCHW%dc" % ic_bn
        new_attrs["out_layout"] = "NCHW%dc" % oc_bn

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
        return relay.nn.contrib_conv2d_nchwc(data_expr, kernel_OIHWioe, **new_attrs)

    if topi_tmpl == "conv2d_NHWC_quantized_interleaved.arm_cpu":
        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        KH, KW, _, OC = get_const_tuple(kernel.shape)
        new_workload_name = "conv2d_NHWC_quantized_interleaved_without_transform.arm_cpu"
        new_kernel, new_kernel_expr = interleave_transpose_weights(
            inputs, data, kernel, interleave_A=True
        )
        new_workload = autotvm.task.args_to_workload(
            [data, new_kernel, strides, padding, dilation, out_dtype, (KH, KW), OC],
            new_workload_name,
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.contrib_conv2d_gemm_without_weight_transform(
            inputs[0], new_kernel_expr, **new_attrs
        )
    if topi_tmpl == "conv2d_NHWC_quantized_native.arm_cpu":
        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        KH, KW, _, OC = get_const_tuple(kernel.shape)
        new_workload_name = "conv2d_NHWC_quantized_native_without_transform.arm_cpu"
        new_kernel, new_kernel_expr = interleave_transpose_weights(
            inputs, data, kernel, interleave_A=False
        )
        new_workload = autotvm.task.args_to_workload(
            [data, new_kernel, strides, padding, dilation, out_dtype, (KH, KW), OC],
            new_workload_name,
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_gemm_without_weight_transform(
            inputs[0], new_kernel_expr, **new_attrs
        )
    return None


@conv2d_legalize.register("arm_cpu")
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

    # Dilation not supported yet. Return None if dilation is not (1, 1)
    dilation = attrs.get_int_tuple("dilation")
    if not (dilation[0] == 1 and dilation[1] == 1):
        return None

    # No legalization for depthwise convolutions yet.
    groups = attrs.get_int("groups")
    if groups != 1:
        return None

    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    # Get the conv attrs
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    is_int8_inputs = False
    # ARM intrinsics need the datatypes of data and kernel to be the same
    # Original --> C = A (conv) B
    # A and B are int8
    #   C = (A + 128 - 128) (conv) B
    #   C = (A' conv B) - 128 (conv) B
    # where A' = A + 128
    # and 128 (conv) B is basically a reduce on CRS axis for weights.
    #   C = (A - 128 + 128) (conv) B
    #   C = (A' conv B) + 128 (conv) B
    # where A' = A - 128
    if (
        data_tensor.dtype == "uint8"
        and kernel_tensor.dtype == "int8"
        or data_tensor.dtype == "int8"
        and kernel_tensor.dtype == "uint8"
    ):
        if data_tensor.dtype == "uint8" and kernel_tensor.dtype == "int8":
            # shift data to int8
            before_shift = relay.add
            after_shift = relay.subtract
            data_dtype = "int8"
        else:
            # shift data to uint8
            before_shift = relay.subtract
            after_shift = relay.add
            data_dtype = "uint8"
        is_int8_inputs = True
        padding = attrs.get_int_tuple("padding")
        kh, kw = attrs.get_int_tuple("kernel_size")
        pt, pl, pb, pr = get_pad_tuple(padding, (kh, kw))

        if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
            adjust_shift = relay.sum(relay.cast(kernel, dtype="int32"), axis=(0, 1, 2))
            pad_width = ((0, 0), (pt, pb), (pl, pr), (0, 0))
        elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
            pad_width = ((0, 0), (0, 0), (pt, pb), (pl, pr))
            adjust_shift = relay.sum(relay.cast(kernel, dtype="int32"), axis=(1, 2, 3))
            adjust_shift = relay.expand_dims(adjust_shift, axis=1, num_newaxis=2)
        else:
            return None

        data = relay.cast(data, "int32")
        data = before_shift(data, relay.const(128, "int32"))
        data = relay.cast(data, data_dtype)

        # Do external padding as pad value has to be 128.
        if any(padding):
            data = relay.nn.pad(data, pad_width=pad_width, pad_value=128)
        new_attrs["padding"] = (0, 0)

        # Multiply 128 to adjust shift.
        adjust_shift = relay.multiply(adjust_shift, relay.const(128, "int32"))

    # Legalize if the datatypes are suitable for fast Int8 instructions. Int8 instructions require
    # input channel to be a multiple of 8 and output channels to be a multiple of 8. For input
    # channels, we pad both the inputs and weights input channels. For output channels, we pad the
    # weight and stride_slice the output.
    if is_int8_hw_support(data_dtype, kernel_dtype):
        # Flags to remember if the expr is modified
        ic_modified = False
        oc_modified = False

        # Find the value of input and output channel.
        in_channel = -1
        out_channel = -1
        if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value
        elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
            in_channel = data_tensor.shape[1].value
            out_channel = kernel_tensor.shape[0].value
        else:
            return None

        if in_channel % 8 != 0:
            new_in_channel = ((in_channel + 8) // 8) * 8
            diff = new_in_channel - in_channel
            if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
                data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
                kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, diff), (0, 0)))
                ic_modified = True
            elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
                pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
                data = relay.nn.pad(data, pad_width=pad_width)
                kernel = relay.nn.pad(kernel, pad_width=pad_width)
                ic_modified = True
            else:
                return None

        new_out_channel = out_channel
        if out_channel % 8 != 0:
            new_out_channel = ((out_channel + 8) // 8) * 8
            diff = new_out_channel - out_channel
            if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
                kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
                oc_modified = True
            elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
                kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
                oc_modified = True
            else:
                return None

        if oc_modified:
            new_attrs["channels"] = new_out_channel
            out = tvm.relay.nn.conv2d(data, kernel, **new_attrs)
            original_out_shape = [x.value for x in output_tensor.shape]
            out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
        else:
            out = relay.nn.conv2d(data, kernel, **new_attrs)

        if is_int8_inputs:
            out = after_shift(out, adjust_shift)

        return out
    return None
