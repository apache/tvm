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

import numpy as np

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.target.target import Target

from ..nn import conv2d_alter_layout, conv2d_legalize
from ..utils import get_const_tuple
from ..x86.conv2d import _get_default_config as _get_x86_default_config
from ..x86.conv2d_int8 import _get_default_config_int8
from .conv2d_int8 import is_int8_hw_support
from .arm_utils import get_tiling_B_transformed, get_conv2d_weights_padding
from ..generic.conv2d import conv2d_alter_int8_common
from .mprofile.dsp.micro_kernel.common import num_simd_lanes_per_word

logger = logging.getLogger("topi")


def transform_weights(inputs, data, kernel, interleave_A):
    """Transform the weight matrix by tiling, interleaving (and transposing it)

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

    KH, KW, IC, OC = get_const_tuple(kernel.shape)
    K = KH * KW * IC
    N = OC

    # Get tiling information for the transformed version of B
    tile_N, tile_K = get_tiling_B_transformed(interleave_A, data.dtype)
    pad_N, pad_K = get_conv2d_weights_padding(N, K, tile_N, tile_K)

    N_padded = N + pad_N
    K_padded = K + pad_K
    new_kernel_expr = relay.nn.contrib_conv2d_gemm_weight_transform(inputs[1], tile_N, tile_K)
    if data.dtype in ["int8", "uint8"]:
        new_kernel = te.placeholder(
            (N_padded // tile_N, K_padded // tile_K, tile_N, tile_K),
            kernel.dtype,
        )
    else:
        new_kernel = te.placeholder(
            (N_padded // tile_N, K_padded // tile_K, tile_K, tile_N),
            kernel.dtype,
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

    if topi_tmpl == "depthwise_conv2d_nhwc_dsp.arm_cpu":
        assert data_layout == "NHWC" and kernel_layout == "HWOI"

        # We are not able to check if inputs[1] (the kernel) is a constant in the
        # strategy function, so as a stopgap solution we use an assert here.
        assert isinstance(
            inputs[1], relay.Constant
        ), "depthwise_conv2d_nhwc_dsp.arm_cpu requires kernel be a relay Constant"

        channels = get_const_tuple(data.shape)[3]
        KH, KW, _, _ = get_const_tuple(kernel.shape)
        simd_lanes = num_simd_lanes_per_word(data.dtype)

        HWOI_kernel_np = inputs[1].data.numpy()
        CHWc_kernel_np = np.zeros((channels // simd_lanes, KH, KW, simd_lanes), dtype=kernel.dtype)
        for i in range(channels // simd_lanes):
            CHWc_kernel_np[i] = HWOI_kernel_np[:, :, simd_lanes * i : simd_lanes * (i + 1), 0]
        reshaped_new_kernel = CHWc_kernel_np.reshape((KH, KW, channels, 1))

        # Store the same config for the altered operator (workload)
        new_data = data
        new_kernel = te.placeholder((KH, KW, channels, 1), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "depthwise_conv2d_nhwc_dsp.arm_cpu",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(
            inputs[0], relay.Constant(tvm.nd.array(reshaped_new_kernel)), **new_attrs
        )

    if topi_tmpl == "conv2d_NHWC_hybrid.arm_cpu":
        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        KH, KW, _, OC = get_const_tuple(kernel.shape)
        new_workload_name = "conv2d_NHWC_hybrid_without_transform.arm_cpu"
        new_kernel, new_kernel_expr = transform_weights(inputs, data, kernel, interleave_A=False)
        new_workload = autotvm.task.args_to_workload(
            [data, new_kernel, strides, padding, dilation, out_dtype, (KH, KW), OC],
            new_workload_name,
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_gemm_without_weight_transform(
            inputs[0], new_kernel_expr, **new_attrs
        )

    if (
        topi_tmpl == "conv2d_NHWC_hybrid_SME.arm_cpu"
        and data_dtype == "float16"
        and kernel_dtype == "float16"
        and out_dtype == "float32"
    ):
        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        KH, KW, IC, OC = get_const_tuple(kernel.shape)
        K = KH * KW * IC
        N = OC
        # The SME schedule for float16->float32 prearranges the two matrices to be multiplied
        # using the ARM_SME_BLOCK2_2SVLx1SVL_FP16_TRANSPOSE_INTERLEAVE intrinsic which expects
        # the reduction axis K as the second dimension of the matrix (i.e. shape = (_, K)).
        # This means that the flattened weights matrix B needs to be transposed to (N, K).
        transposed_kernel_expr = relay.transpose(inputs[1], axes=[3, 0, 1, 2])
        transposed_flattened_kernel_expr = relay.reshape(transposed_kernel_expr, newshape=(N, K))
        new_kernel_expr = transposed_flattened_kernel_expr
        new_kernel = te.placeholder((N, K), kernel.dtype)
        new_workload_name = "conv2d_NHWC_hybrid_SME_transposed_B.arm_cpu"
        new_workload = autotvm.task.args_to_workload(
            [data, new_kernel, strides, padding, dilation, out_dtype, (KH, KW), OC],
            new_workload_name,
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_gemm_without_weight_transform(
            inputs[0], new_kernel_expr, **new_attrs
        )

    # Only microTVM does layout alteration for NHWC layout with real data types
    if data_layout == "NHWC" and data_dtype not in ["uint8", "int8"]:
        return None

    if topi_tmpl == "conv2d_nchw_spatial_pack.arm_cpu":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg["tile_co"].size[-1]

        new_attrs["kernel_layout"] = f"OIHW{VC}o"

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
        # for winograd_nnpack_fp16, the precompute prune pass must run on device,
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

        new_attrs["kernel_layout"] = f"OIHW{cfg['tile_co'].size[-1]}o"

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
        new_attrs["data_layout"] = f"NCHW{ic_bn}c"
        new_attrs["kernel_layout"] = f"OIHW1i{oc_bn}o"
        new_attrs["out_layout"] = f"NCHW{oc_bn}c"

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
        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, _, kh, kw = get_const_tuple(kernel_tensor.shape)

        n_elems = 4

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
                int32_lanes=4,
            )

        ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

        if cfg.is_fallback:
            # ic_bn needs to be divided by n_elems below
            ic_bn = max(ic_bn, n_elems)

        # update new attrs
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

    if topi_tmpl == "conv2d_NHWC_quantized_interleaved.arm_cpu":
        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        KH, KW, _, OC = get_const_tuple(kernel.shape)
        new_workload_name = "conv2d_NHWC_quantized_interleaved_without_transform.arm_cpu"
        new_kernel, new_kernel_expr = transform_weights(inputs, data, kernel, interleave_A=True)
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
        new_kernel, new_kernel_expr = transform_weights(inputs, data, kernel, interleave_A=False)
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
    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    # Determine conv2d implementation
    target = Target.current(allow_none=False)
    _, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.conv2d"),
        attrs,
        [
            te.placeholder(data_tensor.shape, data_dtype),
            te.placeholder(kernel_tensor.shape, kernel_dtype),
        ],
        output_tensor,
        target,
    )
    workload = autotvm.task.get_workload(outs)
    if workload is not None:
        topi_tmpl = workload[0]

    # ARM vector instructions operate on the same dtype for data and kernel, we
    # provide those here and conv2d_alter_int8_common will convert to the
    # correct datatype.
    if is_int8_hw_support(kernel_dtype, kernel_dtype):
        # ARM intrinsics need the datatypes of data and kernel to be the same
        if (
            attrs["data_layout"] == "NHWC"
            and attrs["kernel_layout"] == "HWIO"
            and topi_tmpl == "conv2d_NHWC_quantized_native.arm_cpu"
        ):
            in_channel_vector_length = data_tensor.shape[3]
        else:
            in_channel_vector_length = 8

        return conv2d_alter_int8_common(
            data,
            data_tensor,
            kernel,
            kernel_tensor,
            output_tensor,
            attrs,
            kernel_dtype,
            in_channel_vector_length,
            8,
        )
    return None
