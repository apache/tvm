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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Conv2D alter op and legalize functions for cuda backend"""

import logging
import tvm
from tvm import te, relay, autotvm

from .. import nn
from ..utils import get_const_tuple
from .conv2d_winograd import _infer_tile_size
from .tensorcore_alter_op import pad_to_tensorcore
from ..nn import conv2d_legalize


logger = logging.getLogger("topi")


@nn.conv2d_alter_layout.register(["cuda", "gpu"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current

    new_attrs = {k: attrs[k] for k in attrs.keys()}
    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data, kernel = tinfos
    out_dtype = out_type.dtype

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
            N, H, W, CI = get_const_tuple(data.shape)
            KH, KW, _, CO = get_const_tuple(kernel.shape)

            # Pre-compute weight transformation in winograd
            tile_size = _infer_tile_size(tinfos[0], tinfos[1], layout="NHWC")

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
    if cfg.is_fallback:  # if is fallback, clear query cache and return None
        autotvm.task.clear_fallback_cache(target, workload)
        return None

    topi_tmpl = workload[0]
    if topi_tmpl == "conv2d_NCHWc_int8.cuda":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        new_layout = "NCHW4c"
        new_attrs["channels"] = CO
        new_attrs["data_layout"] = new_layout
        new_attrs["out_layout"] = new_layout
        new_attrs["kernel_layout"] = "OIHW4o4i"
        ic_block_factor = oc_block_factor = 4

        # Store the same config for the altered operator (workload)
        new_data = te.placeholder(
            (N, CI // ic_block_factor, H, W, ic_block_factor), dtype=data.dtype
        )
        new_kernel = te.placeholder(
            (
                CO // oc_block_factor,
                CI // ic_block_factor,
                KH,
                KW,
                oc_block_factor,
                ic_block_factor,
            ),
            dtype=kernel.dtype,
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_layout, out_dtype],
            "conv2d_NCHWc_int8.cuda",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_nchw_winograd.cuda":
        if dilation != (1, 1):
            logger.warning("Does not support weight pre-transform for dilated convolution.")
            return None

        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        # pre-compute weight transformation in winograd
        tile_size = _infer_tile_size(tinfos[0], tinfos[1])

        weight = relay.nn.contrib_conv2d_winograd_weight_transform(inputs[1], tile_size=tile_size)
        weight = relay.transpose(weight, axes=[0, 1, 3, 2])
        new_attrs["tile_size"] = tile_size
        new_attrs["channels"] = CO

        # Store the same config for the altered operator (workload)
        new_data = data
        new_weight = te.placeholder(
            (KH + tile_size - 1, KW + tile_size - 1, CI, CO), dtype=kernel.dtype
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_weight, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd_without_weight_transform.cuda",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight, **new_attrs
        )

    if topi_tmpl in ("conv2d_nhwc_winograd_direct.cuda", "conv2d_nhwc_winograd_tensorcore.cuda"):
        if dilation != (1, 1):
            logger.warning("Does not support weight pre-transform for dilated convolution.")
            return None

        assert data_layout == "NHWC" and kernel_layout == "HWIO"
        N, H, W, CI = get_const_tuple(data.shape)
        KH, KW, _, CO = get_const_tuple(kernel.shape)

        # Pre-compute weight transformation in winograd
        tile_size = _infer_tile_size(data, kernel, layout="NHWC")
        kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
        weight = relay.nn.contrib_conv2d_winograd_weight_transform(
            kernel_transform, tile_size=tile_size
        )
        weight = relay.transpose(weight, axes=[0, 1, 3, 2])
        new_attrs["tile_size"] = tile_size
        new_attrs["channels"] = CO
        # Store the same config for the altered operator (workload)
        new_data = data
        new_weight = te.placeholder(
            (KH + tile_size - 1, KW + tile_size - 1, CI, CO), dtype=kernel.dtype
        )
        if topi_tmpl == "conv2d_nhwc_winograd_direct.cuda":
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_weight, strides, padding, dilation, out_dtype],
                "conv2d_nhwc_winograd_direct_without_weight_transform.cuda",
            )
        elif topi_tmpl == "conv2d_nhwc_winograd_tensorcore.cuda":
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_weight, strides, padding, dilation, out_dtype],
                "conv2d_nhwc_winograd_tensorcore_without_weight_transform.cuda",
            )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight, **new_attrs
        )

    if topi_tmpl == "group_conv2d_NCHWc_int8.cuda":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        new_layout = "NCHW4c"
        new_attrs["channels"] = CO
        new_attrs["data_layout"] = new_layout
        new_attrs["out_layout"] = new_layout
        new_attrs["kernel_layout"] = "OIHW4o4i"
        ic_block_factor = oc_block_factor = 4

        # Store the same config for the altered operator (workload)
        new_data = te.placeholder(
            (N, CI // ic_block_factor, H, W, ic_block_factor), dtype=data.dtype
        )
        new_kernel = te.placeholder(
            (
                CO // oc_block_factor,
                CI // ic_block_factor // groups,
                KH,
                KW,
                oc_block_factor,
                ic_block_factor,
            ),
            dtype=kernel.dtype,
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, groups, out_dtype],
            "group_conv2d_NCHWc_int8.cuda",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    if topi_tmpl == "conv2d_HWNCnc_tensorcore.cuda":
        assert data_layout == "HWNC" and kernel_layout == "HWOI"
        assert float(tvm.cuda(0).compute_version) >= 7.5
        H, W, N, CI = get_const_tuple(data.shape)
        KH, KW, CO, _ = get_const_tuple(kernel.shape)

        if (
            kernel.dtype in ["int4", "uint4"]
            and (CI % 32 != 0 or CO % 8 != 0)
            or kernel.dtype in ["int8", "uint8"]
            and (CI % 16 != 0 or CO % 32 != 0)
        ):
            return relay.nn.conv2d(*inputs, **new_attrs)

        new_attrs["channels"] = CO
        if kernel.dtype in ["int4", "uint4"]:
            new_attrs["kernel_layout"] = "HWOI8o32i"
            ic_block_factor = 32
            oc_block_factor = 8
        else:
            new_attrs["kernel_layout"] = "HWOI32o16i"
            ic_block_factor = 16
            oc_block_factor = 32

        new_kernel = te.placeholder(
            (
                KH,
                KW,
                CO // oc_block_factor,
                CI // ic_block_factor,
                oc_block_factor,
                ic_block_factor,
            ),
            dtype=kernel.dtype,
        )

        new_workload = autotvm.task.args_to_workload(
            [data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_HWNCnc_tensorcore.cuda",
        )

        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)

    return None


def _pad_conv2d_HWNC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor):
    # Pad batch size
    if db != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, db), (0, 0)))

    # Pad input channel
    if di != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))

    # Pad output channel
    if do != 0:
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, do), (0, 0)))

    if do != 0:
        new_out_channel = out_channel + do
        new_attrs["channels"] = new_out_channel

    out = relay.nn.conv2d(data, kernel, **new_attrs)

    if db != 0 or do != 0:
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)

    return out


def _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor):
    # Pad batch size
    if db != 0:
        data = relay.nn.pad(data, pad_width=((0, db), (0, 0), (0, 0), (0, 0)))

    # Pad input channel
    if di != 0:
        data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, di)))
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, di), (0, 0)))

    # Pad output channel
    if do != 0:
        kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, do)))

    if do != 0:
        new_out_channel = out_channel + do
        new_attrs["channels"] = new_out_channel

    out = relay.nn.conv2d(data, kernel, **new_attrs)

    if db != 0 or do != 0:
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)

    return out


@conv2d_legalize.register("cuda")
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

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    # Get the conv attrs
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    # Get data layout. Return None if not NCHW
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]

    # Pad input and output channels to use int8 schedule.
    if data_dtype in ["int8", "uint8"]:
        if data_layout == "NCHW" and kernel_layout == "OIHW":
            oc_modified = False
            in_channel = data_tensor.shape[1].value
            out_channel = kernel_tensor.shape[0].value

            # Pad input channel
            if in_channel % 4 != 0:
                new_in_channel = ((in_channel + 4) // 4) * 4
                diff = new_in_channel - in_channel
                pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
                data = relay.nn.pad(data, pad_width=pad_width)
                kernel = relay.nn.pad(kernel, pad_width=pad_width)

            # Pad output channel
            new_out_channel = out_channel
            if out_channel % 4 != 0:
                new_out_channel = ((out_channel + 4) // 4) * 4
                diff = new_out_channel - out_channel
                kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
                oc_modified = True

            if oc_modified:
                new_attrs["channels"] = new_out_channel
                out = tvm.relay.nn.conv2d(data, kernel, **new_attrs)
                original_out_shape = [x.value for x in output_tensor.shape]
                out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
            else:
                out = relay.nn.conv2d(data, kernel, **new_attrs)
            return out

        if data_layout == "NHWC" and kernel_layout == "HWIO":
            batch = data_tensor.shape[0].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value

            if (
                (batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0)
                or (batch % 16 == 0 and in_channel % 16 == 0 and out_channel % 16 == 0)
                or (batch % 32 == 0 and in_channel % 16 == 0 and out_channel % 8 == 0)
            ):
                # no need to pad
                return None

            candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None

            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

        if data_layout == "HWNC" and kernel_layout == "HWOI":
            batch = data_tensor.shape[2].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[2].value

            if batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0:
                return None

            candidates = [(8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None
            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_HWNC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

    elif data_dtype in ["float16"]:
        if data_layout == "NHWC" and kernel_layout == "HWIO":
            batch = data_tensor.shape[0].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value

            if (
                (batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0)
                or (batch % 16 == 0 and in_channel % 16 == 0 and out_channel % 16 == 0)
                or (batch % 32 == 0 and in_channel % 16 == 0 and out_channel % 8 == 0)
            ):
                # no need to pad
                return None

            candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None

            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

    elif data_dtype in ["int4", "uint4"]:
        if data_layout == "NHWC" and kernel_layout == "HWIO":
            batch = data_tensor.shape[0].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value

            if (
                (batch % 8 == 0 and in_channel % 16 == 0 and out_channel % 32 == 0)
                or (batch % 16 == 0 and in_channel % 16 == 0 and out_channel % 16 == 0)
                or (batch % 32 == 0 and in_channel % 16 == 0 and out_channel % 8 == 0)
            ):
                # no need to pad
                return None

            candidates = [(16, 16, 16), (32, 16, 8), (8, 16, 32)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None

            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_NHWC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

        if data_layout == "HWNC" and kernel_layout == "HWOI":
            batch = data_tensor.shape[2].value
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[2].value

            if batch % 8 == 0 and in_channel % 32 == 0 and out_channel % 8 == 0:
                return None

            candidates = [(8, 32, 8)]
            (db, di, do), extra_flops = pad_to_tensorcore(
                batch, in_channel, out_channel, candidates
            )

            if extra_flops > 2:
                logger.info("conv2d pad_to_tensorcore skipped, extra_flops %s", extra_flops)
                return None
            logger.info("conv2d pad_to_tensorcore, extra_flops %s", extra_flops)

            return _pad_conv2d_HWNC(db, di, do, data, kernel, out_channel, new_attrs, output_tensor)

    return None
