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
# pylint: disable=no-else-return, invalid-name, unused-argument, too-many-arguments, consider-using-in
"""Backend compiler related feature registration"""
from __future__ import absolute_import
import re

from tvm import relay, topi
from tvm.runtime import convert
from tvm.te.hybrid import script
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple

from ....ir import container
from ....tir import expr
from ...transform import LayoutConfig
from .. import op as reg
from .. import strategy
from .._tensor import elemwise_shape_func
from ..op import OpPattern
from ..strategy.generic import is_depthwise_conv2d

# relu
reg.register_broadcast_schedule("nn.relu")
reg.register_pattern("nn.relu", OpPattern.ELEMWISE)


# softmax
reg.register_strategy("nn.softmax", strategy.softmax_strategy)
reg.register_pattern("nn.softmax", OpPattern.OUT_ELEMWISE_FUSABLE)


# fast softmax
reg.register_strategy("nn.fast_softmax", strategy.fast_softmax_strategy)
reg.register_pattern("nn.fast_softmax", OpPattern.OUT_ELEMWISE_FUSABLE)


# log_softmax
reg.register_strategy("nn.log_softmax", strategy.log_softmax_strategy)
reg.register_pattern("nn.log_softmax", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_legalize("nn.matmul")
def legalize_matmul(attrs, inputs, types):
    """Legalize matmul op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current matmul
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    return topi.nn.matmul_legalize(attrs, inputs, types)


# matmul
reg.register_strategy("nn.matmul", strategy.matmul_strategy)
reg.register_pattern("nn.matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_legalize("nn.dense")
def legalize_dense(attrs, inputs, types):
    """Legalize dense op.

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
    return topi.nn.dense_legalize(attrs, inputs, types)


# dense
reg.register_strategy("nn.dense", strategy.dense_strategy)
reg.register_pattern("nn.dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_alter_op_layout("nn.dense")
def alter_op_layout_dense(attrs, inputs, tinfos, out_type):
    """Alternate the layout of dense"""
    return topi.nn.dense_alter_layout(attrs, inputs, tinfos, out_type)


# dense_pack
reg.register_strategy("nn.contrib_dense_pack", strategy.dense_pack_strategy)
reg.register_pattern("nn.contrib_dense_pack", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# fifo_buffer
@reg.register_compute("nn.fifo_buffer")
def compute_fifo_buffer(attrs, inputs, out_type):
    return [topi.nn.fifo_buffer(inputs[0], inputs[1], axis=attrs.get_int("axis"))]


reg.register_injective_schedule("nn.fifo_buffer")
reg.register_pattern("nn.fifo_buffer", OpPattern.OPAQUE)


@reg.register_legalize("nn.batch_matmul")
def legalize_batch_matmul(attrs, inputs, types):
    """Legalize batch_matmul op.

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
    return topi.nn.batch_matmul_legalize(attrs, inputs, types)


# batch_matmul
reg.register_strategy("nn.batch_matmul", strategy.batch_matmul_strategy)
reg.register_pattern("nn.batch_matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# batch_norm
reg.register_strategy("nn.batch_norm", strategy.batch_norm_strategy)
reg.register_pattern("nn.batch_norm", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# sparse_dense
@reg.register_compute("nn.sparse_dense")
def compute_sparse_dense(attrs, inputs, out_type):
    """Compute definition of sparse_dense"""
    return [topi.nn.sparse_dense(inputs[0], inputs[1], inputs[2], inputs[3], attrs["sparse_lhs"])]


reg.register_strategy("nn.sparse_dense", strategy.sparse_dense_strategy)
reg.register_pattern("nn.sparse_dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_alter_op_layout("nn.sparse_dense")
def alter_op_layout_sparse_dense(attrs, inputs, tinfos, out_type):
    """Alternate the layout of sparse_dense"""
    return topi.nn.sparse_dense_alter_layout(attrs, inputs, tinfos, out_type)


# sparse_add
reg.register_strategy("nn.sparse_add", strategy.sparse_add_strategy)
reg.register_pattern("nn.sparse_add", reg.OpPattern.OPAQUE)


@reg.register_compute("nn.internal.sparse_dense_padded")
def compute_sparse_dense_padded(attrs, inputs, out_type):
    """Compute definition of sparse_dense_padded"""
    raise NotImplementedError("nn.internal.sparse_dense_padded is only available on cuda")


reg.register_strategy("nn.internal.sparse_dense_padded", strategy.sparse_dense_padded_strategy)
reg.register_pattern("nn.internal.sparse_dense_padded", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# sparse_transpose
@reg.register_compute("nn.sparse_transpose")
def compute_sparse_transpose(attrs, inputs, out_type):
    """Compute definition of sparse_transpose"""
    return topi.nn.sparse_transpose(inputs[0], inputs[1], inputs[2])


reg.register_schedule("nn.sparse_transpose", strategy.schedule_sparse_transpose)
reg.register_pattern("nn.sparse_transpose", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# sparse_conv2d
@reg.register_compute("nn.sparse_conv2d")
def compute_sparse_conv2d(attrs, inputs, out_type):
    """Compute definition of sparse_conv2d"""
    return [
        topi.nn.sparse_conv2d(
            inputs[0], inputs[1], inputs[2], inputs[3], attrs["layout"], attrs["kernel_size"]
        )
    ]


reg.register_strategy("nn.sparse_conv2d", strategy.sparse_conv2d_strategy)
reg.register_pattern("nn.sparse_conv2d", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# conv1d
reg.register_strategy("nn.conv1d", strategy.conv1d_strategy)
reg.register_pattern("nn.conv1d", OpPattern.OUT_ELEMWISE_FUSABLE)


# conv2d
reg.register_strategy("nn.conv2d", strategy.conv2d_strategy)
reg.register_pattern("nn.conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_alter_op_layout("nn.conv2d")
def alter_op_layout_conv2d(attrs, inputs, tinfos, out_type):
    """Alternate the layout of conv2d"""
    return topi.nn.conv2d_alter_layout(attrs, inputs, tinfos, out_type)


@reg.register_legalize("nn.conv2d")
def legalize_conv2d(attrs, inputs, types):
    """Legalize conv2d op.

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
    return topi.nn.conv2d_legalize(attrs, inputs, types)


@reg.register_convert_op_layout("nn.conv2d")
def convert_conv2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for conv2d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    data, weight = inputs

    # First check if there is a LayoutConfig scope, and if so, whether
    # it indicates we should ignore this layer or not.
    layout_config = LayoutConfig.current
    if layout_config is not None:
        skip_layer = layout_config.check_skip()
        if skip_layer:
            return relay.nn.conv2d(data, weight, **attrs)

    # Prepare new layout.
    new_attrs = dict(attrs)
    assert len(desired_layouts) == 2, "A desired layout is expected for both of nn.conv2d's inputs"
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"
    new_attrs["data_layout"] = desired_data_layout
    need_tile = re.match(r"NCHW(\d*)c", desired_data_layout)

    if desired_kernel_layout != "default" and not need_tile:
        new_attrs["kernel_layout"] = desired_kernel_layout
        return relay.nn.conv2d(data, weight, **new_attrs)

    # Handle default kernel layouts
    if desired_data_layout == "NCHW":
        new_attrs["kernel_layout"] = "OIHW"
        return relay.nn.conv2d(data, weight, **new_attrs)
    elif desired_data_layout == "NHWC":
        # Check for depthwise convolution.
        data_info, weight_info = tinfos
        if is_depthwise_conv2d(
            data_info.shape,
            attrs["data_layout"],
            weight_info.shape,
            attrs["kernel_layout"],
            attrs["groups"],
        ):
            new_attrs["kernel_layout"] = "HWOI"
        else:
            new_attrs["kernel_layout"] = "HWIO"
        return relay.nn.conv2d(data, weight, **new_attrs)
    elif desired_data_layout == "HWNC":
        new_attrs["kernel_layout"] = "HWOI"
        return relay.nn.conv2d(data, weight, **new_attrs)
    elif need_tile:
        assert desired_kernel_layout != "default", "Kernel layout cannot be default."
        tile = int(need_tile.group(1))
        if isinstance(data, relay.expr.Var) and data.checked_type.shape[1] % tile != 0:
            return relay.nn.conv2d(data, weight, **attrs)
        else:
            new_attrs["kernel_layout"] = desired_kernel_layout
            return relay.nn.contrib_conv2d_nchwc(data, weight, **new_attrs)

    raise ValueError("Layout %s is not yet supported." % desired_data_layout)


# conv2d_transpose
reg.register_strategy("nn.conv2d_transpose", strategy.conv2d_transpose_strategy)
reg.register_pattern("nn.conv2d_transpose", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_legalize("nn.conv2d_transpose")
def legalize_conv2d_transpose(attrs, inputs, types):
    """Legalize conv2d_transpose op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current Transposed convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    return topi.nn.conv2d_transpose_legalize(attrs, inputs, types)


@reg.register_convert_op_layout("nn.conv2d_transpose")
def convert_conv2d_transpose(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for conv2d_transpose op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    data, weight = inputs
    new_attrs = dict(attrs)
    assert len(desired_layouts) == 2, "A desired layout is expected for both of nn.conv2d's inputs"
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"
    new_attrs["data_layout"] = desired_data_layout

    if desired_kernel_layout != "default":
        new_attrs["kernel_layout"] = desired_kernel_layout
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)

    # Handle default kernel layouts
    if desired_data_layout == "NCHW":
        new_attrs["kernel_layout"] = "IOHW"
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)
    elif desired_data_layout == "NHWC":
        new_attrs["kernel_layout"] = "HWIO"
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)

    raise ValueError("Layout %s is not yet supported." % desired_data_layout)


# conv3d_transpose
reg.register_strategy("nn.conv3d_transpose", strategy.conv3d_transpose_strategy)
reg.register_pattern("nn.conv3d_transpose", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_legalize("nn.conv3d_transpose")
def legalize_conv3d_transpose(attrs, inputs, types):
    """Legalize conv3d_transpose op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current Transposed convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    return topi.nn.conv3d_transpose_legalize(attrs, inputs, types)


# conv3d
reg.register_strategy("nn.conv3d", strategy.conv3d_strategy)
reg.register_pattern("nn.conv3d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_alter_op_layout("nn.conv3d")
def alter_op_layout_conv3d(attrs, inputs, tinfos, out_type):
    """Alternate the layout of conv3d"""
    return topi.nn.conv3d_alter_layout(attrs, inputs, tinfos, out_type)


@reg.register_convert_op_layout("nn.conv3d")
def convert_conv3d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for conv3d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    data, weight = inputs
    new_attrs = dict(attrs)
    assert len(desired_layouts) == 2, "A desired layout is expected for both of nn.conv3d's inputs"
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"
    new_attrs["data_layout"] = desired_data_layout

    if desired_kernel_layout != "default":
        new_attrs["kernel_layout"] = desired_kernel_layout
        return relay.nn.conv3d(data, weight, **new_attrs)

    # Handle default kernel layouts
    if desired_data_layout == "NCDHW":
        new_attrs["kernel_layout"] = "OIDHW"
        return relay.nn.conv3d(data, weight, **new_attrs)
    elif desired_data_layout == "NDHWC":
        new_attrs["kernel_layout"] = "DHWIO"
        return relay.nn.conv3d(data, weight, **new_attrs)

    raise ValueError("Layout %s is not yet supported" % desired_data_layout)


# conv3d_winograd related operators
reg.register_strategy(
    "nn.contrib_conv3d_winograd_without_weight_transform",
    strategy.conv3d_winograd_without_weight_transfrom_strategy,
)
reg.register_pattern(
    "nn.contrib_conv3d_winograd_without_weight_transform", OpPattern.OUT_ELEMWISE_FUSABLE
)


@reg.register_compute("nn.contrib_conv3d_winograd_weight_transform")
def compute_contrib_conv3d_winograd_weight_transform(attrs, inputs, out_dtype):
    """Compute definition of contrib_conv3d_winograd_weight_transform"""
    out = topi.nn.conv3d_winograd_weight_transform(inputs[0], attrs.get_int("tile_size"))
    return [out]


reg.register_schedule(
    "nn.contrib_conv3d_winograd_weight_transform",
    strategy.schedule_conv3d_winograd_weight_transform,
)
reg.register_pattern("nn.contrib_conv3d_winograd_weight_transform", OpPattern.OUT_ELEMWISE_FUSABLE)


# conv1d_transpose
reg.register_strategy("nn.conv1d_transpose", strategy.conv1d_transpose_strategy)
reg.register_pattern("nn.conv1d_transpose", OpPattern.OUT_ELEMWISE_FUSABLE)


# bias_add
reg.register_injective_schedule("nn.bias_add")
reg.register_pattern("nn.bias_add", OpPattern.BROADCAST)


# max_pool1d
reg.register_schedule("nn.max_pool1d", strategy.schedule_pool)
reg.register_pattern("nn.max_pool1d", OpPattern.OUT_ELEMWISE_FUSABLE)


# max_pool2d
reg.register_schedule("nn.max_pool2d", strategy.schedule_pool)
reg.register_pattern("nn.max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_convert_op_layout("nn.max_pool2d")
def convert_max_pool2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for max_pool2d op.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current pooling
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of one layout string
        layout string defining our desired layout for input and output.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    new_attrs = dict(attrs)
    new_attrs["layout"] = str(desired_layouts[0])
    new_attrs["out_layout"] = str(desired_layouts[0])
    return relay.nn.max_pool2d(*inputs, **new_attrs)


# max_pool3d
reg.register_schedule("nn.max_pool3d", strategy.schedule_pool)
reg.register_pattern("nn.max_pool3d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool1d
reg.register_schedule("nn.avg_pool1d", strategy.schedule_pool)
reg.register_pattern("nn.avg_pool1d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool2d
reg.register_schedule("nn.avg_pool2d", strategy.schedule_pool)
reg.register_pattern("nn.avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_convert_op_layout("nn.avg_pool2d")
def convert_avg_pool2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for avg_pool2d op.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current pooling
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of one layout string
        layout string defining our desired layout for input and output.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    new_attrs = dict(attrs)
    new_attrs["layout"] = str(desired_layouts[0])
    new_attrs["out_layout"] = str(desired_layouts[0])
    return relay.nn.avg_pool2d(*inputs, **new_attrs)


# avg_pool3d
reg.register_schedule("nn.avg_pool3d", strategy.schedule_pool)
reg.register_pattern("nn.avg_pool3d", OpPattern.OUT_ELEMWISE_FUSABLE)


# max_pool2d_grad
reg.register_schedule("nn.max_pool2d_grad", strategy.schedule_pool_grad)
reg.register_pattern("nn.max_pool2d_grad", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool2d_grad
reg.register_schedule("nn.avg_pool2d_grad", strategy.schedule_pool_grad)
reg.register_pattern("nn.avg_pool2d_grad", OpPattern.OUT_ELEMWISE_FUSABLE)


# adaptive_max_pool1d
reg.register_schedule("nn.adaptive_max_pool1d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.adaptive_max_pool1d", OpPattern.OUT_ELEMWISE_FUSABLE)


# adaptive_avg_pool1d
reg.register_schedule("nn.adaptive_avg_pool1d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.adaptive_avg_pool1d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_max_pool2d
reg.register_schedule("nn.global_max_pool2d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.global_max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_convert_op_layout("nn.global_max_pool2d")
def convert_global_max_pool2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for global_max_pool2d op.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current pooling
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of one layout string
        layout string defining our desired layout for input and output.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    new_attrs = dict(attrs)
    new_attrs["layout"] = str(desired_layouts[0])
    new_attrs["out_layout"] = str(desired_layouts[0])
    return relay.nn.global_max_pool2d(*inputs, **new_attrs)


# global_avg_pool2d
reg.register_schedule("nn.global_avg_pool2d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.global_avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_convert_op_layout("nn.global_avg_pool2d")
def convert_global_avg_pool2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for global_avg_pool2d op.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current pooling
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of one layout string
        layout string defining our desired layout for input and output.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    new_attrs = dict(attrs)
    new_attrs["layout"] = str(desired_layouts[0])
    new_attrs["out_layout"] = str(desired_layouts[0])
    return relay.nn.global_avg_pool2d(*inputs, **new_attrs)


# adaptive_max_pool2d
reg.register_schedule("nn.adaptive_max_pool2d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.adaptive_max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# adaptive_avg_pool2d
reg.register_schedule("nn.adaptive_avg_pool2d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.adaptive_avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# adaptive_max_pool3d
reg.register_schedule("nn.adaptive_max_pool3d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.adaptive_max_pool3d", OpPattern.OUT_ELEMWISE_FUSABLE)


# adaptive_avg_pool3d
reg.register_schedule("nn.adaptive_avg_pool3d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.adaptive_avg_pool3d", OpPattern.OUT_ELEMWISE_FUSABLE)


# leaky_relu
reg.register_broadcast_schedule("nn.leaky_relu")
reg.register_pattern("nn.leaky_relu", OpPattern.ELEMWISE)


# prelu
reg.register_broadcast_schedule("nn.prelu")
reg.register_pattern("nn.prelu", OpPattern.BROADCAST)


# flatten
reg.register_broadcast_schedule("nn.batch_flatten")
reg.register_pattern("nn.batch_flatten", OpPattern.INJECTIVE)


# lrn
@reg.register_compute("nn.lrn")
def compute_lrn(attrs, inputs, out_dtype):
    """Compute definition of lrn"""
    assert len(inputs) == 1
    return [topi.nn.lrn(inputs[0], attrs.size, attrs.axis, attrs.alpha, attrs.beta, attrs.bias)]


reg.register_schedule("nn.lrn", strategy.schedule_lrn)
reg.register_pattern("nn.lrn", OpPattern.OPAQUE)


# upsampling
@reg.register_compute("nn.upsampling")
def compute_upsampling(attrs, inputs, out_dtype):
    scale_h = attrs.scale_h
    scale_w = attrs.scale_w
    layout = attrs.layout
    method = attrs.method
    align_corners = attrs.align_corners
    return [topi.nn.upsampling(inputs[0], scale_h, scale_w, layout, method, align_corners)]


reg.register_injective_schedule("nn.upsampling")


# upsampling3d
@reg.register_compute("nn.upsampling3d")
def compute_upsampling3d(attrs, inputs, out_dtype):
    scale_d = attrs.scale_d
    scale_h = attrs.scale_h
    scale_w = attrs.scale_w
    layout = attrs.layout
    method = attrs.method
    coordinate_transformation_mode = attrs.coordinate_transformation_mode
    return [
        topi.nn.upsampling3d(
            inputs[0], scale_d, scale_h, scale_w, layout, method, coordinate_transformation_mode
        )
    ]


reg.register_injective_schedule("nn.upsampling3d")


# pad
reg.register_broadcast_schedule("nn.pad")


# mirror_pad
@reg.register_compute("nn.mirror_pad")
def compute_mirror_pad(attrs, inputs, out_dtype):
    pad_before, pad_after = list(zip(*attrs.pad_width))
    mode = attrs.mode
    out = topi.nn.mirror_pad(inputs[0], pad_before=pad_before, pad_after=pad_after, mode=mode)
    return [out]


reg.register_broadcast_schedule("nn.mirror_pad")


@script
def _mirror_pad_func(data_shape, pad_width):
    out = output_tensor((data_shape.shape[0],), "int64")
    for i in const_range(data_shape.shape[0]):
        out[i] = data_shape[i] + int64(pad_width[i][0]) + int64(pad_width[i][1])
    return out


@reg.register_shape_func("nn.mirror_pad", False)
def mirror_pad_func(attrs, inputs, _):
    pad_width_tuple = [get_const_tuple(p) for p in attrs.pad_width]
    return [_mirror_pad_func(inputs[0], convert(pad_width_tuple))]


# conv2d_winograd related operators
reg.register_strategy(
    "nn.contrib_conv2d_winograd_without_weight_transform",
    strategy.conv2d_winograd_without_weight_transfrom_strategy,
)
reg.register_pattern(
    "nn.contrib_conv2d_winograd_without_weight_transform", OpPattern.OUT_ELEMWISE_FUSABLE
)

# conv2d_gemm related operators
reg.register_strategy(
    "nn.contrib_conv2d_gemm_without_weight_transform",
    strategy.conv2d_gemm_without_weight_transform_strategy,
)
reg.register_pattern(
    "nn.contrib_conv2d_gemm_without_weight_transform", OpPattern.OUT_ELEMWISE_FUSABLE
)


@reg.register_compute("nn.contrib_conv2d_gemm_weight_transform")
def compute_contrib_conv2d_gemm_weight_transform(attrs, inputs, out_dtype):
    """Compute definition of contrib_conv2d_gemm_weight_transform"""
    out = topi.nn.conv2d_gemm_weight_transform(inputs[0], attrs.tile_rows, attrs.tile_cols)
    return [out]


reg.register_schedule(
    "nn.contrib_conv2d_gemm_weight_transform", strategy.schedule_conv2d_gemm_weight_transform
)
reg.register_pattern("nn.contrib_conv2d_gemm_weight_transform", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_compute("nn.contrib_conv2d_winograd_weight_transform")
def compute_contrib_conv2d_winograd_weight_transform(attrs, inputs, out_dtype):
    """Compute definition of contrib_conv2d_winograd_weight_transform"""
    out = topi.nn.conv2d_winograd_weight_transform(inputs[0], attrs.get_int("tile_size"))
    return [out]


reg.register_schedule(
    "nn.contrib_conv2d_winograd_weight_transform",
    strategy.schedule_conv2d_winograd_weight_transform,
)
reg.register_pattern("nn.contrib_conv2d_winograd_weight_transform", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_compute("nn.contrib_conv2d_winograd_nnpack_weight_transform")
def compute_contrib_conv2d_winograd_nnpack_weight_transform(attrs, inputs, out_dtype):
    """Compute definition of contrib_conv2d_winograd_nnpack_weight_transform"""
    convolution_algorithm = attrs.get_int("convolution_algorithm")
    out = topi.nn.conv2d_winograd_nnpack_weight_transform(
        inputs[0], convolution_algorithm, out_dtype
    )
    return [out]


reg.register_schedule(
    "nn.contrib_conv2d_winograd_nnpack_weight_transform",
    strategy.schedule_conv2d_winograd_nnpack_weight_transform,
)
reg.register_pattern("nn.contrib_conv2d_winograd_nnpack_weight_transform", OpPattern.OPAQUE)


# conv2d_NCHWc
reg.register_strategy("nn.contrib_conv2d_NCHWc", strategy.conv2d_NCHWc_strategy)
reg.register_pattern("nn.contrib_conv2d_NCHWc", OpPattern.OUT_ELEMWISE_FUSABLE)

# depthwise_conv2d_NCHWc
reg.register_strategy("nn.contrib_depthwise_conv2d_NCHWc", strategy.depthwise_conv2d_NCHWc_strategy)
reg.register_pattern("nn.contrib_depthwise_conv2d_NCHWc", OpPattern.OUT_ELEMWISE_FUSABLE)


# deformable_conv2d
reg.register_strategy("nn.deformable_conv2d", strategy.deformable_conv2d_strategy)
reg.register_pattern("nn.deformable_conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_alter_op_layout("nn.deformable_conv2d")
def alter_op_layout_deformable_conv2d(attrs, inputs, tinfos, out_type):
    """Alternate the layout of deformable conv2d"""
    return None


@reg.register_legalize("nn.deformable_conv2d")
def legalize_deformable_conv2d(attrs, inputs, types):
    """Legalize deformable conv2d op.
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
    return None


@reg.register_convert_op_layout("nn.deformable_conv2d")
def convert_deformable_conv2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for deformable conv2d op.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    data, offset, weight = inputs
    new_attrs = dict(attrs)
    for attr in new_attrs:
        if isinstance(new_attrs[attr], container.Array):
            new_attrs[attr] = list(new_attrs[attr])
        elif isinstance(new_attrs[attr], expr.IntImm):
            new_attrs[attr] = new_attrs[attr].value

    # First check if there is a LayoutConfig scope, and if so, whether
    # it indicates we should ignore this layer or not.
    layout_config = LayoutConfig.current
    if layout_config is not None:
        skip_layer = layout_config.check_skip()
        if skip_layer:
            return relay.nn.deformable_conv2d(data, offset, weight, **new_attrs)

    # Prepare new layout.
    assert len(desired_layouts) == 2, "A desired layout is expected for data and kernel"
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"
    new_attrs["data_layout"] = desired_data_layout

    if desired_kernel_layout != "default":
        new_attrs["kernel_layout"] = desired_kernel_layout
        return relay.nn.deformable_conv2d(data, offset, weight, **new_attrs)

    # Handle default kernel layouts
    if desired_data_layout == "NCHW":
        new_attrs["kernel_layout"] = "OIHW"
    elif desired_data_layout == "NHWC":
        new_attrs["kernel_layout"] = "HWIO"
    else:
        raise ValueError("Layout %s is not yet supported." % desired_data_layout)

    return relay.nn.deformable_conv2d(data, offset, weight, **new_attrs)


# bitpack
@reg.register_compute("nn.bitpack")
def compute_bitpack(attrs, inputs, out_dtype):
    """Compute definition for bitpack"""
    bits = attrs.bits
    pack_axis = attrs.pack_axis
    bit_axis = attrs.bit_axis
    pack_type = attrs.pack_type
    name = attrs.name
    out = topi.nn.bitpack(inputs[0], bits, pack_axis, bit_axis, pack_type, name)
    return [out]


reg.register_schedule("nn.bitpack", strategy.schedule_bitpack)
reg.register_pattern("nn.bitpack", OpPattern.INJECTIVE)


# bitserial_conv2d
reg.register_strategy("nn.bitserial_conv2d", strategy.bitserial_conv2d_strategy)
reg.register_pattern("nn.bitserial_conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_legalize("nn.bitserial_conv2d")
def legalize_bitserial_conv2d(attrs, inputs, types):
    """Legalize bitserial_conv2d op.

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
    return topi.nn.bitserial_conv2d_legalize(attrs, inputs, types)


# bitserial_dense
reg.register_strategy("nn.bitserial_dense", strategy.bitserial_dense_strategy)
reg.register_pattern("nn.bitserial_dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# cross_entropy
@reg.register_compute("nn.cross_entropy")
def compute_cross_entropy(attrs, inputs, out_dtype):
    x, y = inputs
    return [-topi.sum(topi.log(x) * y) / x.shape[0]]


reg.register_reduce_schedule("nn.cross_entropy")
reg.register_pattern("nn.cross_entropy", OpPattern.OPAQUE)


# dilate
@reg.register_compute("nn.dilate")
def compute_dilate(attrs, inputs, out_dtype):
    return [topi.nn.dilate(inputs[0], attrs.strides, attrs.dilation_value)]


reg.register_broadcast_schedule("nn.dilate")
reg.register_pattern("nn.dilate", OpPattern.INJECTIVE)


# cross_entropy_with_logits
@reg.register_compute("nn.cross_entropy_with_logits")
def compute_cross_entropy_with_logits(attrs, inputs, out_dtype):
    x, y = inputs
    return [-topi.sum(x * y) / x.shape[0]]


reg.register_reduce_schedule("nn.cross_entropy_with_logits")
reg.register_pattern("nn.cross_entropy_with_logits", OpPattern.OPAQUE)


# nll_loss
@reg.register_compute("nn.nll_loss")
def compute_nll_loss(attrs, inputs, out_dtype):
    predictions, targets, weights = inputs
    return [topi.nn.nll_loss(predictions, targets, weights, attrs.reduction, attrs.ignore_index)]


reg.register_reduce_schedule("nn.nll_loss")
reg.register_pattern("nn.nll_loss", OpPattern.OUT_ELEMWISE_FUSABLE)


# depth_to_space
@reg.register_compute("nn.depth_to_space")
def compute_depth_to_space(attrs, inputs, out_dtype):
    block_size = attrs.block_size
    layout = attrs.layout
    mode = attrs.mode
    return [topi.nn.depth_to_space(inputs[0], block_size, layout=layout, mode=mode)]


reg.register_injective_schedule("nn.depth_to_space")
reg.register_pattern("nn.depth_to_space", OpPattern.INJECTIVE)


# space_to_depth
@reg.register_compute("nn.space_to_depth")
def compute_space_to_depth(attrs, inputs, out_dtype):
    block_size = attrs.block_size
    layout = attrs.layout
    return [topi.nn.space_to_depth(inputs[0], block_size, layout=layout)]


reg.register_injective_schedule("nn.space_to_depth")
reg.register_pattern("nn.space_to_depth", OpPattern.INJECTIVE)


# correlation
reg.register_strategy("nn.correlation", strategy.correlation_strategy)
reg.register_pattern("nn.correlation", OpPattern.OUT_ELEMWISE_FUSABLE)


# space_to_batch_nd and batch_to_space_nd
reg.register_injective_schedule("nn.space_to_batch_nd")
reg.register_injective_schedule("nn.batch_to_space_nd")


reg.register_strategy("nn.conv2d_backward_weight", strategy.conv2d_backward_weight_strategy)
reg.register_pattern("nn.conv2d_backward_weight", OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_legalize("nn.conv2d_backward_weight")
def legalize_conv2d_backward_weight(attrs, inputs, types):
    """Legalize conv2d_backward_weight op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current op
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    grad, data = inputs
    data_shape = get_const_tuple(data.checked_type.shape)
    weight_shape = get_const_tuple(types[2].shape)
    _, out_channel, grad_h, grad_w = get_const_tuple(grad.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    _, _, filter_h, filter_w = weight_shape
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)

    grad = relay.tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = relay.reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = relay.reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = relay.nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
        out_dtype=attrs.out_dtype,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1

    backward_weight = relay.reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            out_channel,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )
    backward_weight = relay.sum(backward_weight, axis=0)
    backward_weight = relay.transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w

    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = relay.strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    return backward_weight


@reg.register_convert_op_layout("nn.conv2d_backward_weight")
def convert_conv2d_backward_weight(attrs, inputs, _, desired_layouts):
    """Convert Layout pass registration for conv2d_backward_weight op.
    Note that `desired_layouts` must be a pair [`data_layout`, `kernel_layouts`],
    where `kernel_layouts` affects the output of this op (since the output of this op
    is the weight gradient). The layout of the output gradient (the second input to this op)
    is assumed to be the same as `data_layout`.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current op
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    new_attrs = dict(attrs)
    assert len(desired_layouts) == 2, "A desired layout is expected for both of data and gradient."
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"
    new_attrs["grad_layout"] = desired_data_layout
    new_attrs["data_layout"] = desired_data_layout
    new_attrs["kernel_layout"] = desired_kernel_layout
    new_attrs.pop("out_layout")
    return relay.nn.conv2d_backward_weight(inputs[0], inputs[1], **new_attrs)


#####################
#  Shape functions  #
#####################


@script
def _conv_shape_func_nchw(dshape, kshape, strides, padding, dilation):
    """Shape function for conv*d op with nchw & oihw layout."""
    out = output_tensor((dshape.shape[0],), "int64")
    out[0] = dshape[0]
    out[1] = kshape[0]

    for i in const_range(dshape.shape[0] - 2):
        dilated_k = (kshape[i + 2] - 1) * dilation[i] + 1
        out[i + 2] = (dshape[i + 2] + 2 * padding[i] - dilated_k) // strides[i] + 1
    return out


@script
def _conv_shape_func_nhwc_hwio(dshape, kshape, strides, padding, dilation):
    """Shape function for conv*d op with nhwc & hwio layout."""
    out = output_tensor((dshape.shape[0],), "int64")
    out[0] = dshape[0]
    out[dshape.shape[0] - 1] = kshape[kshape.shape[0] - 1]

    for i in const_range(dshape.shape[0] - 2):
        dilated_k = (kshape[i] - 1) * dilation[i] + 1
        out[i + 1] = (dshape[i + 1] + 2 * padding[i] - dilated_k) // strides[i] + 1
    return out


@script
def _conv_shape_func_nhwc_hwoi(dshape, kshape, strides, padding, dilation):
    """Shape function for conv*d op with nhwc & hwoi layout."""
    out = output_tensor((dshape.shape[0],), "int64")
    out[0] = dshape[0]
    out[dshape.shape[0] - 1] = kshape[kshape.shape[0] - 2]

    for i in const_range(dshape.shape[0] - 2):
        dilated_k = (kshape[i] - 1) * dilation[i] + 1
        out[i + 1] = (dshape[i + 1] + 2 * padding[i] - dilated_k) // strides[i] + 1
    return out


@script
def _conv_shape_func_nhwc_ohwi(dshape, kshape, strides, padding, dilation):
    """Shape function for conv*d op with nhwc & ohwi layout."""
    out = output_tensor((dshape.shape[0],), "int64")
    out[0] = dshape[0]
    out[dshape.shape[0] - 1] = kshape[0]

    for i in const_range(dshape.shape[0] - 2):
        dilated_k = (kshape[i + 1] - 1) * dilation[i] + 1
        out[i + 1] = (dshape[i + 1] + 2 * padding[i] - dilated_k) // strides[i] + 1
    return out


def conv_shape_func(attrs, inputs, _):
    """Shape function for conv*d op."""
    strides = get_const_tuple(attrs.strides)
    padding = get_const_tuple(attrs.padding)
    dilation = get_const_tuple(attrs.dilation)

    shape_func = None
    if attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
        shape_func = _conv_shape_func_nchw
    elif attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
        shape_func = _conv_shape_func_nhwc_hwio
    elif attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWOI":
        shape_func = _conv_shape_func_nhwc_hwoi
    elif attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "OHWI":
        shape_func = _conv_shape_func_nhwc_ohwi
    else:
        raise ValueError(
            "Unsupported data/kernel layout: %s, %s"
            % (attrs["data_layout"], attrs["kernel_layout"])
        )

    return [shape_func(inputs[0], inputs[1], convert(strides), convert(padding), convert(dilation))]


reg.register_shape_func("nn.conv1d", False, conv_shape_func)
reg.register_shape_func("nn.conv2d", False, conv_shape_func)
reg.register_shape_func("nn.conv3d", False, conv_shape_func)


@script
def _conv2d_NCHWc_shape_func(dshape, kshape, strides, padding, dilation, oc_bn):
    out = output_tensor((dshape.shape[0],), "int64")
    ic_chunk = dshape[1]
    height = dshape[2]
    width = dshape[3]
    ic_bn = dshape[4]
    kheight = kshape[2]
    kwidth = kshape[3]
    dilated_kh = (kheight - 1) * dilation[0] + 1
    dilated_kw = (kwidth - 1) * dilation[1] + 1
    kflatten = int64(1)
    for i in const_range(kshape.shape[0]):
        kflatten *= kshape[i]

    oc = kflatten // (kheight * kwidth * ic_chunk * ic_bn)
    oc_chunk = oc // oc_bn

    out_height = (height + 2 * padding[0] - dilated_kh) // strides[0] + 1
    out_width = (width + 2 * padding[1] - dilated_kw) // strides[1] + 1

    out[0] = dshape[0]
    out[1] = oc_chunk
    out[2] = out_height
    out[3] = out_width
    out[4] = int64(oc_bn)
    return out


@reg.register_shape_func("nn.contrib_conv2d_NCHWc", False)
def conv2d_NCHWc_shape_func(attrs, inputs, _):
    """
    Shape function for contrib_conv2d_NCHWc op.
    """
    strides = get_const_tuple(attrs.strides)
    padding = get_const_tuple(attrs.padding)
    dilation = get_const_tuple(attrs.dilation)
    out_layout = attrs.out_layout
    oc_bn = int(out_layout[4:-1])

    return [
        _conv2d_NCHWc_shape_func(
            inputs[0],
            inputs[1],
            convert(strides),
            convert(padding),
            convert(dilation),
            convert(oc_bn),
        )
    ]


@script
def _conv_transpose_shape_func(dshape, kshape, strides, padding, dilation, output_padding):
    out = output_tensor((dshape.shape[0],), "int64")
    out[0] = dshape[0]
    out[1] = kshape[1]

    for i in const_range(dshape.shape[0] - 2):
        dilated_k = (kshape[i + 2] - 1) * dilation[i] + 1
        out[i + 2] = (
            strides[i] * (dshape[i + 2] - 1) + dilated_k - 2 * padding[i] + output_padding[i]
        )
    return out


def conv_transpose_shape_func(attrs, inputs, _):
    """
    Shape function for contrib_conv2d_NCHWc op.
    """
    strides = get_const_tuple(attrs.strides)
    padding = get_const_tuple(attrs.padding)
    dilation = get_const_tuple(attrs.dilation)
    output_padding = get_const_tuple(attrs.output_padding)

    return [
        _conv_transpose_shape_func(
            inputs[0],
            inputs[1],
            convert(strides),
            convert(padding),
            convert(dilation),
            convert(output_padding),
        )
    ]


reg.register_shape_func("nn.conv1d_transpose", False, conv_transpose_shape_func)
reg.register_shape_func("nn.conv2d_transpose", False, conv_transpose_shape_func)


@script
def _pool2d_shape_func(data_shape, pool_size, strides, padding, height_axis, width_axis):
    out = output_tensor((data_shape.shape[0],), "int64")
    for i in const_range(data_shape.shape[0]):
        if i == height_axis:
            out[i] = (data_shape[i] + padding[0] + padding[2] - pool_size[0]) // strides[0] + 1
        elif i == width_axis:
            out[i] = (data_shape[i] + padding[1] + padding[3] - pool_size[1]) // strides[1] + 1
        else:
            out[i] = data_shape[i]

    return out


def pool2d_shape_func(attrs, inputs, _):
    """
    Shape function for pool2d op.
    """
    pool_size = get_const_tuple(attrs.pool_size)
    strides = get_const_tuple(attrs.strides)
    padding = get_const_tuple(attrs.padding)
    layout = attrs.layout
    height_axis = layout.index("H")
    width_axis = layout.index("W")
    if len(padding) == 1:
        padding = [padding[0]] * 4
    elif len(padding) == 2:
        padding = [padding[0], padding[1], padding[0], padding[1]]

    return [
        _pool2d_shape_func(
            inputs[0],
            convert(pool_size),
            convert(strides),
            convert(padding),
            convert(height_axis),
            convert(width_axis),
        )
    ]


reg.register_shape_func("nn.max_pool2d", False, pool2d_shape_func)
reg.register_shape_func("nn.avg_pool2d", False, pool2d_shape_func)


@script
def _global_pool2d_shape_func(data_shape, height_axis, width_axis):
    out = output_tensor((data_shape.shape[0],), "int64")
    for i in const_range(out.shape[0]):
        if i == height_axis or i == width_axis:
            out[i] = int64(1)
        else:
            out[i] = data_shape[i]

    return out


def global_pool2d_shape_func(attrs, inputs, _):
    """
    Shape function for global pool2d op.
    """
    layout = attrs.layout
    height_axis = width_axis = 1
    for i, letter in enumerate(layout):
        if letter == "H":
            height_axis = i
        if letter == "W":
            width_axis = i
    return [_global_pool2d_shape_func(inputs[0], convert(height_axis), convert(width_axis))]


reg.register_shape_func("nn.global_max_pool2d", False, global_pool2d_shape_func)
reg.register_shape_func("nn.global_avg_pool2d", False, global_pool2d_shape_func)


@script
def _batch_flatten_shape_func(data_shape):
    out = output_tensor((2,), "int64")
    out[0] = data_shape[0]
    out[1] = int64(1)
    for i in const_range(data_shape.shape[0] - 1):
        out[1] *= data_shape[i + 1]

    return out


@reg.register_shape_func("nn.batch_flatten", False)
def batch_flatten_shape_func(attrs, inputs, _):
    """
    Shape function for batch_flatten op.
    """
    return [_batch_flatten_shape_func(inputs[0])]


@script
def _matmul_shape_func(tensor_a_shape, tensor_b_shape, transpose_a, transpose_b):
    out = output_tensor((tensor_a_shape.shape[0],), "int64")
    for i in const_range(out.shape[0] - 1):
        out[i] = tensor_a_shape[i]
    if transpose_a:
        out[out.shape[0] - 2] = out[out.shape[0] - 1]
    out[out.shape[0] - 1] = tensor_b_shape[0] if transpose_b else tensor_b_shape[1]

    return out


@reg.register_shape_func("nn.matmul", False)
def matmul_shape_func(attrs, inputs, _):
    """Shape function for matmul op."""
    ret = [
        _matmul_shape_func(
            inputs[0],
            inputs[1],
            expr.IntImm("bool", attrs.transpose_a),
            expr.IntImm("bool", attrs.transpose_b),
        )
    ]
    return ret


@reg.register_shape_func("nn.dense", False)
def dense_shape_func(attrs, inputs, _):
    """Shape function for dense op. This is an alias of matmul_nt operator for data tensor in
    non-transposed format and weight tensor in transposed format.
    """
    ret = [
        _matmul_shape_func(
            inputs[0],
            inputs[1],
            expr.IntImm("bool", False),
            expr.IntImm("bool", True),
        )
    ]
    return ret


@script
def _dense_pack_shape_func(data_shape, weight_shape):
    out = output_tensor((data_shape.shape[0],), "int64")
    assert data_shape.shape[0] == 2, "Input data must be 2D"
    out[0] = data_shape[0]
    out[1] = weight_shape[0] * weight_shape[2]

    return out


@reg.register_shape_func("nn.contrib_dense_pack", False)
def dense_pack_shape_func(attrs, inputs, _):
    """
    Shape function for dense_pack op.
    """
    ret = [_dense_pack_shape_func(inputs[0], inputs[1])]
    return ret


@script
def _batch_matmul_shape_func(tensor_a_shape, tensor_b_shape, transpose_a, transpose_b):
    out = output_tensor((tensor_a_shape.shape[0],), "int64")
    out[0] = max(tensor_a_shape[0], tensor_b_shape[0])
    out[1] = tensor_a_shape[2] if transpose_a else tensor_a_shape[1]
    out[2] = tensor_b_shape[1] if transpose_b else tensor_b_shape[2]

    return out


@reg.register_shape_func("nn.batch_matmul", False)
def batch_matmul_shape_func(attrs, inputs, _):
    """
    Shape function for batch matmul op.
    """
    ret = [
        _batch_matmul_shape_func(
            inputs[0],
            inputs[1],
            expr.IntImm("bool", attrs.transpose_a),
            expr.IntImm("bool", attrs.transpose_b),
        )
    ]
    return ret


@script
def _pad_shape_func(data_shape, pad_width):
    out = output_tensor((data_shape.shape[0],), "int64")
    for i in const_range(out.shape[0]):
        out[i] = data_shape[i] + pad_width[i][0] + pad_width[i][1]

    return out


@reg.register_shape_func("nn.pad", False)
def pad_shape_func(attrs, inputs, _):
    """
    Shape function for pad op.
    """
    pad_width = []
    for pair in attrs.pad_width:
        pad_width.append(get_const_tuple(pair))
    return [_pad_shape_func(inputs[0], convert(pad_width))]


@script
def _dilate_shape_func(data_shape, strides):
    out = output_tensor((data_shape.shape[0],), "int64")
    for i in const_range(out.shape[0]):
        out[i] = (data_shape[i] - 1) * strides[i] + 1

    return out


@reg.register_shape_func("nn.dilate", False)
def dilate_shape_func(attrs, inputs, _):
    """
    Shape function for dilate op.
    """
    return [_dilate_shape_func(inputs[0], convert(attrs.strides))]


reg.register_shape_func("nn.bias_add", False, elemwise_shape_func)
reg.register_shape_func("nn.softmax", False, elemwise_shape_func)
reg.register_shape_func("nn.fast_softmax", False, elemwise_shape_func)
reg.register_shape_func("nn.relu", False, elemwise_shape_func)
reg.register_shape_func("nn.leaky_relu", False, elemwise_shape_func)
reg.register_shape_func("nn.prelu", False, elemwise_shape_func)
