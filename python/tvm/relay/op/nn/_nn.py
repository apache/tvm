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

import topi
from topi.util import get_const_tuple

from tvm.runtime import convert
from tvm.te.hybrid import script
from .. import op as reg
from .. import strategy
from ..op import OpPattern
from .._tensor import elemwise_shape_func
from ..strategy.generic import is_depthwise_conv2d

# relu
reg.register_broadcast_schedule("nn.relu")
reg.register_pattern("nn.relu", OpPattern.ELEMWISE)


# softmax
reg.register_strategy("nn.softmax", strategy.softmax_strategy)
reg.register_pattern("nn.softmax", OpPattern.OPAQUE)


# log_softmax
reg.register_schedule("nn.log_softmax", strategy.schedule_log_softmax)
reg.register_pattern("nn.log_softmax", OpPattern.OPAQUE)


# dense
reg.register_strategy("nn.dense", strategy.dense_strategy)
reg.register_pattern("nn.dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# fifo_buffer
@reg.register_compute('nn.fifo_buffer')
def compute_fifo_buffer(attrs, inputs, out_type):
    return [topi.nn.fifo_buffer(inputs[0], inputs[1], axis=attrs.get_int('axis'))]

reg.register_injective_schedule("nn.fifo_buffer")
reg.register_pattern("nn.fifo_buffer", OpPattern.OPAQUE)


# batch_matmul
reg.register_strategy("nn.batch_matmul", strategy.batch_matmul_strategy)
reg.register_pattern("nn.batch_matmul", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# sparse_dense
@reg.register_compute("nn.sparse_dense")
def compute_sparse_dense(attrs, inputs, out_type):
    """Compute definition of sparse_dense"""
    return [topi.nn.sparse_dense(inputs[0], inputs[1], inputs[2], inputs[3])]

reg.register_schedule("nn.sparse_dense", strategy.schedule_sparse_dense)
reg.register_pattern("nn.sparse_dense", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


# sparse_transpose
@reg.register_compute("nn.sparse_transpose")
def compute_sparse_transpose(attrs, inputs, out_type):
    """Compute definition of sparse_transpose"""
    return topi.nn.sparse_transpose(inputs[0], inputs[1], inputs[2])

reg.register_schedule("nn.sparse_transpose", strategy.schedule_sparse_transpose)
reg.register_pattern("nn.sparse_transpose", reg.OpPattern.OUT_ELEMWISE_FUSABLE)


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
def convert_conv2d(attrs, inputs, tinfos, desired_layout):
    """Convert Layout pass registration for conv2d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layout : str
        The desired layout

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = desired_layout
    if desired_layout == 'NCHW':
        new_attrs['kernel_layout'] = 'OIHW'
        return relay.nn.conv2d(data, weight, **new_attrs)
    elif desired_layout == 'NHWC':
        # Check for depthwise convolution.
        if is_depthwise_conv2d(data.shape, attrs['data_layout'], weight.shape,
                               attrs['kernel_layout'], attrs['groups']):
            new_attrs['kernel_layout'] = 'HWOI'
        else:
            new_attrs['kernel_layout'] = 'HWIO'
        return relay.nn.conv2d(data, weight, **new_attrs)
    else:
        assert "Layout %s is not yet supported." % (desired_layout)
    return None


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


# conv3d
reg.register_strategy("nn.conv3d", strategy.conv3d_strategy)
reg.register_pattern("nn.conv3d", OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_alter_op_layout("nn.conv3d")
def alter_op_layout_conv3d(attrs, inputs, tinfos, out_type):
    """Alternate the layout of conv3d"""
    return topi.nn.conv3d_alter_layout(attrs, inputs, tinfos, out_type)

@reg.register_convert_op_layout("nn.conv3d")
def convert_conv3d(attrs, inputs, tinfos, desired_layout):
    """Convert Layout pass registration for conv3d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layout : str
        The desired layout

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = desired_layout
    if desired_layout == 'NCDHW':
        new_attrs['kernel_layout'] = 'OIDHW'
        return relay.nn.conv3d(data, weight, **new_attrs)
    elif desired_layout == "NDHWC":
        new_attrs['kernel_layout'] = 'DHWIO'
        return relay.nn.conv3d(data, weight, **new_attrs)
    else:
        assert "Layout %s is not yet supported" % desired_layout
    return None

# conv3d_winograd related operators
reg.register_strategy("nn.contrib_conv3d_winograd_without_weight_transform",
                      strategy.conv3d_winograd_without_weight_transfrom_strategy)
reg.register_pattern("nn.contrib_conv3d_winograd_without_weight_transform",
                     OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("nn.contrib_conv3d_winograd_weight_transform")
def compute_contrib_conv3d_winograd_weight_transform(attrs, inputs, out_dtype):
    """Compute definition of contrib_conv3d_winograd_weight_transform"""
    out = topi.nn.conv3d_winograd_weight_transform(
        inputs[0], attrs.get_int('tile_size'))
    return [out]

reg.register_schedule("nn.contrib_conv3d_winograd_weight_transform",
                      strategy.schedule_conv3d_winograd_weight_transform)
reg.register_pattern("nn.contrib_conv3d_winograd_weight_transform",
                     OpPattern.OUT_ELEMWISE_FUSABLE)


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


# max_pool3d
reg.register_schedule("nn.max_pool3d", strategy.schedule_pool)
reg.register_pattern("nn.max_pool3d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool1d
reg.register_schedule("nn.avg_pool1d", strategy.schedule_pool)
reg.register_pattern("nn.avg_pool1d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool2d
reg.register_schedule("nn.avg_pool2d", strategy.schedule_pool)
reg.register_pattern("nn.avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool3d
reg.register_schedule("nn.avg_pool3d", strategy.schedule_pool)
reg.register_pattern("nn.avg_pool3d", OpPattern.OUT_ELEMWISE_FUSABLE)


# max_pool2d_grad
reg.register_schedule("nn.max_pool2d_grad", strategy.schedule_pool_grad)
reg.register_pattern("nn.max_pool2d_grad", OpPattern.OUT_ELEMWISE_FUSABLE)


# avg_pool2d_grad
reg.register_schedule("nn.avg_pool2d_grad", strategy.schedule_pool_grad)
reg.register_pattern("nn.avg_pool2d_grad", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_max_pool2d
reg.register_schedule("nn.global_max_pool2d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.global_max_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# global_avg_pool2d
reg.register_schedule("nn.global_avg_pool2d", strategy.schedule_adaptive_pool)
reg.register_pattern("nn.global_avg_pool2d", OpPattern.OUT_ELEMWISE_FUSABLE)


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
    return [topi.nn.lrn(inputs[0], attrs.size, attrs.axis,
                        attrs.alpha, attrs.beta, attrs.bias)]

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
    return [topi.nn.upsampling3d(inputs[0], scale_d, scale_h, scale_w, layout, method,\
        coordinate_transformation_mode)]

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


# conv2d_winograd related operators
reg.register_strategy("nn.contrib_conv2d_winograd_without_weight_transform",
                      strategy.conv2d_winograd_without_weight_transfrom_strategy)
reg.register_pattern("nn.contrib_conv2d_winograd_without_weight_transform",
                     OpPattern.OUT_ELEMWISE_FUSABLE)


@reg.register_compute("nn.contrib_conv2d_winograd_weight_transform")
def compute_contrib_conv2d_winograd_weight_transform(attrs, inputs, out_dtype):
    """Compute definition of contrib_conv2d_winograd_weight_transform"""
    out = topi.nn.conv2d_winograd_weight_transform(
        inputs[0], attrs.get_int('tile_size'))
    return [out]

reg.register_schedule("nn.contrib_conv2d_winograd_weight_transform",
                      strategy.schedule_conv2d_winograd_weight_transform)
reg.register_pattern("nn.contrib_conv2d_winograd_weight_transform",
                     OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("nn.contrib_conv2d_winograd_nnpack_weight_transform")
def compute_contrib_conv2d_winograd_nnpack_weight_transform(attrs, inputs, out_dtype):
    """Compute definition of contrib_conv2d_winograd_nnpack_weight_transform"""
    convolution_algorithm = attrs.get_int('convolution_algorithm')
    out = topi.nn.conv2d_winograd_nnpack_weight_transform(
        inputs[0], convolution_algorithm, out_dtype)
    return [out]

reg.register_schedule("nn.contrib_conv2d_winograd_nnpack_weight_transform",
                      strategy.schedule_conv2d_winograd_nnpack_weight_transform)
reg.register_pattern("nn.contrib_conv2d_winograd_nnpack_weight_transform",
                     OpPattern.OPAQUE)


# conv2d_NCHWc
reg.register_strategy("nn.contrib_conv2d_NCHWc", strategy.conv2d_NCHWc_strategy)
reg.register_pattern("nn.contrib_conv2d_NCHWc",
                     OpPattern.OUT_ELEMWISE_FUSABLE)

# depthwise_conv2d_NCHWc
reg.register_strategy("nn.contrib_depthwise_conv2d_NCHWc",
                      strategy.depthwise_conv2d_NCHWc_strategy)
reg.register_pattern("nn.contrib_depthwise_conv2d_NCHWc",
                     OpPattern.OUT_ELEMWISE_FUSABLE)


# deformable_conv2d
reg.register_strategy("nn.deformable_conv2d", strategy.deformable_conv2d_strategy)
reg.register_pattern("nn.deformable_conv2d", OpPattern.OUT_ELEMWISE_FUSABLE)


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
    return [topi.nn.dilate(inputs[0], attrs.strides)]

reg.register_broadcast_schedule("nn.dilate")
reg.register_pattern("nn.dilate", OpPattern.INJECTIVE)


# cross_entropy_with_logits
@reg.register_compute("nn.cross_entropy_with_logits")
def compute_cross_entropy_with_logits(attrs, inputs, out_dtype):
    x, y = inputs
    return [-topi.sum(x * y) / x.shape[0]]

reg.register_reduce_schedule("nn.cross_entropy_with_logits")
reg.register_pattern("nn.cross_entropy_with_logits", OpPattern.OPAQUE)


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


#####################
#  Shape functions  #
#####################

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

    return [_conv2d_NCHWc_shape_func(inputs[0], inputs[1],
                                     convert(strides), convert(padding),
                                     convert(dilation), convert(oc_bn))]

@script
def _pool2d_shape_func(data_shape, pool_size, strides,
                       padding, height_axis, width_axis):
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

    return [_pool2d_shape_func(inputs[0], convert(pool_size),
                               convert(strides), convert(padding),
                               convert(height_axis), convert(width_axis))]

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
def _dense_shape_func(data_shape, weight_shape):
    out = output_tensor((data_shape.shape[0],), "int64")
    for i in const_range(out.shape[0] - 1):
        out[i] = data_shape[i]
    out[out.shape[0] - 1] = weight_shape[0]

    return out

@reg.register_shape_func("nn.dense", False)
def dense_shape_func(attrs, inputs, _):
    """
    Shape function for dense op.
    """
    ret = [_dense_shape_func(inputs[0], inputs[1])]
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
reg.register_shape_func("nn.relu", False, elemwise_shape_func)
