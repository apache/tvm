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
# pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import

from tvm.te.hybrid import script
from tvm.runtime import convert

from tvm import topi
from tvm.topi.utils import get_const_tuple
from .. import op as reg
from .. import strategy
from ..op import OpPattern
from .image import resize2d


# resize
@reg.register_compute("image.resize1d")
def compute_resize1d(attrs, inputs, out_type):
    """compute definition for resize1d op"""
    size = attrs.size
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    rounding_method = attrs.rounding_method
    cubic_alpha = attrs.cubic_alpha
    cubic_exclude = attrs.cubic_exclude
    out_dtype = attrs.out_dtype
    return [
        topi.image.resize1d(
            inputs[0],
            size,
            layout,
            method,
            coord_trans,
            rounding_method,
            cubic_alpha,
            cubic_exclude,
            out_dtype,
        )
    ]


reg.register_injective_schedule("image.resize1d")


@reg.register_convert_op_layout("image.resize1d")
def convert_image_resize1d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for image resize1d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current resize op
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data input.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """

    new_attrs = dict(attrs)
    assert len(desired_layouts) == 1, "Only one desired layout is expected"
    desired_layout = str(desired_layouts[0])
    assert desired_layout != "default", "Layout cannot be default"
    new_attrs["layout"] = desired_layout
    return resize1d(*inputs, **new_attrs)


@script
def _resize1d_shape_func(image_shape, size, batch_axis, width_axis, channel_axis):
    out = output_tensor((3,), "int64")
    out[batch_axis] = int64(image_shape[0])
    out[width_axis] = int64(size[1])
    out[channel_axis] = image_shape[channel_axis]
    return out


@reg.register_shape_func("image.resize1d", False)
def resize1d_shape_func(attrs, inputs, _):
    """
    Shape function for resize2d op.
    """
    layout = attrs.layout
    width_axis = channel_axis = 1
    for i, letter in enumerate(layout):
        if letter == "N":
            batch_axis = i
        if letter == "W":
            width_axis = i
        if letter == "C":
            channel_axis = i
    size = get_const_tuple(attrs.size)
    return [
        _resize1d_shape_func(
            inputs[0],
            convert(size),
            convert(batch_axis),
            convert(width_axis),
            convert(channel_axis),
        )
    ]


@reg.register_compute("image.resize2d")
def compute_resize2d(attrs, inputs, out_type):
    """compute definition for resize2d op"""
    size = attrs.size
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    rounding_method = attrs.rounding_method
    cubic_alpha = attrs.cubic_alpha
    cubic_exclude = attrs.cubic_exclude
    out_dtype = attrs.out_dtype
    return [
        topi.image.resize2d(
            inputs[0],
            size,
            layout,
            method,
            coord_trans,
            rounding_method,
            cubic_alpha,
            cubic_exclude,
            out_dtype,
        )
    ]


reg.register_injective_schedule("image.resize2d")


@reg.register_convert_op_layout("image.resize2d")
def convert_image_resize2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for image resize2d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current resize op
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data input.
    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """

    new_attrs = dict(attrs)
    assert len(desired_layouts) == 1, "Only one desired layout is expected"
    desired_layout = str(desired_layouts[0])
    assert desired_layout != "default", "Layout cannot be default"
    new_attrs["layout"] = desired_layout
    return resize2d(*inputs, **new_attrs)


@script
def _resize2d_shape_func(image_shape, size, batch_axis, height_axis, width_axis, channel_axis):
    out = output_tensor((4,), "int64")
    out[batch_axis] = int64(image_shape[0])
    out[height_axis] = int64(size[0])
    out[width_axis] = int64(size[1])
    out[channel_axis] = image_shape[channel_axis]
    return out


@reg.register_shape_func("image.resize2d", False)
def resize2d_shape_func(attrs, inputs, _):
    """
    Shape function for resize2d op.
    """
    layout = attrs.layout
    height_axis = width_axis = channel_axis = 1
    for i, letter in enumerate(layout):
        if letter == "N":
            batch_axis = i
        if letter == "H":
            height_axis = i
        if letter == "W":
            width_axis = i
        if letter == "C":
            channel_axis = i
    size = get_const_tuple(attrs.size)
    return [
        _resize2d_shape_func(
            inputs[0],
            convert(size),
            convert(batch_axis),
            convert(height_axis),
            convert(width_axis),
            convert(channel_axis),
        )
    ]


@reg.register_compute("image.resize3d")
def compute_resize3d(attrs, inputs, out_type):
    """compute definition for resize3d op"""
    size = attrs.size
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    rounding_method = attrs.rounding_method
    cubic_alpha = attrs.cubic_alpha
    cubic_exclude = attrs.cubic_exclude
    out_dtype = attrs.out_dtype
    return [
        topi.image.resize3d(
            inputs[0],
            size,
            layout,
            method,
            coord_trans,
            rounding_method,
            cubic_alpha,
            cubic_exclude,
            out_dtype,
        )
    ]


reg.register_injective_schedule("image.resize3d")


# crop and resize
@reg.register_compute("image.crop_and_resize")
def compute_crop_and_resize(attrs, inputs, out_type):
    crop_size = attrs.crop_size
    layout = attrs.layout
    method = attrs.method
    extrapolation_value = attrs.extrapolation_value
    out_dtype = attrs.out_dtype
    return [
        topi.image.crop_and_resize(
            inputs[0],
            inputs[1],
            inputs[2],
            crop_size,
            layout,
            method,
            extrapolation_value,
            out_dtype,
        )
    ]


reg.register_injective_schedule("image.crop_and_resize")


@script
def _crop_and_resize_func(
    image_shape, boxes_shape, crop_size, height_axis, width_axis, channel_axis
):
    out = output_tensor((4,), "int64")
    out[0] = boxes_shape[0]
    out[height_axis] = int64(crop_size[0])
    out[width_axis] = int64(crop_size[1])
    out[channel_axis] = image_shape[channel_axis]
    return out


@reg.register_shape_func("image.crop_and_resize", False)
def crop_and_resize_func(attrs, inputs, _):
    """
    Shape function for crop_and_resize op.
    """
    layout = attrs.layout
    height_axis = width_axis = channel_axis = 1
    for i, letter in enumerate(layout):
        if letter == "H":
            height_axis = i
        if letter == "W":
            width_axis = i
        if letter == "C":
            channel_axis = i
    crop_size = get_const_tuple(attrs.crop_size)
    return [
        _crop_and_resize_func(
            inputs[0],
            inputs[1],
            convert(crop_size),
            convert(height_axis),
            convert(width_axis),
            convert(channel_axis),
        )
    ]


# dilation2d
reg.register_strategy("image.dilation2d", strategy.dilation2d_strategy)
reg.register_pattern("image.dilation2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# affine_grid
@reg.register_compute("image.affine_grid")
def compute_affine_grid(attrs, inputs, out_dtype):
    target_shape = get_const_tuple(attrs.target_shape)
    return [topi.image.affine_grid(inputs[0], target_shape)]


reg.register_injective_schedule("image.affine_grid")


@script
def _affine_grid_func(data, target_shape):
    out = output_tensor((4,), "int64")
    out[0] = int64(data[0])
    out[1] = int64(2)
    out[2] = int64(target_shape[0])
    out[3] = int64(target_shape[1])
    return out


@reg.register_shape_func("image.affine_grid", False)
def affine_grid_func(attrs, inputs, _):
    """
    Shape function for affine_grid op.
    """
    target_shape = get_const_tuple(attrs.target_shape)
    return [_affine_grid_func(inputs[0], convert(target_shape))]


# grid_sample
@reg.register_compute("image.grid_sample")
def compute_grid_sample(attrs, inputs, out_dtype):
    method = attrs.method
    layout = attrs.layout
    return [topi.image.grid_sample(inputs[0], inputs[1], method, layout)]


reg.register_injective_schedule("image.grid_sample")


@script
def _grid_sample_func(data, grid):
    out = output_tensor((4,), "int64")
    out[0] = int64(data[0])
    out[1] = int64(data[1])
    out[2] = int64(grid[2])
    out[3] = int64(grid[3])
    return out


@reg.register_shape_func("image.grid_sample", False)
def grid_sample_func(attrs, inputs, _):
    """
    Shape function for grid_sample op.
    """
    return [_grid_sample_func(inputs[0], inputs[1])]
