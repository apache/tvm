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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""util functions to be reused in different compute/schedule on Qualcomm Adreno GPU"""

import tvm
import numpy
from tvm import te
from tvm.topi.utils import simplify
from tvm.topi import nn
from ..utils import get_const_tuple


def get_div(value, start):
    """Returns the maximum divider for `value` starting from `start` value"""
    div = 1
    for d in range(start, 0, -1):
        if (value % d) == 0:
            div = d
            break
    return div


def split_to_chunks(extent, block):
    """
    Splits the trip count value to chunks and block, returns the remainder as well
    the chunks and blocks covers or overlaps the origin value

    If extent can be divisible by block:
        extent = chunks * block
    else
        extent = (chunks - 1) * block + tail

    Parameters
    ----------
    extent: int
        tripcount for original compute

    block: int
        size of the block

    Returns
    ----------
    out: tuple of the (chunks, block, tail)
         chunks = ceildiv(extent, block)
         tail = number of origin elements in the latest chunk
    """
    tail = extent % block
    chunks = extent // block
    if tail == 0:
        tail = block
    else:
        chunks += 1
    return chunks, block, tail


def pack_input(Input, layout, batch, chunks, block, original_tail, in_height, in_width):
    """
    Adds compute stages for packing of the data in runtime. Extends channel dimensions
    to be dividable by factor 4

    This function should be substituted by Schedule.transform_layout() in the future: see
    https://github.com/apache/tvm-rfcs/blob/main/rfcs/0039-buffer-physical-layout.md

    Parameters
    ----------
    Input: tvm.te.Tensor
        Input tensor to be repacked in runtime

    layout: string
        Layout of origin 4d tensor
        NCHW or NHWC are acceptable

    batch: int
        Batch size

    chunks: int
        Number of channel chunks been in the final tensor

    block: int
        size of the channel block

    original_tail: int
        Tail in the latest chunk diffing original number of channels vs blocked one
        If original_tail != block:
          original_channels = chunks * block - original_tail
        else
          original_channels = chunks * block

    in_height: int
        Height of the feature map

    in_width: int
        Width of the feature map
    """

    pad_value = tvm.tir.const(0, Input.dtype)

    def _reorder_data_nchw(*indices):
        condition = []
        condition.append(indices[1] == chunks - 1)
        condition.append(indices[4] >= original_tail)
        condition = tvm.tir.all(*condition)
        return tvm.tir.if_then_else(
            condition,
            pad_value,
            Input[indices[0], indices[1] * block + indices[4], indices[2], indices[3]],
        )

    def _reorder_data_nhwc(*indices):
        condition = []
        condition.append(indices[3] == chunks - 1)
        condition.append(indices[4] >= original_tail)
        condition = tvm.tir.all(*condition)
        return tvm.tir.if_then_else(
            condition,
            pad_value,
            Input[indices[0], indices[1], indices[2], indices[3] * block + indices[4]],
        )

    # compute:
    if layout == "NCHW":
        reordered_data = te.compute(
            [batch, chunks, in_height, in_width, block],
            _reorder_data_nchw,
            name="input_pack",
            tag="input_pack",
        )
    elif layout == "NHWC":
        reordered_data = te.compute(
            [batch, in_height, in_width, chunks, block],
            _reorder_data_nhwc,
            name="input_pack",
            tag="input_pack",
        )
    else:
        assert False, "Adreno util function pack_input does not accept unknown layout"
    return reordered_data


def pack_filter(
    Filter,
    layout,
    out_chunks,
    out_block,
    out_original_tail,
    in_filter_channels,
    in_chunks,
    in_block,
    in_original_tail,
    kernel_h,
    kernel_w,
):
    """
    Adds compute stages for packing of the filter in runtime. Extends channels dimensions
    to be dividable by factor 4

    This function should be substituted by Schedule.transform_layout() in the future: see
    https://github.com/apache/tvm-rfcs/blob/main/rfcs/0039-buffer-physical-layout.md

    Parameters
    ----------
    Filter: tvm.te.Tensor
        Filter tensor to be repacked in runtime

    layout: string
        Layout of origin 4d tensor
        NCHW or NHWC are acceptable

    out_chunks: int
        Number of chunks for filters

    out_block: int
        Size of the block for output channels

    out_original_tail: int
        Original size of the latest chunk of output filters

    in_filter_channels: int
        Number of filter channels. might be different vs input channels in the
        data due to groups/depthwise nature

    in_chunks: int
        Number of input data channel chunks

    in_block: int
        Size of the block for input data channels

    in_original_tail
        Original size of the latest chunk for input data channels

    kernel_h: int
        Height of the conv2d kernel

    kernel_w: int
        Width of the conv2d kernel
    """
    pad_value = tvm.tir.const(0, Filter.dtype)

    def _reorder_weights_depthwise_oihw(*indices):
        conditionA = []
        conditionA.append(indices[0] == out_chunks - 1)
        conditionA.append(indices[4] >= out_original_tail)
        conditionAT = tvm.tir.all(*conditionA)

        return tvm.tir.if_then_else(
            conditionAT,
            pad_value,
            Filter[indices[0] * out_block + indices[4], indices[1], indices[2], indices[3]],
        )

    def _reorder_weights_depthwise_hwoi(*indices):
        conditionA = []
        conditionA.append(indices[2] == out_chunks - 1)
        conditionA.append(indices[4] >= out_original_tail)
        conditionAT = tvm.tir.all(*conditionA)

        return tvm.tir.if_then_else(
            conditionAT,
            pad_value,
            Filter[indices[0], indices[1], indices[2] * out_block + indices[4], indices[3]],
        )

    def _reorder_weights_oihw(*indices):
        conditionA = []
        conditionA.append(indices[0] == out_chunks - 1)
        conditionA.append(indices[4] >= out_original_tail)
        conditionAT = tvm.tir.all(*conditionA)

        conditionO = []
        conditionO.append(conditionAT)
        conditionO.append(indices[1] >= in_chunks * in_block + in_original_tail)
        conditionOT = tvm.tir.any(*conditionO)
        return tvm.tir.if_then_else(
            conditionOT,
            pad_value,
            Filter[indices[0] * out_block + indices[4], indices[1], indices[2], indices[3]],
        )

    def _reorder_weights_hwio(*indices):
        conditionA = []
        conditionA.append(indices[3] == out_chunks - 1)
        conditionA.append(indices[4] >= out_original_tail)
        conditionAT = tvm.tir.all(*conditionA)

        conditionO = []
        conditionO.append(conditionAT)
        conditionO.append(indices[2] >= in_chunks * in_block + in_original_tail)
        conditionOT = tvm.tir.any(*conditionO)
        return tvm.tir.if_then_else(
            conditionOT,
            pad_value,
            Filter[indices[0], indices[1], indices[2], indices[3] * out_block + indices[4]],
        )

    if in_filter_channels == 1:
        if layout == "OIHW":
            reordered_filter = te.compute(
                [out_chunks, in_filter_channels, kernel_h, kernel_w, out_block],
                _reorder_weights_depthwise_oihw,
                name="filter_pack",
                tag="filter_pack",
            )
        elif layout == "HWOI":
            reordered_filter = te.compute(
                [kernel_h, kernel_w, out_chunks, in_filter_channels, out_block],
                _reorder_weights_depthwise_hwoi,
                name="filter_pack",
                tag="filter_pack",
            )
        else:
            assert False, "Adreno util function def pack_filter does not accept unknown layout"
    else:
        if layout == "OIHW":
            reordered_filter = te.compute(
                [out_chunks, in_filter_channels, kernel_h, kernel_w, out_block],
                _reorder_weights_oihw,
                name="filter_pack",
                tag="filter_pack",
            )
        elif layout == "HWIO":
            reordered_filter = te.compute(
                [kernel_h, kernel_w, in_filter_channels, out_chunks, out_block],
                _reorder_weights_hwio,
                name="filter_pack",
                tag="filter_pack",
            )
        else:
            assert False, "Adreno util function def pack_filter does not accept unknown layout"
    return reordered_filter


def expand_spatial_dimensions(
    in_height, in_width, kernel_h, kernel_w, dilation_h, dilation_w, padding, stride_h, stride_w
):
    """
    Expands spatial dimensions to be dividable by factor 4. This will allow us to do extrimely
    better parallel computation on GPU. The drawback of this solution - it will be number of
    useless computations. By fact the speed-up of parallelism significantly overcomes the slowdown
    of extra compute and eventuially this is useful approach, at least for GPU

    Parameters
    ----------
    in_height: int
        Height of the feature map

    in_width: int
        Width of the feature map

    kernel_h: int
        Height of the conv2d kernel

    kernel_w: int
        Width of the conv2d kernel

    dilation_h: int
        Vertical dilation of the conv2d kernel

    dilation_w: int
        Horizontal dilation of the conv2d kernel

    padding: tuple or list
        Conv2d paddings

    stride_h: int
        Vertical stride  of the conv2d kernel

    stride_w: int
        Horizontal stride  of the conv2d kernel
    """
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height_orig = out_height = simplify(
        (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    )
    out_width_orig = out_width = simplify(
        (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    )

    # can output shape be divded by 2 or even 4?
    # if it cannot be divided, need to extend for further help with split
    # theortically there should be addition padding for inputs, but it will be optimized by
    # cache_read InferBound. We must proceed pad here exactly to produce tensor which is
    # required for calculation of original out size, not more! In other case intermediate
    # tensor might be allcoated with less sizes while compute will try to fill the expanded
    # one - data discrepancy as a result
    # And in case of textures it is not a problem if we provide texture of less size because
    # 1. It is not important which values would be for extra calc - these calculations are
    #    required only for better utilizatin of GPU fit to working groups
    # 2. When we request pixel out opf bound, texture will handle this correctly. As mentioned
    #    above, the value itself is not important
    if out_height % 2 != 0:
        out_height += 1
    if out_width % 2 != 0:
        out_width += 1

    if out_height % 4 != 0:
        out_height += 2
    if out_width % 4 != 0:
        out_width += 2
    return out_height_orig, out_height, out_width_orig, out_width


def add_pad(
    data,
    layout,
    out_height,
    out_width,
    kernel_h,
    kernel_w,
    dilation_h,
    dilation_w,
    padding,
    stride_h,
    stride_w,
):
    """Computes required padding values by the parameters of conv2d and adds
        compute for extending of original tensor

    Parameters
    ----------
    data: tvm.te.Tensor
        5d tensor, the layout of spatial dimensions are defined as separate argument

    layout: string
        Layout of origin 4d tensor

    out_height: int
        Height of the output feature map

    out_width: int
        Width of the output feature map

    kernel_h: int
        Height of the conv2d kernel

    kernel_w: int
        Width of the conv2d kernel

    dilation_h: int
        Height dilation value from conv2d attributes

    dilation_w: int
        Width dilation value from conv2d attributes

    padding: list / tuple of n ints
        Padding values from conv2d attributes

    stride_h: int
        Height stride value from conv2d attributes

    stride_w: int
        Width stride value from conv2d attributes

    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as Input.
    """
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    # compute graph
    if layout == "NCHW":
        y_axis = 2
        x_axis = 3
        if len(data.shape) == 4:
            _, _, in_height, in_width = data.shape
        else:
            _, _, in_height, in_width, _ = data.shape
    elif layout == "NHWC":
        y_axis = 1
        x_axis = 2
        if len(data.shape) == 4:
            _, in_height, in_width, _ = data.shape
        else:
            _, in_height, in_width, _, _ = data.shape
    else:
        assert False, "not supported layout in adreno util add_pad"
    pad_before = [0, 0, 0, 0, 0]
    pad_after = [0, 0, 0, 0, 0]
    pad_before[y_axis] = pad_top
    pad_before[x_axis] = pad_left
    pad_after[y_axis] = pad_down
    pad_after[x_axis] = pad_right

    # calculation of real used input size:
    input_latest_w = (out_width - 1) * stride_w + (kernel_w - 1) * dilation_w + 1
    input_latest_h = (out_height - 1) * stride_h + (kernel_h - 1) * dilation_h + 1
    if input_latest_w < in_width + pad_before[x_axis] + pad_after[x_axis]:
        pad_after[x_axis] -= in_width + pad_before[x_axis] + pad_after[x_axis] - input_latest_w
    if input_latest_h < in_height + pad_before[y_axis] + pad_after[y_axis]:
        pad_after[y_axis] -= in_height + pad_before[y_axis] + pad_after[y_axis] - input_latest_h
    return nn.pad(data, pad_before, pad_after, name="pad_temp")


def bind_data_copy(stage, axis_to_vectorize=None):
    """
    Schedules the eltwise stages like copying of data or postops

    Parameters
    ----------
    stage: tvm.te.Tensor

    axis_to_vectorize:
        Causes to split certain axis, moves inner part to the end of schedule
        and enable vectorization by this axis
        If parameter is not pointed, the schedule will be vectorized if the most inner
        dim is eq to 4 (size of the vector in texture)
    """
    shape = get_const_tuple(stage.op.output(0).shape)
    if axis_to_vectorize and len(shape) == 4 and shape[axis_to_vectorize] % 4 == 0:
        ax0, ax1, ax2, ax3 = stage.op.axis
        if axis_to_vectorize == 1:
            oax1, iax1 = stage.split(ax1, factor=4)
            stage.reorder(ax0, oax1, ax2, ax3, iax1)
            stage.vectorize(iax1)
            fused = stage.fuse(ax0, oax1, ax2, ax3)
        elif axis_to_vectorize == 3:
            oax3, iax3 = stage.split(ax3, factor=4)
            stage.reorder(ax0, ax1, ax2, oax3, iax3)
            stage.vectorize(iax3)
            fused = stage.fuse(ax0, ax1, ax2, oax3)

        ftc = numpy.prod(shape) / 4
        div = get_div(ftc, 128)
        block, thread = stage.split(fused, factor=div)

        stage.bind(block, te.thread_axis("blockIdx.z"))
        stage.bind(thread, te.thread_axis("threadIdx.z"))
    else:
        axes = stage.op.axis
        fused = stage.fuse(*axes[:-1])
        if shape[-1] <= 32:
            ftc = numpy.prod(shape[:-1])
            div = get_div(ftc, 64)
            block, thread = stage.split(fused, factor=div)
            stage.bind(block, te.thread_axis("blockIdx.x"))
            stage.bind(thread, te.thread_axis("threadIdx.x"))
            if shape[-1] == 4:
                stage.vectorize(axes[-1])
        else:
            stage.bind(fused, te.thread_axis("blockIdx.x"))
            stage.bind(*axes[-1:], te.thread_axis("threadIdx.x"))


def get_texture_storage(shape):
    """
    Returns the texture layout acceptable for the shape

    Parameters
    ----------
    shape: array
        Shape of the tensor to be packed to texture
    """
    # certain limitation of the Qualcomm devices. Subject to be determined for certain device
    # individually, but until we have access to remote device during compilation, we have to
    # define it uniformly for all target devices
    # limit = 16384
    limit = tvm.target.Target.current().attrs["texture_spatial_limit"]

    if shape[0] * shape[1] * shape[2] < limit and shape[3] < limit:
        return "global.texture"
    elif shape[0] * shape[1] < limit and shape[2] * shape[3] < limit:
        return "global.texture-nhwc"
    else:
        return "global.texture-weight"
