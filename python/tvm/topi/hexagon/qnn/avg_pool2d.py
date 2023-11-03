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
# pylint: disable=invalid-name, unused-variable, unused-argument, too-many-locals

""" Compute and schedule for quantized avg_pool2d op """

import tvm
from tvm import te
from tvm import tir
from ..utils import (
    get_layout_transform_fn,
    get_fixed_point_value,
    is_scalar,
    get_const_int_value,
    get_const_float_value,
)
from ...utils import get_const_tuple
from ...nn.utils import get_pad_tuple
from ...nn.pad import pad
from ..compute_poolarea import compute_PoolArea


def saturate(x: te.Tensor, dtype: str):
    """Saturate value for the specified data type"""
    return te.max(te.min_value(dtype), te.min(x, te.max_value(dtype)))


def get_temp_dtype(h, w, dtype):
    temp_dtype = "int16" if h * w < 256 else "int32"
    if dtype in ("uint8", "int8"):
        return temp_dtype
    else:
        raise RuntimeError(f"Unsupported output dtype, {odtype}'")


def qnn_avg_pool2d_NCHW(
    data: te.Tensor,
    kernel: list,
    stride: list,
    padding: list,
    dilation: list,
    count_include_pad: bool,
    oshape: list,
    odtype: str,
    # quantization params:
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
):
    """Compute for quantized avg_pool2d"""
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")

    temp_dtype = get_temp_dtype(kh, kw, odtype)

    sh, sw = stride
    dh, dw = dilation

    scale = input_scale / output_scale
    scale_fixed_point, rsh = get_fixed_point_value(scale, "int16")
    corr = (output_zero_point << rsh) - input_zero_point * scale_fixed_point

    dilated_kh = (kh - 1) * dh + 1
    dilated_kw = (kw - 1) * dw + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        get_const_tuple(padding), (dilated_kh, dilated_kw)
    )

    # DOPAD
    if pad_top != 0 or pad_down != 0 or pad_left != 0 or pad_right != 0:
        pad_before = (0, 0, pad_top, pad_left)
        pad_after = (0, 0, pad_down, pad_right)
        data_pad = pad(data, pad_before, pad_after, pad_value=input_zero_point, name="data_pad")
    else:
        # By definition when True, zero-padding will be included in the averaging calculation
        # This is equivalent to PoolArea = (kh * kw)
        count_include_pad = True
        data_pad = data

    Sum = te.compute(
        oshape,
        lambda b, c, h, w: te.sum(
            data_pad[b, c, h * sh + dh * rh, w * sw + dw * rw].astype(temp_dtype), axis=[rh, rw]
        ),
        name="pool_sum",
    )

    if not count_include_pad:
        # Compute PoolArea using unpadded input tensor
        _, _, oh, ow = oshape
        _, _, ih, iw = data.shape

        PoolArea = te.compute(
            (oh, ow),
            lambda i, j: compute_PoolArea(i, j, ih, iw, kh, kw, sh, sw, dh, dw, pad_top, pad_left),
            name="pool_area",
        )

        ScaleWithArea = te.compute(
            (oh, ow),
            lambda i, j: (scale_fixed_point // PoolArea[i, j]).astype("int32"),
            name="scale_with_area",
        )

        Avg = te.compute(
            oshape,
            lambda b, c, h, w: saturate(
                ((Sum[b, c, h, w] * ScaleWithArea[h, w]) + corr + (1 << (rsh - 1))) >> rsh, odtype
            ).astype(odtype),
            name="pool_avg",
        )
    else:
        ScaleWithArea = scale_fixed_point // (kh * kw)
        Avg = te.compute(
            oshape,
            lambda b, c, h, w: saturate(
                ((Sum[b, c, h, w] * ScaleWithArea) + corr + (1 << (rsh - 1))) >> rsh, odtype
            ).astype(odtype),
            name="pool_avg",
        )
    return Avg


def qnn_avg_pool2d_NHWC(
    data: te.Tensor,
    kernel: list,
    stride: list,
    padding: list,
    dilation: list,
    count_include_pad: bool,
    oshape: list,
    odtype: str,
    # quantization params:
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
):
    """Compute for quantized avg_pool2d"""
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")

    temp_dtype = get_temp_dtype(kh, kw, odtype)

    sh, sw = stride
    dh, dw = dilation

    scale = input_scale / output_scale
    scale_fixed_point, rsh = get_fixed_point_value(scale, "int16")
    corr = (output_zero_point << rsh) - input_zero_point * scale_fixed_point

    dilated_kh = (kh - 1) * dh + 1
    dilated_kw = (kw - 1) * dw + 1
    # Compute Area

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        get_const_tuple(padding), (dilated_kh, dilated_kw)
    )
    # DOPAD
    if pad_top != 0 or pad_down != 0 or pad_left != 0 or pad_right != 0:
        pad_before = (0, pad_top, pad_left, 0)
        pad_after = (0, pad_down, pad_right, 0)
        data_pad = pad(data, pad_before, pad_after, pad_value=input_zero_point, name="data_pad")
    else:
        # By definition when True, zero-padding will be included in the averaging calculation
        # This is equivalent to PoolArea = (kh * kw)
        count_include_pad = True
        data_pad = data

    Sum = te.compute(
        oshape,
        lambda b, h, w, c: te.sum(
            data_pad[b, h * sh + dh * rh, w * sw + dw * rw, c].astype(temp_dtype), axis=[rh, rw]
        ),
        name="pool_sum",
    )

    if not count_include_pad:
        # Compute PoolArea using unpadded input tensor
        _, oh, ow, _ = oshape
        _, ih, iw, _ = data.shape

        PoolArea = te.compute(
            (oh, ow),
            lambda i, j: compute_PoolArea(i, j, ih, iw, kh, kw, sh, sw, dh, dw, pad_top, pad_left),
            name="pool_area",
        )

        ScaleWithArea = te.compute(
            (oh, ow),
            lambda i, j: tir.if_then_else(
                tir.all(PoolArea[i, j] > 0),
                (scale_fixed_point // PoolArea[i, j]).astype("int32"),
                0,
            ),
            name="scale_with_area",
        )

        Avg = te.compute(
            oshape,
            lambda b, h, w, c: saturate(
                ((Sum[b, h, w, c] * ScaleWithArea[h, w]) + corr + (1 << (rsh - 1))) >> rsh, odtype
            ).astype(odtype),
            name="pool_avg",
        )
    else:
        ScaleWithArea = scale_fixed_point // (kh * kw)
        Avg = te.compute(
            oshape,
            lambda b, h, w, c: saturate(
                ((Sum[b, h, w, c] * ScaleWithArea) + corr + (1 << (rsh - 1))) >> rsh, odtype
            ).astype(odtype),
            name="pool_avg",
        )

    return Avg


def qnn_avg_pool2d_wrapper_compute_NCHW(
    data: te.Tensor,
    kernel: list,
    stride: list,
    padding: list,
    dilation: list,
    count_include_pad: bool,
    oshape: list,
    odtype: str,
    # quantization params:
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
):
    """Extract qnn params"""
    if (
        is_scalar(input_scale)
        and is_scalar(output_scale)
        and is_scalar(input_zero_point)
        and is_scalar(output_zero_point)
    ):
        iscale = get_const_float_value(input_scale)
        oscale = get_const_float_value(output_scale)
        izero_point = get_const_int_value(input_zero_point)
        ozero_point = get_const_int_value(output_zero_point)
        return qnn_avg_pool2d_NCHW(
            data,
            kernel,
            stride,
            padding,
            dilation,
            count_include_pad,
            oshape,
            odtype,
            iscale,
            izero_point,
            oscale,
            ozero_point,
        )
    else:
        raise RuntimeError("quantization parameters should be scalar tensors")


def qnn_avg_pool2d_wrapper_compute_NHWC(
    data: te.Tensor,
    kernel: list,
    stride: list,
    padding: list,
    dilation: list,
    count_include_pad: bool,
    oshape: list,
    odtype: str,
    # quantization params:
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
):
    """Extract qnn params"""
    if (
        is_scalar(input_scale)
        and is_scalar(output_scale)
        and is_scalar(input_zero_point)
        and is_scalar(output_zero_point)
    ):
        iscale = get_const_float_value(input_scale)
        oscale = get_const_float_value(output_scale)
        izero_point = get_const_int_value(input_zero_point)
        ozero_point = get_const_int_value(output_zero_point)
        return qnn_avg_pool2d_NHWC(
            data,
            kernel,
            stride,
            padding,
            dilation,
            count_include_pad,
            oshape,
            odtype,
            iscale,
            izero_point,
            oscale,
            ozero_point,
        )
    else:
        raise RuntimeError("quantization parameters should be scalar tensors")


def schedule_qnn_avg_pool2d(outs):
    """Schedule for qnn.avg_pool2d
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of qnn.avg_pool2d
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = tvm.te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)
    return s


def schedule_8h8w32c(outs: te.Tensor, ins: te.Tensor, output_layout: str, input_layout: str):
    """Schedule for input and output layout 8h8w32c"""

    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("pool_sum")
    Avg = s.get_block("pool_avg")
    mem_scope = "global.vtcm"
    sum_read = s.cache_read(Sum, 0, mem_scope)
    avg_read = s.cache_read(Avg, 0, mem_scope)
    avg_write = s.cache_write(Avg, 0, mem_scope)
    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Sum, ("read", 0), input_transform_fn, pad_value=0)
    s.transform_layout(Avg, ("read", 0), input_transform_fn, pad_value=0)
    s.transform_layout(Avg, ("write", 0), output_transform_fn, pad_value=0)
    return s


def schedule_2048c(outs: te.Tensor, ins: te.Tensor, output_layout: str, input_layout: str):
    """Schedule for output layout: 2048c, input layout: 8h8w32c"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("pool_sum")
    Avg = s.get_block("pool_avg")

    mem_scope = "global.vtcm"
    sum_read = s.cache_read(Sum, 0, mem_scope)
    avg_write = s.cache_write(Avg, 0, mem_scope)
    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Sum, ("read", 0), input_transform_fn, pad_value=0)
    s.transform_layout(Avg, ("write", 0), output_transform_fn, pad_value=0)

    # Schedule 'Avg'
    # Split and reorder the axes to iterate over the output tensor chunks.
    # Each chunk consists for 2048 bytes. For n11c-2048c tensor layout, each chunk
    # only contains 2048 channels which get split by a factor of 128 to be vectorized.
    # NOTE: These schedules are a work in progress and may require
    # adjustments in future as some of the missing features for 2-d tensors
    # become available.

    if output_layout == "n11c-2048c-2d":
        _, _, _, c = s.get_loops(Avg)
    else:
        _, c, _, _ = s.get_loops(Avg)

    # n, h, w, c = s.get_loops(Avg)
    co, ci = s.split(c, [None, 2048])
    cio, cii = s.split(ci, [None, 128])
    s.vectorize(cii)

    # Schedule 'Sum'
    # Compute for 'Sum' includes reduction along height and width. The axes are being
    # reordered so that 128 channels become the inner-most loop and can be vectorized.
    # However, vectorization of the 2-d tensors doesn't work when reduction is
    # involved and requires codegen support that is yet to be added.
    s.compute_at(Sum, cio)
    Sum_axis = s.get_loops(Sum)
    s.reorder(Sum_axis[-2], Sum_axis[-1], Sum_axis[-3])
    # s.vectorize(Sum_axis[-3]) # Doesn't work
    return s


def qnn_avg_pool2d_schedule(outs: te.Tensor, ins: te.Tensor, output_layout: str, input_layout: str):
    """Quantized avg_pool2d schedule"""
    if output_layout == "nhwc-8h8w32c-2d" or output_layout == "nchw-8h8w32c-2d":
        return schedule_8h8w32c(outs, ins, output_layout, input_layout)
    if output_layout == "n11c-2048c-2d" or output_layout == "nc11-2048c-2d":
        return schedule_2048c(outs, ins, output_layout, input_layout)

    raise RuntimeError(f"Unexpected layout '{output_layout}'")
