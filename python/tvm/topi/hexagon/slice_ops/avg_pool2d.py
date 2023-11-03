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
# pylint: disable=invalid-name, unused-variable, unused-argument, too-many-locals, pointless-exception-statement

""" Compute and schedule for avg_pool2d slice op """

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn
from ...utils import get_const_tuple
from ...nn.utils import get_pad_tuple
from ...nn.pad import pad
from ..compute_poolarea import compute_PoolArea


def avg_pool2d_NCHW(
    data, kernel, stride, padding, dilation, count_include_pad, oshape, odtype="float16"
):
    """avg_pool2d compute"""
    if odtype != "float16":
        raise RuntimeError(f"Unsupported output dtype '{odtype}'")
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")
    sh, sw = stride
    dh, dw = dilation

    dilated_kh = (kh - 1) * dh + 1
    dilated_kw = (kw - 1) * dw + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        get_const_tuple(padding), (dilated_kh, dilated_kw)
    )

    # DOPAD

    if pad_top != 0 or pad_down != 0 or pad_left != 0 or pad_right != 0:
        pad_before = (0, 0, pad_top, pad_left)
        pad_after = (0, 0, pad_down, pad_right)
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        # By definition when True, zero-padding will be included in the averaging calculation
        # This is equivalent to PoolArea = (kh * kw)
        count_include_pad = True
        data_pad = data

    Sum = te.compute(
        oshape,
        lambda b, c, h, w: te.sum(
            data_pad[b, c, h * sh + dh * rh, w * sw + dw * rw].astype("float32"), axis=[rh, rw]
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

        InvArea = te.compute(
            (oh, ow),
            lambda i, j: tir.if_then_else(
                tir.all(PoolArea[i, j] > 0), (float(1) / PoolArea[i, j]), 0
            ),
            name="inverse_area",
        )

        Avg = te.compute(
            oshape,
            lambda b, c, h, w: (Sum[b, c, h, w] * InvArea[h, w]).astype(odtype),
            name="pool_avg",
        )
    else:
        InvArea = float(1) / (kh * kw)
        Avg = te.compute(
            oshape, lambda b, c, h, w: (Sum[b, c, h, w] * InvArea).astype(odtype), name="pool_avg"
        )

    return Avg


def avg_pool2d_NHWC(
    data, kernel, stride, padding, dilation, count_include_pad, oshape, odtype="float16"
):
    """avg_pool2d compute"""
    if odtype != "float16":
        raise RuntimeError(f"Unsupported output dtype '{odtype}'")
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")

    sh, sw = stride
    dh, dw = dilation
    InvArea = float(1) / (kh * kw)

    dilated_kh = (kh - 1) * dh + 1
    dilated_kw = (kw - 1) * dw + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        get_const_tuple(padding), (dilated_kh, dilated_kw)
    )

    # DOPAD
    if pad_top != 0 or pad_down != 0 or pad_left != 0 or pad_right != 0:
        pad_before = (0, pad_top, pad_left, 0)
        pad_after = (0, pad_down, pad_right, 0)
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        # By definition when True, zero-padding will be included in the averaging calculation
        # This is equivalent to PoolArea = (kh * kw)
        count_include_pad = True
        data_pad = data

    Sum = te.compute(
        oshape,
        lambda b, h, w, c: te.sum(
            data_pad[b, h * sh + dh * rh, w * sw + dw * rw, c].astype("float32"), axis=[rh, rw]
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

        InvArea = te.compute(
            (oh, ow),
            lambda i, j: tir.if_then_else(
                tir.all(PoolArea[i, j] > 0), (float(1) / PoolArea[i, j]), 0
            ),
            name="inverse_area",
        )

        Avg = te.compute(
            oshape,
            lambda b, h, w, c: (Sum[b, h, w, c] * InvArea[h, w]).astype(odtype),
            name="pool_avg",
        )
    else:
        InvArea = float(1) / (kh * kw)
        Avg = te.compute(
            oshape, lambda b, h, w, c: (Sum[b, h, w, c] * InvArea).astype(odtype), name="pool_avg"
        )

    return Avg


def schedule_8h2w32c2w(outs, ins, output_layout: str, input_layout: str):
    """Schedule for input and output layout 8h2w32c2w"""
    func = te.create_prim_func([ins, outs])
    print(func)
    s = tir.Schedule(func)
    Sum = s.get_block("pool_sum")
    Avg = s.get_block("pool_avg")

    mem_scope = "global.vtcm"
    sum_read = s.cache_read(Sum, 0, mem_scope)
    avg_write = s.cache_write(Avg, 0, mem_scope)
    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Sum, ("read", 0), input_transform_fn, pad_value=0.0)
    s.transform_layout(Avg, ("write", 0), output_transform_fn, pad_value=0.0)
    return s


def schedule_1024c(outs, ins, output_layout: str, input_layout: str):
    """Schedule for output layout: 1024c, input layout: 8h2w32c2w"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("pool_sum")
    Avg = s.get_block("pool_avg")

    mem_scope = "global.vtcm"
    sum_read = s.cache_read(Sum, 0, mem_scope)
    avg_write = s.cache_write(Avg, 0, mem_scope)
    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Sum, ("read", 0), input_transform_fn, pad_value=0.0)
    s.transform_layout(Avg, ("write", 0), output_transform_fn, pad_value=0.0)

    # Schedule 'Avg'
    if output_layout == "n11c-1024c-2d":
        n, h, w, c = s.get_loops(Avg)
    else:
        n, c, h, w = s.get_loops(Avg)
    _, ci = s.split(c, [None, 1024])
    cio, cii = s.split(ci, [None, 64])
    s.vectorize(cii)

    # Schedule 'Sum'
    Sum_axis = s.get_loops(Sum)
    s.reorder(Sum_axis[-2], Sum_axis[-1], Sum_axis[-3])
    return s


def avg_pool2d_schedule(outs, ins, output_layout: str, input_layout: str):
    """avg_pool2d schedule"""
    if output_layout == "nhwc-8h2w32c2w-2d" or output_layout == "nchw-8h2w32c2w-2d":
        return schedule_8h2w32c2w(outs, ins, output_layout, input_layout)
    if output_layout == "n11c-1024c-2d" or output_layout == "nc11-1024c-2d":
        return schedule_1024c(outs, ins, output_layout, input_layout)
    raise RuntimeError(f"Unexpected layout '{output_layout}'")
