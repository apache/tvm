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

""" Compute and schedule for quantized avg_pool2d op

Please note the following assumptions made by the implementation:

1) The input must be padded in advance to account for 'padding'. In addition,
   both input and output must be padded as per the physical buffer layout.
2) The current implementation assumes 'count_include_pad' to be 'True'. It can be
   modified to support 'False' case but the element count for the pooling window
   must be pre-computed and provided as an input to reduce the run-time overhead.
3) 'padding' is ignored. It must be handled outside of the sliced op.
4) Please note that this implementation will not work if the output includes any
   physical layout related padding as it can result into out-of-bound access
   for the input.
"""

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn, get_fixed_point_value


def validate_out_shape(out_shape, in_shape, kernel, stride, dilation):
    """Validate output shape"""
    _, oh, ow, _ = out_shape
    _, ih, iw, _ = in_shape
    kh, kw = kernel
    sh, sw = stride
    dh, dw = dilation
    if ih < (oh - 1) * sh + dh * (kh - 1) + 1:
        raise RuntimeError("Output height is too large")
    if iw < (ow - 1) * sw + dw * (kw - 1) + 1:
        raise RuntimeError("Output width is too large")


def saturate(x, dtype):
    """Saturate value for the specified data type"""
    if dtype == "uint8":
        return te.max(0, te.min(x, 255))
    elif dtype == "int8":
        return te.max(-127, te.min(x, 128))
    return x


def qnn_avg_pool2d_compute(
    data,
    kernel,
    stride,
    dilation,
    oshape,
    odtype,
    # quantization params:
    input_zero_point,
    input_scale,
    output_zero_point,
    output_scale,
):
    """Compute for quantized avg_pool2d"""
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")
    ob, oh, ow, oc = oshape
    if isinstance(ob, int):
        validate_out_shape(oshape, data.shape, kernel, stride, dilation)

    if odtype == "uint8":
        temp_dtype = "uint16"
    elif odtype == "int8":
        temp_dtype = "int16"
    else:
        raise RuntimeError(f"Unsupported output dtype, {odtype}'")

    sh, sw = stride
    dh, dw = dilation

    PoolArea = kh * kw

    scale = input_scale / output_scale
    scale_fixed_point, rsh = get_fixed_point_value(scale, "int16")
    scale_with_area = scale_fixed_point // PoolArea
    corr = (output_zero_point << rsh) - input_zero_point * scale_fixed_point

    Sum = te.compute(
        oshape,
        lambda b, h, w, c: te.sum(
            data[b, h * sh + dh * rh, w * sw + dw * rw, c].astype(temp_dtype), axis=[rh, rw]
        ),
        name="sum",
    )

    Avg = te.compute(
        oshape,
        lambda b, h, w, c: saturate(
            ((Sum[b, h, w, c] * scale_with_area) + corr) >> rsh, odtype
        ).astype(odtype),
        name="avg",
    )
    return Avg


def schedule_nhwc_8h8w32c(outs, ins, output_layout: str, input_layout: str):
    """Schedule for input and output layout nhwc-8h8w32c"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("sum")
    Avg = s.get_block("avg")

    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Sum, ("read", 0), input_transform_fn)
    s.transform_layout(Avg, ("write", 0), output_transform_fn)

    # Schedule 'Avg'
    n, h, w, c = s.get_loops(Avg)
    ho, hi = s.split(h, [None, 8])
    wo, wi = s.split(w, [None, 8])
    wio, wii = s.split(wi, [None, 4])
    co, ci = s.split(c, [None, 32])
    s.reorder(n, ho, wo, co, hi, wio, wii, ci)
    wii_ci = s.fuse(wii, ci)
    s.vectorize(wii_ci)

    # Schedule 'Sum'
    s.compute_at(Sum, wio)
    Sum_axis = s.get_loops(Sum)
    s.reorder(Sum_axis[-2], Sum_axis[-1], Sum_axis[-4], Sum_axis[-3])
    ci_wii = s.fuse(Sum_axis[-4], Sum_axis[-3])
    # s.vectorize(ci_wii) # Doesn't work
    return s


def schedule_n11c_2048c(outs, ins, output_layout: str, input_layout: str):
    """Schedule for output layout: n11c-2048c, input layout: nhwc-8h8w32c"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)
    Sum = s.get_block("sum")
    Avg = s.get_block("avg")

    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Sum, ("read", 0), input_transform_fn)
    s.transform_layout(Avg, ("write", 0), output_transform_fn)

    # Schedule 'Avg'
    n, h, w, c = s.get_loops(Avg)
    co, ci = s.split(c, [None, 2048])
    cio, cii = s.split(ci, [None, 128])
    s.vectorize(cii)

    # Schedule 'Sum'
    s.compute_at(Sum, cio)
    Sum_axis = s.get_loops(Sum)
    s.reorder(Sum_axis[-2], Sum_axis[-1], Sum_axis[-3])
    # s.vectorize(Sum_axis[-3]) # Doesn't work
    return s


def qnn_avg_pool2d_schedule(outs, ins, output_layout: str, input_layout: str):
    """Quantized avg_pool2d schedule"""
    if output_layout == "nhwc-8h8w32c-2d":
        return schedule_nhwc_8h8w32c(outs, ins, output_layout, input_layout)
    if output_layout == "n11c-2048c-2d":
        return schedule_n11c_2048c(outs, ins, output_layout, input_layout)
    raise RuntimeError(f"Unexpected layout '{output_layout}'")
