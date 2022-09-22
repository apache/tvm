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

""" Compute and schedule for avg_pool2d slice op

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
from ..utils import get_layout_transform_fn


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


def avg_pool2d_compute(A, kernel, stride, dilation, oshape, odtype="float16"):
    """avg_pool2d compute"""
    if odtype != "float16":
        RuntimeError(f"Unsupported output dtype '{odtype}'")
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")
    ob, oh, ow, oc = oshape
    if isinstance(ob, int):
        validate_out_shape(oshape, A.shape, kernel, stride, dilation)

    sh, sw = stride
    dh, dw = dilation
    InvArea = float(1) / (kh * kw)

    Sum = te.compute(
        oshape,
        lambda b, h, w, c: te.sum(
            A[b, h * sh + dh * rh, w * sw + dw * rw, c].astype("float32"), axis=[rh, rw]
        ),
        name="sum",
    )
    Avg = te.compute(
        oshape, lambda b, h, w, c: (Sum[b, h, w, c] * InvArea).astype(A.dtype), name="avg"
    )
    return Avg


def schedule_nhwc_8h2w32c2w(outs, ins, output_layout: str, input_layout: str):
    """Schedule for input and output layout nhwc-8h2w32c2w"""
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
    wo, wi = s.split(w, [None, 4])
    wio, wii = s.split(wi, [None, 2])
    co, ci = s.split(c, [None, 32])
    s.reorder(n, ho, wo, co, hi, wio, ci, wii)
    ci_wii = s.fuse(ci, wii)
    s.vectorize(ci_wii)

    # Schedule 'Sum'
    s.compute_at(Sum, wio)
    Sum_axis = s.get_loops(Sum)
    s.reorder(Sum_axis[-2], Sum_axis[-1], Sum_axis[-4], Sum_axis[-3])
    ci_wii = s.fuse(Sum_axis[-4], Sum_axis[-3])
    # s.vectorize(ci_wii) # Doesn't work
    return s


def schedule_n11c_1024c(outs, ins, output_layout: str, input_layout: str):
    """Schedule for output layout: n11c-1024c, input layout: nhwc-8h2w32c2w"""
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
    co, ci = s.split(c, [None, 1024])
    cio, cii = s.split(ci, [None, 64])
    s.vectorize(cii)

    # Schedule 'Sum'
    s.compute_at(Sum, cio)
    Sum_axis = s.get_loops(Sum)
    s.reorder(Sum_axis[-2], Sum_axis[-1], Sum_axis[-3])
    # s.vectorize(Sum_axis[-3]) # Doesn't work
    return s


def avg_pool2d_schedule(outs, ins, output_layout: str, input_layout: str):
    """avg_pool2d schedule"""
    if output_layout == "nhwc-8h2w32c2w-2d":
        return schedule_nhwc_8h2w32c2w(outs, ins, output_layout, input_layout)
    if output_layout == "n11c-1024c-2d":
        return schedule_n11c_1024c(outs, ins, output_layout, input_layout)
    raise RuntimeError(f"Unexpected layout '{output_layout}'")
