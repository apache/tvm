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
# pylint: disable=invalid-name, unused-variable, unused-argument, too-many-locals, condition-evals-to-constant

""" Compute and schedule for max_pool2d slice op

Please note the following assumptions made by the implementation:

1) The input must be padded in advance to account for 'padding'. In addition,
   both input and output must be padded as per the physical buffer layout.

2) The current implementation assumes 'count_include_pad' to be 'True'. It can be
   modified to support 'False' case but the element count for the pooling window
   must be pre-computed and provided as an input to reduce the run-time overhead.

3) 'padding' is ignored. It must be handled outside of the sliced op.

4) This implementation will not work if the output includes any physical layout
   related padding, as it can result into out-of-bound access for the input.
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


def max_pool2d_compute(A, out_shape, kernel, stride, dilation):
    """max_pool2d compute"""
    kh, kw = kernel
    rh = te.reduce_axis((0, kh), name="rh")
    rw = te.reduce_axis((0, kw), name="rw")
    ob, oh, ow, oc = out_shape
    if isinstance(ob, int):
        validate_out_shape(out_shape, A.shape, kernel, stride, dilation)

    sh, sw = stride
    dh, dw = dilation

    Max = te.compute(
        out_shape,
        lambda b, h, w, c: te.max(
            A[b, h * sh + dh * rh, w * sw + dw * rw, c].astype(A.dtype), axis=[rh, rw]
        ),
        name="max",
    )
    return Max


def STIR_schedule_nhwc_8h2w32c2w_nhwc_8h8w32c(
    outs: te.Tensor, ins: te.Tensor, output_layout: str, input_layout: str
):
    """Schedule for input and output layout nhwc-8h2w32c2w and nhwc-8h8w32c"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)

    # NOTE!!! This scheduling logic is a work in progress.
    # It is not known to ultimately result in near-optimal Hexagon performance.
    # The schedule below strives to implement these heuristics:
    #
    # (1) For mathematical operations on tensor values, prefer HVX SIMD operations
    #     over per-element scalar operations.
    #
    # (2) Minimize the number of memory transfers used to operate on tensor values:
    #     host-memory <--> Hexagon DDR <--> VTCM <--> HVX registers
    #
    # As a consequence of (1) + (2), prefer TIR schedules that load each value
    # into an HVX SIMD tensor exactly once.

    Max = s.get_block("max")

    if input_layout in (
        "nhwc-8h2w32c2w-2d",
        "nhwc-8h8w32c-2d",
    ):
        input_transform_fn = get_layout_transform_fn(input_layout)
        s.transform_layout(Max, ("read", 0), input_transform_fn)

    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Max, ("write", 0), output_transform_fn)

    # pylint: disable=line-too-long
    #
    # Restructure the loop nestings to have this overall structure:
    # (loop over different 128-byte output-tensor chunks) : n, ho, wo, co   }- the first level of a two-level tensor layout
    #    (loop within one 128-byte output-tensor chunk) : hi, wio, ci, wii  }- the second level of a two-level tensor layout
    #        (loop over reduction axes) : rh, rw                            }- loop over multiple elements of the input tensor
    #
    # Note: This schedule is a work in progress.  We *expect* that it's
    # crucially important for the loops to have this relative ordering:
    #    n ... ho ... wo ... co ... hi ... wio ... ci ... wii
    # because it lets us visit each of the 128-byte output chunks precisely once.

    (
        n,
        h,
        w,
        c,
        rh,
        rw,
    ) = s.get_loops(Max)

    # Restructure the loops from NHWC to nhwc_8h2w32c2w or nhwc_8h8w32c, with loops for 'max's reduction
    # axes at the very end.
    # nhwc_8h2w32c2w layout is for float16 and nhwc-8h8w32c-2d layout is for uint8/int8
    if output_layout == "nhwc-8h2w32c2w-2d":
        ho, hi = s.split(h, [None, 8])
        wo, wi = s.split(w, [None, 4])
        wio, wii = s.split(wi, [None, 2])
        co, ci = s.split(c, [None, 32])
        s.reorder(n, ho, wo, co, hi, wio, ci, wii, rh, rw)
    elif output_layout == "nhwc-8h8w32c-2d":
        ho, hi = s.split(h, [None, 8])
        wo, wi = s.split(w, [None, 8])
        co, ci = s.split(c, [None, 32])

        s.reorder(n, ho, wo, co, hi, wi, ci, rh, rw)

    # TODO: Enable vectorization.
    # Hexagon v69's HVX units support SIMD operations on 64-element float16 vectors.
    #
    # TVM's 'vectorize' schedule primitive is the idiomatic way to encourage lower layers of the
    # compiler to generate this kind of SIMD object code.
    #
    # Several requirements must be met to use 'vectorize':
    #
    # 1) It can only be applied to a schedule's innermost loop variable.
    #
    # 2) Any block-iterator(s) bound to that innermost loop variable must be
    #    *data-parallel* block iterators.
    #
    # 3) Ideally, the innermost loop variable will iterate only over the output
    #    tensor's fastest-changing indices and nothing else.  But in our case,
    #    our two innermost loops correspond to the max operator's reduction axes.
    #
    # Finding a good way to satisfy all of these requirements at the same time is
    # left for future work.

    # ci_wii = s.fuse(ci, wii)
    # s.vectorize(ci_wii_rh_rw)

    return s


def STIR_schedule_n11c(outs, ins, output_layout: str, input_layout: str):
    """Schedule for output layout: n11c-1024c, n11c-2048c-2d;"""

    # NOTE: This function is a variation of the STIR_schedule_maxpool2d
    # functions.  Most of that function's code comments apply to this function
    # as well, but are ommited for brevity.

    # NOTE: the "n11c-1024c" output layout is shorthand for this axis mapping:
    # [n, h, w, c // 1024, te.AXIS_SEPARATOR, c % 1024]
    func = te.create_prim_func([ins, outs])

    s = tir.Schedule(func)
    Max = s.get_block("max")

    input_transform_fn = get_layout_transform_fn(input_layout)
    output_transform_fn = get_layout_transform_fn(output_layout)
    s.transform_layout(Max, ("read", 0), input_transform_fn)
    s.transform_layout(Max, ("write", 0), output_transform_fn)

    (
        n,
        h,
        w,
        c,
        rh,
        rw,
    ) = s.get_loops(Max)
    if output_layout == "n11c-1024c-2d":
        co, ci = s.split(c, [None, 1024])
    else:
        co, ci = s.split(c, [None, 2048])
    # s.vectorize(ci)

    return s


def max_pool2d_STIR_schedule(outs, ins, output_layout: str, input_layout: str):
    """STIR based schedule"""
    if output_layout == "nhwc-8h2w32c2w-2d" or "nhwc-8h8w32c-2d":
        return STIR_schedule_nhwc_8h2w32c2w_nhwc_8h8w32c(outs, ins, output_layout, input_layout)
    if output_layout == "n11c-1024c-2d" or "n11c-2048c-2d":
        return STIR_schedule_n11c(outs, ins, output_layout, input_layout)
    raise RuntimeError(f"Unexpected layout '{output_layout}'")
