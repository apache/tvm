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
# pylint: disable=invalid-name, no-value-for-parameter
"""Direct implementation of conv2d."""

from tvm import autotvm
from tvm.autotvm.task import deserialize_args
from tvm import te
from tvm.topi.utils import simplify, traverse_inline
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from tvm.tir.expr import Mul

from .micro_kernel.gemm import (
    intrin_gemm_MxKxN,
    gemm_MxKxN_impl,
)


def conv2d_nhwc_dsp(*args, **kwargs):
    """Defines the v7e-m DSP instructions of conv2d."""
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    layout = args[-2]
    cfg = autotvm.get_config()
    args = [cfg] + args
    assert layout == "NHWC"
    conv = conv2d_nhwc_dsp_compute(*args)
    sched = conv2d_nhwc_dsp_schedule(cfg, [data, kernel, conv])
    return sched, [data, kernel, conv]


conv2d_nhwc_dsp.template_key = "dsp"
conv2d_nhwc_dsp.default_data_layout = "NHWC"
conv2d_nhwc_dsp.default_kernel_layout = "HWOI"


def conv2d_nhwc_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute function for v7e-m DSP instructions of conv2d."""
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch_size, in_height, in_width, in_channels = data.shape
    kernel_h, kernel_w, out_channels, _ = kernel.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    padded_data = pad(data, pad_before, pad_after, name="padded_data")

    rc = te.reduce_axis((0, in_channels), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    conv = te.compute(
        (batch_size, out_height, out_width, out_channels),
        lambda nn, yy, xx, ff: te.sum(
            padded_data[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * kernel[ry, rx, ff, rc].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="conv2d",
        tag="conv2d_nhwc",
    )

    ###########################
    # Config Space Definition #
    ###########################
    n, oh, ow, co = (
        cfg.axis(batch_size.value),
        cfg.axis(out_height.value),
        cfg.axis(out_width.value),
        cfg.axis(out_channels.value),
    )
    kh, kw, ci = (
        cfg.reduce_axis(kernel_h.value),
        cfg.reduce_axis(kernel_w.value),
        cfg.reduce_axis(in_channels.value),
    )

    owo, owi = cfg.define_split("tile_ow", ow, policy="factors", num_outputs=2)
    cio, cii = cfg.define_split(
        "tile_ci",
        ci,
        policy="factors",
        num_outputs=2,
        # TODO: check case with in_channels.value % 4 != 0 with AutoTVM
        filter=None if cfg.is_fallback else lambda x: x.size[-1] % 4 == 0,
    )
    coo, coi = cfg.define_split("tile_co", co, policy="factors", num_outputs=2)

    cfg.define_reorder(
        "reorder_0_simd",
        [n, oh, owo, owi, coo, coi, kh, kw, cio, cii],
        policy="candidate",
        candidate=[
            [n, oh, kh, kw, owo, coo, cio, owi, coi, cii],
            [n, oh, kh, kw, coo, owo, cio, owi, coi, cii],
            [n, kh, kw, oh, owo, coo, cio, owi, coi, cii],
            [n, kh, kw, oh, coo, owo, cio, owi, coi, cii],
        ],
    )

    cfg.define_knob("auto_unroll_max_step", [0, 2, 4, 8, 16, 32])
    cfg.define_knob("unroll_explicit", [0, 1])

    if cfg.is_fallback:
        cfg.fallback_split("tile_ow", [-1, out_width.value])
        cfg.fallback_split("tile_ci", [-1, in_channels.value])
        cfg.fallback_split("tile_co", [-1, out_channels.value])

    return conv


def conv2d_nhwc_dsp_schedule(cfg, outs):
    """Schedule function for v7e-m DSP instructions of conv2d."""
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc" not in op.tag:
            return

        # extract tensors
        output = op.output(0)
        conv = op
        data_vec = conv.input_tensors[0]
        kernel = conv.input_tensors[1]  # pylint: disable=unused-variable
        last = outs[0]  # pylint: disable=unused-variable

        source_index_w = output.op.body[0].source[0].a.value.indices[2].a
        stride_w = source_index_w.b.value if isinstance(source_index_w, Mul) else 1

        # tile reduction axes
        n, oh, ow, co = sched[conv].op.axis
        kh, kw, ci = sched[conv].op.reduce_axis

        M = cfg["tile_ow"].size[-1]
        K = cfg["tile_ci"].size[-1]
        N = cfg["tile_co"].size[-1]

        owo, owi = cfg["tile_ow"].apply(sched, conv, ow)
        cio, cii = cfg["tile_ci"].apply(sched, conv, ci)
        coo, coi = cfg["tile_co"].apply(sched, conv, co)

        cfg["reorder_0_simd"].apply(sched, conv, [n, oh, owo, owi, coo, coi, kh, kw, cio, cii])

        gemm, uniq_id = intrin_gemm_MxKxN(M, K, N, data_vec.dtype, output.dtype, stride_w)
        sched[output].tensorize(owi, gemm)
        sched[output].pragma(n, "import_c", gemm_MxKxN_impl(M, K, N, uniq_id))

        # this is the scope to attach global config inside this kernel
        kernel_scope = n

        # tune unroll
        sched[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        sched[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    traverse_inline(sched, outs[-1].op, _callback)
    return sched
