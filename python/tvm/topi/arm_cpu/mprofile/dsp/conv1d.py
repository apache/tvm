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
"""Direct implementation of conv1d."""
from tvm import autotvm
from tvm.autotvm.task import deserialize_args
from tvm import te
from tvm.topi.utils import simplify, traverse_inline
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple1d
from tvm.tir.expr import Mul

from .micro_kernel.gemm import (
    intrin_gemm_MxKxN,
    gemm_MxKxN_impl,
)


def conv1d_nwc_dsp(*args, **kwargs):
    """Defines the v7e-m DSP instructions of conv1d on NWC layout."""
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    layout = args[-2]
    cfg = autotvm.get_config()
    args = [cfg] + args
    assert layout == "NWC"
    conv = conv1d_nwc_dsp_compute(*args)
    sched = conv1d_nwc_dsp_schedule(cfg, [data, kernel, conv])
    return sched, [data, kernel, conv]


conv1d_nwc_dsp.template_key = "dsp"
conv1d_nwc_dsp.default_data_layout = "NWC"
conv1d_nwc_dsp.default_kernel_layout = "WOI"


def conv1d_nwc_dsp_compute(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute function for v7e-m DSP instructions of conv1d on NWC layout."""
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    batch_size, data_width, in_channels = data.shape
    kernel_size, out_channels, _ = kernel.shape

    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_channels = simplify(out_channels)
    out_width = simplify((data_width - dilated_kernel_size + pad_left + pad_right) // strides + 1)

    # Apply padding
    pad_before = [0, pad_left, 0]
    pad_after = [0, pad_right, 0]
    padded_data = pad(data, pad_before, pad_after, name="padded_data")

    # Compute graph
    rc = te.reduce_axis((0, in_channels), name="rc")
    rw = te.reduce_axis((0, kernel_size), name="rw")

    conv = te.compute(
        (batch_size, out_width, out_channels),
        lambda b, w, c: te.sum(
            padded_data[b, w * strides + rw * dilation, rc].astype(out_dtype)
            * kernel[rw, c, rc].astype(out_dtype),
            axis=[rw, rc],
        ),
        name="conv1d",
        tag="conv1d_nwc",
    )

    ###########################
    # Config Space Definition #
    ###########################
    n, ow, co = (
        cfg.axis(batch_size.value),
        cfg.axis(out_width.value),
        cfg.axis(out_channels.value),
    )
    kw, ci = (
        cfg.reduce_axis(kernel_size.value),
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
        [n, owo, owi, coo, coi, kw, cio, cii],
        policy="candidate",
        candidate=[
            [n, kw, owo, coo, cio, owi, coi, cii],
            [n, kw, coo, owo, cio, owi, coi, cii],
            [n, kw, owo, coo, cio, owi, coi, cii],
            [n, kw, coo, owo, cio, owi, coi, cii],
        ],
    )

    cfg.define_knob("auto_unroll_max_step", [0, 2, 4, 8, 16, 32])
    cfg.define_knob("unroll_explicit", [0, 1])

    if cfg.is_fallback:
        cfg.fallback_split("tile_ow", [-1, out_width.value])
        cfg.fallback_split("tile_ci", [-1, in_channels.value])
        cfg.fallback_split("tile_co", [-1, out_channels.value])

    return conv


def conv1d_nwc_dsp_schedule(cfg, outs):
    """Schedule function for v7e-m DSP instructions of conv1d on NWC layout."""
    sched = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv1d_nwc" not in op.tag:
            return

        # extract tensors
        output = op.output(0)
        conv = op
        data_vec = conv.input_tensors[0]

        source_index_w = output.op.body[0].source[0].a.value.indices[1].a
        stride_w = source_index_w.b.value if isinstance(source_index_w, Mul) else 1

        # tile reduction axes
        n, ow, co = sched[conv].op.axis
        kw, ci = sched[conv].op.reduce_axis

        M = cfg["tile_ow"].size[-1]
        K = cfg["tile_ci"].size[-1]
        N = cfg["tile_co"].size[-1]

        owo, owi = cfg["tile_ow"].apply(sched, conv, ow)
        cio, cii = cfg["tile_ci"].apply(sched, conv, ci)
        coo, coi = cfg["tile_co"].apply(sched, conv, co)

        cfg["reorder_0_simd"].apply(sched, conv, [n, owo, owi, coo, coi, kw, cio, cii])

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
