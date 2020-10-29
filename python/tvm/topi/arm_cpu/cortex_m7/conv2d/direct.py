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
# pylint: disable=invalid-name
"""Direct implementation of conv2d."""

import tvm
from tvm import autotvm
from tvm.autotvm.task import deserialize_args
from tvm.topi.nn.conv2d import conv2d_nchw, conv2d_nhwc
from tvm.topi.utils import get_const_tuple, get_const_int, traverse_inline


def conv2d_direct(*args, **kwargs):
    """Schedule function for directly-scheduled conv2d."""
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    layout = args[-2]
    cfg = autotvm.get_config()
    args = [cfg] + args
    conv = conv2d_direct_compute(*args)
    if layout == "NHWC":
        sched = conv2d_direct_nhwc_schedule(cfg, [data, kernel, conv])
    elif layout == "NCHW":
        sched = conv2d_direct_nchw_schedule(cfg, [data, kernel, conv])
    else:
        raise RuntimeError(f'unsupported data layout "{layout}"')
    return sched, [data, kernel, conv]


conv2d_direct.template_key = "direct"
conv2d_direct.default_data_layout = "NHWC"
conv2d_direct.default_kernel_layout = "HWIO"


@autotvm.register_topi_compute("conv2d_direct.micro_dev")
def conv2d_direct_compute(*args):
    layout = args[-2]
    if layout == "NHWC":
        return _conv2d_direct_nhwc_compute(*args)
    if layout == "NCHW":
        return _conv2d_direct_nchw_compute(*args)

    raise RuntimeError(f'unsupported data layout "{layout}"')


def _conv2d_direct_nhwc_compute(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    assert layout == "NHWC"
    conv = conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)

    # Config Space Definition
    N, H, W, CI = get_const_tuple(data.shape)
    KH, KW, _, CO = get_const_tuple(kernel.shape)
    n, oh, ow, co = cfg.axis(N), cfg.axis(H), cfg.axis(W), cfg.axis(CO)
    kh, kw, ci = cfg.reduce_axis(KH), cfg.reduce_axis(KW), cfg.reduce_axis(CI)

    # TODO should we add a max_factor attr to these splits?
    co, vc = cfg.define_split("tile_co", co, num_outputs=2)
    oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
    ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)

    cfg.define_reorder(
        "reorder_0",
        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
        policy="candidate",
        candidate=[
            [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
            [n, co, oh, ow, ci, kh, kw, vc, vh, vw],
            [n, co, oh, ow, ci, vh, vw, vc, kh, kw],
            [n, co, oh, ow, ci, vc, vh, vw, kh, kw],
        ],
    )

    cfg.define_annotate("ann_reduce", [kh, kw], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy="try_unroll")

    cfg.define_knob("auto_unroll_max_step", [0, 2, 4, 8, 16, 32])
    cfg.define_knob("unroll_explicit", [0, 1])

    return conv


def _conv2d_direct_nchw_compute(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    assert layout == "NCHW"
    conv = conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)

    ###########################
    # Config Space Definition #
    ###########################
    cfg.define_knob("auto_unroll_max_step", [0, 2, 4, 8, 16, 32])
    cfg.define_knob("unroll_explicit", [0, 1])

    return conv


@autotvm.register_topi_schedule("conv2d_direct_nhwc.micro_dev")
def conv2d_direct_nhwc_schedule(cfg, outs):
    """Schedule function for directly-scheduled conv2d on NHWC layout."""
    sched = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc" not in op.tag:
            return

        ### extract tensors ###
        output = op.output(0)
        conv = op
        data_vec = conv.input_tensors[0]
        kernel = conv.input_tensors[1]  # pylint: disable=unused-variable
        last = outs[0]  # pylint: disable=unused-variable

        # tile reduction axes
        n, oh, ow, co = sched[conv].op.axis
        kh, kw, ci = sched[conv].op.reduce_axis
        # NOTE we can't inline data padding in the SIMD path, because it
        # introduces conditionals in the inner loop.
        data_pad = data_vec.op
        sched[data_pad].compute_inline()

        co, vc = cfg["tile_co"].apply(sched, conv, co)
        oh, vh = cfg["tile_oh"].apply(sched, conv, oh)
        ow, vw = cfg["tile_ow"].apply(sched, conv, ow)
        cfg["reorder_0"].apply(sched, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
        cfg["ann_reduce"].apply(
            sched,
            conv,
            [kh, kw],
            axis_lens=[get_const_int(kh.dom.extent), get_const_int(kw.dom.extent)],
            max_unroll=8,
            cfg=cfg,
        )
        cfg["ann_spatial"].apply(
            sched,
            conv,
            [vh, vw, vc],
            axis_lens=[cfg["tile_oh"].size[-1], cfg["tile_ow"].size[-1], cfg["tile_co"].size[-1]],
            max_unroll=8,
            cfg=cfg,
        )

        kernel_scope = n  # this is the scope to attach global config inside this kernel

        # tune unroll
        sched[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        sched[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    traverse_inline(sched, outs[-1].op, _callback)
    return sched


@autotvm.register_topi_schedule("conv2d_direct_nchw.micro_dev")
def conv2d_direct_nchw_schedule(cfg, outs):
    """Schedule function for Cortex-M7 direct implementation of conv2d."""
    # use default schedule
    sched = tvm.create_schedule([x.op for x in outs])

    conv = outs[-1].op
    output = conv.output(0)
    data_vec = conv.input_tensors[0]
    data_pad = data_vec.op
    sched[data_pad].compute_inline()

    # TODO add more schedule opts (similar to the NHWC template)

    n, _, _, _ = sched[conv].op.axis
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    # tune unroll
    sched[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    sched[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    return sched
