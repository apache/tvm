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
"""Correlation operators on CUDA"""
import tvm
from tvm import te
from tvm import autotvm

from .. import nn
from ..util import traverse_inline


@autotvm.register_topi_compute("correlation_nchw.cuda")
def correlation_nchw(cfg, data1, data2, kernel_size, max_displacement, stride1, stride2, padding,
                     is_multiply):
    """Correlation operator in NCHW layout.

    Parameters
    ----------
    data1 : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]

    data2 : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]

    kernel_size: int
        Kernel size for correlation, must be an odd number

    max_displacement: int
        Max displacement of Correlation

    stride1: int
        Stride for data1

    stride2: int
        Stride for data2 within the neightborhood centered around data1

    padding : int or a list/tuple of 2 or 4 ints
        Padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    is_multiply: bocorrelation
        operation type is either multiplication or substraction

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # pylint: disable=unused-argument
    return nn.correlation_nchw(data1, data2, kernel_size, max_displacement, stride1, stride2,
                               padding, is_multiply)


def _schedule_correlation_nchw(cfg, s, correlation):
    """Schedule correlation_nchw direct template"""
    # pylint: disable=invalid-name
    ##### space definition begin #####
    n, f, y, x = s[correlation].op.axis
    rc, ry, rx = s[correlation].op.reduce_axis
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.id.name in ['nvptx', 'rocm']:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    ##### space definition end #####

    padded_data1, padded_data2 = s[correlation].op.input_tensors
    s[padded_data1].compute_inline()
    s[padded_data2].compute_inline()

    # create cache stage
    s[correlation].set_scope('local')
    AA = s.cache_read(padded_data1, 'shared', [correlation])
    BB = s.cache_read(padded_data2, 'shared', [correlation])

    output = s.outputs[0].output(0)

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[correlation].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[correlation].op.axis
    rc, ry, rx = s[correlation].op.reduce_axis
    rco, rci = cfg['tile_rc'].apply(s, correlation, rc)
    ryo, ryi = cfg['tile_ry'].apply(s, correlation, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, correlation, rx)
    s[correlation].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[correlation], rxo)
    s[BB].compute_at(s[correlation], rxo)

    # cooperative fetching
    for load in [AA, BB]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)


@autotvm.register_topi_schedule("correlation_nchw.cuda")
def schedule_correlation_nchw(cfg, outs):
    """schedule of correlation_nchw for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of correlation
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for correlation.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'correlation_nchw':
            _schedule_correlation_nchw(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
