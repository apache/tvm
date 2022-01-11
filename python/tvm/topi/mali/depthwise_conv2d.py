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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""depthwise_conv2d schedule on ARM Mali GPU"""

import tvm
from tvm import te
from tvm import autotvm

from .. import nn
from ..utils import traverse_inline

# register original implementation of depthwise_conv2d_nchw since we don't need to change this part
@autotvm.register_topi_compute("depthwise_conv2d_nchw.mali")
def depthwise_conv2d_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype):
    return nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


# register customized schedule for Mali.
@autotvm.register_topi_schedule("depthwise_conv2d_nchw.mali")
def schedule_depthwise_conv2d_nchw(cfg, outs):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """traverse to find op to schedule"""
        # schedule depthwise_conv2d
        if op.tag == "depthwise_conv2d_nchw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)
            _schedule(cfg, s, pad_data, kernel, conv, "NCHW")

    traverse_inline(s, outs[0].op, _callback)
    return s


# register original implementation of depthwise_conv2d_nhwc since we don't need to change this part
@autotvm.register_topi_compute("depthwise_conv2d_nhwc.mali")
def depthwise_conv2d_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype):
    return nn.depthwise_conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)


# register customized schedule for Mali.
@autotvm.register_topi_schedule("depthwise_conv2d_nhwc.mali")
def schedule_depthwise_conv2d_nhwc(cfg, outs):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """traverse to find op to schedule"""
        # schedule depthwise_conv2d
        if op.tag == "depthwise_conv2d_nhwc":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)
            _schedule(cfg, s, pad_data, kernel, conv, "NHWC")

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule(cfg, s, pad_data, kernel, conv, layout):
    """schedule depthwise_conv2d"""
    assert layout in ("NCHW", "NHWC")

    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]

    ##### space definition begin #####
    if layout == "NCHW":
        n, c, h, w = s[conv].op.axis
    else:
        n, h, w, c = s[conv].op.axis

    bc, tc, ci = cfg.define_split("tile_c", c, num_outputs=3)
    bh, th, hi = cfg.define_split("tile_y", h, num_outputs=3)
    bw, tw, wi = cfg.define_split("tile_x", w, num_outputs=3)
    cfg.define_annotate("ann_spatial", [ci, hi, wi], policy="try_unroll_vec")

    # fallback support
    if cfg.is_fallback:
        if layout == "NCHW":
            ref_log = autotvm.tophub.load_reference_log(
                "mali", "rk3399", "depthwise_conv2d_nchw.mali"
            )
            cfg.fallback_with_reference_log(ref_log)
        else:
            cfg.fallback_split("tile_c", [-1, 4, 2])
            cfg.fallback_split("tile_y", [-1, 4, 2])
            cfg.fallback_split("tile_x", [-1, 4, 2])
    ###### space definition end ######

    # schedule padding
    if layout == "NCHW":
        n, c, h, w = s[pad_data].op.axis
        z, y, x = c, h, w
        z_factor, y_factor, x_factor = cfg["tile_c"].size[1], 1, 1
    else:
        n, h, w, c = s[pad_data].op.axis
        z, y, x = h, w, c
        z_factor, y_factor, x_factor = 1, 1, cfg["tile_c"].size[1]
    tile_and_bind3d(s, pad_data, z, y, x, z_factor, y_factor, x_factor)

    # schedule dilation
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    # schedule conv
    if conv.op not in s.outputs:
        s[conv].set_scope("local")
        OL = conv
        output = s.outputs[0].output(0)
    else:
        OL = s.cache_write(conv, "local")
        output = conv

    if layout == "NCHW":
        n, c, h, w = s[output].op.axis
    else:
        n, h, w, c = s[output].op.axis

    bc, tc, ci = cfg["tile_c"].apply(s, output, c)
    bh, th, hi = cfg["tile_y"].apply(s, output, h)
    bw, tw, wi = cfg["tile_x"].apply(s, output, w)

    if layout == "NCHW":
        bz, tz, by, ty, bx, tx = bc, tc, bh, th, bw, tw
    else:
        bz, tz, by, ty, bx, tx = bh, th, bw, tw, bc, tc

    bz = s[output].fuse(n, bz)
    s[output].bind(bz, te.thread_axis("blockIdx.z"))
    s[output].bind(tz, te.thread_axis("threadIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))

    di, dj = s[OL].op.reduce_axis
    s[OL].unroll(di)
    s[OL].unroll(dj)

    s[OL].compute_at(s[output], tx)

    if layout == "NCHW":
        n, ci, hi, wi = s[OL].op.axis
    else:
        n, hi, wi, ci = s[OL].op.axis

    cfg["ann_spatial"].apply(
        s,
        OL,
        [ci, hi, wi],
        axis_lens=[cfg["tile_c"].size[2], cfg["tile_y"].size[2], cfg["tile_x"].size[2]],
        max_unroll=max_unroll,
        vec_size=vec_size,
        cfg=cfg,
    )


def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """tile and bind 3d"""
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, te.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, te.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
    return zo, zi, yo, yi, xo, xi
