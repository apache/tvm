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
# pylint: disable=invalid-name, unused-argument
"""Compute definition for conv1d with cuda backend"""
import tvm
from tvm import te
from tvm import autotvm

from .. import nn
from ..utils import traverse_inline, get_const_tuple


@autotvm.register_topi_compute("conv1d_ncw.cuda")
def conv1d_ncw(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    return nn.conv1d_ncw(data, kernel, strides, padding, dilation, out_dtype)


def _schedule_conv1d_ncw(cfg, outs):
    """TOPI schedule callback of conv1d ncw for cuda gpu

    Parameters
    ----------
    cfg : ConfigEntity
        the config for this template.

    outs : Array of Tensor
        The computation graph description of conv1d
        in the format of an array of tensors.

    Returns
    -------
    s : Schedule
        The computation schedule for conv1d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_ncw" or op.tag == "group_conv1d_ncw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, x = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            cfg.define_split("tile_n", cfg.axis(n), num_outputs=4)
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.Target.current()
            if target.kind.name in ["nvptx", "rocm"]:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            ##### space definition end #####

            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, "local")
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope("local")
                OL = conv

            # create cache stage
            s[pad_data].set_scope("shared")
            AA = pad_data
            WW = s.cache_read(kernel, "shared", [OL])

            # tile and bind spatial axes
            n, f, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].reorder(bn, bf, bx, vn, vf, vx, tn, tf, tx, ni, fi, xi)
            s[output].bind(bn, te.thread_axis("blockIdx.z"))
            s[output].bind(bf, te.thread_axis("blockIdx.y"))
            s[output].bind(bx, te.thread_axis("blockIdx.x"))
            s[output].bind(vn, te.thread_axis("vthread"))
            s[output].bind(vf, te.thread_axis("vthread"))
            s[output].bind(vx, te.thread_axis("vthread"))

            s[output].bind(tx, te.thread_axis("threadIdx.x"))
            s[OL].compute_at(s[output], tx)
            # number of threads
            n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
            n_tx = cfg["tile_x"].size[2]

            # tile reduction axes
            n, f, x = s[OL].op.axis
            rc, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, rx, rci, n, f, x)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, f, x = s[load].op.axis
                fused = s[load].fuse(f, x)
                tz, fused = s[load].split(fused, nparts=n_tz)
                tx, fused = s[load].split(fused, nparts=n_tx)
                s[load].bind(tz, te.thread_axis("threadIdx.y"))
                s[load].bind(tx, te.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

            N, CO, OW = get_const_tuple(output.shape)
            _, CI, KW = get_const_tuple(kernel.shape)
            cfg.add_flop(2 * N * OW * CO * KW * CI)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_schedule("conv1d_ncw.cuda")
def schedule_conv1d_ncw(cfg, outs):
    return _schedule_conv1d_ncw(cfg, outs)


@autotvm.register_topi_compute("group_conv1d_ncw.cuda")
def group_conv1d_ncw(cfg, data, kernel, strides, padding, dilation, groups, out_dtype="float32"):
    return nn.group_conv1d_ncw(data, kernel, strides, padding, dilation, groups, out_dtype)


@autotvm.register_topi_schedule("group_conv1d_ncw.cuda")
def schedule_group_conv1d_ncw(cfg, outs):
    return _schedule_conv1d_ncw(cfg, outs)


@autotvm.register_topi_compute("conv1d_nwc.cuda")
def conv1d_nwc(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    return nn.conv1d_nwc(data, kernel, strides, padding, dilation, out_dtype)


def _schedule_conv1d_nwc(cfg, outs):
    """TOPI schedule callback of conv1d nwc for cuda gpu

    Parameters
    ----------
    cfg : ConfigEntity
        the config for this template.

    outs : Array of Tensor
        The computation graph description of conv1d
        in the format of an array of tensors.

    Returns
    -------
    s : Schedule
        The computation schedule for conv1d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_nwc" or op.tag == "group_conv1d_nwc":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, x, f = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            cfg.define_split("tile_n", cfg.axis(n), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.Target.current()
            if target.kind.name in ["nvptx", "rocm"]:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            ##### space definition end #####

            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, "local")
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope("local")
                OL = conv

            # create cache stage
            s[pad_data].set_scope("shared")
            AA = pad_data
            WW = s.cache_read(kernel, "shared", [OL])

            # tile and bind spatial axes
            n, f, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)

            s[output].reorder(bn, bx, bf, vn, vx, vf, tn, tx, tf, ni, xi, fi)
            s[output].bind(bn, te.thread_axis("blockIdx.z"))
            s[output].bind(bx, te.thread_axis("blockIdx.y"))
            s[output].bind(bf, te.thread_axis("blockIdx.x"))
            s[output].bind(vn, te.thread_axis("vthread"))
            s[output].bind(vx, te.thread_axis("vthread"))
            s[output].bind(vf, te.thread_axis("vthread"))

            s[output].bind(tf, te.thread_axis("threadIdx.x"))
            s[OL].compute_at(s[output], tf)
            # number of threads
            n_tz = cfg["tile_n"].size[2] * cfg["tile_x"].size[2]
            n_tx = cfg["tile_f"].size[2]

            # tile reduction axes
            n, x, f = s[OL].op.axis
            rc, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, rx, rci, n, x, f)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, x, f = s[load].op.axis
                fused = s[load].fuse(x, f)
                tz, fused = s[load].split(fused, nparts=n_tz)
                tx, fused = s[load].split(fused, nparts=n_tx)
                s[load].bind(tz, te.thread_axis("threadIdx.y"))
                s[load].bind(tx, te.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

            N, OW, CO = get_const_tuple(output.shape)
            KW, CI, _ = get_const_tuple(kernel.shape)
            cfg.add_flop(2 * N * OW * CO * KW * CI)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_schedule("conv1d_nwc.cuda")
def schedule_conv1d_nwc(cfg, outs):
    return _schedule_conv1d_nwc(cfg, outs)


@autotvm.register_topi_compute("group_conv1d_nwc.cuda")
def group_conv1d_nwc(cfg, data, kernel, strides, padding, dilation, groups, out_dtype="float32"):
    return nn.group_conv1d_nwc(data, kernel, strides, padding, dilation, groups, out_dtype)


@autotvm.register_topi_schedule("group_conv1d_nwc.cuda")
def schedule_group_conv1d_nwc(cfg, outs):
    return _schedule_conv1d_nwc(cfg, outs)
