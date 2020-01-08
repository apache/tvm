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
from tvm import autotvm

from .. import nn, generic
from ..util import traverse_inline, get_const_tuple


@autotvm.register_topi_compute(nn.conv1d, ['cuda', 'gpu'], ['direct'])
def conv1d_cuda(cfg,
                data,
                kernel,
                strides,
                padding,
                dilation,
                layout='NCW',
                out_dtype='float32'):
    """ 1D convolution forward operator for cuda backend.

    Parameters
    ----------
    cfg : ConfigEntity
        The config for this template

    data : tvm.Tensor
        3-D input shape [batch, in_channel, in_width] for layout == 'NCW'
        and [batch, in_width, in_channel] for layout == 'NWC'

    kernel : tvm.Tensor
        3-D kernel with shape [num_filter, in_channel, filter_size] for layout == 'NCW'
        and [filter_size, in_channel, num_filter] for layout == 'NWC'

    strides : int or tuple
        The spatial stride along width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    layout : str
        How input data is laid out, must be one of ['NCW', 'NWC']

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    if layout == 'NCW':
        return nn.conv1d_ncw(data, kernel, strides, padding, dilation,
                             out_dtype)
    if layout == 'NWC':
        return nn.conv1d_nwc(data, kernel, strides, padding, dilation,
                             out_dtype)
    raise ValueError("This layout is not yet supported: {}".format(layout))


@autotvm.register_topi_schedule(generic.schedule_conv1d_ncw, ["cuda", "gpu"],
                                ["direct"])
def schedule_conv1d_ncw(cfg, outs):
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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv1d_ncw':
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

            target = tvm.target.current_target()
            if target.target_name in ['nvptx', 'rocm']:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            ##### space definition end #####

            if isinstance(kernel.op,
                          tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, 'local')
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope('local')
                OL = conv

            # create cache stage
            s[pad_data].set_scope('shared')
            AA = pad_data
            WW = s.cache_read(kernel, 'shared', [OL])

            # tile and bind spatial axes
            n, f, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].reorder(bn, bf, bx, vn, vf, vx, tn, tf, tx, ni, fi, xi)
            s[output].bind(bn, tvm.thread_axis("blockIdx.z"))
            s[output].bind(bf, tvm.thread_axis("blockIdx.y"))
            s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[output].bind(vn, tvm.thread_axis("vthread"))
            s[output].bind(vf, tvm.thread_axis("vthread"))
            s[output].bind(vx, tvm.thread_axis("vthread"))

            s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
            s[OL].compute_at(s[output], tx)
            # number of threads
            n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
            n_tx = cfg["tile_x"].size[2]

            # tile reduction axes
            n, f, x = s[OL].op.axis
            rc, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, rx, rci, n, f, x)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, f, x = s[load].op.axis
                fused = s[load].fuse(f, x)
                tz, fused = s[load].split(fused, nparts=n_tz)
                tx, fused = s[load].split(fused, nparts=n_tx)
                s[load].bind(tz, tvm.thread_axis("threadIdx.y"))
                s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                             cfg['auto_unroll_max_step'].val)
            s[output].pragma(kernel_scope, 'unroll_explicit',
                             cfg['unroll_explicit'].val)

            N, CO, OW = get_const_tuple(output.shape)
            _, CI, KW = get_const_tuple(kernel.shape)
            cfg.add_flop(2 * N * OW * CO * KW * CI)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_schedule(generic.schedule_conv1d_nwc, ["cuda", "gpu"],
                                ["direct"])
def schedule_conv1d_nwc(cfg, outs):
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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv1d_nwc':
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

            target = tvm.target.current_target()
            if target.target_name in ['nvptx', 'rocm']:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            ##### space definition end #####

            if isinstance(kernel.op,
                          tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, 'local')
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope('local')
                OL = conv

            # create cache stage
            s[pad_data].set_scope('shared')
            AA = pad_data
            WW = s.cache_read(kernel, 'shared', [OL])

            # tile and bind spatial axes
            n, f, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)

            s[output].reorder(bn, bx, bf, vn, vx, vf, tn, tx, tf, ni, xi, fi)
            s[output].bind(bn, tvm.thread_axis("blockIdx.z"))
            s[output].bind(bx, tvm.thread_axis("blockIdx.y"))
            s[output].bind(bf, tvm.thread_axis("blockIdx.x"))
            s[output].bind(vn, tvm.thread_axis("vthread"))
            s[output].bind(vx, tvm.thread_axis("vthread"))
            s[output].bind(vf, tvm.thread_axis("vthread"))

            s[output].bind(tf, tvm.thread_axis("threadIdx.x"))
            s[OL].compute_at(s[output], tf)
            # number of threads
            n_tz = cfg["tile_n"].size[2] * cfg["tile_x"].size[2]
            n_tx = cfg["tile_f"].size[2]

            # tile reduction axes
            n, x, f = s[OL].op.axis
            rc, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, rx, rci, n, x, f)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, x, f = s[load].op.axis
                fused = s[load].fuse(x, f)
                tz, fused = s[load].split(fused, nparts=n_tz)
                tx, fused = s[load].split(fused, nparts=n_tx)
                s[load].bind(tz, tvm.thread_axis("threadIdx.y"))
                s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                             cfg['auto_unroll_max_step'].val)
            s[output].pragma(kernel_scope, 'unroll_explicit',
                             cfg['unroll_explicit'].val)

            N, OW, CO = get_const_tuple(output.shape)
            KW, CI, _ = get_const_tuple(kernel.shape)
            cfg.add_flop(2 * N * OW * CO * KW * CI)

    traverse_inline(s, outs[0].op, _callback)

    return s
