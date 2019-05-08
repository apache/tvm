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
"""Conv2d transpose template for cuda backend"""

import tvm
from tvm import autotvm

from .. import nn, generic
from ..util import equal_const_int, get_const_tuple, traverse_inline

@autotvm.task.register_topi_compute(nn.conv2d_transpose_nchw, ['cuda', 'gpu'], "direct")
def conv2d_transpose_nchw_cuda(cfg, Input, Filter, strides, padding, out_dtype):
    """Transposed 2D convolution nchw forward operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.Tensor
        4-D with shape [in_channel, num_filter, filter_height, filter_width]
    strides : tuple of two ints
        The spatial stride along height and width
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    out_dtype: str
        The output type. This is used in mixed precision

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_c, in_h, in_w = get_const_tuple(Input.shape)
    _, out_c, filter_h, filter_w = get_const_tuple(Filter.shape)
    stride_h, stride_w = strides

    # attach stride info to config, this is used in schedule space definition
    cfg.stride = strides

    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = nn.get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    FirstPad = nn.pad(Input,
                      [0, 0, (bpad_top + stride_h - 1) // stride_h,
                       (bpad_left + stride_w - 1) // stride_w],
                      [0, 0, (bpad_bottom + stride_h - 1) // stride_h,
                       (bpad_right + stride_w - 1) // stride_w], name='FirstPad')

    # remove extra padding introduced by dilatation
    border_h = (stride_h - bpad_top % stride_h) % stride_h
    border_w = (stride_w - bpad_left % stride_w) % stride_w

    # dilation stage
    data = FirstPad
    strides = [1, 1, stride_h, stride_w]
    n = len(data.shape)

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not equal_const_int(strides[i], 1):
                index_tuple.append(indices[i] // strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.if_then_else(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    Output = tvm.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            _dilate(b, dc, h + dh + border_h, w + dw + border_w).astype(out_dtype) *
            Filter[dc, c, filter_h - 1 - dh, filter_w - 1 - dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")

    return Output

@autotvm.task.register_topi_schedule(generic.schedule_conv2d_transpose_nchw,
                                     ['cuda', 'gpu'], 'direct')
def schedule_conv2d_transpose_nchw_cuda(cfg, outs):
    """TOPI Schedule callback for conv2d transpose operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The parameters for this template

    outs: Array of Tensor
        The computation graph description of conv2d transpose
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d transpose.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv2d_transpose_nchw':
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.current_target()
            if target.target_name in ['nvptx', 'rocm']:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])
            ##### space definition end #####

            if isinstance(kernel.op, tvm.tensor.ComputeOp) and 'dilate' in kernel.op.tag:
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
            n, f, y, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
            s[output].bind(by, tvm.thread_axis("blockIdx.y"))
            s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[output].bind(vf, tvm.thread_axis("vthread"))
            s[output].bind(vy, tvm.thread_axis("vthread"))
            s[output].bind(vx, tvm.thread_axis("vthread"))
            s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
            s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
            s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
            s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
            s[OL].compute_at(s[output], tx)

            # tile reduction axes
            n, f, y, x = s[OL].op.axis
            rc, ry, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, ry, rx, rci, n, f, y, x)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, f, y, x = s[load].op.axis
                fused = s[load].fuse(f, y, x)
                tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
                ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
                tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
                s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
                s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
                s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
            s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    traverse_inline(s, outs[0].op, _callback)

    return s
