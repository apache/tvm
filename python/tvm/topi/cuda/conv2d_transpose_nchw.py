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
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from .. import nn
from ..utils import get_const_tuple, traverse_inline


@autotvm.register_topi_compute("conv2d_transpose_nchw.cuda")
def conv2d_transpose_nchw(cfg, data, kernel, stride, padding, out_dtype, output_padding):
    """Transposed 2D convolution nchw forward operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    Filter : tvm.te.Tensor
        4-D with shape [in_channel, num_filter, filter_height, filter_width]
    strides : tuple of two ints
        The spatial stride along height and width
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    out_dtype: str
        The output type. This is used in mixed precision
    output_padding : tuple of two ints
        Used to disambiguate output shape.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, inp_channels, inp_height, inp_width = get_const_tuple(data.shape)
    _, out_channels, kernel_height, kernel_width = get_const_tuple(kernel.shape)
    stride_height, stride_width = stride
    outpad_height, outpad_width = output_padding
    assert outpad_height < stride_height and outpad_width < stride_width
    cfg.stride = stride
    pad_top, pad_left, pad_bottom, pad_right = nn.get_pad_tuple(
        padding, (kernel_height, kernel_width)
    )

    out_width = (inp_width - 1) * stride_width + kernel_width - pad_left - pad_right + outpad_width
    pad_left = kernel_width - 1 - pad_left
    pad_right = kernel_width - 1 - pad_right + outpad_width
    dilated_width = stride_width * (inp_width - 1) + 1

    out_height = (
        (inp_height - 1) * stride_height + kernel_height - pad_top - pad_bottom + outpad_height
    )
    pad_top = kernel_height - 1 - pad_top
    pad_bottom = kernel_height - 1 - pad_bottom + outpad_height
    dilated_height = stride_height * (inp_height - 1) + 1

    # compute pad
    data = te.compute(
        (
            batch,
            inp_channels,
            pad_top + dilated_height + pad_bottom,
            pad_left + dilated_width + pad_right,
        ),
        lambda n, c, y, x: tvm.tir.if_then_else(
            tvm.tir.all(
                x >= pad_left,
                x < pad_left + dilated_width,
                tvm.tir.indexmod(x - pad_left, stride_width).equal(0),
                y >= pad_top,
                y < pad_top + dilated_height,
                tvm.tir.indexmod(y - pad_top, stride_height).equal(0),
            ),
            data[
                n,
                c,
                tvm.tir.indexdiv(y - pad_top, stride_height),
                tvm.tir.indexdiv(x - pad_left, stride_width),
            ],
            tvm.tir.const(0.0, data.dtype),
        ),
        name="data_pad",
    )

    # compute transposed conv
    dc = te.reduce_axis((0, inp_channels), name="dc")
    dh = te.reduce_axis((0, kernel_height), name="dh")
    dw = te.reduce_axis((0, kernel_width), name="dw")
    data_out = te.compute(
        (batch, out_channels, out_height, out_width),
        lambda b, c, h, w: te.sum(
            data[b, dc, h + dh, w + dw].astype(out_dtype)
            * kernel[dc, c, kernel_height - 1 - dh, kernel_width - 1 - dw].astype(out_dtype),
            axis=[dc, dh, dw],
        ),
        tag="conv2d_transpose_nchw",
    )

    return data_out


@autotvm.register_topi_schedule("conv2d_transpose_nchw.cuda")
def schedule_conv2d_transpose_nchw(cfg, outs):
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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _fallback_schedule(N, F, Y, X):
        # pylint: disable=unused-argument
        # split N (batch dimension)
        if N > 1:
            cfg["tile_n"] = SplitEntity([-1, 1, 1, 4])
        else:
            cfg["tile_n"] = SplitEntity([1, 1, 1, 1])
        # split F (output channel dimension)
        if F > 1:
            cfg["tile_f"] = SplitEntity([-1, 1, 64, 1])
        # split Y (height dimension)
        y_split_factor = 1
        for candidate in range(5, 17):
            if Y % candidate == 0:
                y_split_factor = candidate
                break
        cfg["tile_y"] = SplitEntity([-1, 1, 1, y_split_factor])
        # split X (width dimension)
        x_split_factor = 1
        for candidate in range(5, 17):
            if X % candidate == 0:
                x_split_factor = candidate
                break
        cfg["tile_x"] = SplitEntity([-1, x_split_factor, 1, 1])
        # split RC (input channel dimension, which is a reduction axis)
        cfg["tile_rc"] = SplitEntity([-1, 1, 16])
        # other configurations
        cfg["fuse_yx"] = OtherOptionEntity(False)
        cfg["unroll_explicit"] = OtherOptionEntity(True)
        cfg["auto_unroll_max_step"] = OtherOptionEntity(1500)

    def _callback(op):
        if op.tag == "conv2d_transpose_nchw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            # TODO(@kevinthesun): Support tuning/optimization for dynamic shape.
            bs = pad_data.shape[0]
            n_tuning_axis = n if isinstance(bs, tvm.tir.IntImm) else 1
            cfg.define_split("tile_n", cfg.axis(n_tuning_axis), num_outputs=4)
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.Target.current()
            if target.kind.name in ["nvptx", "rocm"]:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            if cfg.is_fallback:
                N, F, Y, X = get_const_tuple(conv.shape)
                if not isinstance(N, int):
                    N = 1
                _fallback_schedule(N, F, Y, X)

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
            n, f, y, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
            s[output].bind(bn, te.thread_axis("blockIdx.z"))
            s[output].bind(bf, te.thread_axis("blockIdx.y"))
            s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
            s[output].bind(vn, te.thread_axis("vthread"))
            s[output].bind(vf, te.thread_axis("vthread"))
            s[output].bind(vy, te.thread_axis("vthread"))
            s[output].bind(vx, te.thread_axis("vthread"))

            cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf

            if cfg["fuse_yx"].val:
                s[output].bind(tn, te.thread_axis("threadIdx.z"))
                s[output].bind(tf, te.thread_axis("threadIdx.y"))
                tyx = s[output].fuse(ty, tx)
                s[output].bind(s[output].fuse(ty, tx), te.thread_axis("threadIdx.x"))
                s[OL].compute_at(s[output], tyx)

                # number of threads
                n_tz = cfg["tile_n"].size[2]
                n_ty = cfg["tile_f"].size[2]
                n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
            else:
                s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
                s[output].bind(ty, te.thread_axis("threadIdx.y"))
                s[output].bind(tx, te.thread_axis("threadIdx.x"))
                s[OL].compute_at(s[output], tx)

                # number of threads
                n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
                n_ty = cfg["tile_y"].size[2]
                n_tx = cfg["tile_x"].size[2]

            # tile reduction axes
            n, f, y, x = s[OL].op.axis
            rc, ry, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, ry, rx, rci, n, f, y, x)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, f, y, x = s[load].op.axis
                fused = s[load].fuse(f, y, x)
                tz, fused = s[load].split(fused, nparts=n_tz)
                ty, fused = s[load].split(fused, nparts=n_ty)
                tx, fused = s[load].split(fused, nparts=n_tx)
                s[load].bind(tz, te.thread_axis("threadIdx.z"))
                s[load].bind(ty, te.thread_axis("threadIdx.y"))
                s[load].bind(tx, te.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    traverse_inline(s, outs[0].op, _callback)

    return s
