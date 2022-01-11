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
"""Conv1d transpose template for cuda backend"""

import tvm
from tvm import te
from tvm import autotvm
from .. import nn
from ..utils import get_const_tuple, traverse_inline


@autotvm.task.register_topi_compute("conv1d_transpose_nchw.cuda")
def conv1d_transpose_ncw(cfg, data, kernel, stride, padding, out_dtype, output_padding):
    """Transposed 1D convolution ncw forward operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template
    Input : tvm.te.Tensor
        3-D with shape [batch, in_channel, inp_width]
    Filter : tvm.te.Tensor
        3-D with shape [in_channel, num_filter, kernel_size]
    stride : tuple of one int
        The spatial stride along width
    padding : int, tuple, or string
        int: padding size
        tuple of 2 ints: (pad_left, pad_right) for left and right padding
        string: ['VALID', 'SAME']
    out_dtype: str
        The output type. This is used in mixed precision
    output_padding : ints
        Used to disambiguate the output shape.

    Returns
    -------
    Output : tvm.te.Tensor
    u    3-D with shape [batch, out_channel, out_width]
    """
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(output_padding, (tuple, list)):
        output_padding = output_padding[0]
    assert output_padding < stride
    cfg.stride = stride
    cfg.output_padding = output_padding
    batch, inp_channels, inp_width = get_const_tuple(data.shape)
    _, out_channels, kernel_size = get_const_tuple(kernel.shape)
    pad_left, pad_right = nn.get_pad_tuple1d(padding, kernel_size)
    out_width = (inp_width - 1) * stride + kernel_size - pad_left - pad_right + output_padding
    pad_left = kernel_size - 1 - pad_left
    pad_right = kernel_size - 1 - pad_right + output_padding
    padded_width = pad_left + inp_width + pad_right

    padded_data = te.compute(
        (batch, inp_channels, padded_width),
        lambda n, c, x: tvm.tir.if_then_else(
            tvm.tir.all(x >= pad_left, x < pad_left + inp_width),
            data[n, c, x - pad_left],
            tvm.tir.const(0.0, "float32"),
        ),
        name="data_pad",
    )

    padded_kernel = te.compute(
        (inp_channels, out_channels, kernel_size + stride - 1),
        lambda ci, co, k: tvm.tir.if_then_else(
            tvm.tir.all(k < kernel_size),
            kernel[ci, co, kernel_size - k - 1],
            tvm.tir.const(0.0, "float32"),
        ),
        name="kernel_pad",
    )

    ci = te.reduce_axis((0, inp_channels), name="ci")
    k = te.reduce_axis((0, tvm.tir.indexdiv(kernel_size + stride - 1, stride)), name="k")
    border = pad_left * (stride - 1)

    # Skip multiplication by 0 values in the input data inserted when stride is greater then 1.
    # During multiplication of kernel by padded data:
    #  Kernel indices are: 0, 1 * stride, 2 * stride, ..., ceil(kernel_size / stride) plus
    #  data offset mod stride
    data_out = te.compute(
        (batch, out_channels, out_width),
        lambda b, co, w: te.sum(
            padded_data[b, ci, tvm.tir.indexdiv(border + w + stride - 1, stride) + k].astype(
                out_dtype
            )
            * padded_kernel[
                ci, co, k * stride + tvm.tir.indexmod(stride - w - border, stride)
            ].astype(out_dtype),
            axis=[ci, k],
        ),
        tag="conv1d_transpose_ncw",
    )

    return data_out


@autotvm.task.register_topi_schedule("conv1d_transpose_nchw.cuda")
def schedule_conv1d_transpose_ncw(cfg, outs):
    """TOPI Schedule callback for conv1d_transpose operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The parameters for this template

    outs: Array of Tensor
        The computation graph description of conv1d transpose
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv1d transpose.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_transpose_ncw":
            padded_data = op.input_tensors[0]
            padded_kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, x = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            cfg.define_split("tile_n", cfg.axis(n if isinstance(n, int) else 1), num_outputs=4)
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

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, "local")
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope("local")
                OL = conv

            s[padded_kernel].compute_inline()
            s[padded_data].compute_inline()

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

            # tile reduction axes
            n, f, x = s[OL].op.axis
            rc, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, rx, rci, n, f, x)

            s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    traverse_inline(s, outs[0].op, _callback)

    return s
