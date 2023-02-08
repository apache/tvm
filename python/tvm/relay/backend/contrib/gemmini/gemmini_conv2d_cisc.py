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
# "AS IS" BASIsch, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument
"""
Conv2d operator declaration and schedule registration for Gemmini's CISC instructions
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from tvm.contrib.gemmini.environment import Environment

ENV = Environment.instance()


@autotvm.register_topi_compute("contrib.gemmini.conv2d_cisc")
def conv2d_cisc(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity,
    orig_data: tvm.te.tensor.Tensor,
    kernel: tvm.te.tensor.Tensor,
    bias: tvm.te.tensor.Tensor,
    strides: tvm.ir.container.Array,
    padding: tvm.ir.container.Array,
    ifm_offset: int,
    activation: int,
    gemmini_scale: float,
    pool_size: tvm.ir.container.Array,
    pool_strides: tvm.ir.container.Array,
    pool_dilation: tvm.ir.container.Array,
    pool_padding: tvm.ir.container.Array,
) -> tvm.te.tensor.Tensor:
    """Computation definition for Gemmini's conv2d operator using CISC instructions

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        orig_data (tvm.te.tensor.Tensor): Input feature map
        kernel (tvm.te.tensor.Tensor): Layer weights
        bias (tvm.te.tensor.Tensor): Layer biases
        strides (tvm.ir.container.Array): convolution strides
        padding (tvm.ir.container.Array): input feature map padding
        ifm_offset (int): input feature map offset (used for the padding of the input feature map)
        activation (int): has activation?
        gemmini_scale (float): output scaling factor
        pool_size (tvm.ir.container.Array): size of the output pooling window
        pool_strides (tvm.ir.container.Array): strides for the output pooling window
        pool_dilation (tvm.ir.container.Array): dilation for the output pooling window (not used!)
        pool_padding (tvm.ir.container.Array): padding for the output pooling window

    Returns:
        tvm.te.tensor.Tensor: conv2d operator result
    """
    assert len(orig_data.shape) == 4
    assert len(kernel.shape) == 4
    assert len(bias.shape) == 1
    assert (
        orig_data.shape[1] == orig_data.shape[2]
    ), "GEMMINIs Conv2d CISC schedule only supports square inputs!"

    o_c = kernel.shape[3]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    n = orig_data.shape[0]
    i_h = orig_data.shape[1]
    i_w = orig_data.shape[2]
    i_c = orig_data.shape[3]

    hstr = strides[0]
    wstr = strides[1]
    top_pad = padding[0]
    left_pad = padding[1]
    bottom_pad = padding[2]
    right_pad = padding[3]

    o_h = topi.utils.get_const_int(tvm.tir.div((i_h + (top_pad + bottom_pad) - k_h), hstr) + 1)
    o_w = topi.utils.get_const_int(tvm.tir.div((i_w + (left_pad + right_pad) - k_w), wstr) + 1)

    ric = te.reduce_axis((0, i_c), name="ric")
    rkh = te.reduce_axis((0, k_h), name="rkh")
    rkw = te.reduce_axis((0, k_w), name="rkw")

    oshape = (n, o_h, o_w, o_c)

    if len(set(padding)) == 1 and (ENV.supports_non_zero_padding or ifm_offset == 0):
        # If the padding is the same for all borders, there is no need to use topi.nn.pad,
        # because Gemminis CISC instructions support equal padding
        data = orig_data
    else:
        # If not, then pad before calling Gemminis functions
        data = topi.nn.pad(
            orig_data,
            [0, top_pad, left_pad, 0],
            [0, bottom_pad, right_pad, 0],
            pad_value=ifm_offset,
            name="pad_data",
        )

    res = te.compute(
        oshape,
        lambda b_o, i, j, c_o: te.sum(
            data[b_o, i * hstr + rkh, j * wstr + rkw, ric].astype(ENV.inp_dtype)
            * kernel[rkh, rkw, ric, c_o].astype(ENV.inp_dtype)
            + bias[c_o].astype(ENV.inp_dtype),
            axis=[rkh, rkw, ric],
        ),
        name="res",
        tag="conv2d",
        attrs={
            "activation": activation,
            "strides": [hstr, wstr],
            "padding": padding,
            "padding_value": ifm_offset,
            "scale": gemmini_scale,
            "pool_size": pool_size,
            "pool_strides": pool_strides,
            "pool_dilation": pool_dilation,
            "pool_padding": pool_padding,
        },
    )

    cfg.add_flop(
        np.prod(topi.utils.get_const_tuple(oshape)) * k_h * k_w * i_c
        + np.prod(topi.utils.get_const_tuple(oshape))
        * (k_h * k_w * i_c - 1)  # Multiplications and additions needed
        + np.prod(  # Additions needed
            topi.utils.get_const_tuple(oshape)
        )  # Output scaling multiplications
    )

    return res


@autotvm.register_topi_schedule("contrib.gemmini.conv2d_cisc")
def schedule_conv2d_cisc(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity, outs: tvm.ir.container.Array
) -> tvm.te.schedule.Schedule:
    """Schedule definition for Gemmini's conv2d operator using CISC instructions

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        outs (tvm.ir.container.Array): Output tensors

    Returns:
        tvm.te.schedule.Schedule: transformed schedule
    """
    assert len(outs) == 1
    output = outs[0]
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                if not op.axis:
                    const_ops.append(op)
                else:
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            if op.tag == "conv2d":
                conv2d_res.append(op)
            else:
                for tensor in op.input_tensors:
                    _traverse(tensor.op)

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)
    sch = te.create_schedule(output.op)

    data, kernel, bias = conv2d_stage.op.input_tensors

    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = data

    x_bo, _, _, _ = sch[conv2d_stage].op.axis

    x_bo_o, x_bo_i = sch[conv2d_stage].split(x_bo, factor=pad_data.shape[0])

    axis_for_start = x_bo_o

    # If topi.nn.pad was added, its because the padding was not equal in all dimensions.
    padding = conv2d_stage.op.attrs["padding"] if pad_data == data else [0, 0, 0, 0]
    padding_value = conv2d_stage.op.attrs["padding_value"] if pad_data == data else 0

    # Apply tensorization
    sch[conv2d_stage].tensorize(
        x_bo_i,
        ENV.conv2d_cisc(
            pad_data.shape,
            kernel.shape,
            bias.shape,
            conv2d_stage.shape,
            conv2d_stage.op.attrs["strides"],
            padding,
            padding_value,
            conv2d_stage.op.attrs["activation"],
            conv2d_stage.op.attrs["scale"],
            conv2d_stage.op.attrs["pool_size"],
            conv2d_stage.op.attrs["pool_strides"],
            conv2d_stage.op.attrs["pool_dilation"],
            conv2d_stage.op.attrs["pool_padding"],
        ),
    )

    # Tag loops with pragmas to delimit the start and end of the Gemmini related code
    sch[conv2d_stage].pragma(axis_for_start, "conv2d_cisc_start")
    sch[conv2d_stage].pragma(axis_for_start, "gemm_end")

    return sch
