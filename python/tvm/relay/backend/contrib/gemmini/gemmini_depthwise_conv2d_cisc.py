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
Depthwise conv2d operator declaration and schedule registration for Gemmini's CISC instructions
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from tvm.contrib.gemmini.environment import Environment

env = Environment.instance()


@autotvm.register_topi_compute("contrib.gemmini.depthwiseconv2d_cisc")
def depthwise_conv2d_cisc(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity,
    orig_data: tvm.te.tensor.Tensor,
    orig_kernel: tvm.te.tensor.Tensor,
    bias: tvm.te.tensor.Tensor,
    strides: tvm.ir.container.Array,
    padding: tvm.ir.container.Array,
    ifm_offset: int,
    activation: int,
    gemmini_scale: float,
) -> tvm.te.tensor.Tensor:
    """Computation definition for Gemmini's depthwise conv2d operator using CISC instructions

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        orig_data (tvm.te.tensor.Tensor): Input feature map
        orig_kernel (tvm.te.tensor.Tensor): Layer weights
        bias (tvm.te.tensor.Tensor): Layer biases
        strides (tvm.ir.container.Array): convolution strides
        padding (tvm.ir.container.Array): input feature map padding
        ifm_offset (int): input feature map offset (used for the padding of the input feature map)
        activation (int): has activation?
        gemmini_scale (float): output scaling factor

    Returns:
        tvm.te.tensor.Tensor: depthwise conv2d operator result
    """

    assert len(orig_data.shape) == 4
    assert len(orig_kernel.shape) == 3
    assert len(bias.shape) == 1
    assert (
        orig_data.shape[1] == orig_data.shape[2]
    ), "GEMMINIs depthwise conv2d CISC schedule only supports square inputs!"

    OC = orig_kernel.shape[0]
    KH = orig_kernel.shape[1]
    KW = orig_kernel.shape[2]

    kernel = orig_kernel

    N = orig_data.shape[0]
    IH = orig_data.shape[1]
    IW = orig_data.shape[2]
    orig_data.shape[3]

    HSTR = strides[0]
    WSTR = strides[1]
    TOP_PAD = padding[0]
    LEFT_PAD = padding[1]
    BOTTOM_PAD = padding[2]
    RIGHT_PAD = padding[3]

    OH = topi.utils.get_const_int(tvm.tir.div((IH + (TOP_PAD + BOTTOM_PAD) - KH), HSTR) + 1)
    OW = topi.utils.get_const_int(tvm.tir.div((IW + (LEFT_PAD + RIGHT_PAD) - KW), WSTR) + 1)

    if len(set(padding)) == 1 and env.supports_non_zero_padding:
        # If the padding is the same for all borders, there is no need to use topi.nn.pad, because Gemminis CISC instructions support equal padding
        data = orig_data
    else:
        # If not, then pad before calling Gemminis functions
        data = topi.nn.pad(
            orig_data,
            [0, TOP_PAD, LEFT_PAD, 0],
            [0, BOTTOM_PAD, RIGHT_PAD, 0],
            pad_value=ifm_offset,
            name="pad_data",
        )

    rkh = te.reduce_axis((0, KH), name="rkh")
    rkw = te.reduce_axis((0, KW), name="rkw")

    oshape = (N, OH, OW, OC)

    res = te.compute(
        oshape,
        lambda b_o, i, j, c_o: te.sum(
            data[b_o, i * HSTR + rkh, j * WSTR + rkw, c_o].astype(env.inp_dtype)
            * kernel[c_o, rkh, rkw].astype(env.inp_dtype)
            + bias[c_o].astype(env.inp_dtype),
            axis=[rkh, rkw],
        ),
        name="res",
        tag="conv2d",
        attrs={
            "activation": activation,
            "strides": [HSTR, WSTR],
            "padding": padding,
            "padding_value": ifm_offset,
            "scale": gemmini_scale,
        },
    )

    cfg.add_flop(
        np.prod(topi.utils.get_const_tuple(oshape)) * KH * KW
        + np.prod(topi.utils.get_const_tuple(oshape))
        * (KH * KW - 1)  # Multiplications and additions needed
        + np.prod(topi.utils.get_const_tuple(oshape))  # Output scaling factor multiplications
    )

    return res


@autotvm.register_topi_schedule("contrib.gemmini.depthwiseconv2d_cisc")
def schedule_depthwise_conv2d_cisc(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity, outs: tvm.ir.container.Array
) -> tvm.te.schedule.Schedule:
    """Schedule definition for Gemmini's depthwise conv2d operator using CISC instructions

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

    x_bo, x_i, x_j, x_co = sch[conv2d_stage].op.axis
    rkh, rkw = sch[conv2d_stage].op.reduce_axis

    x_bo_o, x_bo_i = sch[conv2d_stage].split(x_bo, factor=pad_data.shape[0])

    axis_for_start = x_bo_o

    # If topi.nn.pad was added, its because the padding was not equal in all dimensions.
    padding_for_C_code = conv2d_stage.op.attrs["padding"] if pad_data == data else [0, 0, 0, 0]
    padding_value_for_C_code = conv2d_stage.op.attrs["padding_value"] if pad_data == data else 0

    # Apply tensorization
    sch[conv2d_stage].tensorize(
        x_bo_i,
        env.dw_conv2d_cisc(
            pad_data.shape,
            kernel.shape,
            bias.shape,
            conv2d_stage.shape,
            conv2d_stage.op.attrs["strides"],
            padding_for_C_code,
            padding_value_for_C_code,
            conv2d_stage.op.attrs["activation"],
            conv2d_stage.op.attrs["scale"],
        ),
    )

    # Tag loops with pragmas to delimit the start and end of the Gemmini related code
    sch[conv2d_stage].pragma(axis_for_start, "dw_conv2d_cisc_start")
    sch[conv2d_stage].pragma(axis_for_start, "gemm_end")

    return sch
