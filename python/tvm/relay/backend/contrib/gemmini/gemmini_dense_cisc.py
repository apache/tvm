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
Dense (GEMM) operator declaration and schedule registration for Gemmini's CISC instructions
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
from tvm.autotvm.task.space import OtherOptionEntity

from tvm.contrib.gemmini.environment import Environment

env = Environment.instance()


@autotvm.register_topi_compute("contrib.gemmini.gemm_cisc")
def gemm_cisc(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity,
    data: tvm.te.tensor.Tensor,
    weight: tvm.te.tensor.Tensor,
    bias: tvm.te.tensor.Tensor,
    scale: float,
) -> tvm.te.tensor.Tensor:
    """Computation definition for Gemmini's dense operator using CISC instructions

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        data (tvm.te.tensor.Tensor): Input feature map
        weight (tvm.te.tensor.Tensor): Layer weights
        bias (tvm.te.tensor.Tensor): Layer biases
        scale (float): output scaling factor

    Returns:
        tvm.te.tensor.Tensor: dense operator result
    """

    # Derive shapes
    ishape = topi.utils.get_const_tuple(data.shape)
    wshape = topi.utils.get_const_tuple(weight.shape)
    oshape = (data.shape[0], weight.shape[1])

    # Reduction axes (input channel)
    assert ishape[1] == wshape[0]
    k_o = te.reduce_axis((0, wshape[0]), name="k_o")

    res = te.compute(
        oshape,
        lambda x_o, y_o: te.sum(
            data[x_o, k_o].astype(env.inp_dtype) * weight[k_o, y_o].astype(env.inp_dtype)
            + bias[y_o].astype(env.inp_dtype),
            axis=[k_o],
        ),
        name="res",
        tag="dense",
        attrs={"scale": scale},
    )

    cfg.add_flop(
        (2 * np.prod(topi.utils.get_const_tuple(oshape)) * ishape[1])  # element multiplications
        + np.prod(topi.utils.get_const_tuple(oshape))  # bias additions
    )

    return res


@autotvm.register_topi_schedule("contrib.gemmini.gemm_cisc")
def schedule_gemm_cisc(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity, outs: tvm.ir.container.Array
) -> tvm.te.schedule.Schedule:
    """Schedule definition for Gemmini's dense operator using CISC instructions

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        outs (tvm.ir.container.Array): Output tensors

    Returns:
        tvm.te.schedule.Schedule: transformed schedule
    """
    assert len(outs) == 1
    output = outs[0]

    dense_stage = output.op.output(0)
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    sch = te.create_schedule([x.op for x in outs])

    data, weight, bias = dense_stage.op.input_tensors

    # WS/OS
    #   0: Gemmini will be configured as output stationary
    #   1: Gemmini will be configured as weight stationary
    cfg.define_knob("WS/OS", [env.WEIGHT_STATIONARY, env.OUTPUT_STATIONARY])
    if cfg.is_fallback:
        cfg["WS/OS"] = OtherOptionEntity(env.WEIGHT_STATIONARY)

    x_, y_ = sch[dense_stage].op.axis

    x_o, x_i = sch[dense_stage].split(x_, factor=data.shape[0])

    axis_for_start = x_o

    # Apply tensorization
    sch[dense_stage].tensorize(
        x_i,
        env.gemm_cisc(
            data.shape, weight.shape, bias.shape, dense_stage.op.attrs["scale"], cfg["WS/OS"].val
        ),
    )

    # Tag loops with pragmas to delimit the start and end of the Gemmini related code
    sch[dense_stage].pragma(axis_for_start, "gemm_cisc_start")
    sch[dense_stage].pragma(axis_for_start, "gemm_end")

    return sch
