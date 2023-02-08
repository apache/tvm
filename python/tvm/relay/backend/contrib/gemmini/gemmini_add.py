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
Add operator declaration and schedule registration for Gemmini
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from tvm.contrib.gemmini.environment import Environment
from tvm.contrib.gemmini.helpers import get_greater_div


ENV = Environment.instance()


@autotvm.register_topi_compute("contrib.gemmini.add")
def add(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity,
    ifm1: tvm.te.tensor.Tensor,
    ifm2: tvm.te.tensor.Tensor,
    ofm_offset: tvm.te.tensor.Tensor,
    ifm1_scale: float,
    ifm2_scale: float,
) -> tvm.te.tensor.Tensor:
    """Computation definition for Gemmini's add operator

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        ifm1 (tvm.te.tensor.Tensor): input tensor 1
        ifm2 (tvm.te.tensor.Tensor): input tensor 2
        ofm_offset (tvm.te.tensor.Tensor): offset tensor
        ifm1_scale (float): scaling factor for input tensor 1
        ifm2_scale (float): scaling factor for input tensor 2

    Raises:
        topi.InvalidShapeError: if input shapes are not supported

    Returns:
        tvm.te.tensor.Tensor: add operator result
    """

    # Make sure that the input shapes make sense
    if len(ifm1.shape) != 4 or len(ifm2.shape) != 4 or len(ofm_offset.shape) != 4:
        raise topi.InvalidShapeError()

    # Derive shapes
    oshape = topi.utils.get_const_tuple(ifm1.shape)

    tensor_type = ENV.inp_dtype

    ofm_offset_stage = te.compute(
        oshape,
        lambda b, x, y, c: ofm_offset[b, x, y, c].astype(tensor_type),
        name="ofm_offset.local",
        tag="ofm_offset",
    )
    ifm2_stage = te.compute(
        oshape,
        lambda b, x, y, c: ifm2[b, x, y, c].astype(tensor_type)
        + ofm_offset_stage[b, x, y, c].astype(tensor_type),
        name="ifm2.local",
        tag="ifm2",
    )
    res = te.compute(
        oshape,
        lambda b, x, y, c: ifm1[b, x, y, c].astype(tensor_type)
        + ifm2_stage[b, x, y, c].astype(tensor_type),
        name="res",
        tag="add",
        attrs={
            "ifm1_scale": ifm1_scale,
            "ifm2_scale": ifm2_scale,
        },
    )

    cfg.add_flop(
        3 * np.prod(topi.utils.get_const_tuple(oshape))
        + 2  # element additions needed
        * np.prod(
            topi.utils.get_const_tuple(oshape)
        )  # element multiplications needed (input scaling)
    )

    return res


@autotvm.register_topi_schedule("contrib.gemmini.add")
def schedule_add(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity, outs: tvm.ir.container.Array
) -> tvm.te.schedule.Schedule:
    """Schedule definition for Gemmini's add operator

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        outs (tvm.ir.container.Array): Output tensors

    Returns:
        tvm.te.schedule.Schedule: transformed schedule
    """

    assert len(outs) == 1
    output = outs[0]

    add_stage = output.op.output(0)
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    sch = te.create_schedule([x.op for x in outs])

    ifm1, ifm2_op = add_stage.op.input_tensors
    ifm2, ofm_offset_op = ifm2_op.op.input_tensors
    ofm_offset_op.op.input_tensors[0]

    # Prepare the scope of each buffer
    cifm1 = sch.cache_read(ifm1, ENV.acc_scope, [add_stage])
    sch[ifm2_op].set_scope(ENV.acc_scope)
    sch[ofm_offset_op].set_scope(ENV.acc_scope)

    # Split axis, taking into account the maximum value of rows and columns that can be moved into Gemminis accumulator (DIM)
    y_factor = get_greater_div(int(sch[add_stage].op.axis[3].dom.extent))
    x_factor = get_greater_div(int(sch[add_stage].op.axis[2].dom.extent))
    y_o, y_i = sch[add_stage].split(sch[add_stage].op.axis[3], factor=y_factor)
    x_o, x_i = sch[add_stage].split(sch[add_stage].op.axis[2], factor=x_factor)
    sch[add_stage].reorder(x_o, y_o, x_i, y_i)

    # Compute the stages in the correct position
    sch[cifm1].compute_at(sch[add_stage], y_o)
    sch[ifm2_op].compute_at(sch[add_stage], y_o)
    sch[ofm_offset_op].compute_at(sch[add_stage], y_o)

    # Split axis, taking into account the maximum value of rows and columns that can be moved into Gemminis accumulator (DIM)
    cifm1_ax_0_1, cifm1_ax_0_2 = sch[cifm1].split(sch[cifm1].op.axis[2], factor=ENV.DIM)
    cifm1_ax_1_1, cifm1_ax_1_2 = sch[cifm1].split(
        sch[cifm1].op.axis[3], factor=ENV.MAX_BLOCK_LEN_ACC * ENV.DIM
    )
    sch[cifm1].reorder(cifm1_ax_0_1, cifm1_ax_1_1, cifm1_ax_0_2, cifm1_ax_1_2)

    cifm2_ax_0_1, cifm2_ax_0_2 = sch[ifm2_op].split(sch[ifm2_op].op.axis[2], factor=ENV.DIM)
    cifm2_ax_1_1, cifm2_ax_1_2 = sch[ifm2_op].split(
        sch[ifm2_op].op.axis[3], factor=ENV.MAX_BLOCK_LEN_ACC * ENV.DIM
    )
    sch[ifm2_op].reorder(cifm2_ax_0_1, cifm2_ax_1_1, cifm2_ax_0_2, cifm2_ax_1_2)

    cofm_offset_ax_0_1, cofm_offset_ax_0_2 = sch[ofm_offset_op].split(
        sch[ofm_offset_op].op.axis[2], factor=ENV.DIM
    )
    cofm_offset_ax_1_1, cofm_offset_ax_1_2 = sch[ofm_offset_op].split(
        sch[ofm_offset_op].op.axis[3], factor=ENV.MAX_BLOCK_LEN_ACC * ENV.DIM
    )
    sch[ofm_offset_op].reorder(
        cofm_offset_ax_0_1, cofm_offset_ax_1_1, cofm_offset_ax_0_2, cofm_offset_ax_1_2
    )

    # Set pragmas to insert mvin instructions
    oshape = (x_factor, y_factor)
    if x_factor == 1:
        sch[cifm1].pragma(cifm1_ax_0_2, ENV.C_mvin + "_t")
        sch[ofm_offset_op].pragma(cofm_offset_ax_0_2, ENV.C_mvin_accum + "_t")
    else:
        sch[cifm1].pragma(cifm1_ax_0_2, ENV.C_mvin)
        sch[ofm_offset_op].pragma(cofm_offset_ax_0_2, ENV.C_mvin_accum)

    # Tensorize
    sch[ifm2_op].tensorize(cifm2_ax_0_2, ENV.add_tensorize(oshape))
    sch[add_stage].tensorize(x_i, ENV.add_mvout_tensorize(oshape))

    # Create configuration dictionary
    config_dict = {}
    config_dict["A_size"] = int(ifm1.shape[3])
    config_dict["B_size"] = int(ifm2.shape[3])
    config_dict["C_size"] = int(output.shape[3])
    config_dict["A_private_stride"] = ENV.DIM
    config_dict["B_private_stride"] = ENV.DIM
    config_dict["execution_stride"] = 1
    config_dict["activation"] = 0
    config_dict["mode"] = ENV.WEIGHT_STATIONARY
    config_dict["max_pixels_per_row"] = 1
    config_dict["ifm1_scale"] = float(add_stage.op.attrs["ifm1_scale"])
    config_dict["ifm2_scale"] = float(add_stage.op.attrs["ifm2_scale"])
    config_dict["scale"] = 1.0

    # Set pragmas to configure the start and end of the Gemmini code
    sch[output].pragma(sch[output].op.axis[0], "add_start")
    sch[output].pragma(sch[output].op.axis[0], "configs", str(config_dict))
    sch[output].pragma(sch[output].op.axis[0], "gemm_end")

    # print(lower(sch,[ifm1,ifm2,ofm_offset,output]))
    # breakpoint()

    return sch
