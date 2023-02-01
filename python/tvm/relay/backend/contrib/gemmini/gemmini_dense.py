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
Dense (GEMM) operator declaration and schedule registration for Gemmini's intrinsic instructions
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from tvm.contrib.gemmini.environment import Environment
from tvm.contrib.gemmini.helpers import get_greater_div

env = Environment.instance()


@autotvm.register_topi_compute("contrib.gemmini.gemm")
def gemm(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity,
    data: tvm.te.tensor.Tensor,
    weight: tvm.te.tensor.Tensor,
    bias: tvm.te.tensor.Tensor,
    scale: float,
) -> tvm.te.tensor.Tensor:
    """Computation definition for Gemmini's dense operator using intrinsic instructions

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

    bias_stage = te.compute(
        oshape,
        lambda x_o, y_o: bias[y_o].astype(env.inp_dtype),
        name="bias.local.accumulator",
        tag="bias_add",
    )

    res = te.compute(
        oshape,
        lambda x_o, y_o: te.sum(
            data[x_o, k_o].astype(env.inp_dtype) * weight[k_o, y_o].astype(env.inp_dtype)
            + bias_stage[x_o, y_o].astype(env.inp_dtype),
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


@autotvm.register_topi_schedule("contrib.gemmini.gemm")
def schedule_gemm(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity, outs: tvm.ir.container.Array
) -> tvm.te.schedule.Schedule:
    """Schedule definition for Gemmini's dense operator using intrinsic instructions

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

    data, weight, bias_op = dense_stage.op.input_tensors
    bias_op.op.input_tensors[0]

    ##### space definition begin #####
    x, y = sch[dense_stage].op.axis
    (z,) = sch[dense_stage].op.reduce_axis

    # TODO (FP): add limits for scratchpad and accumulator sizes perhaps?
    cfg.define_split(
        "tile_xo",
        x,
        num_outputs=3,
        policy="power2",
        filter=lambda ax: (
            ax.size[-1] == get_greater_div(int(data.shape[0]))
            if (data.shape[0] >= env.DIM)
            else ax.size[-1] <= env.DIM
        ),
    )

    cfg.define_split(
        "tile_yo",
        y,
        num_outputs=3,
        policy="power2",
        filter=lambda ax: (
            ax.size[-1] == get_greater_div(int(weight.shape[1]))
            if (weight.shape[1] >= env.DIM)
            else ax.size[-1] <= env.DIM
        ),
    )

    cfg.define_split(
        "tile_zo",
        z,
        num_outputs=3,
        policy="power2",
        filter=lambda ax: (
            ax.size[-1] == get_greater_div(int(weight.shape[0]))
            if (weight.shape[0] >= env.DIM)
            else ax.size[-1] <= env.DIM
        ),
    )

    # accumulate_multiple_patches knob
    #   2: only one patch is computed in the accumulator
    #   1: More than one patch is computed in the accumulator, depends on tile_yo
    #   0: More than one patch is computed in the accumulator, depends on tile_yo AND tile_xo
    cfg.define_knob("accumulate_multiple_patches", [0, 1, 2])
    # exchange axis
    #   exchange the order of axis x and y
    cfg.define_knob("exchange_axis", [False, True])
    # WS/OS
    #   0: Gemmini will be configured as output stationary
    #   1: Gemmini will be configured as weight stationary
    cfg.define_knob("WS/OS", [env.WEIGHT_STATIONARY, env.OUTPUT_STATIONARY])
    # mvout_big_block
    #   False: generate mvout instructions moving as maximum DIM columns
    #   True: generate mvout instructions moving more than DIM columns
    cfg.define_knob("mvout_big_block", [True, False])
    if cfg.is_fallback:
        # Load default split values
        cfg["tile_xo"] = SplitEntity([-1, 8, get_greater_div(int(data.shape[0]))])
        cfg["tile_yo"] = SplitEntity([-1, 8, get_greater_div(int(weight.shape[1]))])
        cfg["tile_zo"] = SplitEntity([-1, 8, get_greater_div(int(weight.shape[0]))])
        cfg["accumulate_multiple_patches"] = OtherOptionEntity(0)
        cfg["exchange_axis"] = OtherOptionEntity(False)
        cfg["mvout_big_block"] = OtherOptionEntity(True)
        cfg["WS/OS"] = OtherOptionEntity(env.WEIGHT_STATIONARY)

    ###### space definition end ######

    cdata = sch.cache_read(data, env.scr_scope, [dense_stage])
    cweight = sch.cache_read(weight, env.scr_wgt_scope, [dense_stage])
    dense_stage_acc = sch.cache_write(output, env.acc_scope)
    sch[bias_op].set_scope(env.acc_scope)
    (x_, y_) = sch[dense_stage_acc].op.axis
    (z_,) = sch[dense_stage_acc].op.reduce_axis

    # Split loops to generate the inner dimensions specified by knobs tile_xo and tile_yo
    b_y, yo, yi = cfg["tile_yo"].apply(sch, output, sch[output].op.axis[1])
    b_x, xo, xi = cfg["tile_xo"].apply(sch, output, sch[output].op.axis[0])

    # Apply the exchange_axis knob
    if cfg["exchange_axis"].val:
        sch[output].reorder(b_y, b_x, yo, xo, yi, xi)
    else:
        sch[output].reorder(b_x, b_y, xo, yo, xi, yi)

    # Apply the accumulate_multiple_patches knob
    if cfg["accumulate_multiple_patches"].val == 0:
        axis_for_output = b_x if cfg["exchange_axis"].val else b_y
    elif cfg["accumulate_multiple_patches"].val == 1:
        axis_for_output = yo if cfg["exchange_axis"].val else xo
    else:
        axis_for_output = xo if cfg["exchange_axis"].val else yo

    axis_gemm_start = b_y if cfg["exchange_axis"].val else b_x

    # Move the dense_stage_acc stage to the correct axis of the output stage
    sch[dense_stage_acc].compute_at(sch[output], axis_for_output)

    # # Split loops to generate the inner dimensions specified by knob tile_zo
    xo_o, xi_o = sch[dense_stage_acc].split(x_, factor=env.DIM)
    yo_o, yi_o = sch[dense_stage_acc].split(y_, factor=env.DIM)
    b_z, zo_o, zi_o = cfg["tile_zo"].apply(sch, dense_stage_acc, z_)

    # Apply the exchange_axis knob
    if cfg["exchange_axis"].val:
        sch[dense_stage_acc].reorder(b_z, xo_o, yo_o, zo_o, xi_o, yi_o, zi_o)
    else:
        sch[dense_stage_acc].reorder(b_z, yo_o, xo_o, zo_o, yi_o, xi_o, zi_o)

    # Generate knobs to move the copy of data across different loops
    axis_to_input_data = [b_x, b_z, xo_o, zo_o]
    axis_to_input_weights = [b_y, b_z, yo_o, zo_o]
    stages_to_input_data = [output, dense_stage_acc, dense_stage_acc, dense_stage_acc]
    cfg.define_knob("axis_for_cdata", [0, 1, 2, 3])
    cfg.define_knob("axis_for_cweight", [0, 1, 2, 3])
    if cfg.is_fallback:
        cfg["axis_for_cdata"] = OtherOptionEntity(0)
        cfg["axis_for_cweight"] = OtherOptionEntity(0)

    # Compute the move of the bias in the correct loop
    sch[bias_op].compute_at(sch[output], axis_for_output)

    # We assert here that the mvin of data does not use more space than the available one in the scratchpad
    if cfg["axis_for_cdata"].val == 0:
        assert (
            cfg["tile_xo"].size[1] * cfg["tile_xo"].size[2] * data.shape[1]
            <= env.INP_SCR_ROWS * env.DIM
        ), "Data matrix will not fit in scratchpad!"
    elif cfg["axis_for_cdata"].val == 1:
        assert (
            cfg["tile_xo"].size[2] * data.shape[1] <= env.INP_SCR_ROWS * env.DIM
        ), "Data matrix will not fit in scratchpad!"
    if cfg["axis_for_cweight"].val == 0:
        assert (
            cfg["tile_yo"].size[1] * cfg["tile_yo"].size[2] * weight.shape[0]
            <= env.WGT_SCR_ROWS * env.DIM
        ), "Weight matrix will not fit in scratchpad!"
    elif cfg["axis_for_cweight"].val == 1:
        assert (
            cfg["tile_yo"].size[2] * weight.shape[0] <= env.WGT_SCR_ROWS * env.DIM
        ), "Weight matrix will not fit in scratchpad!"

    # And here we assert that there is enough place available in the accumulator
    if cfg["accumulate_multiple_patches"].val == 0:
        assert (
            cfg["tile_xo"].size[1]
            * cfg["tile_xo"].size[2]
            * cfg["tile_yo"].size[1]
            * cfg["tile_yo"].size[2]
            <= env.ACC_ROWS * env.DIM
        ), "Result matrix will not fit in accumulator!"
    elif cfg["accumulate_multiple_patches"].val == 1:
        assert (
            cfg["tile_xo"].size[2] * cfg["tile_yo"].size[1] * cfg["tile_yo"].size[2]
            <= env.ACC_ROWS * env.DIM
        ), "Result matrix will not fit in accumulator!"

    # Move the data and weight move instructions into the correct loops selected by the axis_for_cdata and axis_for_cweight knobs
    axis_for_cdata = axis_to_input_data[cfg["axis_for_cdata"].val]
    axis_for_cweight = axis_to_input_weights[cfg["axis_for_cweight"].val]
    sch[cdata].compute_at(sch[stages_to_input_data[cfg["axis_for_cdata"].val]], axis_for_cdata)
    sch[cweight].compute_at(
        sch[stages_to_input_data[cfg["axis_for_cweight"].val]], axis_for_cweight
    )

    # Split input moves because Gemmini's mvin only supports mvins with rows <= DIM and cols <= MAX_BLOCK_LEN
    cdata_ax_0_1, cdata_ax_0_2 = sch[cdata].split(sch[cdata].op.axis[0], factor=env.DIM)
    cdata_ax_1_1, cdata_ax_1_2 = sch[cdata].split(
        sch[cdata].op.axis[1], factor=env.MAX_BLOCK_LEN * env.DIM
    )
    sch[cdata].reorder(cdata_ax_0_1, cdata_ax_1_1, cdata_ax_0_2, cdata_ax_1_2)

    cweight_ax_0_1, cweight_ax_0_2 = sch[cweight].split(sch[cweight].op.axis[0], factor=env.DIM)
    cweight_ax_1_1, cweight_ax_1_2 = sch[cweight].split(
        sch[cweight].op.axis[1], factor=env.MAX_BLOCK_LEN * env.DIM
    )
    sch[cweight].reorder(cweight_ax_0_1, cweight_ax_1_1, cweight_ax_0_2, cweight_ax_1_2)

    cbias_ax_0_1, cbias_ax_0_2 = sch[bias_op].split(sch[bias_op].op.axis[0], factor=env.DIM)
    cbias_ax_1_1, cbias_ax_1_2 = sch[bias_op].split(
        sch[bias_op].op.axis[1], factor=env.MAX_BLOCK_LEN_ACC * env.DIM
    )
    sch[bias_op].reorder(cbias_ax_0_1, cbias_ax_1_1, cbias_ax_0_2, cbias_ax_1_2)

    # Mvout preparation
    if cfg["exchange_axis"].val:
        sch[output].reorder(yo, yi, xo, xi)
    else:
        sch[output].reorder(xo, xi, yo, yi)
    if cfg["accumulate_multiple_patches"].val == 0:
        fused_x = sch[output].fuse(xo, xi)
        fused_y = sch[output].fuse(yo, yi)
    elif cfg["accumulate_multiple_patches"].val == 1:
        if cfg["exchange_axis"].val:
            fused_x = sch[output].fuse(xo, xi)
            fused_y = yi
        else:
            fused_x = xi
            fused_y = sch[output].fuse(yo, yi)
    else:
        fused_x = xi
        fused_y = yi

    fused_x_1, fused_x_2 = sch[output].split(fused_x, factor=env.DIM)
    fused_y_1, fused_y_2 = sch[output].split(
        fused_y, factor=env.MAX_BLOCK_LEN * env.DIM if cfg["mvout_big_block"].val else env.DIM
    )
    sch[output].reorder(fused_x_1, fused_y_1, fused_x_2, fused_y_2)

    # Tag loops with pragmas, in order to insert the move in and move out instructions
    sch[cweight].pragma(cweight_ax_0_2, env.B_mvin)
    if data.shape[0] == 1 and weight.shape[1] > 1:
        sch[cdata].pragma(cdata_ax_0_2, env.A_mvin + "_t")
        sch[bias_op].pragma(cbias_ax_0_2, env.D_mvin + "_t")
        sch[output].pragma(fused_x_2, env.C_mvout + "_t")
    else:
        sch[cdata].pragma(cdata_ax_0_2, env.A_mvin)
        sch[bias_op].pragma(cbias_ax_0_2, env.D_mvin)
        sch[output].pragma(fused_x_2, env.C_mvout)

    # Apply tensorize
    I = data.shape[0] if data.shape[0] < env.DIM else cfg["tile_xo"].size[-1]
    K = weight.shape[0] if weight.shape[0] < env.DIM else cfg["tile_zo"].size[-1]
    J = weight.shape[1] if weight.shape[1] < env.DIM else cfg["tile_yo"].size[-1]

    sch[dense_stage_acc].tensorize(
        xi_o if cfg["exchange_axis"].val else yi_o,
        env.gemm(
            I,
            K,
            J,
            mode=cfg["WS/OS"].val,
            accum_patch=tvm.tir.IntImm("uint8", 0)
            if cfg["exchange_axis"].val or cfg["tile_zo"].size[1] != 1
            else xo_o.var,
        ),
    )

    # Generate configuration dictionary, in order to correctly generate the calls to the configuration instructions
    config_dict = {}
    config_dict["A_size"] = int(data.shape[1])
    config_dict["B_size"] = int(weight.shape[1])
    config_dict["C_size"] = int(output.shape[1])
    config_dict["A_private_stride"] = env.DIM
    config_dict["B_private_stride"] = env.DIM
    config_dict["execution_stride"] = 1
    config_dict["activation"] = 0
    config_dict["mode"] = cfg["WS/OS"].val
    config_dict["max_pixels_per_row"] = 1
    config_dict["scale"] = float(dense_stage.op.attrs["scale"])
    config_dict["padding_value"] = 0

    # Tag loops with pragmas to delimit the start and end of the Gemmini related code
    sch[output].pragma(axis_gemm_start, "gemm_start")
    sch[output].pragma(axis_gemm_start, "configs", str(config_dict))
    sch[output].pragma(axis_gemm_start, "gemm_end")

    return sch
