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

import tvm
from tvm import te
from tvm import autotvm

from tvm.contrib.gemmini.environment import Environment

ENV = Environment.instance()


@autotvm.register_topi_compute("contrib.gemmini.max_pool2d")
# def conv2d(args,attrs):
def max_pool2d(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity,
    data: tvm.te.tensor.Tensor,
    weights: tvm.te.tensor.Tensor,
    pool_size: tvm.ir.container.Array,
    pool_strides: tvm.ir.container.Array,
    pool_dilation: tvm.ir.container.Array,
    pool_padding: tvm.ir.container.Array,
) -> tvm.te.tensor.Tensor:
    """Computation definition to run a max pooling layer on Gemmini.
    Uses a trick: we call a dw convolution + max pooling, but all weights are 1.
    So the depthwise convolution does nothing, and the Gemmini accelerator takes care
    internally of applying the max pooling.

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        data (tvm.te.tensor.Tensor): Input feature map
        weights (tvm.te.tensor.Tensor): Weights... just all ones, needed by the called function
        pool_size (tvm.ir.container.Array): Pooling window size
        pool_strides (tvm.ir.container.Array): Pooling window strides
        pool_dilation (tvm.ir.container.Array): Pooling window dilation (not used for now)
        pool_padding (tvm.ir.container.Array): Pooling window padding

    Returns:
        tvm.te.tensor.Tensor: max pool2d operator result
    """

    assert len(data.shape) == 4

    def irb_builder_func(ins, outs):
        irb = tvm.tir.ir_builder.create()

        if ENV.supports_non_zero_padding:
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "tiled_conv_dw_auto",
                    ins[0].shape[0],  # BATCH_SIZE,
                    ins[0].shape[1],  # IN_DIM,
                    ins[0].shape[3],  # IN_CHANNELS,
                    ins[0].shape[1],  # OUT_DIM,
                    1,
                    0,
                    0,
                    1,
                    ins[0].access_ptr("r"),
                    ins[1].access_ptr("r"),
                    0,
                    outs[0].access_ptr("w"),
                    0,
                    1.0,
                    pool_size[0],
                    pool_strides[0],
                    pool_padding[0],
                    1,
                )
            )
        else:
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "tiled_conv_dw_auto",
                    ins[0].shape[0],  # BATCH_SIZE,
                    ins[0].shape[1],  # IN_DIM,
                    ins[0].shape[3],  # IN_CHANNELS,
                    ins[0].shape[1],  # OUT_DIM,
                    1,
                    0,
                    1,
                    ins[0].access_ptr("r"),
                    ins[1].access_ptr("r"),
                    0,
                    outs[0].access_ptr("w"),
                    0,
                    1.0,
                    pool_size[0],
                    pool_strides[0],
                    pool_padding[0],
                    1,
                )
            )
        irb.emit(tvm.tir.call_extern("", "gemmini_fence"))

        return irb.get()

    res = te.extern(
        (1,),
        [data, weights],
        lambda ins, outs: irb_builder_func(ins, outs),  # pylint: disable=W0108
        dtype="int8",
    )

    # TODO (FP): add correct FLOPS
    # cfg.add_flop(2 * np.prod(topi.utils.get_const_tuple(oshape)) * KH * KW * IC)

    return res


@autotvm.register_topi_schedule("contrib.gemmini.max_pool2d")
def schedule_max_pool2d(
    cfg: tvm.autotvm.task.space.FallbackConfigEntity, outs: tvm.ir.container.Array
) -> tvm.te.schedule.Schedule:
    """Schedule definition for Gemmini's max pool2d operator

    Args:
        cfg (tvm.autotvm.task.space.FallbackConfigEntity): AutoTVM configuration entity
        outs (tvm.ir.container.Array): Output tensors

    Returns:
        tvm.te.schedule.Schedule: transformed schedule
    """
    assert len(outs) == 1
    output = outs[0]
    sch = te.create_schedule(output.op)
    return sch
