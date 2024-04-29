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

"""Dense alter op definitions for the `arm_cpu` device key."""

import tvm
from tvm import relay
from tvm import autotvm
from tvm import te

from ..nn import dense_alter_layout


@dense_alter_layout.register("arm_cpu")
def _alter_dense(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current

    _, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.dense"),
        attrs,
        tinfos,
        out_type,
        target,
    )
    workload = autotvm.task.get_workload(outs)
    if workload is None:
        # The best implementation is not an AutoTVM template,
        # we then assume it's not necessary to alter this op.
        return None

    cfg = dispatch_ctx.query(target, workload)
    topi_impl = workload[0]
    if topi_impl == "matmul.arm_cpu.sme":
        # Pre-compute transposed weights and convert to a matmul
        assert isinstance(
            inputs[1], relay.Constant
        ), "matmul_sme.arm_cpu requires weights be a Relay Constant"

        weight_dtype = tinfos[1].dtype
        weight_data = inputs[1].data.numpy()
        interleaved = weight_data.transpose()
        encoded_weight = relay.const(interleaved, weight_dtype)

        new_weight = te.placeholder((weight_data.shape), dtype=weight_dtype)
        new_workload = autotvm.task.args_to_workload(
            [tinfos[0], new_weight, None, out_type.dtype], topi_impl
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.matmul(
            inputs[0],
            encoded_weight,
            units=attrs.units,
            out_dtype=attrs.out_dtype,
            transpose_a=False,
            transpose_b=False,
        )

    # x86 schedules are used as a fallback
    return tvm.topi.x86.dense_alter_op._alter_dense_layout(attrs, inputs, tinfos, out_type)
