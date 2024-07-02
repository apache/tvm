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

# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Dense alter op definitions for the `arm_cpu` device key."""

import tvm
from tvm import relay
from tvm import autotvm
from tvm import te

from ..nn import dense_alter_layout


@dense_alter_layout.register("arm_cpu")
def _alter_dense(attrs, inputs, tinfos, out_type):
    from tvm.relay.op.nn import _make  # pylint: disable=import-outside-toplevel

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

        weight_dtype = tinfos[1].dtype
        N, K = tinfos[1].shape
        encoded_weight = inputs[1]

        # For dense the weights (rhs) are provided in transposed format,
        # i.e. they are of the shape (n, k).
        transpose_b = True

        # The SME schedule expects the rhs to be in the format (k, n). We can do this
        # transformation at compile time in the case of float32. Note: For the
        # float16->float32 schedule the transformation currently happens at runtime
        # with the ARM_SME_BLOCK2_2SVLx1SVL_FP16_TRANSPOSE_INTERLEAVE intrinsic.
        if weight_dtype == "float32":
            encoded_weight = relay.transpose(encoded_weight)
            transpose_b = False

        new_weight = te.placeholder(([K, N]), dtype=weight_dtype)

        new_workload = autotvm.task.args_to_workload(
            [tinfos[0], new_weight, None, out_type.dtype, False, transpose_b], topi_impl
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return _make.matmul(
            inputs[0],
            encoded_weight,
            attrs.units,
            attrs.out_dtype,
            False,
            transpose_b,
        )
    elif topi_impl == "dense_gemm.arm_cpu":

        weight_dtype = tinfos[1].dtype
        N, K = tinfos[1].shape

        encoded_weight = relay.transpose(inputs[1])
        new_weight = te.placeholder(([K, N]), dtype=weight_dtype)

        new_workload = autotvm.task.args_to_workload(
            [tinfos[0], new_weight, None, out_type.dtype, False, False], topi_impl
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return _make.matmul(
            inputs[0],
            encoded_weight,
            attrs.units,
            attrs.out_dtype,
            False,
            False,
        )

    # x86 schedules are used as a fallback
    return tvm.topi.x86.dense_alter_op._alter_dense_layout(attrs, inputs, tinfos, out_type)
