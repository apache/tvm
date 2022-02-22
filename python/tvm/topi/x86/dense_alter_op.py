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
"""Dense alter op functions for x86"""

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from .dense import _default_dense_pack_config
from ..utils import get_const_tuple
from ..nn import dense_alter_layout
from .utils import target_has_vnni
from .. import nn


def check_vnni_applicable(x, y):
    mcpu = tvm.target.Target.current().mcpu
    return (
        target_has_vnni(mcpu)
        and "int8" in x.dtype
        and "int8" in y.dtype
        and y.shape[-2] % 16 == 0
        and y.shape[-1] % 4 == 0
    )


@dense_alter_layout.register(["cpu", "arm_cpu"])
def _alter_dense_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    data_tensor, weight_tensor = tinfos
    out_dtype = out_type.dtype
    M, K = get_const_tuple(data_tensor.shape)
    N, _ = get_const_tuple(weight_tensor.shape)

    if check_vnni_applicable(data_tensor, weight_tensor) and data_tensor.dtype == "uint8":
        weight_layout = "NC16n4c"
        return relay.nn.contrib_dense_pack(inputs[0], inputs[1], weight_layout, None, out_dtype)

    _, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.dense"), attrs, tinfos, out_type, target
    )
    workload = autotvm.task.get_workload(outs)

    if workload:
        cfg = dispatch_ctx.query(target, workload)
        topi_impl = workload[0]
        if topi_impl == "dense_pack.x86":
            if cfg.is_fallback:
                _default_dense_pack_config(cfg, M, N, K)
            packw_bn = cfg["tile_x"].size[-1]
            weight_layout = "NC%dn" % packw_bn
            new_weight = te.placeholder(
                (N // packw_bn, K, packw_bn),
                dtype=weight_tensor.dtype,
            )
            # Relay dense doesn't have bias.
            new_workload = autotvm.task.args_to_workload(
                [
                    data_tensor,
                    new_weight,
                    None,
                    out_dtype,
                ],
                topi_impl,
            )
            dispatch_ctx.update(target, new_workload, cfg)
            return relay.nn.contrib_dense_pack(inputs[0], inputs[1], weight_layout, None, out_dtype)

    return None


def vnni_legalize(inputs, arg_types, op, attrs, need_expand=False):
    """Legalizes s8, s8 -> s32 GEMM op for VNNI."""
    if check_vnni_applicable(arg_types[0], arg_types[1]) and arg_types[0].dtype == "int8":
        x, y = inputs
        x = relay.cast(x, "int32")
        x = relay.add(x, relay.const(128, "int32"))
        x = relay.cast(x, "uint8")

        adjust_shift = relay.const(128, "int32") * relay.sum(relay.cast(y, "int32"), axis=[-1])

        if need_expand:
            adjust_shift = relay.expand_dims(adjust_shift, axis=1)

        out = op(x, y, **attrs)

        return relay.subtract(out, adjust_shift)

    return None


@nn.dense_legalize.register("cpu")
def _dense_legalize(attrs, inputs, arg_types):
    """Legalizes s8, s8 -> s32 dense for VNNI."""
    return vnni_legalize(inputs, arg_types, relay.nn.dense, attrs)


@nn.batch_matmul_legalize.register("cpu")
def _batch_matmul_legalize(attrs, inputs, arg_types):
    """Legalizes s8, s8 -> s32 batch_matmul for VNNI."""
    if attrs["transpose_a"] or not attrs["transpose_b"]:
        return None
    return vnni_legalize(inputs, arg_types, relay.nn.batch_matmul, attrs, need_expand=True)
