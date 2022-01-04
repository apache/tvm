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


@dense_alter_layout.register(["cpu", "arm_cpu"])
def _alter_dense_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    data_tensor, weight_tensor = tinfos
    out_dtype = out_type.dtype
    M, K = get_const_tuple(data_tensor.shape)
    N, _ = get_const_tuple(weight_tensor.shape)

    impl, outs = relay.backend.te_compiler.select_implementation(
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
