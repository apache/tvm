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

import numpy as np

import tvm
from tvm import relax
from tvm.relax.transform import DefaultSchedule
from tvm.script import relax as R, tir as T
import tvm.testing


def test_expand():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def broadcast_to(
            rxplaceholder: T.Buffer((T.int64(3), T.int64(1)), "float32"), var_T_broadcast_to: T.handle
        ):
            T.func_attr({"tir.noalias": True})
            x_0 = T.var("int64")
            x_1 = T.var("int64")
            T_broadcast_to = T.match_buffer(var_T_broadcast_to, (x_0, x_1))
            # with T.block("root"):
            for ax0, ax1 in T.grid(x_0, x_1):
                with T.block("T_broadcast_to"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, T.int64(0)])
                    T.writes(T_broadcast_to[v_ax0, v_ax1])
                    T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def broadcast_to(
            rxplaceholder: T.Buffer((T.int64(3), T.int64(1)), "float32"),
            var_T_broadcast_to: T.handle,
        ):
            T.func_attr({"tir.noalias": True})
            x_0 = T.int64()
            x_1 = T.int64()
            T_broadcast_to = T.match_buffer(var_T_broadcast_to, (x_0, x_1))
            # with T.block("root"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                for ax0_ax1_fused_2 in T.thread_binding(
                    T.int64(1024), thread="threadIdx.x"
                ):
                    for ax0_ax1_fused_0 in range(
                        (x_0 * x_1 + T.int64(262143)) // T.int64(262144)
                    ):
                        with T.block("T_broadcast_to"):
                            v_ax0 = T.axis.spatial(
                                x_0,
                                (
                                    (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1)
                                    * T.int64(1024)
                                    + ax0_ax1_fused_2
                                )
                                // x_1,
                            )
                            v_ax1 = T.axis.spatial(
                                x_1,
                                (
                                    (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1)
                                    * T.int64(1024)
                                    + ax0_ax1_fused_2
                                )
                                % x_1,
                            )
                            T.where(
                                (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1)
                                * T.int64(1024)
                                + ax0_ax1_fused_2
                                < x_0 * x_1
                            )
                            T.reads(rxplaceholder[v_ax0, T.int64(0)])
                            T.writes(T_broadcast_to[v_ax0, v_ax1])
                            T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]
    # fmt: on
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with tvm.transform.PassContext(opt_level=3):
        After = DefaultSchedule(target)(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
