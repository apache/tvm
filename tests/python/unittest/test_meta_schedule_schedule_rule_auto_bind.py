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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def element_wise(var_A: T.handle, var_B: T.handle) -> None:
    A = T.match_buffer(var_A, [512, 512], dtype="float32")
    B = T.match_buffer(var_B, [512, 512], dtype="float32")
    for i, j in T.grid(512, 512):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] + 1.0


@T.prim_func
def reduction_loop_only(
    A: T.Buffer(2, "float32"),
    B: T.Buffer(2, "float32"),
    C: T.Buffer((), "float32"),
) -> None:
    for i0 in T.serial(2):
        with T.block("C"):
            k0 = T.axis.reduce(2, i0)
            T.reads(A[k0], B[k0])
            T.writes(C[()])
            with T.init():
                C[()] = T.float32(1.0)
            C[()] = T.min(C[()], A[k0] / B[k0])


@T.prim_func
def zero_dim_add(
    A: T.Buffer((), "float32"),
    B: T.Buffer((), "float32"),
    C: T.Buffer((), "float32"),
) -> None:
    with T.block("C"):
        vi = T.axis.spatial(1, 0)
        C[()] = A[()] + B[()]


def test_cuda_element_wise():
    @T.prim_func
    def elementwise_0(
        A: T.Buffer((512, 512), "float32"),
        B: T.Buffer((512, 512), "float32"),
    ) -> None:
        # body
        # with T.block("root")
        for i_j_fused_0 in T.thread_binding(256, thread="blockIdx.x"):
            for i_j_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) // 512)
                    vj = T.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) % 512)
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

    decision_0 = [
        ("SampleCategorical", 5),
    ]
    mod = element_wise
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[elementwise_0],
        expected_decisions=[decision_0],
    )


def test_cuda_reduction_loop_only():
    @T.prim_func
    def reduction_loop_only_0(
        A: T.Buffer(2, "float32"),
        B: T.Buffer(2, "float32"),
        C: T.Buffer((), "float32"),
    ) -> None:
        for u_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
            for u_fused_1 in T.thread_binding(1, thread="threadIdx.x"):
                for i0 in T.serial(2):
                    with T.block("C"):
                        k0 = T.axis.reduce(2, i0)
                        T.reads(A[k0], B[k0])
                        T.writes(C[()])
                        with T.init():
                            C[()] = T.float32(1)
                        C[()] = T.min(C[()], A[k0] / B[k0])

    mod = reduction_loop_only
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[reduction_loop_only_0],
        expected_decisions=[[]],
    )


def test_cuda_zero_dim_add():
    @T.prim_func
    def zero_dim_add_0(
        A: T.Buffer((), "float32"),
        B: T.Buffer((), "float32"),
        C: T.Buffer((), "float32"),
    ) -> None:
        for u_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
            for u_fused_1 in T.thread_binding(1, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(1, 0)
                    T.reads(A[()], B[()])
                    T.writes(C[()])
                    C[()] = A[()] + B[()]

    mod = zero_dim_add
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[zero_dim_add_0],
        expected_decisions=[[]],
    )


if __name__ == "__main__":
    test_cuda_element_wise()
    test_cuda_reduction_loop_only()
    test_cuda_zero_dim_add()
