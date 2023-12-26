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
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.script import tir as T
from tvm.target import Target
from tvm.te import create_prim_func


def test_cpu_matmul():
    @T.prim_func
    def cpu_matmul_0(
        A: T.Buffer((4, 512), "float32"),
        B: T.Buffer((512, 4), "float32"),
        C: T.Buffer((4, 4), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, i1, i2 in T.grid(4, 4, 512):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(A[i, k], B[k, j])
                T.writes(C[i, j])
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[k, j]

    @T.prim_func
    def cpu_matmul_1(
        A: T.Buffer((4, 512), "float32"),
        B: T.Buffer((512, 4), "float32"),
        C: T.Buffer((4, 4), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        C_rf = T.alloc_buffer([4, 4, 128], dtype="float32")
        for i0, i1, i2_0, i2_1 in T.grid(4, 4, 4, 128):
            with T.block("C_rf"):
                vi2_1, i, j, vi2_0 = T.axis.remap("SSSR", [i2_1, i0, i1, i2_0])
                T.reads(A[i, vi2_0 * 128 + vi2_1], B[vi2_0 * 128 + vi2_1, j])
                T.writes(C_rf[i, j, vi2_1])
                with T.init():
                    C_rf[i, j, vi2_1] = T.float32(0)
                C_rf[i, j, vi2_1] = (
                    C_rf[i, j, vi2_1] + A[i, vi2_0 * 128 + vi2_1] * B[vi2_0 * 128 + vi2_1, j]
                )
        for i0, i1, i2_1 in T.grid(4, 4, 128):
            with T.block("C"):
                vi2_1, i, j = T.axis.remap("RSS", [i2_1, i0, i1])
                T.reads(C_rf[i, j, vi2_1])
                T.writes(C[i, j])
                T.block_attr({"meta_schedule.random_compute_producer": 1})
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + C_rf[i, j, vi2_1]

    @T.prim_func
    def cpu_matmul_2(
        A: T.Buffer((4, 512), "float32"),
        B: T.Buffer((512, 4), "float32"),
        C: T.Buffer((4, 4), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        C_rf = T.alloc_buffer([4, 4, 4], dtype="float32")
        for i0, i1, i2_0, i2_1 in T.grid(4, 4, 4, 128):
            with T.block("C_rf"):
                vi2_0, i, j, vi2_1 = T.axis.remap("SSSR", [i2_0, i0, i1, i2_1])
                T.reads(A[i, vi2_0 * 128 + vi2_1], B[vi2_0 * 128 + vi2_1, j])
                T.writes(C_rf[i, j, vi2_0])
                with T.init():
                    C_rf[i, j, vi2_0] = T.float32(0)
                C_rf[i, j, vi2_0] = (
                    C_rf[i, j, vi2_0] + A[i, vi2_0 * 128 + vi2_1] * B[vi2_0 * 128 + vi2_1, j]
                )
        for i0, i1, i2_0 in T.grid(4, 4, 4):
            with T.block("C"):
                vi2_0, i, j = T.axis.remap("RSS", [i2_0, i0, i1])
                T.reads(C_rf[i, j, vi2_0])
                T.writes(C[i, j])
                T.block_attr({"meta_schedule.random_compute_producer": 1})
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + C_rf[i, j, vi2_0]

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SamplePerfectTile", [4, 128]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [4, 128]),
    ]
    mod = create_prim_func(te_workload.matmul(n=4, m=4, k=512))
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target("llvm --num-cores=32"),
        types=ms.schedule_rule.AddRFactor,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cpu_matmul_0, cpu_matmul_1, cpu_matmul_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_argmax():
    @T.prim_func
    def argmax(
        idx: T.Buffer((128, 128), "int32"),
        val: T.Buffer((128, 128), "float32"),
        argmax_v0: T.Buffer((128,), "int32"),
        argmax_v1: T.Buffer((128,), "float32"),
    ) -> None:
        for i0, i1 in T.grid(128, 128):
            with T.block("argmax"):
                i = T.axis.spatial(128, i0)
                k = T.axis.reduce(128, i1)
                T.reads(idx[i, k], val[i, k])
                T.writes(argmax_v0[i], argmax_v1[i])
                with T.init():
                    argmax_v0[i] = -1
                    argmax_v1[i] = T.min_value("float32")
                v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
                v_argmax_v1: T.float32 = T.Select(
                    argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                )
                argmax_v0[i] = v_argmax_v0
                argmax_v1[i] = v_argmax_v1

    @T.prim_func
    def argmax_0(
        idx: T.Buffer((128, 128), "int32"),
        val: T.Buffer((128, 128), "float32"),
        argmax_v0: T.Buffer(128, "int32"),
        argmax_v1: T.Buffer(128, "float32"),
    ) -> None:
        for i0, i1 in T.grid(128, 128):
            with T.block("argmax"):
                i, k = T.axis.remap("SR", [i0, i1])
                T.reads(idx[i, k], val[i, k])
                T.writes(argmax_v0[i], argmax_v1[i])
                with T.init():
                    argmax_v0[i] = -1
                    argmax_v1[i] = T.float32(-3.4028234663852886e38)
                v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
                v_argmax_v1: T.float32 = T.Select(
                    argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                )
                argmax_v0[i] = v_argmax_v0
                argmax_v1[i] = v_argmax_v1

    @T.prim_func
    def argmax_1(
        idx: T.Buffer((128, 128), "int32"),
        val: T.Buffer((128, 128), "float32"),
        argmax_v0: T.Buffer(128, "int32"),
        argmax_v1: T.Buffer(128, "float32"),
    ) -> None:
        argmax_v0_rf = T.alloc_buffer([128, 16], dtype="int32")
        argmax_v1_rf = T.alloc_buffer([128, 16], dtype="float32")
        for i0, i1_0, i1_1 in T.grid(128, 8, 16):
            with T.block("argmax_rf"):
                vi1_1, i, vi1_0 = T.axis.remap("SSR", [i1_1, i0, i1_0])
                T.reads(idx[i, vi1_0 * 16 + vi1_1], val[i, vi1_0 * 16 + vi1_1])
                T.writes(argmax_v0_rf[i, vi1_1], argmax_v1_rf[i, vi1_1])
                with T.init():
                    argmax_v0_rf[i, vi1_1] = -1
                    argmax_v1_rf[i, vi1_1] = T.float32(-3.4028234663852886e38)
                v_argmax_v0_rf: T.int32 = T.Select(
                    argmax_v1_rf[i, vi1_1] >= val[i, vi1_0 * 16 + vi1_1],
                    argmax_v0_rf[i, vi1_1],
                    idx[i, vi1_0 * 16 + vi1_1],
                )
                v_argmax_v1_rf: T.float32 = T.Select(
                    argmax_v1_rf[i, vi1_1] >= val[i, vi1_0 * 16 + vi1_1],
                    argmax_v1_rf[i, vi1_1],
                    val[i, vi1_0 * 16 + vi1_1],
                )
                argmax_v0_rf[i, vi1_1] = v_argmax_v0_rf
                argmax_v1_rf[i, vi1_1] = v_argmax_v1_rf
        for i0, i1_1 in T.grid(128, 16):
            with T.block("argmax"):
                vi1_1, i = T.axis.remap("RS", [i1_1, i0])
                T.reads(argmax_v0_rf[i, vi1_1], argmax_v1_rf[i, vi1_1])
                T.writes(argmax_v0[i], argmax_v1[i])
                T.block_attr({"meta_schedule.random_compute_producer": 1})
                with T.init():
                    argmax_v0[i] = -1
                    argmax_v1[i] = T.float32(-3.4028234663852886e38)
                v_argmax_v0: T.int32 = T.Select(
                    argmax_v1[i] >= argmax_v1_rf[i, vi1_1], argmax_v0[i], argmax_v0_rf[i, vi1_1]
                )
                v_argmax_v1: T.float32 = T.Select(
                    argmax_v1[i] >= argmax_v1_rf[i, vi1_1], argmax_v1[i], argmax_v1_rf[i, vi1_1]
                )
                argmax_v0[i] = v_argmax_v0
                argmax_v1[i] = v_argmax_v1

    @T.prim_func
    def argmax_2(
        idx: T.Buffer((128, 128), "int32"),
        val: T.Buffer((128, 128), "float32"),
        argmax_v0: T.Buffer(128, "int32"),
        argmax_v1: T.Buffer(128, "float32"),
    ) -> None:
        # body
        # with T.block("root")
        argmax_v0_rf = T.alloc_buffer([128, 8], dtype="int32")
        argmax_v1_rf = T.alloc_buffer([128, 8], dtype="float32")
        for i0, i1_0, i1_1 in T.grid(128, 8, 16):
            with T.block("argmax_rf"):
                vi1_0, i, vi1_1 = T.axis.remap("SSR", [i1_0, i0, i1_1])
                T.reads(idx[i, vi1_0 * 16 + vi1_1], val[i, vi1_0 * 16 + vi1_1])
                T.writes(argmax_v0_rf[i, vi1_0], argmax_v1_rf[i, vi1_0])
                with T.init():
                    argmax_v0_rf[i, vi1_0] = -1
                    argmax_v1_rf[i, vi1_0] = T.float32(-3.4028234663852886e38)
                v_argmax_v0_rf: T.int32 = T.Select(
                    argmax_v1_rf[i, vi1_0] >= val[i, vi1_0 * 16 + vi1_1],
                    argmax_v0_rf[i, vi1_0],
                    idx[i, vi1_0 * 16 + vi1_1],
                )
                v_argmax_v1_rf: T.float32 = T.Select(
                    argmax_v1_rf[i, vi1_0] >= val[i, vi1_0 * 16 + vi1_1],
                    argmax_v1_rf[i, vi1_0],
                    val[i, vi1_0 * 16 + vi1_1],
                )
                argmax_v0_rf[i, vi1_0] = v_argmax_v0_rf
                argmax_v1_rf[i, vi1_0] = v_argmax_v1_rf
        for i0, i1_0 in T.grid(128, 8):
            with T.block("argmax"):
                vi1_0, i = T.axis.remap("RS", [i1_0, i0])
                T.reads(argmax_v0_rf[i, vi1_0], argmax_v1_rf[i, vi1_0])
                T.writes(argmax_v0[i], argmax_v1[i])
                T.block_attr({"meta_schedule.random_compute_producer": 1})
                with T.init():
                    argmax_v0[i] = -1
                    argmax_v1[i] = T.float32(-3.4028234663852886e38)
                v_argmax_v0: T.int32 = T.Select(
                    argmax_v1[i] >= argmax_v1_rf[i, vi1_0], argmax_v0[i], argmax_v0_rf[i, vi1_0]
                )
                v_argmax_v1: T.float32 = T.Select(
                    argmax_v1[i] >= argmax_v1_rf[i, vi1_0], argmax_v1[i], argmax_v1_rf[i, vi1_0]
                )
                argmax_v0[i] = v_argmax_v0
                argmax_v1[i] = v_argmax_v1

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SamplePerfectTile", [8, 16]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [8, 16]),
    ]
    mod = argmax
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target("llvm --num-cores=32"),
        types=ms.schedule_rule.AddRFactor,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[argmax_0, argmax_1, argmax_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


if __name__ == "__main__":
    test_cpu_matmul()
    test_cpu_argmax()
