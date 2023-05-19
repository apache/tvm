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

import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.script import tir as T
from tvm.target import Target
from tvm.te import create_prim_func


@tvm.script.ir_module
class Softmax_mn_after_inline:
    @T.prim_func
    def main(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.min_value("float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_2, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_4, i1_1 = T.axis.remap("SS", [i0_3, i1])
                T.block_attr({"axis": 1})
                T_softmax_norm[i0_4, i1_1] = (
                    T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4], dtype="float32")
                    / T_softmax_expsum[i0_4]
                )


def test_gpu_softmax_mn():
    @T.prim_func
    def softmax_mn_0(
        A: T.Buffer((256, 256), "float32"),
        T_softmax_norm: T.Buffer((256, 256), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                T.reads(A[i0_1, k])
                T.writes(T_softmax_maxelem[i0_1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_exp"):
                i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[i0_2, i1_1], T_softmax_maxelem[i0_2])
                T.writes(T_softmax_exp[i0_2, i1_1])
                T_softmax_exp[i0_2, i1_1] = T.exp(
                    A[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_4, k = T.axis.remap("SR", [i0_3, i1])
                T.reads(T_softmax_exp[i0_4, k])
                T.writes(T_softmax_expsum[i0_4])
                with T.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                T.reads(T_softmax_exp[i0_6, i1_2], T_softmax_expsum[i0_6])
                T.writes(T_softmax_norm[i0_6, i1_2])
                T.block_attr({"axis": 1})
                T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]

    @T.prim_func
    def softmax_mn_1(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        T.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_1 = T.axis.spatial(256, i0 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        T.reads(A[i0_1, k])
                        T.writes(T_softmax_maxelem_shared[i0_1])
                        with T.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                        T_softmax_maxelem_shared[i0_1] = T.max(
                            T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_exp"):
                        T.where(i1_0 * 512 + i1_1 < 256)
                        i0_2 = T.axis.spatial(256, i0)
                        i1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
                        T.reads(A[i0_2, i1], T_softmax_maxelem_shared[i0_2])
                        T.writes(T_softmax_exp[i0_2, i1])
                        T_softmax_exp[i0_2, i1] = T.exp(
                            A[i0_2, i1] - T_softmax_maxelem_shared[i0_2], dtype="float32"
                        )
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_4, k = T.axis.remap("SR", [i0_3, i1])
                T.reads(T_softmax_exp[i0_4, k])
                T.writes(T_softmax_expsum[i0_4])
                with T.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                T.reads(T_softmax_exp[i0_6, i1_2], T_softmax_expsum[i0_6])
                T.writes(T_softmax_norm[i0_6, i1_2])
                T.block_attr({"axis": 1})
                T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]

    @T.prim_func
    def softmax_mn_2(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
        T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                T.reads(A[i0_1, k])
                T.writes(T_softmax_maxelem[i0_1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_exp"):
                i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[i0_2, i1_1], T_softmax_maxelem[i0_2])
                T.writes(T_softmax_exp[i0_2, i1_1])
                T_softmax_exp[i0_2, i1_1] = T.exp(
                    A[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 32):
                for ax1_1 in T.thread_binding(8, thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        i0_4 = T.axis.spatial(256, i0_3 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 8 + ax1_1)
                        T.reads(T_softmax_exp[i0_4, k])
                        T.writes(T_softmax_expsum_shared[i0_4])
                        with T.init():
                            T_softmax_expsum_shared[i0_4] = T.float32(0)
                        T_softmax_expsum_shared[i0_4] = (
                            T_softmax_expsum_shared[i0_4] + T_softmax_exp[i0_4, k]
                        )
            for i1_0 in T.serial(32):
                for i1_1_1 in T.thread_binding(8, thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        i0_5 = T.axis.spatial(256, i0_3)
                        i1 = T.axis.spatial(256, i1_0 * 8 + i1_1_1)
                        T.reads(T_softmax_exp[i0_5, i1], T_softmax_expsum_shared[i0_5])
                        T.writes(T_softmax_norm[i0_5, i1])
                        T.block_attr({"axis": 1})
                        T_softmax_norm[i0_5, i1] = (
                            T_softmax_exp[i0_5, i1] / T_softmax_expsum_shared[i0_5]
                        )

    @T.prim_func
    def softmax_mn_3(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        T_softmax_exp = T.alloc_buffer([256, 256], dtype="float32")
        T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        for i0 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        T.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_1 = T.axis.spatial(256, i0 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        T.reads(A[i0_1, k])
                        T.writes(T_softmax_maxelem_shared[i0_1])
                        with T.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                        T_softmax_maxelem_shared[i0_1] = T.max(
                            T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_exp"):
                        T.where(i1_0 * 512 + i1_1 < 256)
                        i0_2 = T.axis.spatial(256, i0)
                        i1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
                        T.reads(A[i0_2, i1], T_softmax_maxelem_shared[i0_2])
                        T.writes(T_softmax_exp[i0_2, i1])
                        T_softmax_exp[i0_2, i1] = T.exp(
                            A[i0_2, i1] - T_softmax_maxelem_shared[i0_2], dtype="float32"
                        )
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 32):
                for ax1_1 in T.thread_binding(8, thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        i0_4 = T.axis.spatial(256, i0_3 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 8 + ax1_1)
                        T.reads(T_softmax_exp[i0_4, k])
                        T.writes(T_softmax_expsum_shared[i0_4])
                        with T.init():
                            T_softmax_expsum_shared[i0_4] = T.float32(0)
                        T_softmax_expsum_shared[i0_4] = (
                            T_softmax_expsum_shared[i0_4] + T_softmax_exp[i0_4, k]
                        )
            for i1_0 in T.serial(32):
                for i1_1 in T.thread_binding(8, thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        i0_5 = T.axis.spatial(256, i0_3)
                        i1 = T.axis.spatial(256, i1_0 * 8 + i1_1)
                        T.reads(T_softmax_exp[i0_5, i1], T_softmax_expsum_shared[i0_5])
                        T.writes(T_softmax_norm[i0_5, i1])
                        T.block_attr({"axis": 1})
                        T_softmax_norm[i0_5, i1] = (
                            T_softmax_exp[i0_5, i1] / T_softmax_expsum_shared[i0_5]
                        )

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 7),
    ]
    decision_2 = [
        ("SampleCategorical", 1),
    ]
    decision_3 = [
        ("SampleCategorical", 1),
        ("SampleCategorical", 7),
    ]
    mod = create_prim_func(te_workload.softmax_mn(n=256, m=256))
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[softmax_mn_0, softmax_mn_1, softmax_mn_2, softmax_mn_3],
        expected_decisions=[decision_0, decision_1, decision_2, decision_3],
    )


def test_gpu_softmax_mn_after_inline():
    @T.prim_func
    def softmax_mn_after_inline_0(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                T.reads(A[i0_1, k])
                T.writes(T_softmax_maxelem[i0_1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_2, k = T.axis.remap("SR", [i0, i1])
                T.reads(A[i0_2, k], T_softmax_maxelem[i0_2])
                T.writes(T_softmax_expsum[i0_2])
                with T.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_4, i1_1 = T.axis.remap("SS", [i0_3, i1])
                T.reads(A[i0_4, i1_1], T_softmax_maxelem[i0_4], T_softmax_expsum[i0_4])
                T.writes(T_softmax_norm[i0_4, i1_1])
                T.block_attr({"axis": 1})
                T_softmax_norm[i0_4, i1_1] = (
                    T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4], dtype="float32")
                    / T_softmax_expsum[i0_4]
                )

    @T.prim_func
    def softmax_mn_after_inline_1(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([256], dtype="float32")
        for i0, i1_0 in T.grid(256, 4):
            for i1_1 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("T_softmax_maxelem"):
                    i0_1 = T.axis.spatial(256, i0)
                    k = T.axis.reduce(256, i1_0 * 64 + i1_1)
                    T.reads(A[i0_1, k])
                    T.writes(T_softmax_maxelem[i0_1])
                    with T.init():
                        T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_expsum"):
                i0_2, k = T.axis.remap("SR", [i0, i1])
                T.reads(A[i0_2, k], T_softmax_maxelem[i0_2])
                T.writes(T_softmax_expsum[i0_2])
                with T.init():
                    T_softmax_expsum[i0_2] = T.float32(0)
                T_softmax_expsum[i0_2] = T_softmax_expsum[i0_2] + T.exp(
                    A[i0_2, k] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3, i1 in T.grid(256, 256):
            with T.block("T_softmax_norm"):
                i0_4, i1_1 = T.axis.remap("SS", [i0_3, i1])
                T.reads(A[i0_4, i1_1], T_softmax_maxelem[i0_4], T_softmax_expsum[i0_4])
                T.writes(T_softmax_norm[i0_4, i1_1])
                T.block_attr({"axis": 1})
                T_softmax_norm[i0_4, i1_1] = (
                    T.exp(A[i0_4, i1_1] - T_softmax_maxelem[i0_4], dtype="float32")
                    / T_softmax_expsum[i0_4]
                )

    @T.prim_func
    def softmax_mn_after_inline_2(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([256], dtype="float32")
        T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        for i0, i1 in T.grid(256, 256):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                T.reads(A[i0_1, k])
                T.writes(T_softmax_maxelem[i0_1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e38)
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        T.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_2 = T.axis.spatial(256, i0_3 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        T.reads(A[i0_2, k], T_softmax_maxelem[i0_2])
                        T.writes(T_softmax_expsum_shared[i0_2])
                        with T.init():
                            T_softmax_expsum_shared[i0_2] = T.float32(0)
                        T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                            A[i0_2, k] - T_softmax_maxelem[i0_2], dtype="float32"
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        T.where(i1_0 * 512 + i1_1 < 256)
                        i0_4 = T.axis.spatial(256, i0_3)
                        i1_1_1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
                        T.reads(
                            A[i0_4, i1_1_1], T_softmax_maxelem[i0_4], T_softmax_expsum_shared[i0_4]
                        )
                        T.writes(T_softmax_norm[i0_4, i1_1_1])
                        T.block_attr({"axis": 1})
                        T_softmax_norm[i0_4, i1_1_1] = (
                            T.exp(A[i0_4, i1_1_1] - T_softmax_maxelem[i0_4], dtype="float32")
                            / T_softmax_expsum_shared[i0_4]
                        )

    @T.prim_func
    def softmax_mn_after_inline_3(
        A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")
    ) -> None:
        T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        for i0_3 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        T.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_1 = T.axis.spatial(256, i0_3 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        T.reads(A[i0_1, k])
                        T.writes(T_softmax_maxelem_shared[i0_1])
                        with T.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e38)
                        T_softmax_maxelem_shared[i0_1] = T.max(
                            T_softmax_maxelem_shared[i0_1], A[i0_1, k]
                        )
            for ax0, ax1_0 in T.grid(1, 1):
                for ax1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        T.where(ax1_0 * 512 + ax1_1 < 256)
                        i0_2 = T.axis.spatial(256, i0_3 + ax0)
                        k = T.axis.reduce(256, ax1_0 * 512 + ax1_1)
                        T.reads(A[i0_2, k], T_softmax_maxelem_shared[i0_2])
                        T.writes(T_softmax_expsum_shared[i0_2])
                        with T.init():
                            T_softmax_expsum_shared[i0_2] = T.float32(0)
                        T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(
                            A[i0_2, k] - T_softmax_maxelem_shared[i0_2], dtype="float32"
                        )
            for i1_0 in T.serial(1):
                for i1_1 in T.thread_binding(512, thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        T.where(i1_0 * 512 + i1_1 < 256)
                        i0_4 = T.axis.spatial(256, i0_3)
                        i1_1_1 = T.axis.spatial(256, i1_0 * 512 + i1_1)
                        T.reads(
                            A[i0_4, i1_1_1],
                            T_softmax_maxelem_shared[i0_4],
                            T_softmax_expsum_shared[i0_4],
                        )
                        T.writes(T_softmax_norm[i0_4, i1_1_1])
                        T.block_attr({"axis": 1})
                        T_softmax_norm[i0_4, i1_1_1] = (
                            T.exp(A[i0_4, i1_1_1] - T_softmax_maxelem_shared[i0_4], dtype="float32")
                            / T_softmax_expsum_shared[i0_4]
                        )

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 4),
    ]
    decision_2 = [
        ("SampleCategorical", 7),
    ]
    decision_3 = [
        ("SampleCategorical", 7),
        ("SampleCategorical", 0),
    ]

    mod = Softmax_mn_after_inline
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[
            softmax_mn_after_inline_0,
            softmax_mn_after_inline_1,
            softmax_mn_after_inline_2,
            softmax_mn_after_inline_3,
        ],
        expected_decisions=[decision_0, decision_1, decision_2, decision_3],
    )


def test_gpu_batch_norm_bmn():
    @T.prim_func
    def batch_norm_bmn_0(A: T.Buffer((1, 512, 512), "float32"), D: T.Buffer(1, "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C = T.alloc_buffer([1], dtype="float32")
        for i0, i1, i2 in T.grid(1, 512, 512):
            with T.block("C"):
                b, i, j = T.axis.remap("SRR", [i0, i1, i2])
                T.reads(A[b, i, j])
                T.writes(C[b])
                with T.init():
                    C[b] = T.float32(0)
                C[b] = C[b] + A[b, i, j] * A[b, i, j]
        for i0 in T.serial(1):
            with T.block("D"):
                b = T.axis.spatial(1, i0)
                T.reads(C[b])
                T.writes(D[b])
                D[b] = T.sqrt(C[b], dtype="float32")

    @T.prim_func
    def batch_norm_bmn_1(A: T.Buffer((1, 512, 512), "float32"), D: T.Buffer(1, "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C_shared = T.alloc_buffer([1], dtype="float32", scope="shared")
        for i0_0 in T.serial(1):
            for ax0, ax1_ax2_fused_0 in T.grid(1, 1024):
                for ax1_ax2_fused_1 in T.thread_binding(256, thread="threadIdx.x"):
                    with T.block("C"):
                        b = T.axis.spatial(1, ax0)
                        i = T.axis.reduce(512, (ax1_ax2_fused_0 * 256 + ax1_ax2_fused_1) // 512)
                        j = T.axis.reduce(512, (ax1_ax2_fused_0 * 256 + ax1_ax2_fused_1) % 512)
                        T.reads(A[b, i, j])
                        T.writes(C_shared[b])
                        with T.init():
                            C_shared[b] = T.float32(0)
                        C_shared[b] = C_shared[b] + A[b, i, j] * A[b, i, j]
            for i0_1 in T.thread_binding(256, thread="threadIdx.x"):
                with T.block("D"):
                    T.where(i0_0 * 256 + i0_1 < 1)
                    b = T.axis.spatial(1, i0_0 * 256 + i0_1)
                    T.reads(C_shared[b])
                    T.writes(D[b])
                    D[b] = T.sqrt(C_shared[b], dtype="float32")

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 6),
    ]

    mod = create_prim_func(te_workload.norm_bmn(B=1, M=512, N=512))
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[batch_norm_bmn_0, batch_norm_bmn_1],
        expected_decisions=[decision_0, decision_1],
    )


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
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_32(
    idx: T.Buffer((1, 32), "int32"),
    val: T.Buffer((1, 32), "float32"),
    argmax_v0: T.Buffer((1,), "int32"),
    argmax_v1: T.Buffer((1,), "float32"),
) -> None:
    for i0, i1 in T.grid(1, 32):
        with T.block("argmax"):
            i = T.axis.spatial(1, i0)
            k = T.axis.reduce(32, i1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


def test_gpu_argmax():
    @T.prim_func
    def argmax_0(
        idx: T.Buffer((128, 128), "int32"),
        val: T.Buffer((128, 128), "float32"),
        argmax_v0: T.Buffer(128, "int32"),
        argmax_v1: T.Buffer(128, "float32"),
    ) -> None:
        # body
        # with T.block("root")
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
        # body
        # with T.block("root")
        for i0, i1_0 in T.grid(128, 2):
            for i1_1 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("argmax"):
                    i = T.axis.spatial(128, i0)
                    k = T.axis.reduce(128, i1_0 * 64 + i1_1)
                    T.reads(idx[i, k], val[i, k])
                    T.writes(argmax_v0[i], argmax_v1[i])
                    with T.init():
                        argmax_v0[i] = -1
                        argmax_v1[i] = T.float32(-3.4028234663852886e38)
                    v_argmax_v0: T.int32 = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
                    )
                    v_argmax_v1: T.float32 = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                    )
                    argmax_v0[i] = v_argmax_v0
                    argmax_v1[i] = v_argmax_v1

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 4),
    ]

    mod = argmax
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[argmax_0, argmax_1],
        expected_decisions=[decision_0, decision_1],
    )


def test_gpu_argmax_32():
    @T.prim_func
    def argmax_0(
        idx: T.Buffer((1, 32), "int32"),
        val: T.Buffer((1, 32), "float32"),
        argmax_v0: T.Buffer((1,), "int32"),
        argmax_v1: T.Buffer((1,), "float32"),
    ) -> None:
        # body
        # with T.block("root")
        for i0, i1 in T.grid(1, 32):
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
        idx: T.Buffer((1, 32), "int32"),
        val: T.Buffer((1, 32), "float32"),
        argmax_v0: T.Buffer((1,), "int32"),
        argmax_v1: T.Buffer((1,), "float32"),
    ) -> None:
        # body
        # with T.block("root")
        for i0, i1_0 in T.grid(1, 1):
            for i1_1 in T.thread_binding(64, thread="threadIdx.x"):
                with T.block("argmax"):
                    i = T.axis.spatial(1, i0)
                    k = T.axis.reduce(32, i1_0 * 64 + i1_1)
                    T.where(i1_0 * 64 + i1_1 < 32)
                    T.reads(idx[i, k], val[i, k])
                    T.writes(argmax_v0[i], argmax_v1[i])
                    with T.init():
                        argmax_v0[i] = -1
                        argmax_v1[i] = T.float32(-3.4028234663852886e38)
                    v_argmax_v0: T.int32 = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
                    )
                    v_argmax_v1: T.float32 = T.Select(
                        argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
                    )
                    argmax_v0[i] = v_argmax_v0
                    argmax_v1[i] = v_argmax_v1

    decision_0 = []  # type: ignore
    decision_1 = [
        ("SampleCategorical", 4),
    ]

    mod = argmax_32
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3090", host="llvm"),
        types=ms.schedule_rule.CrossThreadReduction,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[argmax_0, argmax_1],
        expected_decisions=[decision_0, decision_1],
    )


if __name__ == "__main__":
    test_gpu_softmax_mn()
    test_gpu_softmax_mn_after_inline()
    test_gpu_batch_norm_bmn()
    test_gpu_argmax()
    test_gpu_argmax_32()
