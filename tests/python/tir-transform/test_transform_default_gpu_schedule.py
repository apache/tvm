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
# pylint: disable=invalid-name,,missing-function-docstring
import tvm
from tvm.tir.transform import DefaultGPUSchedule
from tvm.script import tir as T
import tvm.testing


def test_broadcast_to_symbolic():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
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
            for ax0, ax1 in T.grid(x_0, x_1):
                with T.block("T_broadcast_to"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, T.int64(0)])
                    T.writes(T_broadcast_to[v_ax0, v_ax1])
                    T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def broadcast_to(rxplaceholder: T.Buffer((T.int64(3), T.int64(1)), "float32"), var_T_broadcast_to: T.handle):
            T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
            x_0, x_1 = T.int64(), T.int64()
            T_broadcast_to = T.match_buffer(var_T_broadcast_to, (x_0, x_1))
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                for ax0_ax1_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax0_ax1_fused_0 in range((x_0 * x_1 + T.int64(262143)) // T.int64(262144)):
                        with T.block("T_broadcast_to"):
                            v_ax0 = T.axis.spatial(x_0, (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) % (x_1 * x_0) // x_1)
                            v_ax1 = T.axis.spatial(x_1, (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) % x_1)
                            T.where((ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1) * T.int64(1024) + ax0_ax1_fused_2 < x_0 * x_1)
                            T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_matmul():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def matmul(
            A: T.Buffer((32, 32), "float16"),
            B: T.Buffer((32, 32), "float16"),
            C: T.Buffer((32, 32), "float16"),
        ):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # with T.block("root"):
            for i, j, k in T.grid(32, 32, 32):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    T.reads(A[v_i, v_k], B[v_k, v_j])
                    T.writes(C[v_i, v_j])
                    with T.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @T.prim_func
        def matmul_gpu(
            A: T.Buffer((32, 32), "float16"),
            B: T.Buffer((32, 32), "float16"),
            C: T.Buffer((32, 32), "float16"),
        ):
            T.func_attr({"global_symbol": "main",
                         "target": T.target({"arch": "sm_86",
                                             "keys": ["cuda", "gpu"],
                                             "kind": "cuda",
                                             "max_num_threads": 1024,
                                             "tag": "",
                                             "thread_warp_size": 32}),
                         "tir.noalias": True})
            # with T.block("root"):
            for i, j, k in T.grid(32, 32, 32):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    T.reads(A[v_i, v_k], B[v_k, v_j])
                    T.writes(C[v_i, v_j])
                    with T.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @T.prim_func
        def matmul_cpu(
            A: T.Buffer((32, 32), "float16"),
            B: T.Buffer((32, 32), "float16"),
            C: T.Buffer((32, 32), "float16"),
        ):
            T.func_attr({"global_symbol": "main",
                         "target": T.target({"keys": ["cpu"], "kind": "llvm", "tag": ""}),
                        "tir.noalias": True})
            # with T.block("root"):
            for i, j, k in T.grid(32, 32, 32):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    T.reads(A[v_i, v_k], B[v_k, v_j])
                    T.writes(C[v_i, v_j])
                    with T.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def matmul(
            A: T.Buffer((32, 32), "float16"),
            B: T.Buffer((32, 32), "float16"),
            C: T.Buffer((32, 32), "float16"),
        ):
            T.func_attr({"tir.is_scheduled": True, "global_symbol": "main", "tir.noalias": True})
            # with T.block("root"):
            for i_j_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for i_j_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                    for k in range(32):
                        with T.block("C"):
                            v_i = T.axis.spatial(
                                32, (i_j_fused_0 * 1024 + i_j_fused_1) // 32
                            )
                            v_j = T.axis.spatial(
                                32, (i_j_fused_0 * 1024 + i_j_fused_1) % 32
                            )
                            v_k = T.axis.reduce(32, k)
                            T.reads(A[v_i, v_k], B[v_k, v_j])
                            T.writes(C[v_i, v_j])
                            with T.init():
                                C[v_i, v_j] = T.float16(0)
                            C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @T.prim_func
        def matmul_cpu(A: T.Buffer((32, 32), "float16"), B: T.Buffer((32, 32), "float16"), C: T.Buffer((32, 32), "float16")):
            T.func_attr({"global_symbol": "main", "target": T.target({"keys": ["cpu"], "kind": "llvm", "tag": ""}), "tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i, j, k in T.grid(32, 32, 32):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    T.reads(A[v_i, v_k], B[v_k, v_j])
                    T.writes(C[v_i, v_j])
                    with T.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

        @T.prim_func
        def matmul_gpu(A: T.Buffer((32, 32), "float16"), B: T.Buffer((32, 32), "float16"), C: T.Buffer((32, 32), "float16")):
            T.func_attr({"global_symbol": "main", "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i_j_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for i_j_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                    for k in range(32):
                        with T.block("C"):
                            v_i = T.axis.spatial(32, (i_j_fused_0 * 1024 + i_j_fused_1) // 32)
                            v_j = T.axis.spatial(32, (i_j_fused_0 * 1024 + i_j_fused_1) % 32)
                            v_k = T.axis.reduce(32, k)
                            T.reads(A[v_i, v_k], B[v_k, v_j])
                            T.writes(C[v_i, v_j])
                            with T.init():
                                C[v_i, v_j] = T.float16(0)
                            C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_add():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def add(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add(
            rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"),
            rxplaceholder_1: T.Buffer(
                (T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"
            ),
            T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32"),
        ):
            T.func_attr({"tir.is_scheduled": True,  "tir.noalias": True})
            # with T.block("root"):
            for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_i2_i3_fused_1 in T.thread_binding(
                    T.int64(72), thread="threadIdx.x"
                ):
                    with T.block("T_add"):
                        ax0 = T.axis.spatial(
                            T.int64(4),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            // T.int64(18),
                        )
                        ax1 = T.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(18)
                            // T.int64(6),
                        )
                        ax2 = T.axis.spatial(
                            T.int64(2),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(6)
                            // T.int64(3),
                        )
                        ax3 = T.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(3),
                        )
                        T.reads(
                            rxplaceholder[T.int64(0), ax2, ax3],
                            rxplaceholder_1[ax0, ax1, ax2, T.int64(0)],
                        )
                        T.writes(T_add[ax0, ax1, ax2, ax3])
                        T_add[ax0, ax1, ax2, ax3] = (
                            rxplaceholder[T.int64(0), ax2, ax3]
                            + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
                        )

    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_full():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "int32"), T_full: T.Buffer((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def full(
            rxplaceholder: T.Buffer((), "int32"),
            T_full: T.Buffer((T.int64(2), T.int64(3)), "int32"),
        ):
            T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
            # with T.block("root"):
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(6), thread="threadIdx.x"):
                    with T.block("T_full"):
                        ax0 = T.axis.spatial(
                            T.int64(2),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) // T.int64(3),
                        )
                        ax1 = T.axis.spatial(
                            T.int64(3),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) % T.int64(3),
                        )
                        T.reads(rxplaceholder[()])
                        T.writes(T_full[ax0, ax1])
                        T_full[ax0, ax1] = rxplaceholder[()]

    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_scheduled():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off

    @tvm.script.ir_module
    class Scheduled:
        @T.prim_func
        def full(
            rxplaceholder: T.Buffer((), "int32"),
            T_full: T.Buffer((T.int64(2), T.int64(3)), "int32"),
        ):
            T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
            # with T.block("root"):
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(6), thread="threadIdx.x"):
                    with T.block("T_full"):
                        ax0 = T.axis.spatial(
                            T.int64(2),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) // T.int64(3),
                        )
                        ax1 = T.axis.spatial(
                            T.int64(3),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) % T.int64(3),
                        )
                        T.reads(rxplaceholder[()])
                        T.writes(T_full[ax0, ax1])
                        T_full[ax0, ax1] = rxplaceholder[()]

    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=3):
        # should do nothing
        After = DefaultGPUSchedule()(Scheduled)
    tvm.ir.assert_structural_equal(After, Scheduled)


def test_multiple():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def add(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]

        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "int32"), T_full: T.Buffer((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add(
            rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"),
            rxplaceholder_1: T.Buffer(
                (T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"
            ),
            T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32"),
        ):
            T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
            # with T.block("root"):
            for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_i2_i3_fused_1 in T.thread_binding(
                    T.int64(72), thread="threadIdx.x"
                ):
                    with T.block("T_add"):
                        ax0 = T.axis.spatial(
                            T.int64(4),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            // T.int64(18),
                        )
                        ax1 = T.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(18)
                            // T.int64(6),
                        )
                        ax2 = T.axis.spatial(
                            T.int64(2),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(6)
                            // T.int64(3),
                        )
                        ax3 = T.axis.spatial(
                            T.int64(3),
                            (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1)
                            % T.int64(3),
                        )
                        T.reads(
                            rxplaceholder[T.int64(0), ax2, ax3],
                            rxplaceholder_1[ax0, ax1, ax2, T.int64(0)],
                        )
                        T.writes(T_add[ax0, ax1, ax2, ax3])
                        T_add[ax0, ax1, ax2, ax3] = (
                            rxplaceholder[T.int64(0), ax2, ax3]
                            + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
                        )

        @T.prim_func
        def full(
            rxplaceholder: T.Buffer((), "int32"),
            T_full: T.Buffer((T.int64(2), T.int64(3)), "int32"),
        ):
            T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
            # with T.block("root"):
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(6), thread="threadIdx.x"):
                    with T.block("T_full"):
                        ax0 = T.axis.spatial(
                            T.int64(2),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) // T.int64(3),
                        )
                        ax1 = T.axis.spatial(
                            T.int64(3),
                            (i0_i1_fused_0 * T.int64(6) + i0_i1_fused_1) % T.int64(3),
                        )
                        T.reads(rxplaceholder[()])
                        T.writes(T_full[ax0, ax1])
                        T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=3):
        After = DefaultGPUSchedule()(Before)
    assert tvm.ir.structural_equal(After, Expected)


def test_add_on_metal():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def add(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Buffer((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
            for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(72), thread="threadIdx.x"):
                    with T.block("T_add"):
                        ax0 = T.axis.spatial(T.int64(4), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) // T.int64(18))
                        ax1 = T.axis.spatial(T.int64(3), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) % T.int64(18) // T.int64(6))
                        ax2 = T.axis.spatial(T.int64(2), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) % T.int64(6) // T.int64(3))
                        ax3 = T.axis.spatial(T.int64(3), (i0_i1_i2_i3_fused_0 * T.int64(72) + i0_i1_i2_i3_fused_1) % T.int64(3))
                        T.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                        T.writes(T_add[ax0, ax1, ax2, ax3])
                        T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("apple/m1-gpu")
    with target, tvm.transform.PassContext(opt_level=0):
        mod = DefaultGPUSchedule()(Before)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_scalar_add():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def add(rxplaceholder: T.Buffer((), "int64"), T_add: T.Buffer((), "int64")):
            T.func_attr({"tir.noalias": T.bool(True)})
            with T.block("T_add"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(rxplaceholder[()])
                T.writes(T_add[()])
                T_add[()] = rxplaceholder[()] + T.int64(1)

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add(rxplaceholder: T.Buffer((), "int64"), T_add: T.Buffer((), "int64")):
            T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for u_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for u_fused_1 in T.thread_binding(1, thread="threadIdx.x"):
                    with T.block("T_add"):
                        vi = T.axis.spatial(1, T.int64(0))
                        T.reads(rxplaceholder[()])
                        T.writes(T_add[()])
                        T_add[()] = rxplaceholder[()] + T.int64(1)
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=0):
        mod = DefaultGPUSchedule()(Before)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sum():
    # sum has two reduction axes and no spatial axis
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def sum(A: T.Buffer((T.int64(2), T.int64(2)), "float64"), A_red: T.Buffer((), "float64")):
            for k0, k1 in T.grid(T.int64(2), T.int64(2)):
                with T.block("A_red"):
                    v_k0, v_k1 = T.axis.remap("RR", [k0, k1])
                    with T.init():
                        A_red[()] = T.float64(0)
                    A_red[()] = A_red[()] + A[v_k0, v_k1]

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def sum(A: T.Buffer((T.int64(2), T.int64(2)), "float64"), A_red: T.Buffer((), "float64")):
            T.func_attr({"tir.is_scheduled": T.bool(True)})
            # with T.block("root"):
            for u_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
                for u_fused_1 in T.thread_binding(1, thread="threadIdx.x"):
                    for k0, k1 in T.grid(T.int64(2), T.int64(2)):
                        with T.block("A_red"):
                            v_k0, v_k1 = T.axis.remap("RR", [k0, k1])
                            T.reads(A[v_k0, v_k1])
                            T.writes(A_red[()])
                            with T.init():
                                A_red[()] = T.float64(0)
                            A_red[()] = A_red[()] + A[v_k0, v_k1]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=0):
        mod = DefaultGPUSchedule()(Before)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
