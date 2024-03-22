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
# pylint: disable=missing-docstring

import tempfile
import pytest

from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R


from tvm.dlight.benchmark import (
    benchmark,
    benchmark_prim_func,
    benchmark_relax_func,
    extract_prim_func,
    extract_from_relax,
    extract_func_info_from_prim_func,
)
import tvm.testing

# pylint: disable=no-self-argument,invalid-name,line-too-long,no-method-argument
# fmt: off
@I.ir_module
class Module:
    @T.prim_func
    def full1(var_T_full: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        T_full = T.match_buffer(var_T_full, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
            with T.block("T_full"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads()
                T.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                T_full[v_ax0, v_ax1, v_ax2, v_ax3] = T.float16(1.0)

    @T.prim_func
    def full2(var_T_full: T.handle):
        T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
        n = T.int64()
        T_full = T.match_buffer(var_T_full, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, T.int64(128)):
            with T.block("T_full"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads()
                T.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                T_full[v_ax0, v_ax1, v_ax2, v_ax3] = T.float16(1.0)

    @T.prim_func
    def matmul1(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
        T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
        B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), n):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

    @R.function
    def test():
        n = T.int64()
        R.func_attr({"tir_var_upper_bound": {"n": 2048}})
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.full1,(), out_sinfo=R.Tensor((1, 32, 1, n), dtype="float16"))
            lv1_1 = R.call_tir(cls.full1,(), out_sinfo=R.Tensor((1, 32, 1, n), dtype="float16"))
            lv1_2 = R.call_tir(cls.full1,(), out_sinfo=R.Tensor((1, 32, 1, n), dtype="float16"))
            lv2 = R.call_tir(cls.full2,(), out_sinfo=R.Tensor((1, 32, n, 128), dtype="float16"))
            lv2_1 = R.call_tir(cls.full2,(), out_sinfo=R.Tensor((1, 32, n, 128), dtype="float16"))
            lv3 = R.call_tir(cls.matmul1, (lv1, lv2), out_sinfo=R.Tensor((1, 32, 1, 128), dtype="float16"))
            R.output(lv3)
        return lv3

@T.prim_func
def cuda_workload(var_inp0: T.handle, inp1: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), var_matmul: T.handle):
    T.func_attr({"tir.is_scheduled": 1})
    m = T.int64()
    inp0 = T.match_buffer(var_inp0, (T.int64(1), m, T.int64(4096)))
    matmul = T.match_buffer(var_matmul, (T.int64(1), m, T.int64(4096)))
    # with T.block("root"):
    matmul_reindex_pad_local = T.alloc_buffer((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="local")
    inp0_reindex_pad_shared = T.alloc_buffer((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="shared")
    inp1_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
    for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
        for ax1_0 in T.thread_binding((m + T.int64(31)) // T.int64(32), thread="blockIdx.x"):
            for ax2_0 in T.thread_binding(T.int64(64), thread="blockIdx.y"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax2_3_init, ax1_3_init in T.grid(T.int64(4), T.int64(4)):
                                    with T.block("matmul_init"):
                                        v0 = T.axis.spatial(T.int64(1), ax0)
                                        v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                        v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_init)
                                        T.reads()
                                        T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                        matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("inp0_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(inp0[v0, v1, v2])
                                                        T.writes(inp0_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        inp0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("inp1_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(inp1[v2, v1])
                                                        T.writes(inp1_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        inp1_reindex_shared[v0, v1, v2] = inp1[v2, v1]
                                    for ax3_1, ax2_3, ax1_3 in T.grid(T.int64(16), T.int64(4), T.int64(4)):
                                        with T.block("matmul_update"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                            v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3)
                                            v3 = T.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                            T.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], inp0_reindex_pad_shared[T.int64(0), v1, v3], inp1_reindex_shared[T.int64(0), v2, v3])
                                            T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + inp0_reindex_pad_shared[T.int64(0), v1, v3] * inp1_reindex_shared[T.int64(0), v2, v3]
                                for ax0_1, ax1, ax2_0_1 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0_1)
                                            v1 = T.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(4096), ax2_0 * T.int64(64) + ax2_2 * T.int64(4) + ax2_0_1 * T.int64(2) + ax2_1_1)
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[T.int64(0), v1, v2])
                                            if v1 < m:
                                                matmul[T.int64(0), v1, v2] = matmul_reindex_pad_local[v0, v1, v2]
# fmt: on
# pylint: enable=no-self-argument,invalid-name,line-too-long,no-method-argument


@pytest.mark.skip("requires CUDA")
def test_benchmark_prim_func_rpc():
    with LocalRPC() as rpc:
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        input_infos, _, _ = benchmark(
            cuda_workload,
            args=[
                ((1, "m", 4096), "float32"),
                ((4096, 4096), "float32"),
                ((1, "m", 4096), "float32"),
            ],
            dym_var_sample={"m": 128},
            target="nvidia/geforce-rtx-3070",
            rpc_config=rpc_config,
        )
        assert input_infos == [
            ((1, 128, 4096), "float32"),
            ((4096, 4096), "float32"),
            ((1, 128, 4096), "float32"),
        ]


@pytest.mark.skip("requires CUDA")
def test_benchmark_prim_func_local():
    input_infos, _, _ = benchmark(
        cuda_workload,
        args=[
            ((1, "m", 4096), "float32"),
            ((4096, 4096), "float32"),
            ((1, "m", 4096), "float32"),
        ],
        dym_var_sample={"m": 128},
        target="nvidia/geforce-rtx-3070",
    )
    assert input_infos == [
        ((1, 128, 4096), "float32"),
        ((4096, 4096), "float32"),
        ((1, 128, 4096), "float32"),
    ]


@pytest.mark.skip("requires CUDA")
def test_benchmark_prim_func_full_local():
    with tvm.target.Target("nvidia/geforce-rtx-3070"):
        benchmark_prim_func(
            cuda_workload,
        )


@pytest.mark.skip("requires CUDA")
def test_benchmark_prim_func_full_rpc():
    with LocalRPC() as rpc:
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc.tracker_host,
            tracker_port=rpc.tracker_port,
            tracker_key=rpc.tracker_key,
            session_priority=1,
            session_timeout_sec=100,
        )
        benchmark_prim_func(
            cuda_workload,
            target="nvidia/geforce-rtx-3070",
            rpc_config=rpc_config,
            evaluator_config=ms.runner.EvaluatorConfig(
                number=10,
                repeat=10,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            ),
        )


def test_benchmark_relax_func():
    with tvm.target.Target("llvm -num-cores=4"):
        benchmark_relax_func(Module, "test")


def test_extract_prim_func_full1():
    print(
        extract_prim_func(
            model_name="TEST",
            relax_func_name="test",
            prim_func_name="full1",
            func=Module["full1"],  # type: ignore
            func_args=[((1, 32, 1, "n"), "float16")],
            dym_var_dict={"n": "int32"},
            weight=2,
            sample_number=10,
            target="llvm -num-cores=4",
        )
    )


def test_extract_prim_func_matmul1():
    print(
        extract_prim_func(
            model_name="TEST",
            relax_func_name="test",
            prim_func_name="matmul1",
            func=Module["matmul1"],  # type: ignore
            weight=2,
            sample_number=10,
            target="llvm -num-cores=4",
        )
    )


def test_extract_from_relax():
    with tvm.target.Target("llvm -num-cores=4"):
        with tempfile.TemporaryDirectory() as filepath:
            extract_from_relax(
                Module,
                "TEST",
                file_path=filepath,
            )


def test_extract_func_info_from_prim_func():
    assert (
        str(extract_func_info_from_prim_func(cuda_workload))
        == "([((1, m, 4096), 'float32'), ((4096, 4096), 'float32'), ((1, m, 4096), 'float32')], {'m': 'int64'})"
    )
    assert (
        str(extract_func_info_from_prim_func(Module["full1"]))
        == "([((1, 32, 1, n), 'float16')], {'n': 'int64'})"
    )
    assert (
        str(extract_func_info_from_prim_func(Module["matmul1"]))
        == "([((1, 32, 1, n), 'float16'), ((1, 32, n, 128), 'float16'), ((1, 32, 1, 128), 'float16')], {'n': 'int64'})"
    )
    assert (
        str(extract_func_info_from_prim_func(Module["full2"]))
        == "([((1, 32, n, 128), 'float16')], {'n': 'int64'})"
    )


if __name__ == "__main__":
    tvm.testing.main()
