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

""" Test rpc based launcher for hexagon """
import tempfile

import numpy as np
import pytest
import tvm.testing
import tvm.topi.testing
from tvm import meta_schedule as ms
from tvm import relay, te
from tvm.contrib.hexagon.meta_schedule import (
    get_hexagon_local_builder,
    get_hexagon_rpc_runner,
)
from tvm.meta_schedule import postproc, schedule_rule
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput
from tvm.meta_schedule.runner import RunnerInput
from tvm.script import tir as T
from tvm.tir import FloatImm
from tvm.tir.tensor_intrin.hexagon import VRMPY_u8u8i32_INTRIN

from .infrastructure import get_hexagon_target

MATMUL_N = 16
MATMUL_M = 32


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(  # type: ignore  # pylint: disable=no-self-argument
        a: T.handle, b: T.handle, c: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        C = T.match_buffer(c, (16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.testing.requires_hexagon
def test_builder_runner(hexagon_launcher):
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    mod = MatmulModule

    builder = get_hexagon_local_builder()
    runner = get_hexagon_rpc_runner(hexagon_launcher, number=1, repeat=1, min_repeat_ms=0)

    (builder_result,) = builder.build([BuilderInput(mod, get_hexagon_target("v68"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
            TensorInfo("float32", (MATMUL_N, MATMUL_N)),
        ],
    )

    # Run the module
    (runner_future,) = runner.run([runner_input])
    runner_result = runner_future.result()

    assert runner_result.error_msg is None
    for result in runner_result.run_secs:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0


def dense(m, n, k):
    X = te.placeholder((m, k), name="X", dtype="uint8")
    packedW = te.placeholder((n // 32, k // 4, 32, 4), name="packedW", dtype="uint8")

    ak = te.reduce_axis((0, k), name="k")
    out = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packedW[tvm.tir.indexdiv(j, 32), tvm.tir.indexdiv(ak, 4), j % 32, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        name="compute",
    )
    return [X, packedW, out]


def schedule_dense(sch, block, M, do_tune):
    a_y, a_x, _ = sch.get_loops(block)[-3:]

    if do_tune:
        y_factors = sch.sample_perfect_tile(a_y, n=2, max_innermost_factor=128)
        a_yo, a_yi = sch.split(a_y, factors=y_factors)
    else:
        a_yo, a_yi = sch.split(a_y, factors=[None, min(M, 32)])

    a_xo, a_xi = sch.split(a_x, factors=[None, 32])
    sch.reorder(a_yo, a_xo, a_yi, a_xi)

    a_xi, a_k = sch.get_loops(block)[-2:]
    a_ko, a_ki = sch.split(a_k, factors=[None, 4])
    sch.reorder(a_ko, a_xi, a_ki)

    fused = sch.fuse(a_yo, a_xo)

    sch.parallel(fused)

    dec = sch.decompose_reduction(block, a_ko)

    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    sch.tensorize(a_xi, VRMPY_u8u8i32_INTRIN)


def verify_dense(sch, target, M, N, K, hexagon_session):
    f = tvm.build(sch.mod["main"], target=target, name="dense")
    mod = hexagon_session.load_module(f)
    dev = hexagon_session.device

    a_np = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    b_np = np.random.uniform(1, 10, size=(N, K)).astype("uint8")
    c_np = np.dot(a_np.astype("int32"), b_np.transpose().astype("int32"))

    packW = np.random.uniform(1, 10, size=(N // 32, (K // 4), 32, 4)).astype("uint8")

    for r_idx in range(N // 32):
        for ko in range(K // 4):
            for s_idx in range(32):
                for t_idx in range(4):
                    packW[r_idx][ko][s_idx][t_idx] = b_np[r_idx * 32 + s_idx][ko * 4 + t_idx]

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(packW, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype="int32"), dev)

    mod(a, b, c)
    np.testing.assert_equal(c.numpy(), c_np)

    evaluator = mod.time_evaluator(mod.entry_name, dev, number=10)
    gflops = (N * M * K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("%f ms, %f GOPS" % (time_ms, gflops / (time_ms / 1e3)))


@tvm.testing.requires_hexagon
def test_vrmpy_dense(hexagon_launcher):
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    do_tune = True

    M, N, K = 128, 768, 768
    workload = te.create_prim_func(dense(M, N, K))

    if not do_tune:
        ir_module = tvm.IRModule({"main": workload})
        sch = tvm.tir.Schedule(ir_module)
        block = sch.get_block("compute")
        schedule_dense(sch, block, M, do_tune)
    else:
        with tempfile.TemporaryDirectory() as work_dir:

            def schedule_dense_for_tune(sch):
                block = sch.get_block("compute")
                return schedule_dense(sch, block, None, True)

            target = get_hexagon_target("v69")
            database = ms.tir_integration.tune_tir(
                mod=workload,
                target=target,
                work_dir=work_dir,
                max_trials_global=8,
                space=ms.space_generator.ScheduleFn(
                    schedule_dense_for_tune,
                    sch_rules=[],
                    postprocs=[],
                    mutator_probs={},
                ),
                strategy="replay-trace",
                builder=get_hexagon_local_builder(),
                runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
            )
            sch = ms.tir_integration.compile_tir(database, workload, target)

    with hexagon_launcher.start_session() as session:
        verify_dense(sch, get_hexagon_target("v68"), M, N, K, session)


# This is an example of a schedule found by vrmpy auto tensorization.
# It gets 440 GFLOPS on SD888.
@tvm.script.ir_module
class Module_vrmpy_auto_tensorize:
    @T.prim_func
    def main(  # type: ignore
        X: T.Buffer[(128, 768), "uint8"],  # type: ignore
        packedW: T.Buffer[(24, 192, 32, 4), "uint8"],  # type: ignore
        compute: T.Buffer[(128, 768), "int32"],  # type: ignore
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0_0_i1_0_0_fused in T.parallel(
            512, annotations={"pragma_auto_unroll_max_step": 64, "pragma_unroll_explicit": 1}
        ):
            for i0_1_init, i1_0_1_init, i0_2_init, i1_0_2_init in T.grid(2, 3, 1, 1):
                with T.block("compute_o_init"):
                    i = T.axis.spatial(128, i0_0_i1_0_0_fused // 8 * 2 + i0_1_init + i0_2_init)
                    j_o = T.axis.spatial(24, i1_0_2_init + i0_0_i1_0_0_fused % 8 * 3 + i1_0_1_init)
                    T.reads()
                    T.writes(compute[i, j_o * 32 : j_o * 32 + 32])  # type: ignore
                    for i1_1 in T.vectorized(32):
                        with T.block("compute_init"):
                            j_i_init = T.axis.spatial(32, i1_1)
                            T.reads()
                            T.writes(compute[i, j_o * 32 + j_i_init])
                            compute[i, j_o * 32 + j_i_init] = 0  # type: ignore
            for i2_0_0, i0_1, i1_0_1, i2_0_1, i0_2, i1_0_2 in T.grid(32, 2, 3, 6, 1, 1):
                with T.block("compute_o_update"):
                    i = T.axis.spatial(128, i0_0_i1_0_0_fused // 8 * 2 + i0_1 + i0_2)
                    j_o = T.axis.spatial(24, i1_0_2 + i0_0_i1_0_0_fused % 8 * 3 + i1_0_1)
                    k_o = T.axis.reduce(192, i2_0_0 * 6 + i2_0_1)
                    T.reads(
                        compute[i, j_o * 32 : j_o * 32 + 32],  # type: ignore
                        X[i, k_o * 4 : k_o * 4 + 4],  # type: ignore
                        packedW[j_o, k_o, 0:32, 0:4],  # type: ignore
                    )
                    T.writes(compute[i, j_o * 32 : j_o * 32 + 32])  # type: ignore
                    A = T.match_buffer(
                        X[i, k_o * 4 : k_o * 4 + 4], [4], dtype="uint8", offset_factor=1  # type: ignore
                    )
                    B = T.match_buffer(
                        packedW[j_o, k_o, 0:32, 0:4], [32, 4], dtype="uint8", offset_factor=1
                    )
                    C = T.match_buffer(
                        compute[i, j_o * 32 : j_o * 32 + 32], [32], dtype="int32", offset_factor=1  # type: ignore
                    )
                    A_u8x4: T.uint8x4 = A[0:4]  # type: ignore
                    A_i32: T.int32 = T.reinterpret(A_u8x4, dtype="int32")  # type: ignore
                    B_i32x32: T.int32x32 = T.reinterpret(B[0, 0:128], dtype="int32x32")  # type: ignore
                    C[0:32] = T.call_llvm_pure_intrin(  # type: ignore
                        4390, T.uint32(3), C[0:32], B_i32x32, A_i32, dtype="int32x32"
                    )


@tvm.testing.requires_hexagon
def test_vrmpy_dense_auto_tensorize(hexagon_launcher):
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    M, N, K = 128, 768, 768
    workload = te.create_prim_func(dense(M, N, K))

    sch_rules = [
        schedule_rule.MultiLevelTilingWithIntrin(
            VRMPY_u8u8i32_INTRIN,
            structure="SRSRS",
            tile_binds=None,
            max_innermost_factor=64,
            vector_load_lens=None,
            reuse_read=None,
            reuse_write=schedule_rule.ReuseType(
                req="may",
                levels=[1, 2],
                scope="global",
            ),
        ),
        schedule_rule.ParallelizeVectorizeUnroll(
            max_jobs_per_core=16,
            max_vectorize_extent=128,
            unroll_max_steps=[0, 16, 64, 512],
            unroll_explicit=True,
        ),
    ]

    postprocs = [
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorize(vectorize_init_loop=True),
    ]

    # Make this to False to compile and run the best tuned schedule
    if True:
        with tempfile.TemporaryDirectory() as work_dir:
            target = get_hexagon_target("v68")
            database = ms.tir_integration.tune_tir(
                mod=workload,
                target=target,
                max_trials_global=8,
                num_trials_per_iter=8,
                work_dir=work_dir,
                space=ms.space_generator.PostOrderApply(
                    f_block_filter=None,
                    sch_rules=sch_rules,
                    postprocs=postprocs,
                    mutator_probs={},
                ),
                builder=get_hexagon_local_builder(),
                runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
            )
            sch = ms.tir_integration.compile_tir(database, workload, target)
    else:
        sch = tvm.tir.Schedule(Module_vrmpy_auto_tensorize, debug_mask="all")

    with hexagon_launcher.start_session() as session:
        verify_dense(sch, get_hexagon_target("v68"), M, N, K, session)


@tvm.testing.requires_hexagon
def test_conv2d_relay_auto_schedule(hexagon_launcher):
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    I, O, H, W = 64, 64, 56, 56
    kH = kW = 3

    strides = (1, 1)
    padding = (1, 1)

    d_shape = (1, H, W, I)
    w_shape = (kH, kW, I, O)
    bias_shape = (1, 1, 1, w_shape[3])
    out_channel = w_shape[3]

    data = relay.var("data", shape=d_shape, dtype="float16")
    weight = relay.var("weight", shape=w_shape, dtype="float16")
    bias = relay.var("bias", shape=bias_shape, dtype="float16")
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=(kH, kW),
        channels=out_channel,
        padding=padding,
        strides=strides,
        out_dtype="float16",
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    mod = tvm.IRModule.from_expr(conv2d + bias)
    mod = mod.with_attr("executor", relay.backend.Executor("graph", {"link-params": True}))

    data_np = np.random.randn(*d_shape).astype("float16")
    weight_np = np.random.randn(*w_shape).astype("float16")
    bias_np = np.random.randn(*bias_shape).astype("float16")
    params = {"weight": weight_np, "bias": bias_np}

    ref = (
        relay.create_executor("graph", mod=mod, device=tvm.cpu(0), target="llvm")
        .evaluate()(*[data_np, weight_np, bias_np])
        .numpy()
    )

    with tempfile.TemporaryDirectory() as work_dir:
        target = get_hexagon_target("v69")
        database = ms.relay_integration.tune_relay(
            mod=mod,
            params=params,
            target=target,
            max_trials_global=8,
            strategy="replay-trace",
            work_dir=work_dir,
            builder=get_hexagon_local_builder(),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
        )
        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            params=params,
            target=target,
        )

    with hexagon_launcher.start_session() as session:
        rt_mod = session.get_executor_from_factory(lib)

        rt_mod.set_input("data", data_np)

        rt_mod.run()

        out = rt_mod.get_output(0).numpy()
        # Fairly loose check since fp16 results between x86 and Hexagon have
        # non-trivial difference.
        assert np.mean(np.abs(ref - out)) < 0.5


@tvm.testing.requires_hexagon
def test_dense_relay_auto_schedule(hexagon_launcher):
    """
    This is for testing RewriteLayout postproc. Without this postproc,
    dense on Hexagon is extremely slow.
    """
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    target_hexagon = tvm.target.hexagon("v69")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    data_shape = (128, 128)
    weight_shape = (128, 128)

    data = relay.var("data", shape=data_shape, dtype="float16")
    weight = relay.var("weight", shape=weight_shape, dtype="float16")
    dense = relay.nn.dense(data, weight)
    mod = tvm.IRModule.from_expr(dense)
    mod = mod.with_attr("executor", relay.backend.Executor("graph", {"link-params": True}))

    weight_np = np.random.randn(*weight_shape).astype("float32")

    data_np = np.random.randn(*data_shape).astype("float32")
    params = {"weight": weight_np}
    ref = np.dot(data_np, weight_np.transpose())

    with tempfile.TemporaryDirectory() as work_dir:
        target = get_hexagon_target("v69")
        database = ms.relay_integration.tune_relay(
            mod=mod,
            params=params,
            target=target,
            max_trials_global=8,
            strategy="replay-trace",
            work_dir=work_dir,
            builder=get_hexagon_local_builder(),
            runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
        )
        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            params=params,
            target=target,
        )

    with hexagon_launcher.start_session() as session:
        rt_mod = session.get_executor_from_factory(lib)

        rt_mod.set_input("data", data_np)

        rt_mod.run()

        out = rt_mod.get_output(0).numpy()
        # Fairly loose check since fp16 results between x86 and Hexagon have
        # non-trivial difference.
        assert np.mean(np.abs(ref - out)) < 0.1
