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
import pytest
import numpy as np
import tempfile

import tvm.testing
import tvm.topi.testing
from tvm import te, relay
from tvm import meta_schedule as ms
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput
from tvm.meta_schedule import postproc, schedule_rule
from tvm.script import tir as T
from tvm.tir import FloatImm
from tvm.tir.tensor_intrin.hexagon import VRMPY_u8u8i32_INTRIN
from tvm.meta_schedule.runner import RunnerInput
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm.relay.backend import Executor
from tvm.topi.utils import get_const_tuple
from tvm.meta_schedule.testing import te_workload

MATMUL_N = 16
MATMUL_M = 32


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        C = T.match_buffer(c, (16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.testing.requires_hexagon
def test_builder_runner(hexagon_launcher):
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    mod = MatmulModule

    builder = get_hexagon_local_builder()
    runner = get_hexagon_rpc_runner(hexagon_launcher, number=1, repeat=1, min_repeat_ms=0)

    (builder_result,) = builder.build([BuilderInput(mod, target)])
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
    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    M, N, K = 128, 768, 768
    workload = te.create_prim_func(dense(M, N, K))

    if not do_tune:
        ir_module = tvm.IRModule({"main": workload})
        sch = tvm.tir.Schedule(ir_module)
        block = sch.get_block("compute")
        schedule_dense(sch, block, M, do_tune)
    else:
        with tempfile.TemporaryDirectory() as work_dir:
            config = ms.TuneConfig(
                strategy="replay_trace",
                num_trials_per_iter=8,
                max_trials_per_task=8,
                max_trials_global=8,
            )

            def schedule_dense_for_tune(sch):
                block = sch.get_block("compute")
                return schedule_dense(sch, block, None, True)

            sch = ms.tune_tir(
                mod=workload,
                target=target,
                config=config,
                work_dir=work_dir,
                space=ms.space_generator.ScheduleFn(schedule_dense_for_tune),
                builder=get_hexagon_local_builder(),
                runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
            )

    with hexagon_launcher.start_session() as session:
        verify_dense(sch, target, M, N, K, session)


@tvm.testing.requires_hexagon
def test_vrmpy_dense_auto_tensorize(hexagon_launcher):
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    M, N, K = 128, 768, 768
    workload = te.create_prim_func(dense(M, N, K))

    sch_rules = [
        schedule_rule.AutoInline(
            into_producer=False,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        ),
        schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
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
        postproc.DisallowDynamicLoop(),
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorize(vectorize_init_loop=True),
    ]

    # with tempfile.TemporaryDirectory() as work_dir:
    work_dir = "work"
    config = ms.TuneConfig(
        strategy="replay_trace",
        num_trials_per_iter=32,
        max_trials_per_task=128,
        max_trials_global=128,
    )

    sch = ms.tune_tir(
        mod=workload,
        target=target,
        config=config,
        work_dir=work_dir,
        sch_rules=lambda: sch_rules,
        postprocs=lambda: postprocs,
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
    )

    print(sch.mod.script())

    with hexagon_launcher.start_session() as session:
        verify_dense(sch, target, M, N, K, session)


@tvm.testing.requires_hexagon
def test_conv2d_nhwc_auto_schedule(hexagon_launcher):
    if hexagon_launcher._serial_number == "simulator":
        pytest.skip(msg="Tuning on simulator not supported.")

    target_hexagon = tvm.target.hexagon("v69", num_cores=4)
    target = tvm.target.Target(target_hexagon, host=target_hexagon)
    I = 64
    O = 64
    H = 56
    W = 56
    kH = 3
    kW = 3

    dtype = "float16"

    strides = (1, 1)
    padding = (1, 1)

    data, kernel, out = te_workload.conv2d_nhwc(
        1, H, W, I, O, 3, 1, 1, 1, in_dtype="float16", out_dtype="float16"
    )
    workload = te.create_prim_func([data, kernel, out])
    # workload = te.create_prim_func(te_workload.matmul(128, 768, 768))

    # with tempfile.TemporaryDirectory() as work_dir:
    work_dir = "work"
    config = ms.TuneConfig(
        # strategy="replay_trace",
        strategy="evolutionary",
        num_trials_per_iter=32,
        max_trials_per_task=32,
        max_trials_global=32,
    )

    sch = ms.tune_tir(
        mod=workload,
        target=target,
        config=config,
        work_dir=work_dir,
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
    )
    print(sch.trace)

    import time

    t1 = time.time()
    f = tvm.build(sch.mod["main"], [data, kernel, out], target)
    t2 = time.time()

    # print("compiled in", t2 - t1)
    # return

    with hexagon_launcher.start_session() as session:
        # print("session acquired")
        module = session.load_module(f)
        dev = session.device

        a_np = np.random.randn(1, I, H, W).astype("float16")
        w_np = np.random.randn(O, I, kH, kW).astype("float16")
        c_np = tvm.topi.testing.conv2d_nchw_python(
            a_np.astype("float32"), w_np.astype("float32"), strides, padding
        )

        data_np = np.zeros(get_const_tuple(data.shape)).astype(dtype)
        w_np_hwio = np.zeros(get_const_tuple(kernel.shape)).astype(dtype)

        for i in range(I):
            for h in range(H):
                for w in range(W):
                    data_np[0, h, w, i] = a_np[0, i, h, w]

        for o in range(O):
            for i in range(I):
                for h in range(kH):
                    for w in range(kW):
                        w_np_hwio[h, w, i, o] = w_np[o, i, h, w]

        a = tvm.nd.array(data_np.astype(dtype), dev)
        w = tvm.nd.array(w_np_hwio.astype(dtype), dev)

        c = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), dev)

        module(a, w, c)

        P, Q = c.shape[1:3]
        evaluator = module.time_evaluator(module.entry_name, dev, number=20)
        time_ms = evaluator(a, w, c).mean * 1e3
        gflops = (O * P * Q * I * kH * kW) * 2 / 1e9
        print("time elapsed: ", time_ms)
        print("GFLOPS:", gflops / (time_ms / 1e3))

        out_nhwc = c.numpy()

        out = np.zeros(c_np.shape).astype("float16")

        for o in range(O):
            for h in range(P):
                for w in range(Q):
                    out[0, o, h, w] = out_nhwc[0, h, w, o]

        print(np.max(np.abs(out - c_np)), np.mean(np.abs(out - c_np)))

        mx = np.max(np.abs(out - c_np))

        indices = np.where(np.abs(out - c_np) == mx)

        print(out[indices], c_np[indices])


# @tvm.testing.requires_hexagon
# def test_conv2d_relay_auto_schedule(hexagon_launcher):
#     if hexagon_launcher._serial_number == "simulator":
#         pytest.skip(msg="Tuning on simulator not supported.")

#     target_hexagon = tvm.target.hexagon("v69", num_cores=4)
#     target = tvm.target.Target(target_hexagon, host=target_hexagon)
#     I = 64
#     O = 64
#     H = 56
#     W = 56
#     kH = 3
#     kW = 3

#     strides = (1, 1)
#     padding = (1, 1)

#     d_shape = (1, H, W, I)
#     w_shape = (kH, kW, I, O)
#     bias_shape = (w_shape[3],)
#     out_channel = w_shape[0]

#     data = relay.var("data", shape=d_shape, dtype="float16")
#     weight = relay.var("weight", shape=w_shape, dtype="float16")
#     bias = relay.var("bias", shape=bias_shape, dtype="float16")
#     conv2d = relay.nn.conv2d(
#         data=data,
#         weight=weight,
#         kernel_size=w_shape[2:],
#         channels=out_channel,
#         padding=padding,
#         strides=strides,
#         out_dtype="float16",
#         data_layout="NHWC",
#         weight_layout="HWIO",
#     )

#     bias_add = relay.nn.bias_add(conv2d, bias)

#     use_bias = True

#     if use_bias:
#         out = bias_add
#     else:
#         out = conv2d

#     mod = tvm.IRModule.from_expr(out)

#     data_np = np.random.randn(*d_shape).astype("float16")
#     weight_np = np.random.randn(*w_shape).astype("float16")
#     bias_np = np.random.randn(w_shape[0]).astype("float16")
#     params = {"weight": weight_np, "bias": bias_np}

#     target_llvm = tvm.target.Target("llvm")

#     with tvm.transform.PassContext(
#         opt_level=3,
#     ):
#         lib_ref = relay.build(mod, target=target_llvm, params=params)

#     rt_mod_ref = tvm.contrib.graph_executor.GraphModule(lib_ref["default"](tvm.cpu(0)))

#     rt_mod_ref.set_input("data", data_np)

#     rt_mod_ref.run()

#     ref = rt_mod_ref.get_output(0).numpy()

#     config = ms.TuneConfig(
#         strategy="evolutionary",
#         num_trials_per_iter=32,
#         max_trials_per_task=32,
#         max_trials_global=32,
#     )

#     with tempfile.TemporaryDirectory() as work_dir:
#         executor = Executor("graph", {"link-params": True})
#         lib = ms.tune_relay(
#             mod=mod,
#             params=params,
#             target=target,
#             config=config,
#             work_dir=work_dir,
#             builder=get_hexagon_local_builder(),
#             runner=get_hexagon_rpc_runner(hexagon_launcher, number=20),
#             executor=executor,
#         )

#     with hexagon_launcher.start_session() as session:
#         rt_mod = session.get_executor_from_factory(lib)

#         rt_mod.set_input("data", data_np)

#         rt_mod.run()

#         out = rt_mod.get_output(0).numpy()
#         print(np.max(np.abs(ref - out)), np.mean(np.abs(ref - out)))
