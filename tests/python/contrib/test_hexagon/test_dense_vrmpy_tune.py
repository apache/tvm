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

import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.script import tir as T
from tvm.tir import TensorIntrin
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.utils import apply_fixed_schedules
from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner


@T.prim_func
def dot_product_32x4_u8u8i32_desc(
    A: T.Buffer((4,), "uint8", offset_factor=1),
    B: T.Buffer((32, 4), "uint8", offset_factor=1),
    C: T.Buffer((32,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:32], A[0:4], B[0:32, 0:4])
        T.writes(C[0:32])
        for i in T.serial(0, 32):
            with T.init():
                C[i] = T.int32(0)
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


@T.prim_func
def dot_product_32x4_u8u8i32_vrmpy(
    A: T.Buffer((4,), "uint8", offset_factor=1),
    B: T.Buffer((32, 4), "uint8", offset_factor=1),
    C: T.Buffer((32,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:32], A[0:4], B[0:32, 0:4])
        T.writes(C[0:32])

        A_u8x4 = A.vload([0], "uint8x4")
        A_i32 = T.reinterpret(A_u8x4, dtype="int32")

        B_i8x128 = B.vload([0, 0], dtype="uint8x128")
        B_i32x32 = T.reinterpret(B_i8x128, dtype="int32x32")

        C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
            T.uint32(3),
            C[T.ramp(T.int32(0), 1, 32)],
            B_i32x32,
            A_i32,
            dtype="int32x32",
        )


VRMPY_INTRIN = "dot_32x4_vrmpy"

TensorIntrin.register(VRMPY_INTRIN, dot_product_32x4_u8u8i32_desc, dot_product_32x4_u8u8i32_vrmpy)


def schedule_matmul_common(sch, block, batched, M, do_tune=False):
    a_y, a_x, _ = sch.get_loops(block)[-3:]
    outer_block = block

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

    if batched:
        a_b = sch.get_loops(outer_block)[0]
        fused = sch.fuse(a_b, a_yo, a_xo)
    else:
        fused = sch.fuse(a_yo, a_xo)

    # sch.parallel(fused)

    dec = sch.decompose_reduction(block, a_ko)

    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    sch.tensorize(a_xi, VRMPY_INTRIN)

    return fused


def schedule_dense(dense_block, M, sch, do_tune=False):
    schedule_matmul_common(sch, dense_block, False, M, do_tune=do_tune)


@tvm.testing.requires_hexagon
def test_dense_u8u8i32_vrmpy(hexagon_launcher):
    target_hexagon = tvm.target.hexagon("v68")
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    M = 128
    N = 768
    K = 768
    data_shape = (M, K)
    weight_shape = (N, K)

    dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)

    dense = relay.nn.dense(data, weight, out_dtype="int32")

    use_bias = False

    if dtype == "uint8":
        data_np = np.random.uniform(1, 255, size=data_shape).astype(dtype)
        weight_np = np.random.uniform(1, 255, size=weight_shape).astype(dtype)
    else:
        data_np = np.random.uniform(-128, 127, size=data_shape).astype(dtype)
        weight_np = np.random.uniform(-128, 127, size=weight_shape).astype(dtype)

    # data_np = np.ones(data_shape).astype(dtype) * 127
    # weight_np =  np.ones(weight_shape).astype(dtype) * 127

    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    params = {"weight": weight_np, "bias": bias_np}

    if use_bias:
        bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
        out = relay.nn.bias_add(dense, bias)
    else:
        out = dense

    mod = tvm.IRModule.from_expr(out)

    do_tune = True

    if do_tune:
        def schedule_rule_dense_vrmpy(sch, dense_block):
            schedule_dense(dense_block, None, sch, True)
            return [sch]

        from tvm._ffi import register_func
        register_func("meta_schedule.dense_u8u8i32_vrmpy", schedule_rule_dense_vrmpy)

        config = {"relay.backend.use_meta_schedule": True, "relay.FuseOps.link_params": True}
        extracted_tasks = ms.extract_task_from_relay(mod, target, params, pass_config=config)

        config = ms.TuneConfig(
            strategy="replay_trace",
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=32,
        )

        import tempfile
        # with tempfile.TemporaryDirectory() as work_dir:
        work_dir = "work"
        database = ms.tune_extracted_tasks(
            extracted_tasks,
            config,
            work_dir=work_dir,
            postprocs=lambda: [],
            builder=get_hexagon_local_builder(),
            runner=get_hexagon_rpc_runner(hexagon_launcher)
        )
    else:
        def schedule_fn(task, sch):
            if "dense" not in task.task_name:
                return False

            block = sch.get_block("compute")

            schedule_rule = sch.get(block).annotations["schedule_rule"]

            assert "dense_u8u8i32_vrmpy" in schedule_rule

            schedule_dense(block, M, sch)

    #        print(sch.mod.script())

            return True

        with tvm.transform.PassContext(config={"relay.FuseOps.link_params": True}):
            database = apply_fixed_schedules(mod, target, params, schedule_fn)

    with ms.ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            executor = Executor("graph", {"link-params": True})
            lib = relay.build(mod, target=target, params=params, executor=executor)

    asm = lib.lib.get_source("asm")
#    assert "vrmpy" in asm

    with hexagon_launcher.start_session() as hexagon_session:
        rt_mod = hexagon_session.get_executor_from_factory(lib)

        rt_mod.set_input("data", data_np)

        rt_mod.run()

        out = rt_mod.get_output(0).numpy()
        ref = np.dot(data_np.astype("int32"), weight_np.transpose().astype("int32"))

        if use_bias:
            ref += bias_np

        np.testing.assert_equal(out, ref)
        print(ref)

        gops = (N * M * K) * 2 / 1e9
        time_ms = rt_mod.benchmark(hexagon_session.device, number=1, repeat=50).mean * 1e3

        print("time elapsed: ", time_ms)
        print("GOPS:", gops / (time_ms / 1e3))
