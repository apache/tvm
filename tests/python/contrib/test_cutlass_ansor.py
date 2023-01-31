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
import logging
import math
import tvm
from tvm import relay
from tvm.contrib.cudnn import conv_output_shape
import numpy as np
from tvm.runtime.vm import VirtualMachine
from tvm.relay import op as _op
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.relay.transform import FirstOrderGradient, ToMixedPrecision, InferType
from tvm import auto_scheduler
from tvm.contrib.cutlass import (
    has_cutlass,
    num_cutlass_partitions,
    finalize_modules,
    finalize_modules_vm,
)
import tvm.testing

logging.basicConfig(level=logging.INFO)


def get_ref_rt_mod(mod, params, target="cuda"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    dev = tvm.device(target, 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev


def get_random_ndarray(shape, dtype):
    if dtype == "int8":
        return np.random.randint(-128, 128, shape).astype(dtype)
    elif dtype == "uint8":
        return np.random.randint(0, 256, shape).astype(dtype)
    return np.random.uniform(-1, 1, shape).astype(dtype)


def get_output(rt_mod, names, inputs):
    for name, inp in zip(names, inputs):
        rt_mod.set_input(name, inp)
    rt_mod.run()
    return rt_mod.get_output(0).numpy()


def get_dense_transpose_dense(M, N, K, dtype="float16"):
    """
    dense: [M, K] * [N, K] -> [M, N]
    transpose: [M, N] -> [N, M]
    dense: [N, M] * [K, M] -> [N, K]

    input: [M, K]
    weight0: [N, K]
    weight1: [K, M]
    """
    in_shape = (M, K)
    w0_shape = (N, K)
    w1_shape = (K, M)

    input = relay.var("input", shape=in_shape, dtype=dtype)
    w0 = relay.var("weight0", shape=w0_shape, dtype=dtype)
    w1 = relay.var("weight1", shape=w1_shape, dtype=dtype)

    out0 = relay.nn.dense(input, w0, out_dtype=dtype)
    input1 = _op.transpose(out0, axes=(1, 0))
    out1 = relay.nn.dense(input1, w1, out_dtype=dtype)
    return out1


def build_by_cutlass(
    mod,
    params,
    sm,
    split_k_slices=[1],
    tmp_dir="./tmp",
    use_fast_math=False,
    use_3xtf32=True,
):
    logging.info("before partitioning:\n%s", mod)
    mod = partition_for_cutlass(mod)
    logging.info("after partitioning:\n%s", mod)

    num_cutlass_partition = num_cutlass_partitions(mod)
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    cutlass = tvm.target.Target(
        {
            "kind": "cutlass",
            "sm": sm,
            "use_3xtf32": use_3xtf32,
            "split_k_slices": split_k_slices,
            "profile_all_alignments": False,
            "find_first_valid": True,
            "use_multiprocessing": True,
            "use_fast_math": use_fast_math,
            "tmp_dir": tmp_dir,
        },
        host=host,
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=[cuda, cutlass], params=params)
    lib = finalize_modules(lib, "compile.so", tmp_dir)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


def build_by_cutlass_ansor(
    mod,
    params,
    sm,
    split_k_slices=[1],
    tmp_dir="./tmp",
    use_fast_math=False,
    use_3xtf32=True,
    num_trials=10,
):
    logging.info("before partitioning:\n%s", mod)
    mod = partition_for_cutlass(mod)
    logging.info("after partitioning:\n%s", mod)

    num_cutlass_partition = num_cutlass_partitions(mod)
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    cutlass = tvm.target.Target(
        {
            "kind": "cutlass",
            "sm": sm,
            "use_3xtf32": use_3xtf32,
            "split_k_slices": split_k_slices,
            "profile_all_alignments": False,
            "find_first_valid": True,
            "use_multiprocessing": True,
            "use_fast_math": use_fast_math,
            "tmp_dir": tmp_dir,
        },
        host=host,
    )

    # extract tasks
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod, params, cuda, include_simple_tasks=True, opt_level=3, other_targets=[cutlass]
        )
    for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
        print(f"==== Task {idx}: {task.desc} (weight {task_weight} key: {task.workload_key}) =====")
        print(task.compute_dag)

    # auto-tuning
    log_file = "cutlass_ansor.log"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=200, timeout=10)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(
        auto_scheduler.TuningOptions(
            num_measure_trials=num_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[
                auto_scheduler.RecordToFile(log_file),
            ],
        )
    )

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_auto_scheduler": True},
        ):
            lib = relay.build(
                mod,
                target=cuda,
                target_host=host,
                params=params,
            )
    lib = finalize_modules(lib, "compile.so", tmp_dir)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


def verify_dense_transpose_dense(
    func,
    M,
    N,
    K,
    ref_target="cuda",
    sm=80,
    atol=1e-5,
    rtol=1e-5,
    run_benchmark=False,
    dtype="float16",
    use_3xtf32=True,
):
    assert has_cutlass()
    if sm < 80 and dtype == "float32":
        return

    mod = tvm.IRModule.from_expr(func)
    typ = relay.transform.InferType()(mod)["main"].body.checked_type
    np_data = get_random_ndarray((M, K), dtype)
    np_weight0 = get_random_ndarray((N, K), dtype)
    np_weight1 = get_random_ndarray((K, M), dtype)

    params = {"weight0": np_weight0, "weight1": np_weight1}

    rt_mod_ref, dev = get_ref_rt_mod(mod, params, target=ref_target)
    cutlass_rt_mod, dev, num_partition = build_by_cutlass(mod, params, sm, use_3xtf32=use_3xtf32)
    cutlass_ansor_rt_mod, dev, num_partition = build_by_cutlass_ansor(
        mod, params, sm, use_3xtf32=use_3xtf32
    )
    x = tvm.nd.array(np_data, device=dev)
    cutlass_out = get_output(cutlass_rt_mod, ["input"], [x])
    cutlass_ansor_out = get_output(cutlass_ansor_rt_mod, ["input"], [x])
    ref_out = get_output(rt_mod_ref, ["input"], [x])

    assert num_partition > 0
    np.testing.assert_allclose(cutlass_out, ref_out, atol=atol, rtol=rtol)
    np.testing.assert_allclose(cutlass_ansor_out, ref_out, atol=atol, rtol=rtol)

    if run_benchmark:
        print("CUTLASS:", cutlass_rt_mod.benchmark(dev, number=1, repeat=600))
        print("CUTLASS with Ansor:", cutlass_ansor_rt_mod.benchmark(dev, number=1, repeat=600))
        print("TVM with target %s:" % ref_target, rt_mod_ref.benchmark(dev, number=1, repeat=600))


M = 128
N = 128
K = 128

# Use larger M/N/K for significant performance improvement
# M = 1024
# N = 1024
# K = 1024


@tvm.testing.requires_cutlass
def test_dense_transpose_dense():
    verify_dense_transpose_dense(get_dense_transpose_dense(M, N, K), M, N, K)


if __name__ == "__main__":
    tvm.testing.main()
