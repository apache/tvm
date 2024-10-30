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
import tempfile

import ml_dtypes
import numpy as np

import tvm
import tvm.testing
from tvm import auto_scheduler, relay
from tvm.contrib.cudnn import conv_output_shape
from tvm.contrib.cutlass import (
    finalize_modules,
    finalize_modules_vm,
    has_cutlass,
    num_cutlass_partitions,
)
from tvm.contrib.pickle_memoize import memoize
from tvm.relay import op as _op
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.relay.transform import FirstOrderGradient, InferType, ToMixedPrecision
from tvm.runtime.vm import VirtualMachine

logging.basicConfig(level=logging.INFO)


def has_cublas():
    return tvm.get_global_func("tvm.contrib.cublas.matmul", True) != None


def get_ref_rt_mod(mod, params, target="cuda"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    dev = tvm.device(target, 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev


def get_ref_vm(mod, params, target="cuda"):
    with tvm.transform.PassContext(opt_level=3):
        vm_exec = relay.vm.compile(mod, target=target, params=params)
        code, lib = vm_exec.save()
    dev = tvm.device(target, 0)
    vm_exec = tvm.runtime.vm.Executable.load_exec(code, lib)
    return VirtualMachine(vm_exec, dev), dev


def get_output(rt_mod, names, inputs):
    for name, inp in zip(names, inputs):
        rt_mod.set_input(name, inp)
    rt_mod.run()
    return rt_mod.get_output(0).asnumpy()


def get_output_vm(vm, names, inputs):
    params = dict(zip(names, inputs))
    return vm.invoke("main", **params).numpy()


def get_dense_with_shape(
    data_shape, weight_shape, out_dtype="float16", data_dtype="float16", weight_dtype="float16"
):
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    return relay.nn.dense(data, weight, out_dtype=out_dtype)


def get_dense(M, N, K, out_dtype="float16", data_dtype="float16", weight_dtype="float16"):
    return get_dense_with_shape((M, K), (N, K), out_dtype, data_dtype, weight_dtype)


def get_dense_bias(M, N, K, out_dtype="float16"):
    dense = get_dense(M, N, K, out_dtype=out_dtype)
    bias = relay.var("bias", shape=(N,), dtype=out_dtype)
    return relay.nn.bias_add(dense, bias)


def get_dense_bias_relu(M, N, K, out_dtype="float16"):
    return relay.nn.relu(get_dense_bias(M, N, K, out_dtype=out_dtype))


def get_dense_bias_gelu(M, N, K, out_dtype="float16"):
    bias_add = get_dense_bias(M, N, K, out_dtype)
    mul = bias_add * relay.const((1.0 / math.sqrt(2.0)), dtype=out_dtype)
    if out_dtype == "float16":
        erf = relay.cast(relay.op.erf(relay.cast(mul, "float32")), "float16")
    else:
        erf = relay.op.erf(mul)
    mul_half = erf * relay.const(0.5, dtype=out_dtype)
    add = mul_half + relay.const(0.5, dtype=out_dtype)
    return add * bias_add


def get_batch_matmul_with_shape(x_shape, y_shape, out_dtype="float16"):
    x = relay.var("x", shape=x_shape, dtype="float16")
    y = relay.var("y", shape=y_shape, dtype="float16")
    return relay.nn.batch_matmul(x, y, out_dtype=out_dtype)


def get_batch_matmul(batch, M, N, K, out_dtype="float16"):
    return get_batch_matmul_with_shape((batch, M, K), (batch, N, K), out_dtype="float16")


def get_conv2d_nchw(
    d_shape,
    w_shape,
    padding,
    strides=(1, 1),
    out_dtype="float16",
    data_dtype="float16",
    weight_dtype="float16",
):
    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    out_channel = w_shape[0]
    return relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        strides=strides,
        out_dtype=out_dtype,
    )


def get_conv2d_nchw_bias(d_shape, w_shape, padding, out_dtype="float16"):
    conv2d = get_conv2d_nchw(d_shape, w_shape, padding, out_dtype=out_dtype)
    bias = relay.var("bias", shape=(w_shape[0],), dtype=out_dtype)
    return relay.nn.bias_add(conv2d, bias)


def silu(x):
    return x * relay.sigmoid(x)


def hardswish(x, out_dtype="float16"):
    return x * (
        relay.clip(x + relay.const(3, dtype=out_dtype), a_min=0, a_max=6)
        / relay.const(6, dtype=out_dtype)
    )


def get_conv2d_nchw_bias_relu(d_shape, w_shape, padding, out_dtype="float16"):
    return relay.nn.relu(get_conv2d_nchw_bias(d_shape, w_shape, padding, out_dtype=out_dtype))


def get_conv2d_nchw_bias_sigmoid(d_shape, w_shape, padding, out_dtype="float16"):
    return relay.sigmoid(get_conv2d_nchw_bias(d_shape, w_shape, padding, out_dtype=out_dtype))


def get_conv2d_nchw_bias_silu(d_shape, w_shape, padding, out_dtype="float16"):
    conv_out = get_conv2d_nchw_bias(d_shape, w_shape, padding, out_dtype=out_dtype)
    return silu(conv_out)


def get_conv2d_nchw_bias_hardswish(d_shape, w_shape, padding, out_dtype="float16"):
    conv_out = get_conv2d_nchw_bias(d_shape, w_shape, padding, out_dtype=out_dtype)
    return hardswish(conv_out, out_dtype)


def get_conv2d_nchw_bias_residual(d_shape, w_shape, padding, out_dtype="float16"):
    data = relay.var("data", shape=d_shape, dtype="float16")
    weight = relay.var("weight", shape=w_shape, dtype="float16")
    bias = relay.var("bias", shape=(w_shape[0],), dtype=out_dtype)
    out_channel = w_shape[0]
    conv2d = relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        out_dtype=out_dtype,
    )
    bias_add = relay.nn.bias_add(conv2d, bias)
    return bias_add, data


def get_conv2d_transpose_nchw(
    d_shape,
    w_shape,
    padding,
    output_padding,
    strides,
    out_dtype="float32",
    data_dtype="float32",
    weight_dtype="float32",
):
    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    out_channel = w_shape[1]
    return relay.nn.conv2d_transpose(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        output_padding=output_padding,
        strides=strides,
        out_dtype=out_dtype,
    )


def get_conv2d_backward_weight(
    d_shape,
    w_shape,
    o_shape,
    padding,
    strides,
    out_dtype="float32",
    data_dtype="float32",
    weight_dtype="float32",
):
    grad = relay.var("grad", shape=o_shape, dtype=weight_dtype)
    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    out_channel = o_shape[1]
    return relay.nn.conv2d_backward_weight(
        grad=grad,
        data=data,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        strides=strides,
        out_dtype=out_dtype,
    )


def get_dense_transpose_dense(M, N, K, dtype="float16"):
    """
    output = nn.dense(_op.transpose(nn.dense(input, weight0), axes=(1, 0)), weight1)

    dense0: [M, K] * [N, K] -> [M, N]
    transpose: [M, N] -> [N, M]
    dense1: [N, M] * [K, M] -> [N, K]

    input: [M, K]
    weight0: [N, K]
    weight1: [K, M]
    """
    input_shape = (M, K)
    weight0_shape = (N, K)
    weight1_shape = (K, M)

    input = relay.var("input", shape=input_shape, dtype=dtype)
    weight0 = relay.var("weight0", shape=weight0_shape, dtype=dtype)
    weight1 = relay.var("weight1", shape=weight1_shape, dtype=dtype)

    output0 = relay.nn.dense(input, weight0, out_dtype=dtype)
    input1 = _op.transpose(output0, axes=(1, 0))
    output = relay.nn.dense(input1, weight1, out_dtype=dtype)
    return output


def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


def get_random_ndarray(shape, dtype):
    if dtype == "int8":
        return np.random.randint(-128, 128, shape).astype(dtype)
    elif dtype == "uint8":
        return np.random.randint(0, 256, shape).astype(dtype)
    return np.random.uniform(-1, 1, shape).astype(dtype)


def profile_and_build(
    mod,
    params,
    sm,
    split_k_slices=[1],
    tmp_dir="./tmp",
    use_fast_math=False,
    use_3xtf32=True,
    use_ansor=False,
    ansor_tuning=False,
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

    if use_ansor:
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod, params, cuda, include_simple_tasks=True, opt_level=3, other_targets=[cutlass]
            )
        for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
            logging.info(
                f"==== Task {idx}: {task.desc} (weight {task_weight} key: {task.workload_key}) ====="
            )
            logging.info(task.compute_dag)

        with tempfile.NamedTemporaryFile() as fp:
            log_file = fp.name

            # auto-tuning is disabled by default
            if ansor_tuning:
                measure_ctx = auto_scheduler.LocalRPCMeasureContext(
                    repeat=3, min_repeat_ms=200, timeout=10
                )
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                tuner.tune(
                    auto_scheduler.TuningOptions(
                        num_measure_trials=100,
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
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=[cuda, cutlass], params=params)
    lib = finalize_modules(lib, "compile.so", tmp_dir)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


def profile_and_build_vm(
    mod,
    params,
    sm,
    split_k_slices=[1],
    tmp_dir="./tmp",
    use_fast_math=False,
    use_3xtf32=True,
):
    mod = partition_for_cutlass(mod)
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
        vm_exec = relay.vm.compile(mod, target=[cuda, cutlass], params=params)
    vm_exec = finalize_modules_vm(vm_exec, "compile.so", tmp_dir)
    dev = tvm.device("cuda", 0)
    return VirtualMachine(vm_exec, dev), dev, num_cutlass_partition


def verify_dense(
    func,
    M,
    N,
    K,
    ref_target="cuda",
    sm=80,
    atol=1e-5,
    rtol=1e-5,
    run_benchmark=False,
    data_dtype="float16",
    weight_dtype="float16",
    use_3xtf32=True,
):
    assert has_cutlass()
    if sm < 80 and data_dtype == "float32":
        return

    mod = tvm.IRModule.from_expr(func)
    typ = relay.transform.InferType()(mod)["main"].body.checked_type
    out_dtype = typ.dtype
    use_vm = any(isinstance(s, tvm.tir.Any) for s in typ.shape)
    np_data = get_random_ndarray((M, K), data_dtype)
    np_weight = get_random_ndarray((N, K), weight_dtype)
    np_bias = get_random_ndarray((N,), out_dtype)

    params = {"weight": np_weight, "bias": np_bias}

    if use_vm:
        if ref_target == "cuda" and out_dtype == "float16":
            # Uncomment "return" below to see the accuracy difference of static vs dynamic TVM native fp16 dense
            # The static one can use a tensorcore schedule, but the dynamic one cannot
            rt_mod, dev = get_ref_vm(tvm.IRModule.from_expr(get_dense(M, N, K)), params)
            num_partition = 1
            logging.warning(
                "The reference fp16 dense with dynamic shape using fp16 accumulation has accuracy issues."
            )
            return
        else:
            rt_mod, dev, num_partition = profile_and_build_vm(
                mod, params, sm, use_3xtf32=use_3xtf32
            )

        rt_mod_ref, dev = get_ref_vm(mod, params, target=ref_target)
        x = tvm.nd.array(np_data, device=dev)
        out = get_output_vm(rt_mod, ["data"], [x])
        ref_out = get_output_vm(rt_mod_ref, ["data"], [x])
    else:
        rt_mod_ref, dev = get_ref_rt_mod(mod, params, target=ref_target)
        rt_mod, dev, num_partition = profile_and_build(mod, params, sm, use_3xtf32=use_3xtf32)
        x = tvm.nd.array(np_data, device=dev)
        out = get_output(rt_mod, ["data"], [x])
        ref_out = get_output(rt_mod_ref, ["data"], [x])

    assert num_partition > 0
    np.testing.assert_allclose(out, ref_out, atol=atol, rtol=rtol)

    if run_benchmark:
        print("CUTLASS:", rt_mod.benchmark(dev, number=1, repeat=600))
        print("TVM with target %s:" % ref_target, rt_mod_ref.benchmark(dev, number=1, repeat=600))


def verify_batch_matmul(
    func, batch, M, N, K, ref_target="cuda", sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
):
    assert has_cutlass()
    mod = tvm.IRModule.from_expr(func)
    typ = relay.transform.InferType()(mod)["main"].body.checked_type
    use_vm = any(isinstance(s, tvm.tir.Any) for s in typ.shape)
    x_np = np.random.uniform(-1, 1, (batch, M, K)).astype("float16")
    y_np = np.random.uniform(-1, 1, (batch, N, K)).astype("float16")

    if use_vm:
        rt_mod, dev, num_partition = profile_and_build_vm(mod, {}, sm)
        rt_mod_ref, dev = get_ref_vm(mod, {}, target=ref_target)
        assert num_partition > 0
        x = tvm.nd.array(x_np, device=dev)
        y = tvm.nd.array(y_np, device=dev)
        out = get_output_vm(rt_mod, ["x", "y"], [x, y])
        ref_out = get_output_vm(rt_mod_ref, ["x", "y"], [x, y])
    else:
        rt_mod, dev, num_partition = profile_and_build(mod, {}, sm)
        rt_mod_ref, dev = get_ref_rt_mod(mod, {})
        assert num_partition > 0

        x = tvm.nd.array(x_np, device=dev)
        y = tvm.nd.array(y_np, device=dev)
        out = get_output(rt_mod, ["x", "y"], [x, y])
        ref_out = get_output(rt_mod_ref, ["x", "y"], [x, y])

    np.testing.assert_allclose(out, ref_out, atol=atol, rtol=rtol)

    if run_benchmark:
        print("CUTLASS:", rt_mod.benchmark(dev, number=1, repeat=600))
        print("TVM Tensorcore (no tuning):", rt_mod_ref.benchmark(dev, number=1, repeat=600))


M = 96
N = 64
K = 64


@tvm.testing.requires_cutlass
def test_dense():
    verify_dense(get_dense(M, N, K), M, N, K)
    verify_dense(get_dense(M, N, K, out_dtype="float32"), M, N, K)
    # Test align1 case
    verify_dense(get_dense_bias(M, N + 1, K), M, N + 1, K)
    # int8
    verify_dense(
        get_dense(M, N, K, "int32", "int8", "int8"), M, N, K, data_dtype="int8", weight_dtype="int8"
    )

    dense_fp32 = get_dense(M, N, K, "float32", "float32", "float32")
    # fp32
    verify_dense(
        dense_fp32,
        M,
        N,
        K,
        data_dtype="float32",
        weight_dtype="float32",
        use_3xtf32=False,
        sm=75,
    )
    # tf32
    verify_dense(
        dense_fp32,
        M,
        N,
        K,
        data_dtype="float32",
        weight_dtype="float32",
        use_3xtf32=False,
        atol=1e-2,
        rtol=1e-2,
    )
    # 3xtf32
    verify_dense(
        dense_fp32,
        M,
        N,
        K,
        data_dtype="float32",
        weight_dtype="float32",
    )


@tvm.testing.requires_cutlass
def test_dense_bias():
    verify_dense(get_dense_bias(M, N, K), M, N, K)
    verify_dense(get_dense_bias(M, N, K, out_dtype="float32"), M, N, K)


@tvm.testing.requires_cutlass
def test_dense_bias_relu():
    verify_dense(get_dense_bias_relu(M, N, K), M, N, K)
    verify_dense(get_dense_bias_relu(M, N, K, out_dtype="float32"), M, N, K)


@tvm.testing.requires_cutlass
def test_dense_bias_gelu():
    verify_dense(get_dense_bias_gelu(M, N, K), M, N, K, atol=1e-3, rtol=1e-3)
    verify_dense(get_dense_bias_gelu(M, N, K, out_dtype="float32"), M, N, K, atol=1e-3, rtol=1e-3)


@tvm.testing.requires_cutlass
def test_dense_dynamic():
    data_shape = (relay.Any(), K)
    weight_shape = (relay.Any(), K)

    if has_cublas():
        # TVM native fp16 dense (without tensorcore), using fp16 accum, seems to have accuracy issues
        # Use cublas as a reference

        verify_dense(
            get_dense_with_shape(data_shape, weight_shape),
            M,
            N,
            K,
            ref_target="cuda -libs=cublas",
        )

    verify_dense(
        get_dense_with_shape(data_shape, weight_shape, out_dtype="float32"),
        M,
        N,
        K,
        atol=1e-4,
        rtol=1e-4,
    )


@tvm.testing.requires_cutlass
def test_batch_matmul():
    batch = 8
    verify_batch_matmul(get_batch_matmul(batch, M, N, K), batch, M, N, K)
    verify_batch_matmul(get_batch_matmul(batch, M, N, K, out_dtype="float32"), batch, M, N, K)

    if has_cublas():
        # Test dynamic shape batch_matmul
        # AutoTVM does not seem to support it
        x_shape = (relay.Any(), relay.Any(), K)
        y_shape = (relay.Any(), relay.Any(), K)

        verify_batch_matmul(
            get_batch_matmul_with_shape(x_shape, y_shape),
            batch,
            M,
            N,
            K,
            ref_target="cuda -libs=cublas",
        )


def verify_conv2d_common(
    expr_nchw,  # can be dynamic batch
    expr_ref,  # always static batch
    input_names,
    inputs,
    params,
    sm=80,
    split_k_slices=[1],
    atol=1e-5,
    rtol=1e-5,
    use_cudnn_ref=False,
    run_benchmark=False,
    use_fast_math=False,
    ref_target="cuda",
    use_vm=False,
):
    assert has_cutlass()
    if sm < 80 and inputs[0].dtype == "float32":
        return

    mod_nchw = tvm.IRModule.from_expr(expr_nchw)
    mod_ref = tvm.IRModule.from_expr(expr_ref)

    if use_vm:
        profile_and_build_func = profile_and_build_vm
        get_output_func = get_output_vm
        ref_build_func = get_ref_vm
    else:
        profile_and_build_func = profile_and_build
        get_output_func = get_output
        ref_build_func = get_ref_rt_mod

    mod_weight_ohwi = convert_conv2d_layout(
        mod_nchw,
        {
            "nn.conv2d": ["NHWC", "OHWI"],
            "nn.conv2d_transpose": ["NHWC", "IHWO"],
            "nn.conv2d_backward_weight": ["NHWC", "OHWI"],
        },
    )

    rt_mod, _, num_cutlass_partition = profile_and_build_func(
        mod_weight_ohwi, params, sm, split_k_slices, use_fast_math=use_fast_math
    )
    out = get_output_func(rt_mod, input_names, inputs)

    assert num_cutlass_partition > 0

    if use_cudnn_ref:
        rt_mod_ref, dev = ref_build_func(
            convert_conv2d_layout(mod_ref, {"nn.conv2d": ["NHWC", "OHWI"]}),
            params,
            target="cuda -libs=cudnn",
        )
    else:
        rt_mod_ref, dev = ref_build_func(
            convert_conv2d_layout(mod_ref, {"nn.conv2d": ["NHWC", "HWIO"]}),
            params,
            target=ref_target,
        )

    ref_out = get_output_func(rt_mod_ref, input_names, inputs)

    if run_benchmark:
        print("CUTLASS:", rt_mod.benchmark(dev, number=1, repeat=600))
        print("TVM Tensorcore (no tuning):", rt_mod_ref.benchmark(dev, number=1, repeat=600))

    np.testing.assert_allclose(out, ref_out, atol=atol, rtol=rtol)


def verify_conv2d(
    expr_nchw,  # can be dynamic batch
    expr_ref,  # always static batch
    d_shape,
    w_shape,
    sm=80,
    atol=1e-5,
    rtol=1e-5,
    use_cudnn_ref=False,
    run_benchmark=False,
    use_fast_math=False,
    data_dtype="float16",
    weight_dtype="float16",
    ref_target="cuda",
    use_vm=False,
):
    mod_nchw = tvm.IRModule.from_expr(expr_nchw)
    typ = relay.transform.InferType()(mod_nchw)["main"].body.checked_type

    use_vm = use_vm or any(isinstance(s, tvm.tir.Any) for s in typ.shape)

    np_data = get_random_ndarray(d_shape, data_dtype)
    np_weight = get_random_ndarray(w_shape, weight_dtype)
    np_bias = get_random_ndarray((w_shape[0],), typ.dtype)
    params = {"weight": np_weight, "bias": np_bias}

    split_k_slices = [1]

    return verify_conv2d_common(
        expr_nchw,
        expr_ref,
        ["data"],
        [np_data],
        params,
        sm,
        split_k_slices,
        atol,
        rtol,
        use_cudnn_ref,
        run_benchmark,
        use_fast_math,
        ref_target,
        use_vm,
    )


def verify_conv2d_backward_weight(
    expr_nchw,  # can be dynamic batch
    expr_ref,  # always static batch
    grad_shape,
    data_shape,
    sm=80,
    split_k_slices=[1],
    atol=1e-5,
    rtol=1e-5,
    use_cudnn_ref=False,
    use_fast_math=False,
    grad_dtype="float16",
    data_dtype="float16",
    ref_target="cuda",
    use_vm=False,
):
    np_grad = get_random_ndarray(grad_shape, grad_dtype)
    np_data = get_random_ndarray(data_shape, data_dtype)
    params = {}
    input_names = ["grad", "data"]
    return verify_conv2d_common(
        expr_nchw,
        expr_ref,
        input_names,
        [np_grad, np_data],
        params,
        sm,
        split_k_slices,
        atol,
        rtol,
        use_cudnn_ref,
        False,
        use_fast_math,
        ref_target,
        use_vm,
    )


@tvm.testing.requires_cutlass
def test_conv2d():
    d_shape = (16, 16, 32, 32)
    w_shape = (32, 16, 3, 3)
    padding = (1, 1)

    for IC in [3, 16]:
        d_shape = (16, IC, 32, 32)
        w_shape = (32, IC, 3, 3)
        mod_nchw = get_conv2d_nchw(d_shape, w_shape, padding)

        verify_conv2d(
            mod_nchw,
            mod_nchw,
            d_shape,
            w_shape,
            sm=80,
            atol=1e-5,
            rtol=1e-5,
            use_cudnn_ref=(IC == 3),  # The autotvm kernel has an accuracy issue with IC == 3 case
            run_benchmark=False,
        )

    dyn_batch_shape = (relay.Any(),) + d_shape[1:]
    mod_nchw = get_conv2d_nchw(d_shape, w_shape, padding)
    mod_dyn = get_conv2d_nchw(dyn_batch_shape, w_shape, padding)

    verify_conv2d(
        mod_dyn, mod_nchw, d_shape, w_shape, sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
    )

    for data_dtype, weight_dtype, out_dtype in [
        ("float32", "float32", "float32"),  # 3xtf32
        ("int8", "int8", "int32"),
        ("uint8", "int8", "int32"),
    ]:
        expr = get_conv2d_nchw(
            d_shape,
            w_shape,
            padding,
            out_dtype=out_dtype,
            data_dtype=data_dtype,
            weight_dtype=weight_dtype,
        )

        verify_conv2d(
            expr,
            expr,
            d_shape,
            w_shape,
            sm=80,
            atol=1e-5,
            rtol=1e-5,
            run_benchmark=False,
            data_dtype=data_dtype,
            weight_dtype=weight_dtype,
            ref_target="llvm",
        )

    # align1 + int8 case
    d_shape = (16, 3, 32, 32)
    w_shape = (32, 3, 3, 3)
    mod_nchw = get_conv2d_nchw(
        d_shape, w_shape, padding, out_dtype="int32", data_dtype="uint8", weight_dtype="int8"
    )

    verify_conv2d(
        mod_nchw,
        mod_nchw,
        d_shape,
        w_shape,
        sm=80,
        atol=1e-5,
        rtol=1e-5,
        ref_target="llvm",
        data_dtype="uint8",
        weight_dtype="int8",
    )


@tvm.testing.requires_cutlass
def test_conv2d_fusion():
    d_shape = (16, 16, 32, 32)
    w_shape = (32, 16, 3, 3)
    padding = (1, 1)

    mod_nchw = get_conv2d_nchw_bias(d_shape, w_shape, padding)
    verify_conv2d(
        mod_nchw, mod_nchw, d_shape, w_shape, sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
    )

    mod_nchw = get_conv2d_nchw_bias_relu(d_shape, w_shape, padding)
    verify_conv2d(
        mod_nchw, mod_nchw, d_shape, w_shape, sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
    )

    mod_nchw = get_conv2d_nchw_bias_sigmoid(d_shape, w_shape, padding, out_dtype="float16")
    verify_conv2d(
        mod_nchw, mod_nchw, d_shape, w_shape, sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
    )
    verify_conv2d(
        mod_nchw,
        mod_nchw,
        d_shape,
        w_shape,
        sm=80,
        atol=1e-3,
        rtol=1e-3,
        run_benchmark=False,
        use_fast_math=True,
    )

    mod_nchw = get_conv2d_nchw_bias_sigmoid(d_shape, w_shape, padding, out_dtype="float32")
    verify_conv2d(
        mod_nchw, mod_nchw, d_shape, w_shape, sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
    )

    mod_nchw = get_conv2d_nchw_bias_silu(d_shape, w_shape, padding, out_dtype="float32")
    verify_conv2d(
        mod_nchw, mod_nchw, d_shape, w_shape, sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
    )

    mod_nchw = get_conv2d_nchw_bias_hardswish(d_shape, w_shape, padding, out_dtype="float16")
    verify_conv2d(
        mod_nchw, mod_nchw, d_shape, w_shape, sm=80, atol=5e-2, rtol=5e-2, run_benchmark=False
    )


@tvm.testing.requires_cutlass
def test_conv2d_residual_block():
    d_shape = (16, 16, 32, 32)
    w_shape = (16, 16, 3, 3)
    padding = (1, 1)

    bias_add, residual_input = get_conv2d_nchw_bias_residual(d_shape, w_shape, padding)

    for func, tol in [
        (relay.nn.relu(bias_add + residual_input), 1e-5),
        (relay.nn.relu(bias_add) + residual_input, 1e-5),
        (relay.sigmoid(bias_add) * residual_input, 1e-5),
        (relay.nn.relu(silu(bias_add) * residual_input), 1e-5),
        # HardSwish requires higher tolerance since vectoring the residual block epilogue
        # in cutlass.
        # TODO(masahi): Invesitigate this issue
        (relay.nn.relu(hardswish(bias_add) + residual_input), 5e-2),
    ]:
        verify_conv2d(func, func, d_shape, w_shape, sm=80, atol=tol, rtol=tol, run_benchmark=False)


@tvm.testing.requires_cutlass
def test_conv2d_transpose():
    OC = 8
    IC = 16
    d_shape = (16, IC, 32, 32)
    w_shape = (OC, IC, 3, 3)
    padding = (1, 1)
    dtype = "float32"

    for strides in [(1, 1), (2, 2)]:
        o_shape = conv_output_shape(
            0, padding, strides, (1, 1), d_shape, (OC, IC, 3, 3), "float32", "float32"
        )
        output_padding = (1, 1) if strides[0] > 1 else (0, 0)
        mod_nchw = get_conv2d_transpose_nchw(
            o_shape,
            w_shape,
            padding,
            output_padding,
            strides,
            out_dtype=dtype,
            data_dtype=dtype,
            weight_dtype=dtype,
        )

        verify_conv2d(
            mod_nchw,
            mod_nchw,
            o_shape,
            w_shape,
            sm=80,
            atol=1e-3,
            rtol=1e-3,
            use_cudnn_ref=False,
            run_benchmark=False,
            data_dtype=dtype,
            weight_dtype=dtype,
        )


@tvm.testing.requires_cutlass
def test_conv2d_backward_weight():
    OC = 8
    IC = 16
    d_shape = (16, IC, 32, 32)
    w_shape = (OC, IC, 3, 3)
    dtype = "float16"

    for strides in [(1, 1), (2, 2)]:
        o_shape = (16, OC, 32 // strides[0], 32 // strides[1])
        padding = (1, 1)

        mod_nchw = get_conv2d_backward_weight(
            d_shape,
            w_shape,
            o_shape,
            padding,
            strides,
            out_dtype="float32",
            data_dtype=dtype,
            weight_dtype=dtype,
        )

        for split_k_slices in [1, 8]:
            verify_conv2d_backward_weight(
                mod_nchw,
                mod_nchw,
                o_shape,
                d_shape,
                sm=80,
                split_k_slices=[split_k_slices],
                atol=5e-3,
                rtol=5e-3,
                use_cudnn_ref=False,
                grad_dtype=dtype,
                data_dtype=dtype,
            )


@tvm.testing.requires_cutlass
def test_conv2d_bwd():
    IC = 16
    OC = 8
    dshape = (16, IC, 32, 32)
    wshape = (OC, IC, 3, 3)
    padding = (0, 0)
    strides = (1, 1)

    conv = get_conv2d_nchw(
        dshape,
        wshape,
        padding,
        strides=strides,
        out_dtype="float32",
        data_dtype="float32",
        weight_dtype="float32",
    )
    fwd_mod = InferType()(tvm.IRModule.from_expr(conv))

    # Note: large difference in tvm and cutlass Wgrad results if use fp16.
    # Cutlass wgrad uses fp32 accumulation even if the output is fp16.
    use_fp16 = False
    verify_dgrad = False  # False to verify wgrad
    tol = 1e-5 if verify_dgrad else 1e-4  # Wgrad slightly less accurate

    if use_fp16:
        fwd_mod = ToMixedPrecision("float16")(fwd_mod)

    fwd_bwd_func = FirstOrderGradient()(fwd_mod)["main"]

    bwd_func = relay.Function(
        fwd_bwd_func.params,
        relay.TupleGetItem(relay.TupleGetItem(fwd_bwd_func.body, 1), 0 if verify_dgrad else 1),
    )

    verify_conv2d(
        bwd_func,
        bwd_func,
        dshape,
        wshape,
        sm=80,
        atol=1e-2 if use_fp16 else tol,
        rtol=1e-2 if use_fp16 else tol,
        use_cudnn_ref=False,
        data_dtype="float32",
        weight_dtype="float32",
        use_vm=True,
    )


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
    cutlass_rt_mod, dev, num_partition = profile_and_build(
        mod,
        params,
        sm,
        use_3xtf32=use_3xtf32,
        use_ansor=False,
    )
    cutlass_ansor_rt_mod, dev, num_partition = profile_and_build(
        mod,
        params,
        sm,
        use_3xtf32=use_3xtf32,
        use_ansor=True,
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


@tvm.testing.requires_cutlass
def test_dense_transpose_dense():
    verify_dense_transpose_dense(get_dense_transpose_dense(M, N, K), M, N, K)


def verify_group_gemm(
    func_name, M, N, K, num_groups, x_dtype, weight_dtype, out_dtype, use_scale, rtol, atol
):
    group_gemm_func = tvm.get_global_func(func_name, allow_missing=True)
    if group_gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    @memoize("tvm.contrib.cutlass.test_group_gemm_sm90")
    def get_ref_data():
        assert M % num_groups == 0
        M_per_group = M // num_groups
        a_np = get_random_ndarray((M, K), "float16")
        b_np = get_random_ndarray((num_groups, N, K), "float16")
        indptr_np = np.arange(1, num_groups + 1).astype("int64") * M_per_group
        c_np = np.concatenate(
            [a_np[i * M_per_group : (i + 1) * M_per_group] @ b_np[i].T for i in range(num_groups)],
            axis=0,
        )
        return a_np, b_np, indptr_np, c_np

    def to_numpy_dtype(dtype):
        mapping = {"e5m2_float8": ml_dtypes.float8_e5m2, "e4m3_float8": ml_dtypes.float8_e4m3fn}
        return mapping.get(dtype, dtype)

    a_np, b_np, indptr_np, c_np = get_ref_data()
    dev = tvm.cuda(0)
    a_nd = tvm.nd.array(a_np.astype(to_numpy_dtype(x_dtype)), device=dev)
    b_nd = tvm.nd.array(b_np.astype(to_numpy_dtype(weight_dtype)), device=dev)
    c_nd = tvm.nd.empty(c_np.shape, dtype=out_dtype, device=dev)
    indptr_nd = tvm.nd.array(indptr_np, device=dev)
    workspace = tvm.nd.empty((4096 * 1024,), dtype="uint8", device=dev)
    if use_scale:
        scale = tvm.nd.array(np.array([1.0], dtype="float32"), device=dev)
        group_gemm_func(a_nd, b_nd, indptr_nd, workspace, scale, c_nd)
    else:
        group_gemm_func(a_nd, b_nd, indptr_nd, workspace, c_nd)
    tvm.testing.assert_allclose(c_nd.asnumpy(), c_np, rtol=rtol, atol=atol)


@tvm.testing.requires_cutlass
def test_group_gemm_sm90():
    verify_group_gemm(
        "cutlass.group_gemm_fp16_sm90",
        8,
        128,
        128,
        4,
        "float16",
        "float16",
        "float16",
        False,
        rtol=1e-3,
        atol=1e-3,
    )
    verify_group_gemm(
        "cutlass.group_gemm_e5m2_e5m2_fp16",
        8,
        16,
        16,
        4,
        "e5m2_float8",
        "e5m2_float8",
        "float16",
        True,
        rtol=1e-1,
        atol=1,
    )
    verify_group_gemm(
        "cutlass.group_gemm_e4m3_e4m3_fp16",
        8,
        16,
        16,
        4,
        "e4m3_float8",
        "e4m3_float8",
        "float16",
        True,
        rtol=1e-1,
        atol=1,
    )
    verify_group_gemm(
        "cutlass.group_gemm_e5m2_e4m3_fp16",
        8,
        16,
        16,
        4,
        "e5m2_float8",
        "e4m3_float8",
        "float16",
        True,
        rtol=1e-1,
        atol=1,
    )


def verify_gemm(func_name, M, N, K, x_dtype, weight_dtype, out_dtype, scale_value, rtol, atol):
    gemm_func = tvm.get_global_func(func_name, allow_missing=True)
    if gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    @memoize("tvm.contrib.cutlass.test_fp8_gemm_sm90")
    def get_ref_data():
        a_np = get_random_ndarray((M, K), "float16")
        b_np = get_random_ndarray((N, K), "float16")
        c_np = a_np @ b_np.T * scale_value
        return a_np, b_np, c_np

    def to_numpy_dtype(dtype):
        mapping = {"e5m2_float8": ml_dtypes.float8_e5m2, "e4m3_float8": ml_dtypes.float8_e4m3fn}
        return mapping.get(dtype, dtype)

    a_np, b_np, c_np = get_ref_data()
    dev = tvm.cuda(0)
    a_nd = tvm.nd.array(a_np.astype(to_numpy_dtype(x_dtype)), device=dev)
    b_nd = tvm.nd.array(b_np.astype(to_numpy_dtype(weight_dtype)), device=dev)
    c_nd = tvm.nd.empty(c_np.shape, dtype=out_dtype, device=dev)
    workspace = tvm.nd.empty((4096 * 1024,), dtype="uint8", device=dev)
    scale = tvm.nd.array(np.array([scale_value], dtype="float32"), device=dev)
    gemm_func(a_nd, b_nd, workspace, scale, c_nd)
    tvm.testing.assert_allclose(c_nd.asnumpy(), c_np, rtol=rtol, atol=atol)


@tvm.testing.requires_cutlass
def test_fp8_gemm_sm90():
    verify_gemm(
        "cutlass.gemm_e5m2_e5m2_fp16",
        8,
        16,
        16,
        "e5m2_float8",
        "e5m2_float8",
        "float16",
        1.5,
        rtol=1e-1,
        atol=1,
    )
    verify_gemm(
        "cutlass.gemm_e4m3_e4m3_fp16",
        8,
        16,
        16,
        "e4m3_float8",
        "e4m3_float8",
        "float16",
        1.5,
        rtol=1e-1,
        atol=1,
    )
    verify_gemm(
        "cutlass.gemm_e4m3_e4m3_fp16",
        32,
        16,
        16,
        "e4m3_float8",
        "e4m3_float8",
        "float16",
        1.5,
        rtol=1e-1,
        atol=1,
    )
    verify_gemm(
        "cutlass.gemm_e5m2_e4m3_fp16",
        8,
        16,
        16,
        "e5m2_float8",
        "e4m3_float8",
        "float16",
        1.5,
        rtol=1e-1,
        atol=1,
    )


if __name__ == "__main__":
    tvm.testing.main()
