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
import pytest
import tvm
from tvm import relay
import numpy as np
from tvm.runtime.vm import VirtualMachine
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass import (
    tune_cutlass_kernels,
    build_cutlass_kernels,
    build_cutlass_kernels_vm,
)

logging.basicConfig(level=logging.INFO)


def has_cublas():
    return tvm.get_global_func("tvm.contrib.cublas.matmul", True) != None


def has_cutlass():
    return tvm.get_global_func("relay.ext.cutlass", True) != None


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


def get_dense_with_shape(data_shape, weight_shape, out_dtype="float16"):
    data = relay.var("data", shape=data_shape, dtype="float16")
    weight = relay.var("weight", shape=weight_shape, dtype="float16")
    return relay.nn.dense(data, weight, out_dtype=out_dtype)


def get_dense(M, N, K, out_dtype="float16"):
    return get_dense_with_shape((M, K), (N, K), out_dtype)


def get_dense_bias(M, N, K, out_dtype="float16"):
    dense = get_dense(M, N, K, out_dtype=out_dtype)
    bias = relay.var("bias", shape=(N,), dtype=out_dtype)
    return relay.nn.bias_add(dense, bias)


def get_dense_bias_relu(M, N, K, out_dtype="float16"):
    return relay.nn.relu(get_dense_bias(M, N, K, out_dtype="float16"))


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


def profile_and_build(mod, params, sm, tmp_dir="./tmp", lib_path="compile.so"):
    mod = partition_for_cutlass(mod)
    mod, num_cutlass_partition = tune_cutlass_kernels(
        mod, sm, profile_all=False, use_multiprocessing=False, tmp_dir=tmp_dir
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="cuda", params=params)
    lib = build_cutlass_kernels(lib, sm, tmp_dir, lib_path)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


def profile_and_build_vm(
    mod, params, sm, tmp_dir="./tmp", lib_path="compile.so", vmcode_path="vmcode.ro"
):
    mod = partition_for_cutlass(mod)
    mod, num_cutlass_partition = tune_cutlass_kernels(mod, sm, tmp_dir=tmp_dir)
    with tvm.transform.PassContext(opt_level=3):
        vm_exec = relay.vm.compile(mod, target="cuda", params=params)
    vm_exec = build_cutlass_kernels_vm(vm_exec, sm, tmp_dir, lib_path, vmcode_path)
    dev = tvm.device("cuda", 0)
    return VirtualMachine(vm_exec, dev), dev, num_cutlass_partition


def verify_dense(
    func, M, N, K, ref_target="cuda", sm=80, atol=1e-5, rtol=1e-5, run_benchmark=False
):
    if not has_cutlass():
        return
    mod = tvm.IRModule.from_expr(func)
    typ = relay.transform.InferType()(mod)["main"].body.checked_type
    out_dtype = typ.dtype
    use_vm = any(isinstance(s, tvm.tir.Any) for s in typ.shape)
    np_data = np.random.uniform(-1, 1, (M, K)).astype("float16")
    np_weight = np.random.uniform(-1, 1, (N, K)).astype("float16")
    np_bias = np.random.uniform(-1, 1, (N,)).astype(out_dtype)

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
            rt_mod, dev, num_partition = profile_and_build_vm(mod, params, sm)

        rt_mod_ref, dev = get_ref_vm(mod, params, target=ref_target)
        x = tvm.nd.array(np_data, device=dev)
        out = get_output_vm(rt_mod, ["data"], [x])
        ref_out = get_output_vm(rt_mod_ref, ["data"], [x])
    else:
        rt_mod_ref, dev = get_ref_rt_mod(mod, params, target=ref_target)
        rt_mod, dev, num_partition = profile_and_build(mod, params, sm)
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
    if not has_cutlass():
        return
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

    if True:
        print("CUTLASS:", rt_mod.benchmark(dev, number=1, repeat=600))
        print("TVM Tensorcore (no tuning):", rt_mod_ref.benchmark(dev, number=1, repeat=600))


M = 1820
N = 768
K = 768


def test_dense():
    verify_dense(get_dense(M, N, K), M, N, K)
    verify_dense(get_dense(M, N, K, out_dtype="float32"), M, N, K)


def test_dense_bias():
    verify_dense(get_dense_bias(M, N, K), M, N, K)
    verify_dense(get_dense_bias(M, N, K, out_dtype="float32"), M, N, K)


def test_dense_bias_relu():
    verify_dense(get_dense_bias_relu(M, N, K), M, N, K)
    verify_dense(get_dense_bias_relu(M, N, K, out_dtype="float32"), M, N, K)


def test_dense_bias_gelu():
    verify_dense(get_dense_bias_gelu(M, N, K), M, N, K, atol=1e-3, rtol=1e-3)
    verify_dense(get_dense_bias_gelu(M, N, K, out_dtype="float32"), M, N, K, atol=1e-3, rtol=1e-3)


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


if __name__ == "__main__":
    pytest.main([__file__])
