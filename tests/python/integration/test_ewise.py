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
"""Test elementwise integration."""
import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm.contrib import nvcc


@tvm.testing.requires_gpu
def test_exp():
    """Test scheduling and running exponent."""
    # graph
    arr_length = 1024
    arr_length_tvm = tvm.runtime.convert(arr_length)
    placeholder_a = te.placeholder((arr_length_tvm,), name="A")
    placeholder_b = te.compute(placeholder_a.shape, lambda *i: te.exp(placeholder_a(*i)), name="B")
    schedule = te.create_schedule(placeholder_b.op)
    # create iter var and assign them tags.
    num_thread = 8
    axis1, axis2 = schedule[placeholder_b].split(placeholder_b.op.axis[0], factor=num_thread)
    schedule[placeholder_b].bind(axis1, te.thread_axis("blockIdx.x"))
    schedule[placeholder_b].bind(axis2, te.thread_axis("threadIdx.x"))

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.testing.device_enabled(host):
            return
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        fexp = tvm.build(schedule, [placeholder_a, placeholder_b], device, host, name="myexp")
        dev = tvm.device(device, 0)
        # launch the kernel.
        buff_a = tvm.nd.array(np.random.uniform(size=arr_length).astype(placeholder_a.dtype), dev)
        buff_b = tvm.nd.array(np.zeros(arr_length, dtype=placeholder_b.dtype), dev)
        fexp(buff_a, buff_b)
        tvm.testing.assert_allclose(buff_b.numpy(), np.exp(buff_a.numpy()), rtol=1e-5)

    check_device("opencl -device=intel_graphics")
    check_device("cuda", "llvm")
    check_device("vulkan")


@tvm.testing.requires_gpu
def test_fmod():
    """Test scheduling and running fmod."""

    # graph
    def run(dtype):
        size_var_n = te.size_var("n")
        placeholder_a = te.placeholder((size_var_n,), name="A", dtype=dtype)
        placeholder_b = te.placeholder((size_var_n,), name="B", dtype=dtype)
        result_c = te.compute(
            placeholder_a.shape, lambda *i: te.fmod(placeholder_a(*i), placeholder_b(*i)), name="C"
        )
        schedule = te.create_schedule(result_c.op)
        # create iter var and assign them tags.
        num_thread = 8
        axis0, axis1 = schedule[result_c].split(result_c.op.axis[0], factor=num_thread)

        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            target = tvm.target.Target(device)
            if "cpu" not in target.keys:
                schedule[result_c].bind(axis0, te.thread_axis("blockIdx.x"))
                schedule[result_c].bind(axis1, te.thread_axis("threadIdx.x"))
            fmod = tvm.build(
                schedule, [placeholder_a, placeholder_b, result_c], device, name="myfmod"
            )

            # launch the kernel.
            value_n = 1024
            a_np = (np.random.uniform(size=value_n) * 256).astype(placeholder_a.dtype)
            b_np = (np.random.uniform(size=value_n) * 256).astype(placeholder_b.dtype)

            # "fix" the values in a and b to avoid the result being too small
            b_np += (b_np < 2.0) * 2
            a_np[np.abs(np.fmod(a_np, b_np)) < 1] += 1

            buff_a = tvm.nd.array(a_np, dev)
            buff_b = tvm.nd.array(b_np, dev)
            buff_c = tvm.nd.array(np.zeros(value_n, dtype=result_c.dtype), dev)
            ftimer = fmod.time_evaluator(fmod.entry_name, dev, number=1)
            _ = ftimer(buff_a, buff_b, buff_c).mean
            np.testing.assert_allclose(
                buff_c.numpy(), np.mod(buff_a.numpy(), buff_b.numpy()), rtol=1e-5
            )

        check_device("cuda")
        check_device("opencl -device=intel_graphics")
        check_device("metal")

    run("float32")


@tvm.testing.requires_gpu
def test_multiple_cache_write():
    """Test multiple cache writes."""
    # graph
    arr_length = 1024
    arr_length_tvm = tvm.runtime.convert(arr_length)
    placeholder_a0 = te.placeholder((arr_length_tvm,), name="A0", dtype="float32")
    placeholder_a1 = te.placeholder((arr_length_tvm,), name="A1", dtype="float32")
    result_b0, result_b1 = te.compute(
        (arr_length_tvm,),
        lambda *i: (
            placeholder_a0(*i) + placeholder_a1(*i),
            placeholder_a0(*i) * placeholder_a1(*i),
        ),
        name="B",
    )
    result_c = te.compute((arr_length_tvm,), lambda *i: result_b0(*i) + result_b1(*i), name="C")
    schedule = te.create_schedule(result_c.op)
    # create iter var and assign them tags.
    num_thread = 8
    cache_b0, _ = schedule.cache_write([result_b0, result_b1], "local")
    axis0, axis1 = schedule[result_c].split(result_c.op.axis[0], factor=num_thread)
    schedule[result_b0].compute_at(schedule[result_c], axis0)
    schedule[cache_b0].compute_at(schedule[result_c], axis0)
    schedule[result_c].bind(axis0, te.thread_axis("blockIdx.x"))
    schedule[result_c].bind(axis1, te.thread_axis("threadIdx.x"))

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.testing.device_enabled(host):
            return
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            return
        func = tvm.build(
            schedule,
            [placeholder_a0, placeholder_a1, result_c],
            device,
            host,
            name="multiple_cache_write",
        )
        dev = tvm.device(device, 0)
        # launch the kernel.
        buff_a0 = tvm.nd.array(np.random.uniform(size=arr_length).astype(placeholder_a0.dtype), dev)
        buff_a1 = tvm.nd.array(np.random.uniform(size=arr_length).astype(placeholder_a1.dtype), dev)
        buff_c = tvm.nd.array(np.zeros(arr_length, dtype=result_c.dtype), dev)
        func(buff_a0, buff_a1, buff_c)
        tvm.testing.assert_allclose(
            buff_c.numpy(),
            buff_a0.numpy() + buff_a1.numpy() + (buff_a0.numpy() * buff_a1.numpy()),
            rtol=1e-5,
        )

    check_device("cuda", "llvm")
    check_device("vulkan")
    check_device("opencl")


def test_log_pow_llvm():
    """Test log pow using llvm to lower."""
    # graph
    size_var_n = te.size_var("n")
    placeholder_a = te.placeholder((size_var_n,), name="A")
    result_b = te.compute(
        placeholder_a.shape, lambda *i: te.power(te.log(placeholder_a(*i)), 2.0), name="B"
    )
    schedule = te.create_schedule(result_b.op)
    # create iter var and assign them tags.
    schedule[result_b].split(result_b.op.axis[0], factor=32)
    # one line to build the function.
    if not tvm.testing.device_enabled("llvm"):
        return

    flog = tvm.build(schedule, [placeholder_a, result_b], "llvm", name="mylog")
    dev = tvm.cpu(0)
    # launch the kernel.
    size_var_n = 1028
    buff_a = tvm.nd.array(np.random.uniform(size=size_var_n).astype(placeholder_a.dtype), dev)
    buff_b = tvm.nd.array(np.zeros(size_var_n, dtype=result_b.dtype), dev)
    repeat = 10
    ftimer = flog.time_evaluator(flog.entry_name, dev, number=1, repeat=repeat)
    res = ftimer(buff_a, buff_b)
    assert len(res.results) == repeat
    tvm.testing.assert_allclose(buff_b.numpy(), np.power(np.log(buff_a.numpy()), 2.0), rtol=1e-5)


@tvm.testing.uses_gpu
def test_popcount():
    """Test popcount."""

    def run(dtype):
        # graph
        arr_length = 1024
        arr_length_tvm = tvm.runtime.convert(1024)
        placeholder_a = te.placeholder((arr_length_tvm,), name="A", dtype=dtype)
        placeholder_b = te.compute(
            placeholder_a.shape, lambda *i: tvm.tir.popcount(placeholder_a(*i)), name="B"
        )
        schedule = te.create_schedule(placeholder_b.op)
        # simple schedule
        num_thread = 8
        axis1, axis2 = schedule[placeholder_b].split(placeholder_b.op.axis[0], factor=num_thread)

        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            target = tvm.target.Target(device)
            if "cpu" not in target.keys:
                schedule[placeholder_b].bind(axis1, te.thread_axis("blockIdx.x"))
                schedule[placeholder_b].bind(axis2, te.thread_axis("threadIdx.x"))
            func = tvm.build(schedule, [placeholder_a, placeholder_b], device)
            # launch the kernel.
            buff_a = tvm.nd.array(
                np.random.randint(low=0, high=1000, size=arr_length, dtype=placeholder_a.dtype), dev
            )
            buff_b = tvm.nd.array(np.zeros(shape=arr_length, dtype=placeholder_b.dtype), dev)
            func(buff_a, buff_b)
            tvm.testing.assert_allclose(
                buff_b.numpy(), list(map(lambda x: bin(x).count("1"), buff_a.numpy())), rtol=1e-5
            )

        check_device("llvm")
        check_device("cuda")
        check_device("opencl")
        if dtype == "uint32":
            check_device("metal")
            check_device("vulkan")

    run("uint32")
    run("uint64")


@tvm.testing.requires_gpu
def test_add():
    """Test addition."""

    def run(dtype):
        # graph
        size_var_n = te.size_var("n")
        placeholder_a = te.placeholder((size_var_n,), name="A", dtype=dtype)
        placeholder_b = te.placeholder((size_var_n,), name="B", dtype=dtype)
        result_c = te.compute(
            placeholder_a.shape, lambda *i: placeholder_a(*i) + placeholder_b(*i), name="C"
        )
        # schedule
        schedule = te.create_schedule(result_c.op)
        # create iter var and assign them tags.
        num_thread = 16
        axis_bx, axis_x = schedule[result_c].split(result_c.op.axis[0], factor=num_thread * 4)
        axis_tx, axis_x = schedule[result_c].split(axis_x, nparts=num_thread)
        _, axis_x = schedule[result_c].split(axis_x, factor=4)
        schedule[result_c].bind(axis_bx, te.thread_axis("blockIdx.x"))
        schedule[result_c].bind(axis_tx, te.thread_axis("threadIdx.x"))
        schedule[result_c].vectorize(axis_x)

        # one line to build the function.
        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            fadd = tvm.build(
                schedule, [placeholder_a, placeholder_b, result_c], device, name="myadd"
            )

            # launch the kernel.
            n = 1024
            buff_a = tvm.nd.array(
                (np.random.uniform(size=n) * 256).astype(placeholder_a.dtype), dev
            )
            buff_b = tvm.nd.array(
                (np.random.uniform(size=n) * 256).astype(placeholder_b.dtype), dev
            )
            buff_c = tvm.nd.array(np.zeros(n, dtype=result_c.dtype), dev)
            ftimer = fadd.time_evaluator(fadd.entry_name, dev, number=1)
            _ = ftimer(buff_a, buff_b, buff_c).mean
            tvm.testing.assert_allclose(buff_c.numpy(), buff_a.numpy() + buff_b.numpy(), rtol=1e-6)

        check_device("opencl")
        check_device("cuda")
        if dtype == "float32":
            check_device("metal")
            check_device("vulkan")

    run("float32")
    run("int32")
    run("int64")
    run("uint64")


@tvm.testing.requires_gpu
def try_warp_memory():
    """Test using warp memory
    skip this in default test because it require higher arch"""
    arr_size = 128
    placeholder_a = te.placeholder((arr_size,), name="A")
    result_b = te.compute((arr_size,), lambda i: placeholder_a[i] + 3, name="B")
    warp_size = 32
    schedule = te.create_schedule(result_b.op)
    cache_read_aa = schedule.cache_read(placeholder_a, "warp", [result_b])
    axis_x0, axis_xi = schedule[result_b].split(result_b.op.axis[0], warp_size * 2)
    _, axis_xi1 = schedule[result_b].split(axis_xi, factor=warp_size)
    thread_axis_tx = te.thread_axis("threadIdx.x")
    schedule[result_b].bind(axis_xi1, thread_axis_tx)
    schedule[result_b].bind(axis_x0, te.thread_axis("blockIdx.x"))
    schedule[cache_read_aa].compute_at(schedule[result_b], axis_x0)
    axis_x0, axis_xi = schedule[cache_read_aa].split(schedule[cache_read_aa].op.axis[0], warp_size)
    schedule[cache_read_aa].bind(axis_xi, thread_axis_tx)

    @tvm.register_func("tvm_callback_cuda_compile", override=True)
    def tvm_callback_cuda_compile(code, _):  # pylint: disable=unused-variable
        ptx = nvcc.compile_cuda(code)
        return ptx

    # one line to build the function.
    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        myfunc = tvm.build(schedule, [placeholder_a, result_b], device)
        buff_a = tvm.nd.array(
            (np.random.uniform(size=arr_size) * 256).astype(placeholder_a.dtype), dev
        )
        buff_b = tvm.nd.array(np.zeros(arr_size, dtype=result_b.dtype), dev)
        myfunc(buff_a, buff_b)
        tvm.testing.assert_allclose(buff_b.numpy(), buff_a.numpy() + 3, rtol=1e-6)

    check_device("cuda")


if __name__ == "__main__":
    test_exp()
    try_warp_memory()
    test_multiple_cache_write()
    test_add()
    test_log_pow_llvm()
    test_popcount()
    test_fmod()
