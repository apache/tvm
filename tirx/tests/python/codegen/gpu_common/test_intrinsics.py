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
from functools import partial

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import cc, popen_pool, utils
from tvm_tirx.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "supports_int64": True}, marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_int_intrin(target, dtype):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")
    test_funcs = [
        (T.clz, lambda x, dtype: int(dtype[-2:]) - (len(bin(x)) - 2)),
    ]

    for tvm_intrin, np_func in test_funcs:
        n = 128

        @I.ir_module
        class Module:
            @T.function
            def main(
                A: T.Tensor((n,), dtype),
                B: T.Tensor((n,), dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                with T.launch_thread("threadIdx.x", n) as i0:
                    B[i0] = tvm_intrin(A[i0])

        f = tvm_tirx.tirx.compile(Module, target=target)

        def run_and_check():
            dev = tvm_tirx.device_from_target(target)
            a = tvm_tirx.runtime.tensor(np.random.randint(0, 100000, size=n).astype(dtype), dev)
            b = tvm_tirx.runtime.tensor(np.zeros(shape=(n,)).astype(dtype), dev)
            f(a, b)
            ref = np.vectorize(partial(np_func, dtype=dtype))(a.numpy())
            tvm_tirx.testing.assert_allclose(b.numpy(), ref)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_device_module_dump():
    pytest.importorskip("cloudpickle")  # needed by popen_pool.PopenWorker

    @T.function
    def main(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        with T.launch_thread("blockIdx.x", 128) as bx:
            with T.launch_thread("threadIdx.x", 8) as tx:
                B[bx * 8 + tx] = A[bx * 8 + tx] + T.float32(1)

    def check_device(device):
        if not tvm_tirx.testing.device_enabled(device):
            print(f"Skip because {device} is not enabled")
            return
        temp = utils.tempdir()
        f = tvm_tirx.tirx.compile(main, target=device)

        path_dso = temp.relpath("dev_lib.so")
        # test cross compiler function
        f.export_library(path_dso, fcompile=cc.cross_compiler("g++"))

        def run_and_check():
            dev = tvm_ffi.device(device, 0)

            def popen_check():
                import tvm_tirx

                f1 = tvm_tirx.runtime.load_module(path_dso)
                a = tvm_tirx.runtime.tensor(np.random.uniform(size=1024).astype("float32"), dev)
                b = tvm_tirx.runtime.tensor(np.zeros(1024, dtype="float32"), dev)
                f1(a, b)
                np.testing.assert_equal(b.numpy(), a.numpy() + 1)

            worker = popen_pool.PopenWorker()
            try:
                worker.send(popen_check)
                worker.recv()
            finally:
                worker.kill()

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    def check_c(device):
        if not tvm_tirx.testing.device_enabled(device):
            print(f"Skip because {device} is not enabled")
            return
        f = tvm_tirx.tirx.compile(main, target=tvm_tirx.target.Target(device, host="c"))

        def run_and_check():
            dev = tvm_ffi.device(device, 0)
            a = tvm_tirx.runtime.tensor(np.random.uniform(size=1024).astype("float32"), dev)
            b = tvm_tirx.runtime.tensor(np.zeros(1024, dtype="float32"), dev)
            f["main"](a, b)
            np.testing.assert_equal(b.numpy(), a.numpy() + 1)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    for device in ["cuda", "vulkan", "opencl", "metal"]:
        check_device(device)
        check_c(device)


if __name__ == "__main__":
    tvm_tirx.testing.main()
