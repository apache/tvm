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

import ctypes

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.support.nvcc
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import cc, popen_pool, tar, utils
from tvm_tirx.testing import env


@pytest.fixture(params=["nvcc", "nvrtc"])
def backend_config(request):
    if request.param == "nvrtc":
        pytest.importorskip("cuda.bindings.nvrtc")
    return {"cuda": {"compiler": request.param}}


@pytest.mark.gpu
def test_cuda_multi_lib():
    pytest.importorskip("cloudpickle")

    # test combining two system lib together
    # each contains a fatbin component in cuda
    for device in ["llvm", "cuda"]:
        if not tvm_tirx.testing.device_enabled(device):
            print(f"skip because {device} is not enabled...")
            return

    @tvm_tirx.script.ir_module
    class ModA:
        I.module_attrs({"system_lib_prefix": "modA_"})

        @T.function
        def my_inplace_update(x: T.Tensor((12), "float32")) -> None:
            T.func_attr({"global_symbol": "modA_my_inplace_update"})
            with T.launch_thread("blockIdx.x", T.int64(1)) as bx:  # noqa: F841
                with T.launch_thread("threadIdx.x", T.int64(12)) as tx:
                    x[tx] = x[tx] + 1

    @tvm_tirx.script.ir_module
    class ModB:
        I.module_attrs({"system_lib_prefix": "modB_"})

        @T.function
        def my_inplace_update(x: T.Tensor((12), "float32")) -> None:
            T.func_attr({"global_symbol": "modB_my_inplace_update"})
            with T.launch_thread("blockIdx.x", T.int64(1)) as bx:  # noqa: F841
                with T.launch_thread("threadIdx.x", T.int64(12)) as tx:
                    x[tx] = x[tx] + 2

    temp = utils.tempdir()
    target = tvm_tirx.target.Target("cuda", host="llvm")
    libA = tvm_tirx.tirx.compile(ModA, target=target)
    libB = tvm_tirx.tirx.compile(ModB, target=target)

    pathA = temp.relpath("libA.tar")
    pathB = temp.relpath("libB.tar")
    pathAll = temp.relpath("libAll.a")

    path_dso = temp.relpath("mylib.so")
    libA.export_library(pathA, fcompile=tar.tar)
    libB.export_library(pathB, fcompile=tar.tar)
    cc.create_staticlib(pathAll, [pathA, pathB])
    # package two static libs together
    cc.create_shared(path_dso, ["-Wl,--whole-archive", pathAll, "-Wl,--no-whole-archive"])

    def popen_check():
        def run_and_check():
            # Load dll, will trigger system library registration
            ctypes.CDLL(path_dso)
            # Load the system wide library
            dev = tvm_tirx.cuda()
            a_np = np.random.uniform(size=12).astype("float32")
            a_nd = tvm_tirx.runtime.tensor(a_np, dev)
            b_nd = tvm_tirx.runtime.tensor(a_np, dev)
            syslibA = tvm_ffi.system_lib("modA_")
            syslibB = tvm_ffi.system_lib("modB_")
            # reload same lib twice
            syslibA = tvm_ffi.system_lib("modA_")
            syslibA["my_inplace_update"](a_nd)
            syslibB["my_inplace_update"](b_nd)
            np.testing.assert_equal(a_nd.numpy(), a_np + 1)
            np.testing.assert_equal(b_nd.numpy(), a_np + 2)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    # system lib should be loaded in different process
    worker = popen_pool.PopenWorker()
    worker.send(popen_check)
    worker.recv()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_export_load_with_fallback(monkeypatch, tmp_path, *, backend_config):
    """Force the codegen wrapper into the fallback branch, then export+load+run."""
    n = 1024

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            with T.launch_thread("blockIdx.x", n // 32) as i_0:
                with T.launch_thread("threadIdx.x", 32) as i_1:
                    B[i_0 * 32 + i_1] = A[i_0 * 32 + i_1] + 1.0

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    host_lib = tvm_tirx.tirx.compile(
        Module, target={"kind": "cuda", "backend_config": backend_config}
    )
    monkeypatch.delenv("TVM_COMPILE_FORCE_FALLBACK")

    lib_path = str(tmp_path / "lib.so")
    host_lib.export_library(lib_path)
    reloaded = tvm_tirx.runtime.load_module(lib_path)

    a_np = np.random.uniform(size=(n,)).astype("float32")
    b_np = np.zeros((n,), dtype="float32")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(b_np, dev)
        reloaded["main"](a, b)
        np.testing.assert_allclose(b.numpy(), a_np + 1.0, rtol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
