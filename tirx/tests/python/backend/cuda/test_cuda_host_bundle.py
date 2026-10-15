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
import pytest

import tvm_tirx
import tvm_tirx.support.nvcc
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@pytest.fixture(params=["nvcc", "nvrtc"])
def backend_config(request):
    if request.param == "nvrtc":
        pytest.importorskip("cuda.bindings.nvrtc")
    return {"cuda": {"compiler": request.param}}


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_host_bundle(tmp_path, *, backend_config):
    from shutil import which

    import tvm_ffi.cpp

    from tvm_tirx.backend.cuda import export_cuda_host

    if which("nvcc") is None:
        pytest.skip("CUDA-host compilation requires NVCC")

    @T.function
    def add_one(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.launch_thread("threadIdx.x", 32) as tx:
            B[tx] = A[tx] + T.float32(1)

    target = tvm_tirx.target.Target("cuda", host="cuda_host")
    built = tvm_tirx.tirx.compile(
        add_one,
        target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config},
    ).mod
    source = export_cuda_host(built)
    assert "tvm_cuda_binary_0" in source
    assert "__global__" not in source
    assert "cuLaunchKernelEx" in source
    library = tvm_ffi.cpp.build_inline(
        name="cuda_host_add_one",
        cuda_sources=source,
        extra_ldflags=["-lcuda"],
        extra_cuda_cflags=[f"-arch={target.arch}"],
        build_directory=str(tmp_path),
        backend="cuda",
    )
    loaded = tvm_ffi.load_module(library)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        values = np.arange(32, dtype="float32")
        a = tvm_tirx.runtime.tensor(values, dev)
        b = tvm_tirx.runtime.empty((32,), "float32", dev)
        loaded["add_one"](a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), values + 1)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_host_bundle_bf16_cluster(tmp_path, *, backend_config):
    """bfloat16 pointers and cluster launch attributes in a CUDA-host bundle."""
    from shutil import which

    import tvm_ffi.cpp

    from tvm_tirx.backend.cuda import export_cuda_host

    ml_dtypes = pytest.importorskip("ml_dtypes")
    if which("nvcc") is None:
        pytest.skip("CUDA-host compilation requires NVCC")
    if int(tvm_tirx.cuda(0).compute_version.split(".")[0]) < 9:
        pytest.skip("thread block clusters require SM90 or newer")

    @T.function
    def main(
        A: T.Tensor((128,), "bfloat16"),
        B: T.Tensor((128,), "bfloat16"),
        R: T.Tensor((4,), "int32"),
    ):
        T.device_entry()
        with T.launch_thread("clusterCtaIdx.x", 2) as cx:
            with T.launch_thread("blockIdx.x", 4) as bx:
                with T.launch_thread("threadIdx.x", 32) as tx:
                    B[bx * 32 + tx] = A[bx * 32 + tx] + T.bfloat16(1)
                    if tx == 0:
                        R[bx] = cx

    arch = env.cuda_arch()
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch}, host="cuda_host")
    with target:
        built = tvm_tirx.tirx.compile(
            tvm_tirx.IRModule({"main": main}),
            target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config},
            pipeline="tirx",
        ).mod
    source = export_cuda_host(built)
    assert "cudaLaunchAttributeClusterDimension" in source
    library = tvm_ffi.cpp.build_inline(
        name="cuda_host_bf16_cluster",
        cuda_sources=source,
        extra_ldflags=["-lcuda"],
        extra_cuda_cflags=[f"-arch={arch}"],
        build_directory=str(tmp_path),
        backend="cuda",
    )
    loaded = tvm_ffi.load_module(library)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        values = np.arange(128).astype(ml_dtypes.bfloat16)
        a = tvm_tirx.runtime.tensor(values, dev)
        b = tvm_tirx.runtime.empty((128,), "bfloat16", dev)
        r = tvm_tirx.runtime.empty((4,), "int32", dev)
        loaded["main"](a, b, r)
        tvm_tirx.testing.assert_allclose(b.numpy().astype("float32"), np.arange(128) + 1)
        # Each CTA reports its rank within a two-CTA cluster.
        np.testing.assert_array_equal(r.numpy(), [0, 1, 0, 1])

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_host_bundle_programmatic_dependent_launch(tmp_path, *, backend_config):
    """A programmatic-dependent-launch flag becomes a launch attribute."""
    from shutil import which

    import tvm_ffi.cpp

    from tvm_tirx.backend.cuda import export_cuda_host

    if which("nvcc") is None:
        pytest.skip("CUDA-host compilation requires NVCC")
    if int(tvm_tirx.cuda(0).compute_version.split(".")[0]) < 9:
        pytest.skip("programmatic dependent launch requires SM90 or newer")

    @T.function
    def add_one(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        T.func_attr({"tirx.kernel_launch_params": ["tirx.use_programtic_dependent_launch"]})
        with T.launch_thread("threadIdx.x", 32) as tx:
            B[tx] = A[tx] + T.float32(1)

    target = tvm_tirx.target.Target("cuda", host="cuda_host")
    source = export_cuda_host(
        tvm_tirx.tirx.compile(
            add_one,
            target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config},
        ).mod
    )
    assert "cudaLaunchAttributeProgrammaticStreamSerialization" in source
    library = tvm_ffi.cpp.build_inline(
        name="cuda_host_programmatic_dependent_launch",
        cuda_sources=source,
        extra_ldflags=["-lcuda"],
        extra_cuda_cflags=[f"-arch={target.arch}"],
        build_directory=str(tmp_path),
        backend="cuda",
    )
    loaded = tvm_ffi.load_module(library)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        values = np.arange(32, dtype="float32")
        a = tvm_tirx.runtime.tensor(values, dev)
        b = tvm_tirx.runtime.empty((32,), "float32", dev)
        loaded["add_one"](a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), values + 1)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_host_bundle_tensor_map_parameter(tmp_path, monkeypatch, *, backend_config):
    """A tensor-map parameter handle compiles in the C++ host wrapper."""
    from shutil import which

    import tvm_ffi.cpp

    from tvm_tirx.backend.cuda import export_cuda_host

    if which("nvcc") is None:
        pytest.skip("CUDA-host compilation requires NVCC")

    @T.function
    def main(A_map: T.Ptr(T.TensorMap)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.evaluate(T.address_of(A_map))

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"}, host="cuda_host")
    with target:
        built = tvm_tirx.tirx.compile(
            tvm_tirx.IRModule({"main": main}),
            target={**tvm_tirx.target.Target(target).export(), "backend_config": backend_config},
            pipeline="tirx",
        ).mod
    source = export_cuda_host(built)
    assert "(CUtensorMap*)" in source
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "9.0a")
    tvm_ffi.cpp.build_inline(
        name="cuda_host_tensor_map_parameter",
        cuda_sources=source,
        extra_ldflags=["-lcuda"],
        build_directory=str(tmp_path),
        backend="cuda",
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
