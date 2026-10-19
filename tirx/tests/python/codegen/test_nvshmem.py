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
"""NVSHMEM code generation and device linking, without kernel execution."""

import json
import re
import shutil
from pathlib import Path

import pytest
import tvm_ffi

import tvm_tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import nvcc
from tvm_tirx.testing import env


@pytest.fixture
def cuda_source(monkeypatch):
    # Source generation needs neither NVSHMEM headers nor a CUDA device.
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")

    def generate(function):
        target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"})
        executable = tvm_tirx.tirx.compile(function, target=target)
        source = executable.mod.imports[0].inspect_source("cuda")
        assert "#include <nvshmem.h>" in source
        assert "#include <nvshmemx.h>" in source
        return source

    return generate


def test_thread_info(cuda_source):
    @T.function
    def main(res: T.Tensor((2,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        res[0] = T.nvshmem.my_pe()
        res[1] = T.nvshmem.n_pes()

    source = cuda_source(main)
    assert "nvshmem_my_pe()" in source
    assert "nvshmem_n_pes()" in source


@pytest.mark.parametrize("scope,nelems", [("thread", 4), ("warp", 128), ("block", 256)])
@pytest.mark.parametrize("op_name", ["getmem_nbi", "putmem_nbi"])
def test_transfer(cuda_source, scope, nelems, op_name):
    is_block, is_thread = scope == "block", scope == "thread"
    op_func = getattr(T.nvshmem, op_name)
    if scope != "thread":
        op_func = getattr(op_func, scope)

    @T.function
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        tid = T.cuda.thread_idx("x")
        warp_id = T.cuda.warp_id()
        my_pe = T.nvshmem.my_pe()
        n_pes = T.nvshmem.n_pes()
        offset = T.if_then_else(is_block, 0, T.if_then_else(is_thread, tid, warp_id * 32))
        op_func(
            dst=B.ptr_to([offset]),
            src=A.ptr_to([offset]),
            nelems=nelems,
            pe=(my_pe + 1) % n_pes,
        )
        T.nvshmem.quiet()

    source = cuda_source(main)
    call = f"nvshmem_{op_name}" if scope == "thread" else f"nvshmemx_{op_name}_{scope}"
    assert f"{call}(" in source
    assert "nvshmem_quiet()" in source
    suffix = "" if scope == "thread" else f"_{scope}"
    assert re.search(rf"tvm_builtin_nvshmem_{op_name}{suffix}\(.+, {nelems},", source)


@pytest.mark.parametrize("sig_op", ["set", "add"])
def test_signal_op(cuda_source, sig_op):
    cmp_value = 1 if sig_op == "set" else 2

    @T.function
    def main(res: T.Tensor((1,), "uint64")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        my_pe = T.nvshmem.my_pe()
        n_pes = T.nvshmem.n_pes()
        if T.constexpr(sig_op == "add"):
            res[0] = 1
        T.nvshmem.barrier_all()
        T.nvshmem.signal_op(
            sig_addr=res.ptr_to([0]), signal=1, sig_op=sig_op, pe=(my_pe + 1) % n_pes
        )
        T.nvshmem.wait_until(ivar=res.ptr_to([0]), cmp="eq", cmp_value=cmp_value)

    source = cuda_source(main)
    assert "nvshmemx_signal_op(" in source
    assert "nvshmem_uint64_wait_until(" in source
    assert "nvshmem_barrier_all()" in source
    sig_op_value = 0 if sig_op == "set" else 1
    assert re.search(rf"tvm_builtin_nvshmem_signal_op\(.+, 1, {sig_op_value},", source)
    assert re.search(rf"tvm_builtin_nvshmem_uint64_wait_until\(.+, 0, {cmp_value}\);", source)


@pytest.mark.parametrize(
    "scope,nelems,cmp_value", [("thread", 4, 32), ("warp", 128, 2), ("block", 256, 1)]
)
def test_put_signal(cuda_source, scope, nelems, cmp_value):
    is_block, is_thread = scope == "block", scope == "thread"
    op_func = T.nvshmem.putmem_signal_nbi
    if scope != "thread":
        op_func = getattr(op_func, scope)

    @T.function
    def main(
        A: T.Tensor((64,), "float32"),
        B: T.Tensor((64,), "float32"),
        signal_array: T.Tensor((1,), "uint64"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        tid = T.cuda.thread_idx("x")
        warp_id = T.cuda.warp_id()
        my_pe = T.nvshmem.my_pe()
        n_pes = T.nvshmem.n_pes()
        offset = T.if_then_else(is_block, 0, T.if_then_else(is_thread, tid, warp_id * 32))
        op_func(
            dst=B.ptr_to([offset]),
            src=A.ptr_to([offset]),
            nelems=nelems,
            sig_addr=signal_array.ptr_to([0]),
            signal=1,
            sig_op="set",
            pe=(my_pe + 1) % n_pes,
        )
        T.nvshmem.wait_until(ivar=signal_array.ptr_to([0]), cmp="eq", cmp_value=cmp_value)

    source = cuda_source(main)
    call = (
        "nvshmem_putmem_signal_nbi" if scope == "thread" else f"nvshmemx_putmem_signal_nbi_{scope}"
    )
    assert f"{call}(" in source
    assert "nvshmem_uint64_wait_until(" in source
    assert re.search(rf"tvm_builtin_nvshmem_uint64_wait_until\(.+, 0, {cmp_value}\);", source)


def test_fence_barrier(cuda_source):
    @T.function
    def main(
        A: T.Tensor((64,), "float32"),
        B: T.Tensor((64,), "float32"),
        res: T.Tensor((1,), "uint64"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        tid = T.cuda.thread_idx("x")
        my_pe = T.nvshmem.my_pe()
        n_pes = T.nvshmem.n_pes()
        dst_pe = (my_pe + 1) % n_pes
        T.nvshmem.barrier_all()
        T.nvshmem.putmem_nbi.block(dst=B.ptr_to([0]), src=A.ptr_to([0]), nelems=256, pe=dst_pe)
        T.nvshmem.fence()
        if tid == 0:
            T.nvshmem.signal_op(sig_addr=res.ptr_to([0]), signal=1, sig_op="set", pe=dst_pe)
        T.nvshmem.wait_until(ivar=res.ptr_to([0]), cmp="eq", cmp_value=1)

    source = cuda_source(main)
    assert "nvshmem_barrier_all()" in source
    assert "nvshmemx_putmem_nbi_block(" in source
    assert "nvshmem_fence()" in source
    assert "nvshmemx_signal_op(" in source
    assert "nvshmem_uint64_wait_until(" in source
    calls = ["barrier_all", "putmem_nbi_block", "fence", "signal_op", "uint64_wait_until"]
    positions = [source.rfind(f"tvm_builtin_nvshmem_{call}(") for call in calls]
    assert positions == sorted(positions)


NVSHMEM_QUERY_KERNEL_SOURCE = """
#include <nvshmem.h>

extern "C" __global__ void nvshmem_query_kernel(int* my_pe_out, int* n_pes_out) {
    my_pe_out[0] = nvshmem_my_pe();
    n_pes_out[0] = nvshmem_n_pes();
}
"""


def _require_nvshmem_compiler(compiler):
    if tvm_ffi.get_global_func("ffi.Module.create.cuda", allow_missing=True) is None:
        pytest.skip("CUDA runtime module support is not enabled")
    try:
        nvcc.find_cuda_path()
        _, library_path = nvcc.find_nvshmem_paths()
    except RuntimeError as err:
        pytest.skip(str(err))
    if not (Path(library_path) / "libnvshmem_device.a").is_file():
        pytest.skip("NVSHMEM device library is not installed")
    if compiler == "nvcc":
        if shutil.which("nvcc") is None or shutil.which("nvlink") is None:
            pytest.skip("NVCC and nvlink must be available on PATH")
    else:
        pytest.importorskip("cuda.bindings.nvrtc")
        pytest.importorskip("cuda.bindings.driver")
        if not env.has_cuda():
            pytest.skip("NVRTC NVSHMEM linking requires a CUDA device")


@pytest.mark.parametrize("compiler", ["nvcc", "nvrtc"])
def test_kernel_compile(compiler):
    _require_nvshmem_compiler(compiler)
    # NVRTC links against the current device context; NVCC can cross-compile.
    arch = env.cuda_arch() if compiler == "nvrtc" else "sm_80"
    backend_config = {"cuda": {"compiler": compiler, "arch": arch, "ptxas": []}}

    @I.ir_module
    class NvshmemQueryModule:
        @T.function
        def query_pe(my_pe_out: T.Tensor((1,), "int32"), n_pes_out: T.Tensor((1,), "int32")):
            T.call_kernel(
                NVSHMEM_QUERY_KERNEL_SOURCE,
                ((1,), (1,)),
                my_pe_out.data,
                n_pes_out.data,
                kernel_name="nvshmem_query_kernel",
                backend_config=backend_config,
            )

    external_mods = NvshmemQueryModule.get_attr("external_mods")
    assert len(external_mods) == 1
    assert external_mods[0].implements_function("nvshmem_query_kernel")
    config = json.loads(external_mods[0].inspect_source("backend_config"))["cuda"]
    assert config["compiler"] == compiler
    assert config["target_format"] == "cubin"

    @T.function
    def expected(my_pe_out: T.Tensor((1,), "int32"), n_pes_out: T.Tensor((1,), "int32")):
        T.func_attr({"global_symbol": "query_pe"})
        T.call_packed("nvshmem_query_kernel", my_pe_out.data, n_pes_out.data, 1, 1)

    tvm_tirx.ir.assert_structural_equal(NvshmemQueryModule["query_pe"], expected)
