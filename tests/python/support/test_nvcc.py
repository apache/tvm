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
"""Tests for functions in tvm/python/tvm/support/nvcc.py."""

import os
import sys
import types

import pytest

import tvm.testing
from tvm.support import nvcc


def _make_cuda_root(root, triples):
    """Create a fake CUDA toolkit exposing the given ``targets/<triple>`` dirs."""
    for triple in triples:
        os.makedirs(os.path.join(root, "targets", triple, "include"))
    return str(root)


@pytest.mark.parametrize(
    "machine,available,expected",
    [
        # ARM64 server toolkits ship the headers under "sbsa-linux".
        ("aarch64", ["sbsa-linux", "aarch64-linux"], "sbsa-linux"),
        # Embedded/L4T toolkits only provide "aarch64-linux".
        ("aarch64", ["aarch64-linux"], "aarch64-linux"),
        ("arm64", ["sbsa-linux"], "sbsa-linux"),
        ("x86_64", ["x86_64-linux"], "x86_64-linux"),
    ],
)
def test_find_cuda_target_include(tmp_path, monkeypatch, machine, available, expected):
    """The architecture-specific include dir matches the installed toolkit layout."""
    monkeypatch.setattr(nvcc.platform, "machine", lambda: machine)
    monkeypatch.setattr(nvcc.platform, "system", lambda: "Linux")
    cuda_path = _make_cuda_root(tmp_path, available)
    assert nvcc._find_cuda_target_include(cuda_path) == os.path.join(
        cuda_path, "targets", expected, "include"
    )


def test_find_cuda_target_include_absent(tmp_path, monkeypatch):
    """Toolkits without a ``targets/`` layout report no architecture-specific dir."""
    monkeypatch.setattr(nvcc.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(nvcc.platform, "system", lambda: "Linux")
    assert nvcc._find_cuda_target_include(str(tmp_path)) is None


@pytest.mark.parametrize("disable_fast_math", [False, True])
def test_nvrtc_fast_math_opt_out(tmp_path, monkeypatch, disable_fast_math):
    """NVRTC fast math remains the default, but can be disabled explicitly."""
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    (include_dir / "cuda_runtime.h").touch()
    monkeypatch.setattr(nvcc, "find_cuda_path", lambda: str(tmp_path))

    if disable_fast_math:
        monkeypatch.setenv("TVM_CUDA_NVRTC_NO_FAST_MATH", "1")
    else:
        monkeypatch.delenv("TVM_CUDA_NVRTC_NO_FAST_MATH", raising=False)

    success = 0
    compile_options = None

    fake_nvrtc = types.ModuleType("cuda.bindings.nvrtc")
    fake_nvrtc.nvrtcResult = types.SimpleNamespace(NVRTC_SUCCESS=success)
    fake_nvrtc.nvrtcCreateProgram = lambda *args: (success, object())

    def capture_compile_options(_program, num_options, options):
        nonlocal compile_options
        assert num_options == len(options)
        compile_options = options
        return (success,)

    fake_nvrtc.nvrtcCompileProgram = capture_compile_options
    fake_nvrtc.nvrtcGetPTXSize = lambda _program: (success, 1)
    fake_nvrtc.nvrtcGetPTX = lambda _program, _output: (success,)
    fake_nvrtc.nvrtcDestroyProgram = lambda _program: (success,)

    fake_bindings = types.ModuleType("cuda.bindings")
    fake_bindings.nvrtc = fake_nvrtc
    fake_cuda = types.ModuleType("cuda")
    fake_cuda.bindings = fake_bindings
    monkeypatch.setitem(sys.modules, "cuda", fake_cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", fake_bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.nvrtc", fake_nvrtc)

    nvcc.compile_cuda("", target_format="ptx", arch="compute_80", compiler="nvrtc")

    assert (b"--use_fast_math" in compile_options) is not disable_fast_math


if __name__ == "__main__":
    tvm.testing.main()
