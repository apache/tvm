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
"""Configure standalone defaults without requiring installed CUDA or LLVM."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2]


@pytest.fixture
def configure(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    if not (SOURCE / "3rdparty/tvm-ffi/CMakeLists.txt").is_file():
        pytest.skip("Initialize tirx/3rdparty/tvm-ffi before configuring")

    def run(*options, source=SOURCE):
        env = os.environ.copy()
        env.pop("CMAKE_ARGS", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            [
                cmake,
                "-S",
                str(source),
                "-B",
                str(tmp_path / "build"),
                "-DTVM_TIRX_USE_Z3=OFF",
                "-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON",
                *options,
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return run


def test_auto_without_cuda(configure):
    result = configure("-DTVM_TIRX_USE_LLVM=OFF")
    assert result.returncode == 0, result.stdout
    assert "CUDA AUTO: usable toolkit not found; building without CUDA" in result.stdout
    assert "Build cuda device runtime" not in result.stdout


def test_cuda_off(configure):
    result = configure("-DTVM_TIRX_USE_CUDA=OFF", "-DTVM_TIRX_USE_LLVM=OFF")
    assert result.returncode == 0, result.stdout
    assert "CUDA AUTO:" not in result.stdout
    assert "Build cuda device runtime" not in result.stdout


@pytest.mark.parametrize("cuda", ["ON", "/missing-cuda-sdk"])
def test_explicit_cuda_is_required(configure, cuda):
    result = configure(f"-DTVM_TIRX_USE_CUDA={cuda}", "-DTVM_TIRX_USE_LLVM=OFF")
    assert result.returncode != 0
    assert "Cannot find CUDA" in result.stdout


def test_default_llvm_missing_guidance(configure):
    result = configure("-DCMAKE_DISABLE_FIND_PACKAGE_LLVM=ON")
    assert result.returncode != 0
    for hint in ["LLVM development files", "LLVM 15 or newer", "LLVM_DIR", "llvm-config"]:
        assert hint in result.stdout
    assert "TVM_TIRX_USE_LLVM=OFF" in result.stdout


def test_llvm_off_skips_discovery(configure):
    result = configure("-DCMAKE_DISABLE_FIND_PACKAGE_LLVM=ON", "-DTVM_TIRX_USE_LLVM=OFF")
    assert result.returncode == 0, result.stdout
    assert "LLVM development files" not in result.stdout


@pytest.mark.parametrize(
    "found, runtime, driver",
    [("TRUE", "cudart", ""), ("TRUE", "", "cuda-stub"), ("FALSE", "cudart", "cuda-stub")],
)
def test_auto_with_incomplete_toolkit(configure, tmp_path, found, runtime, driver):
    # Simulate a partial toolkit independently of the build host's installation.
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "FindCUDA.cmake").write_text(
        f'set(CUDA_FOUND {found})\nset(CUDA_CUDA_LIBRARY "{driver}")\n'
        f'set(CUDA_CUDART_LIBRARY "{runtime}")\n'
    )
    result = configure(
        "-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=OFF",
        f"-DCMAKE_MODULE_PATH={modules}",
        "-DTVM_TIRX_USE_LLVM=OFF",
    )
    assert result.returncode == 0, result.stdout
    assert "CUDA AUTO: usable toolkit not found; building without CUDA" in result.stdout
    assert "Build cuda device runtime" not in result.stdout


@pytest.mark.parametrize("cuda", ["AUTO", "ON", "path"])
def test_cuda_compiler_overrides(configure, tmp_path, cuda):
    host = shutil.which("g++")
    nvcc = shutil.which("nvcc")
    if not host or not nvcc:
        pytest.skip("Requires a CUDA toolkit and GNU host compiler")
    if cuda == "path":
        cuda = str(Path(nvcc).resolve().parents[1])
    result = configure(
        "-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=OFF",
        f"-DTVM_TIRX_USE_CUDA={cuda}",
        "-DTVM_TIRX_USE_LLVM=OFF",
        f"-DCMAKE_CUDA_HOST_COMPILER={host}",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
    )
    assert result.returncode == 0, result.stdout
    if "building without CUDA" in result.stdout:
        pytest.skip("Requires a complete CUDA toolkit")
    assert "Build cuda device runtime" in result.stdout
    cache = (tmp_path / "build/CMakeCache.txt").read_text()
    compiler_files = list((tmp_path / "build/CMakeFiles").glob("*/CMakeCUDACompiler.cmake"))
    assert compiler_files
    assert f'set(CMAKE_CUDA_HOST_COMPILER "{host}")' in compiler_files[0].read_text()
    assert re.search(r"^CMAKE_CUDA_ARCHITECTURES:[^=]*=75$", cache, re.M)


def test_explicit_llvm_config(configure):
    llvm_config = shutil.which("llvm-config")
    if llvm_config is None:
        pytest.skip("Requires LLVM development files")
    version = subprocess.check_output([llvm_config, "--version"], text=True)
    if int(version.split(".", 1)[0]) < 15:
        pytest.skip("Requires LLVM 15 or newer")
    result = configure(f"-DTVM_TIRX_USE_LLVM={llvm_config}")
    assert result.returncode == 0, result.stdout
    assert f"Use llvm-config={llvm_config}" in result.stdout
    assert "Build with LLVM" in result.stdout


@pytest.mark.parametrize("selection", [None, "AUTO", "OFF"])
def test_embedded_cuda_without_toolkit(configure, tmp_path, selection):
    parent = tmp_path / "parent"
    parent.mkdir()
    option = f"set(TVM_TIRX_USE_CUDA {selection})\n" if selection else ""
    (parent / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\n"
        "project(embedding LANGUAGES C CXX)\n"
        + option
        + f'add_subdirectory("{SOURCE.as_posix()}" core)\n'
        + 'if(TVM_TIRX_USE_CUDA)\nmessage(FATAL_ERROR "Parent CUDA must be disabled")\nendif()\n'
        + 'if(TARGET tvm_runtime_cuda)\nmessage(FATAL_ERROR "Unexpected CUDA runtime")\nendif()\n'
    )
    result = configure("-DTVM_TIRX_USE_LLVM=OFF", source=parent)
    assert result.returncode == 0, result.stdout
    if selection != "OFF":
        cache = (tmp_path / "build/CMakeCache.txt").read_text()
        assert "TVM_TIRX_USE_CUDA:STRING=AUTO" in cache
