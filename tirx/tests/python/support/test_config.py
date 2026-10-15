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
"""Installed discovery stays within its selected prefix and does not load TVM."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "python/tvm_tirx/config.py"
_SPEC = importlib.util.spec_from_file_location("tirx_config", _CONFIG_PATH)
config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config)


def _installation(prefix, libdir="lib", windows=False):
    header = prefix / "include/tvm/runtime/base.h"
    header.parent.mkdir(parents=True)
    header.touch()
    library_dir = prefix / libdir
    cmake_dir = library_dir / "cmake/tvm_tirx"
    cmake_dir.mkdir(parents=True)
    for name in ("tvm_tirx", "tvm_runtime"):
        filenames = (f"{name}.dll", f"{name}.lib") if windows else (f"lib{name}.so",)
        for filename in filenames:
            (library_dir / filename).touch()
    return library_dir


def test_exact_library_directory_and_relocation(tmp_path):
    prefix = tmp_path / "native install"
    libdir = _installation(prefix, "lib/x86_64-linux-gnu")
    (libdir / "libtvm_runtime_static.a").touch()
    (libdir / "libtvm_runtime_cuda.so").touch()
    (libdir / "libtvm_ffi.so").touch()
    result = config.configuration(library_dir=libdir, platform="Linux")
    assert result["include_dir"] == str(prefix / "include")
    assert set(result["libraries"]) == {"tvm_tirx", "tvm_runtime"}
    assert set(result) == {"version", "include_dir", "library_dir", "cmake_dir", "libraries"}
    moved = tmp_path / "relocated"
    shutil.copytree(prefix, moved)
    result = config.configuration(library_dir=moved / "lib/x86_64-linux-gnu", platform="Linux")
    assert result["include_dir"] == str(moved / "include")
    assert all(str(moved) in library["location"] for library in result["libraries"].values())


def test_prefix_selects_installed_lib64(tmp_path):
    prefix = tmp_path / "install"
    libdir = _installation(prefix, "lib64")
    (prefix / "lib").mkdir()
    result = config.configuration(prefix=prefix, platform="Linux")
    assert result["library_dir"] == str(libdir)
    result = subprocess.run(
        [sys.executable, str(_CONFIG_PATH), "--prefix", str(prefix), "--cmakedir"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == str(libdir / "cmake/tvm_tirx")
    with pytest.raises(RuntimeError, match="Cannot find TIRx library"):
        config.configuration(library_dir=prefix / "lib", platform="Linux")


@pytest.mark.parametrize("name", ["tvm_tirx", "tvm_runtime"])
def test_windows_library_pairing(tmp_path, name):
    libdir = _installation(tmp_path / "windows", windows=True)
    (libdir / "tvm_runtime_static.lib").touch()
    (libdir / "tvm_runtime_cuda.dll").touch()
    (libdir / "tvm_runtime_cuda.lib").touch()
    result = config.configuration(library_dir=libdir, platform="Windows")
    assert set(result["libraries"]) == {"tvm_tirx", "tvm_runtime"}
    assert result["libraries"][name] == {
        "location": str(libdir / f"{name}.dll"),
        "implib": str(libdir / f"{name}.lib"),
    }
    (libdir / f"{name}.lib").unlink()
    with pytest.raises(RuntimeError, match="Missing import library"):
        config.configuration(library_dir=libdir, platform="Windows")
    (libdir / f"lib{name}.dll.a").touch()
    result = config.configuration(library_dir=libdir, platform="Windows")
    assert result["libraries"][name]["implib"] == str(libdir / f"lib{name}.dll.a")


def test_explicit_prefix_does_not_fall_back(tmp_path, monkeypatch):
    good = tmp_path / "good"
    _installation(good)
    monkeypatch.setenv("PYTHONPATH", str(good))
    monkeypatch.setenv("TVM_TIRX_LIBRARY_PATH", str(good / "lib"))
    with pytest.raises(RuntimeError, match="Cannot find TIRx public headers"):
        config.configuration(prefix=tmp_path / "missing", platform="Linux")


def test_module_cli_without_native_dependencies(tmp_path):
    prefix = tmp_path / "install"
    _installation(prefix)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_CONFIG_PATH.parents[1])
    result = subprocess.run(
        [sys.executable, "-S", "-m", "tvm_tirx.config", "--prefix", str(prefix), "--json"],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["include_dir"] == str(prefix / "include")


@pytest.mark.skipif(shutil.which("cmake") is None, reason="CMake is not installed")
def test_cmake_targets_without_ffi(tmp_path):
    libdir = _installation(tmp_path / "install")
    cmake_dir = libdir / "cmake/tvm_tirx"
    shutil.copy2(_CONFIG_PATH, cmake_dir / "config.py")
    shutil.copy2(
        _CONFIG_PATH.parents[2] / "cmake/tvm_tirx-config.cmake",
        cmake_dir / "tvm_tirx-config.cmake",
    )
    source = tmp_path / "consumer"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.26)
project(config_contract LANGUAGES NONE)
find_package(tvm_tirx CONFIG REQUIRED)
get_property(imported DIRECTORY PROPERTY IMPORTED_TARGETS)
list(FILTER imported EXCLUDE REGEX "^Python::")
list(SORT imported)
if(NOT "${imported}" STREQUAL "tvm_runtime::shared;tvm_tirx::header;tvm_tirx::shared")
  message(FATAL_ERROR "Unexpected imported targets: ${imported}")
endif()
foreach(target tvm_tirx::header tvm_tirx::shared tvm_runtime::shared)
  get_target_property(links ${target} INTERFACE_LINK_LIBRARIES)
  if(links MATCHES "ffi|OpenMP|Threads")
    message(FATAL_ERROR "Unexpected implicit dependency: ${links}")
  endif()
endforeach()
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(tmp_path / "build"),
            f"-Dtvm_tirx_DIR={cmake_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_DISABLE_FIND_PACKAGE_tvm_ffi=TRUE",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
