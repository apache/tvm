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
    (cmake_dir / "tvm_tirx-build.json").write_text(
        json.dumps({"version": "0.1.0a0", "openmp": "OFF"}), encoding="utf-8"
    )
    filenames = ("tvm_runtime.dll", "tvm_runtime.lib") if windows else ("libtvm_runtime.so",)
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
    assert set(result["libraries"]) == {"tvm_runtime", "tvm_runtime_static", "tvm_runtime_cuda"}
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
    with pytest.raises(RuntimeError, match="Cannot read TIRx build metadata"):
        config.configuration(library_dir=prefix / "lib", platform="Linux")


def test_windows_library_pairing(tmp_path):
    libdir = _installation(tmp_path / "windows", windows=True)
    (libdir / "tvm_runtime_static.lib").touch()
    result = config.configuration(library_dir=libdir, platform="Windows")
    assert result["libraries"]["tvm_runtime"] == {
        "location": str(libdir / "tvm_runtime.dll"),
        "implib": str(libdir / "tvm_runtime.lib"),
    }
    assert result["libraries"]["tvm_runtime_static"]["implib"] == ""
    (libdir / "tvm_runtime.lib").unlink()
    with pytest.raises(RuntimeError, match="Missing import library"):
        config.configuration(library_dir=libdir, platform="Windows")


def test_windows_runtime_only_backend(tmp_path):
    libdir = _installation(tmp_path / "windows", windows=True)
    (libdir / "tvm_runtime_cuda.dll").touch()
    result = config.configuration(library_dir=libdir, platform="Windows")
    assert set(result["libraries"]) == {"tvm_runtime"}
    assert (libdir / "tvm_runtime_cuda.dll").is_file()
    (libdir / "tvm_tirx.dll").touch()
    with pytest.raises(RuntimeError, match="Missing import library.*tvm_tirx.dll"):
        config.configuration(library_dir=libdir, platform="Windows")
    (libdir / "tvm_tirx.lib").touch()
    (libdir / "tvm_runtime_cuda.lib").touch()
    result = config.configuration(library_dir=libdir, platform="Windows")
    assert set(result["libraries"]) == {"tvm_runtime", "tvm_tirx", "tvm_runtime_cuda"}


def test_runtime_interface_metadata(tmp_path):
    libdir = _installation(tmp_path / "install")
    result = config.configuration(library_dir=libdir, platform="Linux")
    assert result["custom_logging"] is False
    assert result["use_threads"] is True
    metadata = {"version": "0.1.0a0", "openmp": "OFF", "custom_logging": True, "use_threads": False}
    (libdir / "cmake/tvm_tirx/tvm_tirx-build.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    result = config.configuration(library_dir=libdir, platform="Linux")
    assert result["custom_logging"] is True
    assert result["use_threads"] is False


@pytest.mark.parametrize(
    "metadata",
    [
        {"version": 1, "openmp": "OFF"},
        {"version": "1"},
        {"version": "1", "openmp": "OFF", "custom_logging": "ON"},
        {"version": "1", "openmp": "OFF", "use_threads": 1},
    ],
)
def test_invalid_metadata(tmp_path, metadata):
    libdir = _installation(tmp_path / "install")
    (libdir / "cmake/tvm_tirx/tvm_tirx-build.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Invalid TIRx"):
        config.configuration(library_dir=libdir, platform="Linux")


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
