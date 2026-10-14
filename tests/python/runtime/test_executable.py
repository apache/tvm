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
"""Executable function lookup, JIT caching, and export delegation."""

import pytest

import tvm
import tvm.testing
from tvm.runtime import Executable


def test_executable_init(source_module):
    executable = Executable(source_module)
    assert executable.mod is source_module
    assert executable._jitted_mod is None


def test_executable_getitem(source_module, native_compile_options):
    executable = Executable(source_module)
    executable.jit(options=native_compile_options)
    assert executable["answer"]() == 42


def test_executable_jit_already_runnable(native_library):
    module = tvm.runtime.load_module(str(native_library))
    executable = Executable(module)
    assert executable.jit() is module
    assert executable.jit(force_recompile=True) is module
    assert executable["answer"]() == 42


def test_executable_jit_force_recompile(source_module, native_compile_options):
    executable = Executable(source_module)
    first = executable.jit(options=native_compile_options)
    assert executable.jit() is first
    rebuilt = executable.jit(force_recompile=True, options=native_compile_options)
    assert rebuilt is not first
    assert first["answer"]() == rebuilt["answer"]() == 42


@pytest.mark.parametrize("use_workspace", [False, True])
def test_executable_export_library(source_module, native_compile_options, tmp_path, use_workspace):
    kwargs = {"options": native_compile_options}
    if use_workspace:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        kwargs["workspace_dir"] = str(workspace)
    path = tmp_path / "module.so"
    Executable(source_module).export_library(str(path), **kwargs)
    assert path.is_file()
    assert tvm.runtime.load_module(str(path))["answer"]() == 42


if __name__ == "__main__":
    tvm.testing.main()
