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
"""Executable export, JIT, and static import coverage."""

import shlex
import subprocess
import sys

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.codegen import Executable
from tvm_tirx.script import tirx as T
from tvm_tirx.support import cc
from tvm_tirx.testing import env


def test_import_static_library(source_module, native_compile_options, tmp_path):
    source = tmp_path / "other.cc"
    source.write_text(
        "#include <tvm/ffi/function.h>\n"
        "int other() { return 17; }\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(other, other);\n"
    )
    obj = tmp_path / "other.o"
    cc.create_shared(str(obj), [str(source)], options=["-c", *native_compile_options])
    imported = tvm_tirx.codegen.load_static_library(str(obj), ["other"])
    assert imported.implements_function("other")
    assert imported.is_compilation_exportable()
    source_module.import_module(imported)
    path = tmp_path / "combined.so"
    Executable(source_module).export_library(str(path), options=native_compile_options)
    loaded = tvm_tirx.runtime.load_module(str(path))
    assert loaded.kind == "library"
    assert len(loaded.imports) == 0
    for name, expected in (("answer", 42), ("other", 17)):
        assert loaded.implements_function(name)
        assert loaded.get_function(name)() == expected
    assert not loaded.is_compilation_exportable()


def test_executable_init(source_module):
    executable = Executable(source_module)
    assert executable.mod is source_module
    assert executable._jitted_mod is None


def test_executable_getitem(source_module, native_compile_options):
    executable = Executable(source_module)
    executable.jit(options=native_compile_options)
    assert executable["answer"]() == 42


def test_executable_jit_already_runnable(native_library):
    module = tvm_tirx.runtime.load_module(str(native_library))
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
    assert tvm_tirx.runtime.load_module(str(path))["answer"]() == 42


@pytest.mark.gpu
def test_cuda_aot(tmp_path):
    if sys.platform != "linux" or not env.has_cuda() or not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("CUDA AOT requires Linux CUDA and an LLVM host compiler")
    pytest.importorskip("torch")

    @T.function
    def add_one(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 16) as i:
                B[i] = A[i] + 1

    module = tvm_tirx.tirx.compile(add_one, target=tvm_tirx.target.Target("cuda", host="llvm"))
    path = tmp_path / "model.so"
    options = shlex.split(
        subprocess.check_output(
            [sys.executable, "-m", "tvm_tirx.config", "--ldflags", "--libs", "runtime", "cuda"],
            text=True,
        )
    )
    library_dir = subprocess.check_output(
        [sys.executable, "-m", "tvm_tirx.config", "--libdir"], text=True
    ).strip()
    module.export_library(path, options=[*options, "-Wl,-rpath," + library_dir])
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
import tvm_ffi
module = tvm_ffi.load_module(sys.argv[1])
a = torch.arange(16, dtype=torch.int32, device="cuda")
b = torch.empty_like(a)
module["add_one"](a, b)
torch.testing.assert_close(b, a + 1)
assert "tvm" not in sys.modules and "tvm_tirx" not in sys.modules
""",
            str(path),
        ],
        check=True,
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
