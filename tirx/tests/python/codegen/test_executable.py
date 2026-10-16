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

import os
import shlex
import shutil
import struct
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path

import pytest
import tvm_ffi

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
    if env.has_llvm() and sys.platform in ("linux", "darwin"):
        devices = ["cpu"]
        if sys.platform == "linux" and env.has_cuda():
            devices.append("cuda")
        for device in devices:
            objects_dir = tmp_path / device
            objects_dir.mkdir()
            _check_aot_object_link(device, objects_dir)
            print(f"AOT object-link execution passed: {device}")


def _check_aot_object_link(device, tmp_path):
    if device == "cuda" and (sys.platform != "linux" or not env.has_cuda()):
        pytest.skip("CUDA execution requires a supported Linux GPU")

    @T.function
    def cpu(A: T.Tensor((16,), "int32", align=4), B: T.Tensor((16,), "int32", align=4)):
        # This service is looked up by name, so no undefined C symbol keeps the
        # runtime library alive. CLI retention is required for successful execution.
        for i in range(16):
            B[i] = A[i] + T.call_packed("runtime.NumThreads")

    @T.function
    def cuda(A: T.Tensor((16,), "int32", align=4), B: T.Tensor((16,), "int32", align=4)):
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 16) as i:
                B[i] = A[i] + 1

    target = "llvm" if device == "cpu" else tvm_tirx.target.Target("cuda", host="llvm")
    module = tvm_tirx.tirx.compile(cpu if device == "cpu" else cuda, target=target)
    archive = tmp_path / "model.tar"
    module.export_library(archive)
    objects = []
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            assert member.isfile() and Path(member.name).suffix == ".o"
            data = bundle.extractfile(member).read()
            # Reject a renamed prelinked shared library as well as source files.
            if data[:4] == b"\x7fELF":
                endian = "<" if data[5] == 1 else ">"
                assert struct.unpack_from(endian + "H", data, 16)[0] == 1  # ET_REL
            else:
                assert data[:4] in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")
                endian = "<" if data[0] == 0xCF else ">"
                assert struct.unpack_from(endian + "I", data, 12)[0] == 1  # MH_OBJECT
            path = tmp_path / f"object-{len(objects)}.o"
            path.write_bytes(data)
            objects.append(path)
    assert objects
    if device == "cuda":
        assert len(objects) > 1  # Host code and packed device imports; consume all of them.

    executable = Path(sysconfig.get_path("scripts")) / "tvm-tirx-config"
    command = (
        [str(executable)] if executable.is_file() else [sys.executable, "-m", "tvm_tirx.config"]
    )
    components = ["runtime"] + (["cuda"] if device == "cuda" else [])

    def config(option):
        return shlex.split(subprocess.check_output([*command, option, *components], text=True))

    libraries = config("--libfiles")
    library_dirs = list(dict.fromkeys(str(Path(lib).parent) for lib in libraries))
    output = tmp_path / ("model.dylib" if sys.platform == "darwin" else "model.so")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    assert shutil.which(compiler[0]), "A C++ linker is required"
    subprocess.run(
        [
            *compiler,
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-o",
            str(output),
            *map(str, objects),
            *config("--ldflags"),
            *config("--libs"),
        ],
        check=True,
    )
    child_env = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "TVM_LIBRARY_PATH",
        "TVM_TIRX_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ):
        child_env.pop(name, None)
    # Loader discovery only: no preloading and no substitute library or retention list.
    loader_path = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    child_env[loader_path] = os.pathsep.join(library_dirs)
    consumer = r"""
import importlib.abc
import importlib.util
import sys
class NoCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # Resolve only FFI when the producer uses a PYTHONPATH install.
        # Do not expose sibling compiler packages to this isolated consumer.
        if fullname == 'tvm_ffi':
            return importlib.util.spec_from_file_location(fullname, sys.argv[3])
        if fullname.split('.')[0] == 'torch':
            raise ImportError('Torch is not part of the FFI consumer')
        if fullname.split('.')[0] in ('tvm', 'tvm_tirx'):
            raise AssertionError('Unexpected compiler/framework import: ' + fullname)
sys.meta_path.insert(0, NoCompiler())
import numpy as np
import tvm_ffi
assert tvm_ffi.get_global_func('runtime.NumThreads', allow_missing=True) is None
module = tvm_ffi.load_module(sys.argv[1])
for name in ('codegen.Build', 'runtime.ModuleTimeEvaluator', 'runtime.LocalTimeEvaluator'):
    assert tvm_ffi.get_global_func(name, allow_missing=True) is None, name
assert tvm_ffi.get_global_func('runtime.NumThreads', allow_missing=True), (
    'runtime.NumThreads missing')
a = np.arange(16, dtype='int32')
b = np.zeros_like(a)
a_ffi, b_ffi = tvm_ffi.from_dlpack(a), tvm_ffi.from_dlpack(b)
if sys.argv[2] == 'cuda':
    allocate = tvm_ffi.get_global_func('runtime.TVMTensorAllocWithScope')
    copy = tvm_ffi.get_global_func('runtime.TVMTensorCopyFromTo')
    def empty():
        return allocate(tvm_ffi.Shape([16]), tvm_ffi.dtype('int32'),
                        tvm_ffi.device('cuda', 0), None)
    a_device, b_device = empty(), empty()
    copy(a_ffi, a_device)
    module['cuda'](a_device, b_device)
    copy(b_device, b_ffi)
    expected = a + 1
else:
    module['cpu'](a_ffi, b_ffi)
    expected = a + tvm_ffi.get_global_func('runtime.NumThreads')()
np.testing.assert_array_equal(b, expected)
assert not any(name.split('.')[0] in ('tvm', 'tvm_tirx', 'torch') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", consumer, str(output), device, tvm_ffi.__file__],
        env=child_env,
        cwd=tmp_path,
        check=True,
    )
    if device == "cpu":
        # Prove the registry-only dependency actually needs retention. This
        # negative control deliberately uses raw inputs, never the positive link.
        unretained = tmp_path / ("unretained" + output.suffix)
        discard = "-Wl,-dead_strip_dylibs" if sys.platform == "darwin" else "-Wl,--as-needed"
        subprocess.run(
            [
                *compiler,
                "-dynamiclib" if sys.platform == "darwin" else "-shared",
                discard,
                "-o",
                str(unretained),
                *map(str, objects),
                *libraries,
            ],
            check=True,
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", consumer, str(unretained), device, tvm_ffi.__file__],
            env=child_env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "runtime.NumThreads" in result.stderr


if __name__ == "__main__":
    tvm_tirx.testing.main()
