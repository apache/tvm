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

"""Canonical runtime memory-scope identity, allocation, and registration."""

import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
from tvm_tirx.runtime import MemoryScope


def new_scope():
    return MemoryScope.register("test.memory_scope." + uuid.uuid4().hex)


def test_identity_and_registration():
    global_scope = MemoryScope("global")
    assert global_scope.same_as(MemoryScope("global"))
    assert MemoryScope(global_scope) is global_scope
    assert global_scope.name == "global"
    assert global_scope != MemoryScope("shared")
    assert global_scope != "global"
    assert {global_scope: 1}[MemoryScope("global")] == 1

    name = "test.memory_scope." + uuid.uuid4().hex
    with pytest.raises(ValueError):
        MemoryScope(name)
    custom = MemoryScope.register(name)
    assert custom.same_as(MemoryScope(name))
    assert custom in MemoryScope.all_entries()
    with pytest.raises(ValueError):
        MemoryScope.register(name)
    with pytest.raises(ValueError):
        MemoryScope.register("")
    with pytest.raises(TypeError):
        MemoryScope.register(123)
    with pytest.raises(ValueError):
        MemoryScope("")
    prefix = "test.memory_scope.nul." + uuid.uuid4().hex
    with pytest.raises(ValueError):
        MemoryScope.register(prefix + "\x00scope")
    with pytest.raises(ValueError):
        MemoryScope(prefix)


def test_shared_cpp_python_registry():
    name = "test.memory_scope.cpp." + uuid.uuid4().hex
    cpp_scope = tvm_ffi.get_global_func("runtime.MemoryScopeRegister")(name)
    assert MemoryScope(name).same_as(cpp_scope)
    with pytest.raises(ValueError):
        tvm_ffi.get_global_func("runtime.MemoryScopeRegister")(name)


def test_runtime_tensor_allocation():
    expected = np.arange(4, dtype="float32")
    for scope in (None, "global", MemoryScope("global")):
        tensor = tvm_tirx.runtime.tensor(expected, tvm_tirx.cpu(), mem_scope=scope)
        np.testing.assert_array_equal(tensor.numpy(), expected)
        np.testing.assert_array_equal(
            tensor.copyto(tvm_tirx.cpu(), mem_scope=scope).numpy(), expected
        )
    for scope in ("", "test.memory_scope.unregistered"):
        with pytest.raises(ValueError):
            tvm_tirx.runtime.empty((4,), mem_scope=scope)
    with pytest.raises(ValueError, match="Unsupported memory scope"):
        tvm_tirx.runtime.empty((4,), mem_scope=new_scope())


def test_runtime_only_registration():
    env = {**os.environ, "TVM_USE_RUNTIME_LIB": "1"}
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import tvm_ffi
import tvm_tirx
from tvm_tirx.runtime import MemoryScope
assert tvm_tirx.base._RUNTIME_ONLY
for name in ("global", "shared", "shared.dyn", "warp", "local"):
    assert MemoryScope(name).name == name
for name, allocator in (("nvshmem", "runtime.disco.nvshmem.empty"),
                        ("ipc_memory", "runtime.disco.cuda_ipc.empty")):
    if tvm_ffi.get_global_func(allocator, allow_missing=True) is not None:
        assert MemoryScope(name).name == name
    else:
        try:
            MemoryScope(name)
        except ValueError:
            pass
        else:
            raise AssertionError("Scope registered without its owning runtime: " + name)
custom = MemoryScope.register("test.runtime_only")
assert custom.same_as(MemoryScope(custom.name))
""",
        ],
        env=env,
        check=True,
    )


@pytest.mark.parametrize("load_order", ["runtime_only", "runtime_first", "compiler_first"])
def test_opencl_memory_scope_registration_load_order(load_order):
    runtime_path = Path(tvm_tirx.base._LOADED_LIBS["tvm_runtime"]._name).resolve()
    opencl_path = runtime_path.with_name(
        runtime_path.name.replace("tvm_runtime", "tvm_runtime_opencl")
    )
    if not opencl_path.is_file():
        pytest.skip("OpenCL runtime sidecar is not built")
    compiler = tvm_tirx.base._LOADED_LIBS.get("tvm_tirx")
    if compiler is None and load_order != "runtime_only":
        pytest.skip("Compiler library is not available")
    env = {
        **os.environ,
        "TVM_USE_RUNTIME_LIB": "1" if load_order == "runtime_only" else "0",
        "TVM_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import ctypes
import sys
import tvm_ffi

runtime_path, opencl_path, load_order = sys.argv[1:]
handles = []
if load_order == "runtime_first":
    # Load the sidecar before the compiler, without creating compiler-only
    # Python type placeholders through a runtime-only import first.
    handles.append(ctypes.CDLL(runtime_path, mode=ctypes.RTLD_GLOBAL))
    handles.append(ctypes.CDLL(opencl_path, mode=ctypes.RTLD_LOCAL))

import tvm_tirx
from tvm_tirx.runtime import MemoryScope

assert tvm_tirx.base._RUNTIME_ONLY == (load_order == "runtime_only")
names = ("texture", "global.texture", "global.texture-1d", "global.texture-nchw",
         "global.texture-nhwc", "global.texture-weight")
identities = None
if load_order == "runtime_only":
    for name in names:
        try:
            MemoryScope(name)
        except ValueError:
            pass
        else:
            raise AssertionError("OpenCL scope registered without an owning library: " + name)
else:
    identities = [MemoryScope(name) for name in names]
handles.append(ctypes.CDLL(opencl_path, mode=ctypes.RTLD_LOCAL))
scopes = [MemoryScope(name) for name in names]
if identities is not None:
    assert all(before.same_as(after) for before, after in zip(identities, scopes))
for scope in scopes:
    assert MemoryScope.attr_dict["opencl.texture_storage"][scope] is True
    expected = 2 if scope.name == "texture" else 1
    assert MemoryScope.attr_dict["runtime.memory_alloc_property"][scope] == expected
""",
            str(runtime_path),
            str(opencl_path),
            load_order,
        ],
        env=env,
        check=True,
        timeout=60,
    )
