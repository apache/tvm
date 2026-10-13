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

"""Canonical memory scopes cross the Python, compiler, and runtime boundaries."""

import os
import subprocess
import sys
import uuid

import numpy as np
import pytest
import tvm_ffi

import tvm
from tvm.runtime import MemoryScope
from tvm.script import tirx as T


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
    prefix = "test.memory_scope.nul." + uuid.uuid4().hex
    with pytest.raises(ValueError):
        MemoryScope.register(prefix + "\x00scope")
    with pytest.raises(ValueError):
        MemoryScope(prefix)


def test_shared_cpp_python_registry_and_typed_conversion():
    name = "test.memory_scope.cpp." + uuid.uuid4().hex
    cpp_scope = tvm_ffi.get_global_func("runtime.MemoryScopeRegister")(name)
    assert MemoryScope(name).same_as(cpp_scope)
    with pytest.raises(ValueError):
        tvm_ffi.get_global_func("runtime.MemoryScopeRegister")(name)

    custom = new_scope()
    for scope in (custom, custom.name):
        ptr_ty = tvm.ir.PtrType(tvm.ir.PrimType("float32"), scope)
        tensor = tvm.tirx.decl_tensor((4,), "float32", scope=scope)
        tensor_ty = tvm.tirx.TensorType((4,), "float32", scope)
        assert tensor_ty.storage_scope.same_as(custom)
        assert ptr_ty.storage_scope.same_as(custom)
        assert tensor.ty.storage_scope.same_as(custom)
        assert tensor.scope().same_as(custom)
        assert tensor.data.ty.storage_scope.same_as(custom)

    unknown = "test.memory_scope.unknown." + uuid.uuid4().hex
    for construct in (
        lambda: tvm.ir.PtrType(tvm.ir.PrimType("float32"), unknown),
        lambda: tvm.tirx.decl_tensor((4,), "float32", scope=unknown),
        lambda: T.Tensor((4,), "float32", scope=unknown),
    ):
        with pytest.raises(ValueError):
            construct()
    with pytest.raises(ValueError):
        MemoryScope(unknown)


def test_global_defaults():
    global_scope = MemoryScope("global")
    assert tvm.ir.PtrType(tvm.ir.PrimType("float32")).storage_scope.same_as(global_scope)
    assert tvm.tirx.decl_tensor((4,), "float32").scope().same_as(global_scope)
    for scope in ("global", global_scope):
        assert tvm.ir.PtrType(tvm.ir.PrimType("float32"), scope).storage_scope.same_as(global_scope)
        assert tvm.tirx.decl_tensor((4,), "float32", scope=scope).scope().same_as(global_scope)
        assert T.Tensor((4,), "float32", scope=scope).storage_scope.same_as(global_scope)
        assert tvm.relax.VDevice("llvm", memory_scope=scope).memory_scope.same_as(global_scope)
    for construct in (
        lambda: MemoryScope(""),
        lambda: tvm.ir.PtrType(tvm.ir.PrimType("float32"), ""),
        lambda: tvm.tirx.decl_tensor((4,), "float32", scope=""),
        lambda: T.Tensor((4,), "float32", scope=""),
        lambda: tvm.relax.VDevice("llvm", memory_scope=""),
        lambda: tvm.runtime.empty((4,), mem_scope=""),
    ):
        with pytest.raises(ValueError):
            construct()


def test_json_and_script_roundtrip():
    custom = new_scope()
    for scope in (MemoryScope("global"), MemoryScope("shared.dyn"), custom):
        assert tvm.ir.load_json(tvm.ir.save_json(scope)).same_as(scope)
        ty = T.Tensor((4,), "float32", scope=scope)
        restored = tvm.ir.load_json(tvm.ir.save_json(ty))
        assert restored.storage_scope.same_as(scope)
        tvm.ir.assert_structural_equal(ty, restored)

        @T.function
        def func(A: T.Tensor((4,), "float32", scope=scope)):
            A[0] = T.float32(1)

        script = func.script()
        if scope != MemoryScope("global"):
            assert f'scope="{scope.name}"' in script
        restored_func = tvm.script.from_source(script, extra_vars={"T": T})
        tvm.ir.assert_structural_equal(func, restored_func)
        assert restored_func.params[0].scope().same_as(scope)


def test_runtime_tensor_allocation():
    expected = np.arange(4, dtype="float32")
    for scope in (None, "global", MemoryScope("global")):
        tensor = tvm.runtime.tensor(expected, tvm.cpu(), mem_scope=scope)
        np.testing.assert_array_equal(tensor.numpy(), expected)
        np.testing.assert_array_equal(tensor.copyto(tvm.cpu(), mem_scope=scope).numpy(), expected)
    with pytest.raises(ValueError):
        tvm.runtime.empty((4,), mem_scope="test.memory_scope.unregistered")
    with pytest.raises(ValueError, match="Unsupported memory scope"):
        tvm.runtime.empty((4,), mem_scope=new_scope())


def test_runtime_only_registration():
    env = {**os.environ, "TVM_USE_RUNTIME_LIB": "1"}
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import tvm
from tvm.runtime import MemoryScope
assert tvm.base._RUNTIME_ONLY
for name in ("global", "shared", "shared.dyn", "warp", "local"):
    assert MemoryScope(name).name == name
for name, allocator in (("nvshmem", "runtime.disco.nvshmem.empty"),
                        ("ipc_memory", "runtime.disco.cuda_ipc.empty")):
    if tvm.get_global_func(allocator, allow_missing=True) is not None:
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


def test_relax_allocation_scope_operands():
    scope = MemoryScope("global")
    for make_call in (
        lambda value: tvm.relax.op.builtin.alloc_tensor((4,), "float32", 0, value),
        lambda value: tvm.relax.op.memory.alloc_storage((16,), 0, value, "float32"),
        lambda value: tvm.relax.op.vm.alloc_storage((16,), 0, "float32", value),
    ):
        tvm.ir.assert_structural_equal(make_call(scope), make_call(scope.name))
