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

import re

import numpy as np
import pytest

import tvm
from tvm.script import ir as I
from tvm.script import tirx as T


def _source(module, target, monkeypatch):
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"} if target == "cuda" else target)
    return tvm.get_global_func("target.build." + target.kind.name)(module, target).inspect_source()


@pytest.mark.parametrize("target", ["c", "cuda", "opencl", "metal"])
@pytest.mark.parametrize("source_dtype", ["void", "int32", "float32"])
def test_argument_tensor_view(target, source_dtype, monkeypatch):
    @I.ir_module
    class Module:
        @T.function
        def main(data: T.handle(source_dtype)):
            T.func_attr(
                {"global_symbol": "kernel", "calling_conv": 2, "tirx.kernel_launch_params": []}
            )
            view = T.decl_tensor((4,), "float32", data=data)
            view[0] = T.float32(1)
            view[1] = view[0] + T.float32(2)

    source = _source(Module, target, monkeypatch)
    name = "data" if source_dtype == "float32" else "view"
    assert re.search(rf"\b{name}\[0\] =", source)
    assert re.search(rf"\b{name}\[1\] =", source)
    if source_dtype != "float32":
        assert re.search(r"float\* view = .*data", source)


@pytest.mark.parametrize("target", ["c", "cuda", "opencl", "metal"])
def test_allocation_tensor_view(target, monkeypatch):
    scope = "global" if target == "c" else "shared"

    @I.ir_module
    class Module:
        @T.function
        def main():
            T.func_attr(
                {"global_symbol": "kernel", "calling_conv": 2, "tirx.kernel_launch_params": []}
            )
            storage = T.alloc_tensor((8,), "uint32", scope=scope)
            view = T.decl_tensor((8,), "float32", data=storage.data, scope=scope)
            same = T.decl_tensor((8,), "float32", data=view.data, scope=scope)
            view[0] = T.float32(1)
            same[1] = view[0]
            storage[2] = T.uint32(3)

    source = _source(Module, target, monkeypatch)
    assert "view[0] =" in source
    assert "view[1] = view[0]" in source
    assert "storage_ptr[2] =" in source or "storage[2] =" in source
    if target == "opencl":
        assert "__local float* view = ((__local float*)" in source
    if target == "metal":
        assert "threadgroup float* view = ((threadgroup float*)" in source


@pytest.mark.parametrize("source_dtype, dtype", [("float32", "float32"), ("uint32", "int8x4")])
def test_webgpu_array_view(source_dtype, dtype, monkeypatch):
    @I.ir_module
    class Module:
        @T.function
        def main(data: T.handle(source_dtype)):
            T.func_attr(
                {"global_symbol": "kernel", "calling_conv": 2, "tirx.kernel_launch_params": []}
            )
            view = T.decl_tensor((4,), dtype, data=data)
            view[0] = view[1]

    source = _source(Module, "webgpu", monkeypatch)
    assert "data[0i] =" in source
    assert "float*" not in source
    if dtype == "int8x4":
        assert "array<u32>" in source


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_cuda_shared_pool_views(scope, monkeypatch):
    @I.ir_module
    class Module:
        @T.function
        def main():
            T.func_attr(
                {"global_symbol": "kernel", "calling_conv": 2, "tirx.kernel_launch_params": []}
            )
            pool = T.alloc_tensor((64,), "uint8", scope=scope)
            floats = T.decl_tensor((8,), "float32", data=pool.data, scope=scope)
            words = T.decl_tensor((8,), "uint32", data=pool.data, scope=scope)
            floats[1] = T.float32(2)
            words[2] = T.uint32(3)

    source = _source(Module, "cuda", monkeypatch)
    assert "uchar pool_ptr[" in source
    assert "float* floats = ((float*)pool_ptr)" in source
    assert "uint* words = ((uint*)pool_ptr)" in source
    assert "floats[1] =" in source
    assert "words[2] =" in source
    if scope == "shared.dyn":
        assert "extern __shared__" in source
        assert "__align__(" in source


def test_packed_arguments_and_byte_offset_view(tmp_path):
    @T.function
    def main(A: T.Tensor((4,), "float32"), B: T.Tensor((2,), "float32")):
        offset = T.handle_add_byte_offset(A.data, 4)
        view = T.decl_tensor((2,), "float32", data=offset)
        B[0] = view[0]
        B[1] = view[1]

    built = tvm.tirx.build(main, target="c")
    source = built.inspect_source()
    assert "float* view = ((float*)offset)" in source
    for name in ["A", "B"]:
        binding = next(line for line in source.splitlines() if f"float* {name} =" in line)
        assert binding.count("(float*)") == 1
    assert "((char*)" in source and " + 4" in source
    assert "view[0]" in source and "view[1]" in source
    path = str(tmp_path / "views.so")
    built.export_library(path)
    loaded = tvm.runtime.load_module(path)
    a = tvm.runtime.tensor(np.array([1, 2, 3, 4], dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(2, dtype="float32"))
    loaded["main"](a, b)
    np.testing.assert_equal(b.numpy(), [2, 3])


@pytest.mark.parametrize("target", ["c", "cuda", "opencl", "metal"])
def test_volatile_tensor_view(target, monkeypatch):
    scope = "global" if target == "c" else "shared"

    @I.ir_module
    class Module:
        @T.function
        def main():
            T.func_attr(
                {"global_symbol": "kernel", "calling_conv": 2, "tirx.kernel_launch_params": []}
            )
            pool = T.alloc_tensor((8,), "uint32", scope=scope, annotations={"tirx.volatile": True})
            view = T.decl_tensor((8,), "float32", data=pool.data, scope=scope)
            view[0] = T.float32(1)
            view[1] = view[0]
            pool[2] = T.uint32(3)

    source = _source(Module, target, monkeypatch)
    assert re.search(r"\(\(volatile [^)]*float\*\)view\)\[0\]", source)
    assert re.search(r"\(\(volatile [^)]*float\*\)view\)\[1\]", source)


def test_cuda_vector_view_preserves_access_cast(monkeypatch):
    @I.ir_module
    class Module:
        @T.function
        def main(data: T.handle("uint8")):
            T.func_attr(
                {"global_symbol": "kernel", "calling_conv": 2, "tirx.kernel_launch_params": []}
            )
            view = T.decl_tensor((8,), "float32", data=data)
            view[T.Ramp(0, 1, 4)] = T.Broadcast(T.float32(1), 4)

    source = _source(Module, "cuda", monkeypatch)
    assert "float* view = ((float*)data)" in source
    assert "float4*" in source


def test_cuda_packed_view_preserves_index_units(monkeypatch):
    @I.ir_module
    class Module:
        @T.function
        def main(data: T.handle("uint8")):
            T.func_attr(
                {"global_symbol": "kernel", "calling_conv": 2, "tirx.kernel_launch_params": []}
            )
            view = T.decl_tensor((32,), "int4", data=data)
            view[16] = T.Cast("int4", 1)

    source = _source(Module, "cuda", monkeypatch)
    assert "int* view = ((int*)data)" in source
    assert "16 / 8" in source
