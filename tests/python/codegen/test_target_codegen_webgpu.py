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

import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def test_codegen_buffer_access_modes():
    """Read-only typed buffer parameters should remain read-only in WGSL."""

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
            for tx in T.thread_binding(8, thread="threadIdx.x"):
                B[tx] = A[tx]

    executable = tvm.compile(Module, target="webgpu")
    source = executable.mod.imports[0].inspect_source("wgsl")

    assert "var<storage, read> A_ptr" in source
    assert "var<storage, read_write> B_ptr" in source


def _build_webgpu(mod, target="webgpu"):
    build = tvm.get_global_func("target.build.webgpu")
    return build(mod, tvm.target.Target(target))


def test_bounded_symbolic_stack_allocation():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((T.min(n, 64), 2), "float32", scope="local")
            T.evaluate(scratch.data)

    source = _build_webgpu(Module).inspect_source()
    assert re.search(r"\bvar\s+\w+\s*:\s*array<f32,\s*128>;", source)


def test_unbounded_symbolic_stack_allocation_rejected():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((n,), "float32", scope="local")
            scratch[0] = 1.0
            T.evaluate(scratch[0])

    with pytest.raises(
        tvm.error.InternalError,
        match="WebGPU allocation extent requires a finite compile-time upper bound",
    ):
        _build_webgpu(Module)


@pytest.mark.parametrize("extent", [0, -1])
def test_nonpositive_stack_allocation_rejected(extent):
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main():
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((extent,), "float32", scope="local")
            T.evaluate(scratch.data)

    with pytest.raises(
        tvm.error.InternalError,
        match="WebGPU allocation extent requires a positive compile-time upper bound",
    ):
        _build_webgpu(Module)


def test_stack_allocation_element_count_overflow_rejected():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(n: T.int32, m: T.int32, k: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer(
                (T.min(n, 1 << 30), T.min(m, 1 << 30), T.min(k, 1 << 30)),
                "uint8",
                scope="local",
            )
            T.evaluate(scratch.data)

    with pytest.raises(
        tvm.error.InternalError, match="WebGPU allocation element count is too large to represent"
    ):
        _build_webgpu(Module)


def test_stack_allocation_byte_size_overflow_rejected():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(n: T.int32, m: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer(
                (T.min(n, 1 << 30), T.min(m, 1 << 30), 4), "float32", scope="local"
            )
            T.evaluate(scratch.data)

    with pytest.raises(
        tvm.error.InternalError, match="WebGPU allocation byte size is too large to represent"
    ):
        _build_webgpu(Module)


def test_workgroup_allocation_at_target_limit():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main():
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((8192,), "float32", scope="shared")
            scratch[0] = 1.0

    source = _build_webgpu(Module).inspect_source()
    assert re.search(r"var<workgroup>\s+\w+\s*:\s*array<f32,\s*8192>;", source)


def test_total_workgroup_allocation_above_target_limit_rejected():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main():
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            first = T.alloc_buffer((4096,), "float32", scope="shared")
            second = T.alloc_buffer((4097,), "float32", scope="shared")
            first[0] = 1.0
            second[0] = 2.0

    with pytest.raises(
        tvm.error.InternalError,
        match=r"WebGPU workgroup allocations use 32784 bytes, .* supports only 32768 bytes",
    ):
        _build_webgpu(Module)


def test_workgroup_allocation_accounts_for_declaration_alignment():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main():
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            first = T.alloc_buffer((1,), "float32", scope="shared")
            second = T.alloc_buffer((1,), "float32", scope="shared")
            first[0] = 1.0
            second[0] = 2.0

    with pytest.raises(
        tvm.error.InternalError,
        match=r"WebGPU workgroup allocations use 32 bytes, .* supports only 16 bytes",
    ):
        _build_webgpu(Module, {"kind": "webgpu", "max_shared_memory_per_block": 16})


def test_workgroup_allocation_uses_target_limit():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main():
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("webgpu"),
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((16384,), "float32", scope="shared")
            scratch[0] = 1.0

    _build_webgpu(Module, {"kind": "webgpu", "max_shared_memory_per_block": 65536})


if __name__ == "__main__":
    tvm.testing.main()
