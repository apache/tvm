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
import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_metal_inf_nan():
    target = "metal"

    def check_inf_nan(n, value, dtype):
        @I.ir_module(s_tir=True)
        class Module:
            @T.prim_func(s_tir=True)
            def main(
                A: T.Buffer((1,), dtype),
                C: T.Buffer((1,), dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                for i in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(1, i)
                        T.reads()
                        T.writes(C[v_i])
                        C[v_i] = T.Cast(dtype, value)

        fun = tvm.compile(Module, target=target)

        def run_and_check():
            dev = tvm.metal(0)
            a = tvm.runtime.empty((n,), dtype, dev)
            c = tvm.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm.testing.run_with_gpu_lock(run_and_check)

    check_inf_nan(1, -float("inf"), "float32")
    check_inf_nan(1, -float("inf"), "float16")
    check_inf_nan(1, float("inf"), "float32")
    check_inf_nan(1, float("inf"), "float16")
    check_inf_nan(1, float("nan"), "float32")
    check_inf_nan(1, float("nan"), "float16")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_unaligned_vectorize():
    @tvm.script.ir_module
    class IRModule:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((2, 3), "float32"), B: T.Buffer((6,), "float32")):
            T.func_attr({"global_symbol": "main"})
            for i0_1 in T.thread_binding(3, thread="threadIdx.x"):
                for i0_0 in T.vectorized(2):
                    with T.sblock("block"):
                        vi0 = T.axis.spatial(6, i0_0 * 3 + i0_1)
                        B[vi0] = A[vi0 // 3, vi0 % 3]

    target = "metal"
    a = (np.arange(6).reshape(2, 3)).astype("float32")
    f = tvm.compile(IRModule, target=target)

    def run_and_check():
        dev = tvm.metal()
        a_nd = tvm.runtime.tensor(a, dev)
        b_nd = tvm.runtime.empty((6,), "float32", dev)
        f(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.numpy(), a.reshape(6), atol=1e-5, rtol=1e-5)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_metal_erf():
    target = "metal"

    def check_erf(n, dtype):
        @I.ir_module(s_tir=True)
        class Module:
            @T.prim_func(s_tir=True)
            def main(
                A: T.Buffer((1,), dtype),
                C: T.Buffer((1,), dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                for i0 in T.thread_binding(1, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i0 = T.axis.spatial(1, i0)
                        T.reads(A[v_i0])
                        T.writes(C[v_i0])
                        C[v_i0] = T.erf(A[v_i0])

        fun = tvm.compile(Module, target=target)

        def run_and_check():
            dev = tvm.metal(0)
            a = tvm.runtime.empty((n,), dtype, dev)
            c = tvm.runtime.empty((n,), dtype, dev)
            fun(a, c)

        tvm.testing.run_with_gpu_lock(run_and_check)

    check_erf(1, "float32")
    check_erf(1, "float16")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_ramp():
    target = "metal"

    @tvm.script.ir_module
    class IRModule:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((1, 2), "int32")):
            T.func_attr({"global_symbol": "main"})
            for i in T.thread_binding(1, thread="threadIdx.x"):
                with T.sblock("block"):
                    tx = T.axis.spatial(1, i)
                    r = T.ramp(tx, 3, 2)
                    A[0, T.ramp(0, 1, 2)] = r

    f = tvm.compile(IRModule, target=target)

    def run_and_check():
        dev = tvm.metal()
        a_nd = tvm.runtime.empty((1, 2), "int32", dev)
        f(a_nd)
        assert tuple(a_nd.numpy()[0, :]) == (0, 3)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_select_vectorize():
    @tvm.script.ir_module
    class IRModule:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((6), "float32"), B: T.Buffer((6,), "float32")):
            T.func_attr({"global_symbol": "main"})
            for i0_1 in T.thread_binding(3, thread="threadIdx.x"):
                for i0_0 in T.vectorized(2):
                    with T.sblock("block"):
                        vi0 = T.axis.spatial(6, i0_0 * 3 + i0_1)
                        B[vi0] = T.Select((vi0 % 2) == 0, A[vi0], T.float32(0))

    target = "metal"
    a = np.arange(6).astype("float32")
    f = tvm.compile(IRModule, target=target)
    a.reshape(3, 2)[:, 1] = 0

    def run_and_check():
        dev = tvm.metal()
        a_nd = tvm.runtime.tensor(a, dev)
        b_nd = tvm.runtime.empty((6,), "float32", dev)
        f(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.numpy(), a, atol=1e-5, rtol=1e-5)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_vectorized_uint8():
    @T.prim_func(s_tir=True)
    def func(A: T.Buffer((16), "uint8"), B: T.Buffer((16), "float32")):
        for i in T.thread_binding(4, thread="threadIdx.x"):
            for j in T.vectorized(4):
                with T.sblock("block"):
                    vi = T.axis.spatial(16, i * 4 + j)
                    B[vi] = T.Cast("float32", A[vi])

    a = np.arange(16).astype("uint8")
    f = tvm.compile(func, target="metal")

    def run_and_check():
        dev = tvm.metal()
        a_nd = tvm.runtime.tensor(a, dev)
        b_nd = tvm.runtime.empty((16,), "float32", dev)
        f(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.numpy(), a.astype("float32"), atol=1e-5, rtol=1e-5)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_func_with_trailing_pod_params():
    from tvm.support import xcode  # pylint: disable=import-outside-toplevel

    @T.prim_func(s_tir=True)
    def func(A: T.Buffer((16), "float32"), B: T.Buffer((16), "float32"), x: T.float32):
        for i in T.thread_binding(16, thread="threadIdx.x"):
            with T.sblock("block"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi] + x

    @tvm.register_global_func("tvm_callback_metal_compile")
    def compile_metal(src, target):
        return xcode.compile_metal(src)

    mod = tvm.IRModule({"main": func})

    f = tvm.tirx.build(mod, target="metal")
    src: str = f.imports[0].inspect_source()
    occurrences = src.count("struct func_kernel_args_t")
    assert occurrences == 1, occurrences


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_metal_compile_callback_source_passthrough():
    n = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i_0 in T.thread_binding(n // 32, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(n, i_0 * 32 + i_1)
                        T.reads(A[v_i])
                        T.writes(B[v_i])
                        B[v_i] = A[v_i] + 1.0

    seen = {}

    def inspect_callback(src, target):
        # Pure inspection callback: capture the source, return it untouched and
        # declare it is still textual MSL so it is compiled at load time.
        seen["src"] = src
        return (src, "metal")

    tvm.register_global_func("tvm_callback_metal_compile", inspect_callback, override=True)
    try:
        f = tvm.compile(Module, target="metal")
        dev = tvm.metal()
        a = np.random.rand(n).astype("float32")
        a_nd = tvm.runtime.tensor(a, dev)
        b_nd = tvm.runtime.empty((n,), "float32", dev)
        f(a_nd, b_nd)
        dev.sync()
    finally:
        tvm_ffi.registry.remove_global_func("tvm_callback_metal_compile")

    assert "src" in seen and len(seen["src"]) > 0
    tvm.testing.assert_allclose(b_nd.numpy(), a + 1.0, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_metal_compile_callback_mixed_formats_rejected():
    n = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((n,), "float32"),
            B: T.Buffer((n,), "float32"),
            C: T.Buffer((n,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # Two independent thread-bound regions -> two device kernels, so the
            # compile callback is invoked twice within one module.
            for i_0 in T.thread_binding(n // 32, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(n, i_0 * 32 + i_1)
                        T.reads(A[v_i])
                        T.writes(B[v_i])
                        B[v_i] = A[v_i] + 1.0
            for j_0 in T.thread_binding(n // 32, thread="blockIdx.x"):
                for j_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_j = T.axis.spatial(n, j_0 * 32 + j_1)
                        T.reads(A[v_j])
                        T.writes(C[v_j])
                        C[v_j] = A[v_j] + 2.0

    calls = {"n": 0}

    def mixed_callback(src, target):
        calls["n"] += 1
        if calls["n"] == 1:
            # Treated as a compiled metallib payload.
            return src
        # Second kernel declares textual MSL, contradicting the metallib above.
        return (src, "metal")

    tvm.register_global_func("tvm_callback_metal_compile", mixed_callback, override=True)
    try:
        with pytest.raises(Exception, match="inconsistent formats"):
            tvm.compile(Module, target="metal")
    finally:
        tvm_ffi.registry.remove_global_func("tvm_callback_metal_compile")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_export_load_with_fallback(monkeypatch, tmp_path):
    """Force the codegen wrapper into the fallback branch, then export."""
    n = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i_0 in T.thread_binding(n // 32, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(n, i_0 * 32 + i_1)
                        T.reads(A[v_i])
                        T.writes(B[v_i])
                        B[v_i] = A[v_i] + 1.0

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    host_lib = tvm.compile(Module, target="metal")
    monkeypatch.delenv("TVM_COMPILE_FORCE_FALLBACK")

    lib_path = str(tmp_path / "lib.so")
    host_lib.export_library(lib_path)


def test_codegen_simdgroup_buffer_data():
    """Simdgroup intrinsics should accept buffer_data projections."""

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def kernel():
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "kernel",
                    "tirx.kernel_launch_params": [],
                }
            )
            A = T.alloc_buffer((64,), "float16", scope="shared")
            A_frag = T.alloc_buffer((64,), "float16", scope="metal.simdgroup")
            B_frag = T.alloc_buffer((64,), "float16", scope="metal.simdgroup")
            C_frag = T.alloc_buffer((64,), "float16", scope="metal.simdgroup")
            T.metal.make_filled_simdgroup_matrix(C_frag.data, 0, T.float32(0), 8, 8)
            T.metal.simdgroup_load(A_frag.data, 0, A.data, 8, 8, 8, T.bool(False))
            T.metal.simdgroup_store(C_frag.data, 0, A.data, 8, 8, 8, T.bool(False))
            T.metal.simdgroup_multiply_accumulate(
                C_frag.data, 0, A_frag.data, 0, B_frag.data, 0, C_frag.data, 0
            )

    metal_codegen = tvm.get_global_func("target.build.metal")
    module = metal_codegen(Module, tvm.target.Target("metal"))
    source = module.inspect_source()

    assert "make_filled_simdgroup_matrix<half, 8, 8>" in source
    assert "simdgroup_load(" in source
    assert "simdgroup_store(" in source
    assert "simdgroup_multiply_accumulate(" in source


def _build_metal(mod):
    build = tvm.get_global_func("target.build.metal")
    return build(mod, tvm.target.Target("metal"))


def test_bounded_symbolic_stack_allocation():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("metal"),
                    "tirx.kernel_launch_params": [],
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((T.min(n, 64), 2), "float32", scope="local")
            T.evaluate(scratch.data)

    source = _build_metal(Module).inspect_source()
    assert "thread float scratch[128]" in source


def test_unbounded_symbolic_stack_allocation_rejected():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(n: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("metal"),
                    "tirx.kernel_launch_params": [],
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((n,), "float32", scope="local")
            scratch[0] = 1.0
            T.evaluate(scratch[0])

    with pytest.raises(
        tvm.error.InternalError,
        match="Metal allocation extent requires a finite compile-time upper bound",
    ):
        _build_metal(Module)


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
                    "target": T.target("metal"),
                    "tirx.kernel_launch_params": [],
                    "tirx.is_global_func": True,
                }
            )
            scratch = T.alloc_buffer((extent,), "float32", scope="local")
            T.evaluate(scratch.data)

    with pytest.raises(
        tvm.error.InternalError,
        match="Metal allocation extent requires a positive compile-time upper bound",
    ):
        _build_metal(Module)


def test_stack_allocation_element_count_overflow_rejected():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def main(n: T.int32, m: T.int32, k: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "main",
                    "target": T.target("metal"),
                    "tirx.kernel_launch_params": [],
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
        tvm.error.InternalError, match="Metal allocation element count is too large to represent"
    ):
        _build_metal(Module)


def test_codegen_pointer_byte_offsets_preserve_storage_scope():
    """Pointer byte offsets should preserve the source Metal address space."""

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def kernel():
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "kernel",
                    "tirx.kernel_launch_params": [],
                }
            )
            shared = T.alloc_buffer((16,), "float16", scope="shared")
            typed_alias = T.ptr_byte_offset(shared.data, 4, "float16")
            typed_buffer = T.decl_buffer((14,), "float16", data=typed_alias, scope="shared")
            void_alias = T.handle_add_byte_offset(shared.data, 8)
            void_buffer = T.decl_buffer((12,), "float16", data=void_alias, scope="shared")
            typed_buffer[0] = T.float16(1)
            void_buffer[0] = T.float16(2)

    metal_codegen = tvm.get_global_func("target.build.metal")
    module = metal_codegen(Module, tvm.target.Target("metal"))
    source = module.inspect_source()

    assert "threadgroup half* typed_alias" in source
    assert "threadgroup void* void_alias" in source
    assert source.count("threadgroup char*") == 2


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_metal(), reason="need metal")
def test_pointer_byte_offsets_execute_in_threadgroup_memory():
    """Pointer byte offsets should execute in Metal threadgroup memory."""

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
            for bx in T.thread_binding(1, thread="blockIdx.x"):
                for tx in T.thread_binding(1, thread="threadIdx.x"):
                    shared = T.alloc_buffer((16,), "float32", scope="shared")
                    typed_alias = T.ptr_byte_offset(shared.data, 4, "float32")
                    typed_buffer = T.decl_buffer((15,), "float32", data=typed_alias, scope="shared")
                    void_alias = T.handle_add_byte_offset(shared.data, 8)
                    void_buffer = T.decl_buffer((14,), "float32", data=void_alias, scope="shared")
                    shared[0] = A[0]
                    typed_buffer[0] = A[1]
                    void_buffer[0] = A[2]
                    B[0] = shared[0]
                    B[1] = typed_buffer[0]
                    B[2] = void_buffer[0]

    executable = tvm.compile(Module, target="metal")
    source = executable.mod.imports[0].inspect_source()

    assert "threadgroup float shared[16]" in source
    assert "threadgroup float* typed_alias" in source
    assert "threadgroup void* void_alias" in source
    assert source.count("threadgroup char*") == 2

    def run_and_check():
        dev = tvm.metal(0)
        host_input = np.arange(16, dtype="float32") + 10
        input_tensor = tvm.runtime.tensor(host_input, dev)
        output_tensor = tvm.runtime.tensor(np.zeros(16, dtype="float32"), dev)
        executable(input_tensor, output_tensor)
        tvm.testing.assert_allclose(output_tensor.numpy()[:3], host_input[:3])

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
