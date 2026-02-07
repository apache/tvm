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
import tvm.testing
from tvm.script import tir as T, ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I_builder
from tvm.script.ir_builder import tir as T_builder


dtype = tvm.testing.parameter("float32", "int32", "float16", "int8")
fuzz_seed = tvm.testing.parameter(range(25))


# Explicitly specify a target, as this test is looking at the
# generated shader code, and is not running on an actual device.
@tvm.testing.parametrize_targets(
    " ".join(
        [
            "vulkan",
            "-supports_int8=1",
            "-supports_8bit_buffer=1",
            "-supports_storage_buffer_storage_class=1",
            "-supports_float16=1",
            "-supports_16bit_buffer=1",
        ]
    )
)
def test_vector_comparison(target, dev, dtype):
    target = tvm.target.Target(target)
    zero = tvm.tir.const(0, dtype)
    one = tvm.tir.const(1, dtype)

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((1024,), dtype), B: T.Buffer((1024,), dtype)):
            for i_0 in T.thread_binding(8, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    for i_2 in T.vectorized(4):
                        with T.sblock("B"):
                            v_i = T.axis.spatial(1024, i_0 * 128 + i_1 * 4 + i_2)
                            B[v_i] = T.Select(A[v_i] >= zero, A[v_i] + one, zero)

    # Build
    f = tvm.tir.build(Module, target=target)

    # Verify we generate the boolx4 type declaration and the OpSelect
    # v4{float,half,int} instruction
    assembly = f.imports[0].inspect_source()
    matches = re.findall("%v4bool = OpTypeVector %bool 4", assembly)
    assert len(matches) == 1
    matches = re.findall("OpSelect %v4.*", assembly)
    assert len(matches) == 1


def test_array_copy(dev, dtype, fuzz_seed):
    np.random.seed(fuzz_seed)

    log_arr_size = np.random.uniform(low=np.log(1), high=np.log(32768))
    arr_size = np.exp(log_arr_size).astype(int)
    a_np = np.random.uniform(size=(arr_size,)).astype(dtype)
    a = tvm.runtime.empty((arr_size,), dtype, dev).copyfrom(a_np)
    b_np = a.numpy()
    tvm.testing.assert_allclose(a_np, b_np)
    tvm.testing.assert_allclose(a_np, a.numpy())


@tvm.testing.exclude_targets("llvm")
def test_array_vectorize_add(target, dev, dtype):
    target = tvm.target.Target(target)
    arr_size = 64
    lanes = 2

    if "opencl" in str(target) and dtype == "float16":
        pytest.xfail("Opencl target does not support float16")

    vec_dtype = f"{dtype}x{lanes}"
    one = tvm.tir.const(1, vec_dtype)

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((64,), vec_dtype), B: T.Buffer((64,), vec_dtype)):
            for i_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1 in T.thread_binding(4, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(64, i_0 * 4 + i_1)
                        B[v_i] = A[v_i] + one

    f = tvm.compile(Module, target=target)

    a = tvm.runtime.empty((arr_size,), vec_dtype, dev).copyfrom(
        np.random.uniform(size=(arr_size, lanes))
    )
    c = tvm.runtime.empty((arr_size,), vec_dtype, dev)
    f(a, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)


@tvm.testing.exclude_targets("llvm")
def test_vulkan_bool_load(target, dev):
    target = tvm.target.Target(target)
    arr_size = 1024

    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((1024,), "bool"), B: T.Buffer((1024,), "int32")):
            for i_0 in T.thread_binding(8, thread="blockIdx.x"):
                for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(1024, i_0 * 128 + i_1)
                        B[v_i] = T.Cast("int32", A[v_i])

    f = tvm.compile(Module, target=target)

    a_np = np.random.uniform(size=arr_size) > 0.5
    b_np = np.zeros((arr_size,), dtype="int32")
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    f(a, b)
    ref = a_np.astype(np.int32)
    tvm.testing.assert_allclose(b.numpy(), ref)


vulkan_parameter_impl = tvm.testing.parameter("push_constants", "ubo")
vulkan_parameter_dtype = tvm.testing.parameter("int32", "float32", "int64")


# Only run on vulkan because extremely large numbers of input
# parameters can crash cuda/llvm compiler.
@tvm.testing.parametrize_targets("vulkan -from_device=0")
def test_vulkan_constant_passing(target, dev, vulkan_parameter_impl, vulkan_parameter_dtype):
    target = tvm.target.Target(target)
    dtype = vulkan_parameter_dtype

    if not target.attrs.get("supports_int64", False):
        pytest.xfail("Vulkan target does not support Int64 variables")

    # f_add has 3+num_int_params scalar parameters.  The other three
    # are length_n, stride1, and stride2.
    if vulkan_parameter_impl == "push_constants":
        # 4 params, 32 bytes.  Within 128-byte spec-guaranteed size of
        # push constants.  Uses push constants.
        num_int_params = 1
    else:
        # 24 params, 192 bytes.  May be above spec-guaranteed size of 128
        # bytes for push constants.  Uses either push constants or UBO,
        # depending on the device.
        max_push_constants_size = int(target.attrs.get("max_push_constants_size", 128))
        max_int_params_in_push = max_push_constants_size // 8 - 3
        num_int_params = max_int_params_in_push + 1

    # Build IRModule programmatically since num_int_params is dynamic
    with IRBuilder() as ib:
        with I_builder.ir_module():
            with T_builder.prim_func():
                T_builder.func_name("main")
                scalar_vars = []
                for i in range(num_int_params):
                    v = T_builder.arg(f"scale{i}", tvm.tir.Var("", dtype))
                    scalar_vars.append(v)
                var_A = T_builder.arg("var_A", T_builder.handle())
                var_B = T_builder.arg("var_B", T_builder.handle())
                T_builder.func_attr({"tir.noalias": True})
                n_var = T_builder.int32(is_size_var=True)
                A = T_builder.match_buffer(var_A, (n_var,), dtype)
                B = T_builder.match_buffer(var_B, (n_var,), dtype)
                scalar_sum = scalar_vars[0]
                for s in scalar_vars[1:]:
                    scalar_sum = scalar_sum + s
                with T_builder.thread_binding(
                    tvm.tir.ceildiv(n_var, 64), thread="blockIdx.x"
                ) as i_0:
                    with T_builder.thread_binding(64, thread="threadIdx.x") as i_1:
                        with T_builder.sblock("B"):
                            v_i = T_builder.axis.spatial(n_var, i_0 * 64 + i_1)
                            T_builder.where(i_0 * 64 + i_1 < n_var)
                            T_builder.reads(A[v_i])
                            T_builder.writes(B[v_i])
                            T_builder.buffer_store(B, scalar_sum + A[v_i], [v_i])
    mod = ib.get()
    f_add = tvm.compile(mod, target=target)

    n = 1024
    scalars = np.array([1 for _ in range(num_int_params)]).astype(dtype)
    a = tvm.runtime.tensor(np.random.uniform(size=n).astype(dtype), dev)
    b = tvm.runtime.tensor(np.zeros(n, dtype=dtype), dev)
    f_add(*scalars, a, b)

    tvm.testing.assert_allclose(a.numpy() + sum(scalars), b.numpy())


def test_vulkan_while_if(target, dev):
    target = tvm.target.Target(target)
    n = 1
    dtype = "int32"

    def get_module(is_gpu):
        if is_gpu:

            @T.prim_func
            def while_if_gpu(A: T.Buffer((1,), "int32"), B: T.Buffer((1,), "int32")):
                for bx in T.thread_binding(1, thread="blockIdx.x"):
                    iterations = T.decl_buffer((1,), "int32", scope="local")
                    iterations[0] = 0
                    B[0] = 0
                    while iterations[0] < T.if_then_else(A[0] > 0, 10, 20):
                        iterations[0] = iterations[0] + 1
                        B[0] = B[0] + iterations[0]

            return tvm.IRModule.from_expr(while_if_gpu.with_attr("target", target))
        else:

            @T.prim_func
            def while_if_cpu(A: T.Buffer((1,), "int32"), B: T.Buffer((1,), "int32")):
                iterations = T.decl_buffer((1,), "int32", scope="local")
                iterations[0] = 0
                B[0] = 0
                while iterations[0] < T.if_then_else(A[0] > 0, 10, 20):
                    iterations[0] = iterations[0] + 1
                    B[0] = B[0] + iterations[0]

            return tvm.IRModule.from_expr(while_if_cpu.with_attr("target", target))

    mod = get_module("gpu" in target.keys)
    compiled_func = tvm.compile(mod, target=target)

    a = tvm.runtime.tensor(np.array([5], dtype=dtype), dev)
    b = tvm.runtime.tensor(np.zeros(n, dtype=dtype), dev)
    compiled_func(a, b)
    tvm.testing.assert_allclose(b.numpy(), [55])

    a = tvm.runtime.tensor(np.array([-5], dtype=dtype), dev)
    b = tvm.runtime.tensor(np.zeros(n, dtype=dtype), dev)
    compiled_func(a, b)
    tvm.testing.assert_allclose(b.numpy(), [210])


@tvm.testing.exclude_targets("llvm")
def test_vulkan_local_threadidx(target, dev):
    target = tvm.target.Target(target)
    n = 32

    @T.prim_func
    def local_threadidx_func(A: T.Buffer((32,), "int32"), B: T.Buffer((32,), "int32")):
        # First block with thread extent 16
        for _ in range(1):
            for tx in T.thread_binding(16, thread="threadIdx.x"):
                B[tx + 0] = A[tx + 0]
        # Second block with thread extent 16
        for _ in range(1):
            for tx in T.thread_binding(16, thread="threadIdx.x"):
                B[tx + 16] = A[tx + 16]

    mod = tvm.IRModule.from_expr(local_threadidx_func)
    func = tvm.compile(mod, target=target)

    a_np = np.arange(n).astype(dtype="int32")
    b_np = np.zeros((n,), dtype="int32")
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    func(a, b)
    tvm.testing.assert_allclose(b.numpy(), a_np)


@tvm.testing.parametrize_targets("vulkan -from_device=0")
def test_vectorized_index_ramp(target, dev):
    """Test vectorized copy with ramp indices (load N values, write to N locations)"""
    n = 4
    ramp_index = tvm.tir.Ramp(0, 1, 4)

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tir.noalias": True})
            A = T.match_buffer(var_A, (n,), "int32", offset_factor=1)
            B = T.match_buffer(var_B, (n,), "int32", offset_factor=1)
            with T.sblock("compute"):
                T.reads()
                T.writes()
                bx = T.launch_thread("blockIdx.x", 1)
                B[ramp_index] = A[ramp_index]

    f = tvm.compile(Module, target=target)

    a_np = np.random.randint(np.iinfo("int32").max, size=n).astype("int32")
    b_np = np.zeros(n, dtype="int32")

    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), a_np)


@tvm.testing.parametrize_targets("vulkan -from_device=0")
def test_vectorized_index_broadcast(target, dev):
    """Test broadcast index (load 1 value, write to N locations)"""
    n = 4
    broadcast_index = tvm.tir.Broadcast(0, 4)
    ramp_index = tvm.tir.Ramp(0, 1, 4)

    @I.ir_module
    class Module:
        @T.prim_func
        def main(var_A: T.handle, var_B: T.handle):
            T.func_attr({"tir.noalias": True})
            A = T.match_buffer(var_A, (n,), "int32", offset_factor=1)
            B = T.match_buffer(var_B, (n,), "int32", offset_factor=1)
            with T.sblock("compute"):
                T.reads()
                T.writes()
                bx = T.launch_thread("blockIdx.x", 1)
                # Load from broadcast index (single element), store to ramp index
                B[ramp_index] = A[broadcast_index]

    f = tvm.compile(Module, target=target)

    a_np = np.random.randint(np.iinfo("int32").max, size=n).astype("int32")
    b_np = np.zeros(n, dtype="int32")

    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    f(a, b)
    # All elements of b should be a[0] (broadcast load)
    tvm.testing.assert_allclose(b.numpy(), np.full(n, a_np[0]))


def test_negative_operand_divmod(target, dev):
    """Test handling of negative offsets to floormod/floordiv

    Even though the SPIR-V spec states that OpSRem and OpSMod can give
    the signed modulo, the Vulkan spec states that any use of negative
    operands is undefined behavior.  This test starts with negative
    operands to floordiv, validating that they are simplified into the
    corresponding positive operands, such that the final TIR can be
    expressed using only positive operands.

    SPIR-V: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSRem
    Vulkan: https://registry.khronos.org/vulkan/specs/1.3/html/chap37.html#spirvenv-op-prec
    """

    N = 32
    offset = 16
    divisor = 5

    if "gpu" in tvm.target.Target(target).keys:

        @T.prim_func
        def func(A: T.Buffer((N, 2), "int32")):
            for i in T.thread_binding(N, thread="threadIdx.x"):
                with T.sblock("A"):
                    v_i = T.axis.spatial(N, i)
                    A[v_i, 0] = T.floordiv(v_i - offset, divisor)
                    A[v_i, 1] = T.floormod(v_i - offset, divisor)

    else:

        @T.prim_func
        def func(A: T.Buffer((N, 2), "int32")):
            for i in T.serial(N):
                with T.sblock("A"):
                    v_i = T.axis.spatial(N, i)
                    A[v_i, 0] = T.floordiv(v_i - offset, divisor)
                    A[v_i, 1] = T.floormod(v_i - offset, divisor)

    built = tvm.compile(func, target=target)

    a_dev = tvm.runtime.empty([N, 2], "int32", dev)
    built(a_dev)
    a = a_dev.numpy()

    np.testing.assert_array_equal(a[:, 0], (np.arange(N) - offset) // divisor)
    np.testing.assert_array_equal(a[:, 1], (np.arange(N) - offset) % divisor)


@pytest.mark.parametrize("out_dtype", ["float32", "float16"])
def test_cooperative_matrix(out_dtype):
    M, N, K = 16, 16, 32

    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def main(X: T.Buffer((16, 32), "float16"), W: T.Buffer((32, 16), "float16"), compute: T.Buffer((16, 16), out_dtype)):
            T.func_attr({"tir.noalias": True})
            X_shared = T.alloc_buffer((16, 32), "float16", scope="shared")
            W_shared = T.alloc_buffer((32, 16), "float16", scope="shared")
            X_shared_wmma_matrix_a = T.alloc_buffer((16, 32), "float16", scope="wmma.matrix_a")
            W_shared_wmma_matrix_b = T.alloc_buffer((32, 16), "float16", scope="wmma.matrix_b")
            compute_wmma_accumulator = T.alloc_buffer((16, 16), out_dtype, scope="wmma.accumulator")
            for i_0_j_0_fused in T.thread_binding(1, thread="blockIdx.x"):
                with T.sblock("compute_init_o"):
                    v_i_o = T.axis.spatial(1, 0)
                    v_j_o = T.axis.spatial(1, 0)
                    T.reads()
                    T.writes(compute_wmma_accumulator[0:16, 0:16])
                    C = T.match_buffer(compute_wmma_accumulator[0:16, 0:16], (16, 16), out_dtype, strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                    T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.float32(0.0))
                for k_0 in range(2):
                    for ax0_ax1_fused_0 in range(2):
                        for ax0_ax1_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                            for ax0_ax1_fused_2 in T.vectorized(4):
                                with T.sblock("X_shared"):
                                    v0 = T.axis.spatial(16, (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 16)
                                    v1 = T.axis.spatial(32, k_0 * 16 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 16)
                                    T.reads(X[v0, v1])
                                    T.writes(X_shared[v0, v1])
                                    X_shared[v0, v1] = X[v0, v1]
                    for ax0_ax1_fused_0 in range(2):
                        for ax0_ax1_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                            for ax0_ax1_fused_2 in T.vectorized(4):
                                with T.sblock("W_shared"):
                                    v0 = T.axis.spatial(32, k_0 * 16 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 16)
                                    v1 = T.axis.spatial(16, (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 16)
                                    T.reads(W[v0, v1])
                                    T.writes(W_shared[v0, v1])
                                    W_shared[v0, v1] = W[v0, v1]
                    for ax0_0 in T.unroll(1):
                        for ax1_0 in T.unroll(1):
                            with T.sblock("X_shared_wmma.matrix_a_o"):
                                v0_o = T.axis.spatial(1, ax0_0)
                                v1_o = T.axis.spatial(2, k_0 + ax1_0)
                                T.reads(X_shared[0:16, v1_o * 16:v1_o * 16 + 16])
                                T.writes(X_shared_wmma_matrix_a[0:16, v1_o * 16:v1_o * 16 + 16])
                                A = T.match_buffer(X_shared[0:16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared", offset_factor=16)
                                C = T.match_buffer(X_shared_wmma_matrix_a[0:16, v1_o * 16:v1_o * 16 + 16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "row_major")
                    for ax0_0 in T.unroll(1):
                        for ax1_0 in T.unroll(1):
                            with T.sblock("W_shared_wmma.matrix_b_o"):
                                v0_o = T.axis.spatial(2, k_0 + ax0_0)
                                v1_o = T.axis.spatial(1, ax1_0)
                                T.reads(W_shared[v0_o * 16:v0_o * 16 + 16, 0:16])
                                T.writes(W_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, 0:16])
                                A = T.match_buffer(W_shared[v0_o * 16:v0_o * 16 + 16, 0:16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="shared", offset_factor=16)
                                C = T.match_buffer(W_shared_wmma_matrix_b[v0_o * 16:v0_o * 16 + 16, 0:16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * 16, 1), A.strides[0], "row_major")
                    with T.sblock("compute_update_o"):
                        v_i_o = T.axis.spatial(1, 0)
                        v_j_o = T.axis.spatial(1, 0)
                        v_k_o = T.axis.reduce(2, k_0)
                        T.reads(compute_wmma_accumulator[0:16, 0:16], X_shared_wmma_matrix_a[0:16, v_k_o * 16:v_k_o * 16 + 16], W_shared_wmma_matrix_b[v_k_o * 16:v_k_o * 16 + 16, 0:16])
                        T.writes(compute_wmma_accumulator[0:16, 0:16])
                        A = T.match_buffer(X_shared_wmma_matrix_a[0:16, v_k_o * 16:v_k_o * 16 + 16], (16, 16), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                        B = T.match_buffer(W_shared_wmma_matrix_b[v_k_o * 16:v_k_o * 16 + 16, 0:16], (16, 16), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                        C = T.match_buffer(compute_wmma_accumulator[0:16, 0:16], (16, 16), out_dtype, strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                        T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16, A.data, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, B.data, B.elem_offset // B.strides[0] // 16 * (B.strides[0] // 16) + B.elem_offset % B.strides[0] // 16, C.data, C.elem_offset // C.strides[0] // 16 * (C.strides[0] // 16) + C.elem_offset % C.strides[0] // 16)
                with T.sblock("compute_wmma.accumulator_o"):
                    v0_o = T.axis.spatial(1, 0)
                    v1_o = T.axis.spatial(1, 0)
                    T.reads(compute_wmma_accumulator[0:16, 0:16])
                    T.writes(compute[0:16, 0:16])
                    A = T.match_buffer(compute_wmma_accumulator[0:16, 0:16], (16, 16), out_dtype, strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                    C = T.match_buffer(compute[0:16, 0:16], (16, 16), out_dtype, strides=("C_s0", "C_s1"), offset_factor=16)
                    T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // 16 * (A.strides[0] // 16) + A.elem_offset % A.strides[0] // 16, T.tvm_access_ptr(T.type_annotation(out_dtype), C.data, C.elem_offset, C.strides[0] * 16, 2), C.strides[0], "row_major")
    # fmt: on

    target = "vulkan -from_device=0"
    tgt_attrs = tvm.target.Target(target).attrs

    if tgt_attrs.get("supports_cooperative_matrix"):
        f = tvm.compile(Module, target=target)

        dev = tvm.device(target, 0)

        A = tvm.runtime.tensor(np.random.randn(M, K).astype("float16"), dev)
        B = tvm.runtime.tensor(np.random.randn(K, N).astype("float16"), dev)
        C = tvm.runtime.tensor(np.random.randn(M, N).astype(out_dtype), dev)

        f(A, B, C)

        A_np = A.numpy()
        B_np = B.numpy()
        ref = np.dot(A_np.astype("float32"), B_np.astype("float32"))

        tvm.testing.assert_allclose(C.numpy(), ref, rtol=1e-2, atol=1e-2)


@tvm.testing.requires_vulkan(support_required="compile-only")
def test_codegen_decl_buffer():
    """The codegen should accept DeclBuffer nodes in its input"""

    @I.ir_module
    class Module:
        @T.prim_func
        def kernel():
            T.func_attr({"calling_conv": 2, "global_symbol": "kernel", "tir.noalias": True})
            A_data = T.allocate([256], dtype="float32", scope="local")
            A_buf = T.decl_buffer([256], dtype="float32", scope="local", data=A_data)

    target = tvm.target.Target("vulkan")
    vulkan_codegen = tvm.get_global_func("target.build.vulkan")
    vulkan_codegen(Module, target)


@tvm.testing.requires_gpu
@tvm.testing.requires_vulkan
def test_unary():
    test_funcs = [
        (tvm.tir.sin, lambda x: np.sin(x)),
        (tvm.tir.cos, lambda x: np.cos(x)),
        (tvm.tir.tan, lambda x: np.tan(x)),
        (tvm.tir.sinh, lambda x: np.sinh(x)),
        (tvm.tir.cosh, lambda x: np.cosh(x)),
        (tvm.tir.tanh, lambda x: np.tanh(x)),
        (tvm.tir.asin, lambda x: np.arcsin(x)),
        (tvm.tir.acos, lambda x: np.arccos(x)),
        (tvm.tir.atan, lambda x: np.arctan(x)),
        (tvm.tir.asinh, lambda x: np.arcsinh(x)),
        (tvm.tir.acosh, lambda x: np.arccosh(x)),
        (tvm.tir.atanh, lambda x: np.arctanh(x)),
    ]

    def run_test(tvm_intrin, np_func):
        n = 16

        @I.ir_module
        class Module:
            @T.prim_func
            def main(var_A: T.handle, var_B: T.handle):
                m = T.int32(is_size_var=True)
                A = T.match_buffer(var_A, (m,), "float32")
                B = T.match_buffer(var_B, (m,), "float32")
                for i_0 in T.thread_binding((m + 63) // 64, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(64, thread="threadIdx.x"):
                        with T.sblock("B"):
                            v_i = T.axis.spatial(m, i_0 * 64 + i_1)
                            T.where(i_0 * 64 + i_1 < m)
                            T.reads(A[v_i])
                            T.writes(B[v_i])
                            B[v_i] = tvm_intrin(A[v_i])

        target = tvm.target.Target("vulkan")
        dev = tvm.device(target.kind.name, 0)
        func = tvm.compile(Module, target=target)

        if tvm_intrin in [tvm.tir.asin, tvm.tir.acos]:
            data = np.random.uniform(-1.0, 1.0, size=n)
        elif tvm_intrin == tvm.tir.atanh:
            data = np.random.uniform(-0.999, 0.999, size=n)
        elif tvm_intrin == tvm.tir.acosh:
            data = np.random.uniform(1.0, 5.0, size=n)
        else:
            data = np.random.uniform(0.1, 0.9, size=n)

        a = tvm.runtime.tensor(data.astype("float32"), dev)
        b = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        func(a, b)
        tvm.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func)


if __name__ == "__main__":
    tvm.testing.main()
