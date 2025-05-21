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

import os
from posixpath import split
import random
import re
import threading

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te, tir
from tvm.topi.math import cast
from tvm.script import tir as T, ir as I
from tvm.tir import TensorIntrin, IntImm, Cast, Schedule
from tvm.tir.tensor_intrin.cuda import (
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
    WMMA_FILL_16x16x16_F16_INTRIN,
    WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN,
)


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
    n = 1024
    A = te.placeholder((n,), dtype=dtype, name="A")
    B = te.compute(
        A.shape,
        lambda i: tvm.tir.Select(
            A[i] >= 0, A[i] + tvm.tir.const(1, dtype), tvm.tir.const(0, dtype)
        ),
        name="B",
    )

    # Create IRModule
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
    sch = tir.Schedule(mod)
    (bx, tx) = sch.split(sch.get_loops("B")[0], factors=[None, 128])
    (tx, vx) = sch.split(tx, factors=[None, 4])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.vectorize(vx)

    # Build
    f = tvm.tir.build(sch.mod, target=target)

    # Verify we generate the boolx4 type declaration and the OpSelect
    # v4{float,half,int} instruction
    assembly = f.imported_modules[0].get_source()
    matches = re.findall("%v4bool = OpTypeVector %bool 4", assembly)
    assert len(matches) == 1
    matches = re.findall("OpSelect %v4.*", assembly)
    assert len(matches) == 1


def test_array_copy(dev, dtype, fuzz_seed):
    np.random.seed(fuzz_seed)

    log_arr_size = np.random.uniform(low=np.log(1), high=np.log(32768))
    arr_size = np.exp(log_arr_size).astype(int)
    a_np = np.random.uniform(size=(arr_size,)).astype(dtype)
    a = tvm.nd.empty((arr_size,), dtype, dev).copyfrom(a_np)
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

    A = te.placeholder((arr_size,), name="A", dtype="%sx%d" % (dtype, lanes))
    B = te.compute(A.shape, lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")

    sch = tir.Schedule(te.create_prim_func([A, B]))
    xo, xi = sch.split(sch.get_loops("B")[0], factors=[None, 4])
    sch.bind(xo, "blockIdx.x")
    sch.bind(xi, "threadIdx.x")
    f = tvm.compile(sch.mod, target=target)

    a = tvm.nd.empty((arr_size,), A.dtype, dev).copyfrom(np.random.uniform(size=(arr_size, lanes)))
    c = tvm.nd.empty((arr_size,), B.dtype, dev)
    f(a, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)


@tvm.testing.exclude_targets("llvm")
def test_vulkan_bool_load(target, dev):
    target = tvm.target.Target(target)
    arr_size = 1024
    A = te.placeholder((arr_size,), name="A", dtype="bool")
    B = te.compute(A.shape, lambda i: A[i].astype("int32"), name="B")

    sch = tir.Schedule(te.create_prim_func([A, B]))
    xo, xi = sch.split(sch.get_loops("B")[0], factors=[None, 128])
    sch.bind(xo, "blockIdx.x")
    sch.bind(xi, "threadIdx.x")

    # Build
    f = tvm.compile(sch.mod, target=target)

    a_np = np.random.uniform(size=arr_size) > 0.5
    b_np = np.zeros((arr_size,), dtype="int32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
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

    n = te.var("n")
    scalars = [te.var("scale{}".format(i), dtype=dtype) for i in range(num_int_params)]
    scalar_sum = scalars[0]
    for s in scalars[1:]:
        scalar_sum += s

    A = te.placeholder((n,), name="A", dtype=dtype)
    B = te.compute(A.shape, lambda i: scalar_sum + A[i], name="B")

    sch = tvm.tir.Schedule(te.create_prim_func(scalars + [A, B]))
    xo, xi = sch.split(sch.get_loops("B")[0], factors=[None, 64])
    sch.bind(xo, "blockIdx.x")
    sch.bind(xi, "threadIdx.x")
    f_add = tvm.compile(sch.mod, target=target)

    n = 1024
    scalars = np.array([1 for _ in scalars]).astype(dtype)
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
    f_add(*scalars, a, b)

    tvm.testing.assert_allclose(a.numpy() + sum(scalars), b.numpy())


def test_vulkan_while_if(target, dev):
    target = tvm.target.Target(target)
    n = 1
    dtype = "int32"
    A = te.placeholder((n,), name="A", dtype=dtype)

    def do_compute(A, B, n):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)

        if "gpu" in target.keys:
            ib.scope_attr(te.thread_axis("blockIdx.x"), "thread_extent", 0)

        iterations = ib.allocate("int32", (1,), name="iterations", scope="local")
        iterations[0] = 0
        B[0] = 0

        loop_condition = iterations[0] < tvm.tir.if_then_else(A[0] > 0, 10, 20)
        with ib.while_loop(loop_condition):
            iterations[0] += 1
            B[0] += iterations[0]

        return ib.get()

    B = te.extern(
        A.shape,
        [A],
        lambda ins, outs: do_compute(ins[0], outs[0], n),
        dtype=dtype,
    )

    # Create IRModule
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
    sch = tir.Schedule(mod)

    # Build
    func = tvm.compile(sch.mod, target=target)

    a = tvm.nd.array(np.array([5], dtype=A.dtype), dev)
    b = tvm.nd.array(np.zeros(n, dtype=A.dtype), dev)
    func(a, b)
    tvm.testing.assert_allclose(b.numpy(), [55])

    a = tvm.nd.array(np.array([-5], dtype=A.dtype), dev)
    b = tvm.nd.array(np.zeros(n, dtype=A.dtype), dev)
    func(a, b)
    tvm.testing.assert_allclose(b.numpy(), [210])


@tvm.testing.exclude_targets("llvm")
def test_vulkan_local_threadidx(target, dev):
    target = tvm.target.Target(target)
    n = 32
    A = te.placeholder((n,), name="A", dtype="int32")

    def do_compute(A, B, n):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)

        tx = te.thread_axis("threadIdx.x")

        with ib.for_range(0, 1):
            ib.scope_attr(tx, "thread_extent", 16)
            B[tx + 0] = A[tx + 0]

        with ib.for_range(0, 1):
            ib.scope_attr(tx, "thread_extent", 16)
            B[tx + 16] = A[tx + 16]

        return ib.get()

    B = te.extern(
        A.shape,
        [A],
        lambda ins, outs: do_compute(ins[0], outs[0], n),
        dtype="int32",
    )

    # Create IRModule
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
    sch = tir.Schedule(mod)

    # Build
    func = tvm.compile(sch.mod, target=target)

    n = 32
    a_np = np.arange(n).astype(dtype=A.dtype)
    b_np = np.zeros((n,), dtype="int32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    func(a, b)
    tvm.testing.assert_allclose(b.numpy(), a_np)


class TestVectorizedIndices:
    load_type, store_type = tvm.testing.parameters(
        # Load N values, write to N locations.
        # Vectorized copy.
        ("ramp", "ramp"),
        # Load 1 value, write to N locations.
        # Scalar load, vectorized store.
        #
        # Most TVM operations (e.g. schedule[tensor].vectorize(axis)) have
        # the broadcast outside of the index, but it is semantically okay
        # for the broadcast to be inside the index, and it shows up with
        # some optimizations.
        ("broadcast", "ramp"),
        # Load 1 values, write to 1 location.
        # Broadcasting on both sides should be equivalent to a scalar copy.
        ("broadcast", "broadcast"),
        # Loads N values, write to 1 location.
        # Disabled as it would have unclear semantics.
        # ("ramp","broadcoast"),
    )
    indirect_indices = tvm.testing.parameter(True, False, ids=["reorder", "no_reorder"])

    @tvm.testing.fixture
    def ref_data(self, load_type, store_type, indirect_indices):
        n = 4

        index_map = {
            "ramp": np.arange(n),
            "broadcast": np.zeros(n, dtype="int32"),
        }

        a_np = np.random.randint(np.iinfo("int32").max, size=n).astype("int32")
        b_np = np.zeros(shape=n, dtype=a_np.dtype)
        reorder_np = np.arange(n, dtype="int32")[::-1]

        load_index = index_map[load_type]
        store_index = index_map[store_type]

        if indirect_indices:
            load_index = reorder_np[load_index]

        b_np[store_index] = a_np[load_index]

        return a_np, reorder_np, b_np

    @tvm.testing.fixture
    def mod(self, target, load_type, store_type, indirect_indices):
        target = tvm.target.Target(target)

        n = 4
        dtype = "int32"
        A = te.placeholder((n,), dtype=dtype, name="A")
        R = te.placeholder((n,), dtype=dtype, name="R")

        def do_compute(ins, outs):
            ib = tvm.tir.ir_builder.create()
            A, R = map(ib.buffer_ptr, ins)
            B = ib.buffer_ptr(outs[0])

            if "gpu" in target.keys:
                ib.scope_attr(te.thread_axis("blockIdx.x"), "thread_extent", 0)

            index_map = {
                "ramp": tvm.tir.Ramp(0, 1, 4),
                "broadcast": tvm.tir.Broadcast(0, 4),
            }

            load_index = index_map[load_type]
            store_index = index_map[store_type]

            if indirect_indices:
                load_index = R[load_index]

            B[store_index] = A[load_index]

            return ib.get()

        B = te.extern(A.shape, [A, R], do_compute, dtype="int32")

        return tvm.IRModule.from_expr(te.create_prim_func([A, R, B]))

    def test_ramp_broadcast_index(self, target, dev, mod, ref_data):
        f = tvm.compile(mod, target=target)

        a_np, reorder_np, b_np = ref_data
        a = tvm.nd.array(a_np, dev)
        r = tvm.nd.array(reorder_np, dev)
        b = tvm.nd.array(np.zeros(shape=b_np.shape, dtype="int32"), dev)
        f(a, r, b)
        tvm.testing.assert_allclose(b.numpy(), b_np)


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

    @T.prim_func
    def func(A: T.Buffer((N, 2), "int32")):
        for i in T.serial(N):
            with T.block("A"):
                v_i = T.axis.spatial(N, i)
                A[v_i, 0] = T.floordiv(v_i - offset, divisor)
                A[v_i, 1] = T.floormod(v_i - offset, divisor)

    if "gpu" in tvm.target.Target(target).keys:
        sch = tvm.tir.Schedule(func)
        sch.bind(sch.get_loops("A")[0], "threadIdx.x")
        func = sch.mod["main"]

    built = tvm.compile(func, target=target)

    a_dev = tvm.nd.empty([N, 2], "int32", dev)
    built(a_dev)
    a = a_dev.numpy()

    np.testing.assert_array_equal(a[:, 0], (np.arange(N) - offset) // divisor)
    np.testing.assert_array_equal(a[:, 1], (np.arange(N) - offset) % divisor)


@pytest.mark.parametrize("out_dtype", ["float32", "float16"])
def test_cooperative_matrix(out_dtype):
    def get_matmul(m, n, k, out_dtype="float32"):
        X = te.placeholder((m, k), name="X", dtype="float16")
        W = te.placeholder((k, n), name="W", dtype="float16")
        ak = te.reduce_axis((0, k), name="k")

        if out_dtype == "float32":
            matmul = te.compute(
                (m, n),
                lambda i, j: te.sum(
                    X[i, ak].astype("float32") * W[ak, j].astype("float32"),
                    axis=ak,
                ),
                name="compute",
            )
        else:
            matmul = te.compute(
                (m, n),
                lambda i, j: te.sum(X[i, ak] * W[ak, j], axis=ak),
                name="compute",
            )

        return te.create_prim_func([X, W, matmul])

    M, N, K = 16, 16, 32
    func = get_matmul(M, N, K, out_dtype)
    sch = Schedule(func)
    block = sch.get_block("compute")

    i, j, k = sch.get_loops(block)
    i_outer, i_inner = sch.split(i, factors=[None, 16])
    j_outer, j_inner = sch.split(j, factors=[None, 16])
    k_outer, k_inner = sch.split(k, factors=[None, 16])
    sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)
    fused_outer = sch.fuse(i_outer, j_outer)
    sch.bind(fused_outer, "blockIdx.x")

    def fetch_to_shared(block, idx):
        block_read = sch.cache_read(block, idx, "shared")
        sch.compute_at(block_read, k_outer)
        warp_size = 32

        fused = sch.fuse(*sch.get_loops(block_read)[-2:])

        vector_size = 4
        _, f_2, f_3 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f_2, "threadIdx.x")
        sch.vectorize(f_3)

    def tensorize_load(block, dim):
        loops = sch.get_loops(block)
        i, j = loops[-dim : (len(loops) - dim + 2)]

        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)
        sch.unroll(i0)
        sch.unroll(j0)
        return i1

    fetch_to_shared(block, 0)
    fetch_to_shared(block, 1)

    c_warp_scope = "wmma.accumulator"
    a_warp_scope = "wmma.matrix_a"
    b_warp_scope = "wmma.matrix_b"

    A_mat = sch.cache_read(block, 0, a_warp_scope)
    B_mat = sch.cache_read(block, 1, b_warp_scope)

    loop_a = tensorize_load(A_mat, 2)
    sch.tensorize(loop_a, WMMA_LOAD_16x16x16_F16_A_INTRIN)

    loop_b = tensorize_load(B_mat, 2)
    sch.tensorize(loop_b, WMMA_LOAD_16x16x16_F16_B_INTRIN)

    store = sch.cache_write(block, 0, c_warp_scope)
    sch.reverse_compute_at(store, fused_outer)
    init = sch.decompose_reduction(block, sch.get_loops(block)[1])

    intrin = WMMA_FILL_16x16x16_F32_INTRIN
    if out_dtype == "float16":
        intrin = WMMA_FILL_16x16x16_F16_INTRIN
    sch.tensorize(sch.get_loops(init)[1], intrin)

    intrin = WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN
    if out_dtype == "float16":
        intrin = WMMA_STORE_16x16x16_F16_GLOBAL_INTRIN
    sch.tensorize(sch.get_loops(store)[1], intrin)

    intrin = WMMA_SYNC_16x16x16_f16f16f32_INTRIN
    if out_dtype == "float16":
        intrin = WMMA_SYNC_16x16x16_f16f16f16_INTRIN
    sch.tensorize(sch.get_loops(block)[2], intrin)

    target = "vulkan -from_device=0"
    tgt_attrs = tvm.target.Target(target).attrs

    if tgt_attrs.get("supports_cooperative_matrix"):
        f = tvm.compile(sch.mod, target=target)

        dev = tvm.device(target, 0)

        A = tvm.nd.array(np.random.randn(M, K).astype("float16"), dev)
        B = tvm.nd.array(np.random.randn(K, N).astype("float16"), dev)
        C = tvm.nd.array(np.random.randn(M, N).astype(out_dtype), dev)

        f(A, B, C)

        A_np = A.numpy()
        B_np = B.numpy()
        ref = np.dot(A_np.astype("float32"), B_np.astype("float32"))

        tvm.testing.assert_allclose(C.numpy(), ref, rtol=1e-2, atol=1e-2)


@tvm.testing.requires_vulkan(support_required="compile-only")
def test_codegen_decl_buffer():
    """The codegen should accept DeclBuffer nodes in its input"""

    @I.ir_module
    class mod:
        @T.prim_func
        def kernel():
            T.func_attr({"calling_conv": 2, "global_symbol": "kernel", "tir.noalias": True})
            A_data = T.allocate([256], dtype="float32", scope="local")
            A_buf = T.decl_buffer([256], dtype="float32", scope="local", data=A_data)

    target = tvm.target.Target("vulkan")
    vulkan_codegen = tvm.get_global_func("target.build.vulkan")
    vulkan_codegen(mod, target)


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
        m = te.var("m")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda *i: tvm_intrin(A(*i)), name="B")

        mod = te.create_prim_func([A, B])
        sch = tir.Schedule(mod)

        block = sch.get_block("B")
        loop = sch.get_loops(block)[0]
        bx, tx = sch.split(loop, factors=[None, 64])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        target = tvm.target.Target("vulkan")
        dev = tvm.device(target.kind.name, 0)
        func = tvm.compile(sch.mod, target=target)

        n = 16
        if tvm_intrin in [tvm.tir.asin, tvm.tir.acos]:
            data = np.random.uniform(-1.0, 1.0, size=n)
        elif tvm_intrin == tvm.tir.atanh:
            data = np.random.uniform(-0.999, 0.999, size=n)
        elif tvm_intrin == tvm.tir.acosh:
            data = np.random.uniform(1.0, 5.0, size=n)
        else:
            data = np.random.uniform(0.1, 0.9, size=n)

        a = tvm.nd.array(data.astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=A.dtype), dev)
        func(a, b)
        tvm.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func)


if __name__ == "__main__":
    tvm.testing.main()
