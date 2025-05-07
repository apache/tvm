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

from itertools import product

import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

native_dtype, promoted_dtype = tvm.testing.parameters(
    ("float4_e2m1fnx2", "float32x2"),
    ("float4_e2m1fnx2", "float16x2"),
)


@tvm.testing.requires_cuda_compute_version(10)
def test_e2m1_vector_conversions(native_dtype, promoted_dtype):
    vector_length = 64

    @T.prim_func
    def add(
        A: T.Buffer((vector_length,), native_dtype),
        B: T.Buffer((vector_length,), native_dtype),
        C: T.Buffer((vector_length,), native_dtype),
    ):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i in range(vector_length):
            with T.block("C"):
                v_i = T.axis.spatial(vector_length, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = T.Cast(
                    native_dtype, T.Cast(promoted_dtype, A[v_i]) + T.Cast(promoted_dtype, B[v_i])
                )

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    fadd = tvm.compile(sch.mod, target=target)
    dev = tvm.device(target, 0)

    numpytype = "float4_e2m1fn"
    if "x" in native_dtype:
        lanes = int(native_dtype.split("x")[-1])
    else:
        lanes = 1

    if "x" in promoted_dtype:
        promoted_base_dtype = promoted_dtype.split("x")[0]
    else:
        promoted_base_dtype = promoted_dtype

    np_shape = (vector_length, lanes) if lanes > 1 else (vector_length,)
    a_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    a = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    b_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    b = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype(promoted_base_dtype), (a_np + b_np).astype(promoted_base_dtype)
    )


@tvm.testing.requires_cuda_compute_version(10)
def test_e2m1_schedule_vectorize():
    native_dtype = "float4_e2m1fn"
    n = 128

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)
    for promoted_dtype, vector_length in product(
        ["float16", "bfloat16", "float32"],
        [1, 2, 4],
    ):

        @T.prim_func
        def add(
            A: T.Buffer((n,), native_dtype),
            B: T.Buffer((n,), native_dtype),
            C: T.Buffer((n,), native_dtype),
        ):
            T.func_attr({"tir.noalias": True})
            for i in range(n):
                with T.block("C"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = T.Cast(
                        native_dtype,
                        T.Cast(promoted_dtype, A[v_i]) + T.Cast(promoted_dtype, B[v_i]),
                    )

        sch = tvm.tir.Schedule(add)
        block = sch.get_block("C")
        b = sch.get_loops(block)
        bx, tx, vec = sch.split(b[0], factors=[None, 32, vector_length])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

        fadd = tvm.compile(sch.mod, target=target)

        numpytype = "float4_e2m1fn"
        promoted_base_dtype = promoted_dtype

        a_np = np.random.uniform(low=-6, high=6, size=(n,)).astype(numpytype)
        a = tvm.nd.empty(shape=(n,), dtype=native_dtype, device=dev)
        a.copyfrom(a_np)
        b_np = np.random.uniform(low=-6, high=6, size=(n,)).astype(numpytype)
        b = tvm.nd.empty(shape=(n,), dtype=native_dtype, device=dev)
        b.copyfrom(b_np)
        c = tvm.nd.empty(shape=(n,), dtype=native_dtype, device=dev)
        fadd(a, b, c)

        if promoted_base_dtype != "bfloat16":
            tvm.testing.assert_allclose(
                c.numpy().astype(promoted_base_dtype), (a_np + b_np).astype(promoted_base_dtype)
            )
        else:
            # assert_allclose with bfloat16 throws an error here.
            # Thus we convert bfloat16 to float32 for comparison.
            tvm.testing.assert_allclose(
                c.numpy().astype(promoted_base_dtype).astype("float32"),
                (a_np + b_np).astype(promoted_base_dtype).astype("float32"),
            )


@tvm.testing.requires_cuda_compute_version(10)
def test_e2m1_reinterpret():
    n = 128

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)

    def get_reinterpret_mod(src_dtype, dst_dtype, vector_length):
        @T.prim_func
        def reinterpret(
            A: T.Buffer((n,), src_dtype),
            B: T.Buffer((n,), dst_dtype),
        ):
            T.func_attr({"tir.noalias": True})
            for i in range(n):
                with T.block("C"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = T.reinterpret(dst_dtype, A[v_i])

        sch = tvm.tir.Schedule(reinterpret)
        block = sch.get_block("C")
        b = sch.get_loops(block)
        bx, tx, vec = sch.split(b[0], factors=[None, 32, vector_length])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)
        return sch.mod

    # Part 1. reinterpret float4_e2m1fn to uint8
    for vector_length in [1, 2, 4]:
        mod = get_reinterpret_mod("float4_e2m1fn", "uint8", vector_length)
        f = tvm.compile(mod, target=target)
        a_np = np.random.uniform(low=-6, high=6, size=(n,)).astype("float4_e2m1fn")
        a = tvm.nd.empty(shape=(n,), dtype="float4_e2m1fn", device=dev)
        a.copyfrom(a_np)
        b = tvm.nd.empty(shape=(n,), dtype="uint8", device=dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), a_np.view("uint8"))

    # Part 2. reinterpret uint8 to float4_e2m1fn
    for vector_length in [1, 2, 4]:
        mod = get_reinterpret_mod("uint8", "float4_e2m1fn", vector_length)
        f = tvm.compile(mod, target=target)
        a_np = np.random.uniform(low=-6, high=6, size=(n,)).astype("uint8")
        a = tvm.nd.empty(shape=(n,), dtype="uint8", device=dev)
        a.copyfrom(a_np)
        b = tvm.nd.empty(shape=(n,), dtype="float4_e2m1fn", device=dev)
        f(a, b)
        tvm.testing.assert_allclose(
            b.numpy().astype("float32"), a_np.view("float4_e2m1fn").astype("float32")
        )


@tvm.testing.requires_cuda_compute_version(10)
def test_e2m1_dequantize():
    n = 128

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)
    num_elem_per_storage = 32 // 4

    def get_reinterpret_mod(func_type, vector_length):
        @T.prim_func
        def shuffle_reinterpret(
            A: T.Buffer((n // num_elem_per_storage,), "uint32"),
            B: T.Buffer((n,), "float16"),
        ):
            T.func_attr({"tir.noalias": True})
            for i in range(n):
                with T.block("C"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = T.Shuffle(
                        [
                            T.reinterpret(
                                "float4_e2m1fnx2",
                                T.bitwise_and(
                                    T.shift_right(
                                        A[v_i // num_elem_per_storage],
                                        ((v_i % num_elem_per_storage) // 2 * 4 * 2).astype(
                                            "uint32"
                                        ),
                                    ),
                                    T.uint32((1 << (4 * 2)) - 1),
                                ).astype("uint8"),
                            ).astype("float16x2")
                        ],
                        indices=[v_i % 2],
                    )

        @T.prim_func
        def scalar_reinterpret(
            A: T.Buffer((n // num_elem_per_storage,), "uint32"),
            B: T.Buffer((n,), "float16"),
        ):
            T.func_attr({"tir.noalias": True})
            for i in range(n):
                with T.block("C"):
                    v_i = T.axis.spatial(n, i)
                    T.reads(A[v_i])
                    T.writes(B[v_i])
                    B[v_i] = T.reinterpret(
                        "float4_e2m1fn",
                        T.bitwise_and(
                            T.shift_right(
                                A[v_i // num_elem_per_storage],
                                (v_i % num_elem_per_storage * 4).astype("uint32"),
                            ),
                            T.uint32((1 << 4) - 1),
                        ).astype("uint8"),
                    ).astype("float16")

        func = shuffle_reinterpret if func_type == "shuffle" else scalar_reinterpret
        sch = tvm.tir.Schedule(func)
        block = sch.get_block("C")
        b = sch.get_loops(block)
        bx, tx, vec = sch.split(b[0], factors=[None, 32, vector_length])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)
        return sch.mod

    # We only test the whether the code can be compiled.
    for func_type, vector_length in product(["shuffle", "scalar"], [1, 2, 4]):
        if func_type == "shuffle" and vector_length == 1:
            # Vectorize is necessary for shuffle.
            continue
        mod = get_reinterpret_mod(func_type, vector_length)
        tvm.compile(mod, target=target)


if __name__ == "__main__":
    tvm.testing.main()
