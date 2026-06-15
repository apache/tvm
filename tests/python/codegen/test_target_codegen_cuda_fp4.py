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
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env

try:
    from ml_dtypes import float4_e2m1fn

    ML_DTYPES_AVAILABLE = True
except ImportError:
    ML_DTYPES_AVAILABLE = False


@pytest.mark.parametrize("promoted_dtype", ["float32x2", "float16x2"])
@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_e2m1_vector_conversions(promoted_dtype):
    native_dtype = "float4_e2m1fnx2"
    vector_length = 64

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((vector_length,), native_dtype),
            B: T.Buffer((vector_length,), native_dtype),
            C: T.Buffer((vector_length,), native_dtype),
        ):
            T.func_attr({"tirx.noalias": True})
            for i_0 in T.thread_binding(vector_length // 32, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.sblock("C"):
                        v_i = T.axis.spatial(vector_length, i_0 * 32 + i_1)
                        T.reads(A[v_i], B[v_i])
                        T.writes(C[v_i])
                        C[v_i] = T.Cast(
                            native_dtype,
                            T.Cast(promoted_dtype, A[v_i]) + T.Cast(promoted_dtype, B[v_i]),
                        )

    target = "cuda"
    fadd = tvm.compile(Module, target=target)
    dev = tvm.device(target, 0)

    if "x" in native_dtype:
        lanes = int(native_dtype.split("x")[-1])
    else:
        lanes = 1

    if "x" in promoted_dtype:
        promoted_base_dtype = promoted_dtype.split("x")[0]
    else:
        promoted_base_dtype = promoted_dtype

    np_shape = (vector_length, lanes) if lanes > 1 else (vector_length,)

    # Create test data - either using ml_dtypes if available, or using int8 with valid FP4 values
    if ML_DTYPES_AVAILABLE:
        a_np = np.random.uniform(low=0, high=5, size=np_shape).astype(float4_e2m1fn)
        b_np = np.random.uniform(low=0, high=5, size=np_shape).astype(float4_e2m1fn)
    else:
        # float4_e2m1fn possible values: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        # We will create int8 arrays with valid FP4 bit patterns
        valid_fp4_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 4-bit values
        a_np = np.random.choice(valid_fp4_values, size=np_shape).astype(np.int8)
        b_np = np.random.choice(valid_fp4_values, size=np_shape).astype(np.int8)

    a = tvm.runtime.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    b = tvm.runtime.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.runtime.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    fadd(a, b, c)

    # For the comparison, we will convert result to the promoted dtype and compare
    # Note: When ml_dtypes is not available, we skip the numpy-level computation comparison
    # and just verify that the CUDA kernel compiles and executes without error
    c_result = c.numpy().astype(promoted_base_dtype)

    if ML_DTYPES_AVAILABLE:
        # Full comparison when ml_dtypes is available
        expected = (a_np + b_np).astype(promoted_base_dtype)
        tvm.testing.assert_allclose(c_result, expected)
    else:
        # When ml_dtypes is not available, we just verify the comparison ran successfully
        # by checking that we got a result with the expected shape and dtype
        assert c_result.shape == np_shape
        assert c_result.dtype == promoted_base_dtype


def _shuffle_reinterpret_module(n, num_blocks, vector_length, num_elem_per_storage):
    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((n // num_elem_per_storage,), "uint32"),
            B: T.Buffer((n,), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i_0 in T.thread_binding(num_blocks, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    for i_2 in T.vectorized(vector_length):
                        with T.sblock("C"):
                            v_i = T.axis.spatial(
                                n, i_0 * 32 * vector_length + i_1 * vector_length + i_2
                            )
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

    return Module


def _scalar_reinterpret_module(n, num_blocks, vector_length, num_elem_per_storage):
    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((n // num_elem_per_storage,), "uint32"),
            B: T.Buffer((n,), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i_0 in T.thread_binding(num_blocks, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    for i_2 in T.vectorized(vector_length):
                        with T.sblock("C"):
                            v_i = T.axis.spatial(
                                n, i_0 * 32 * vector_length + i_1 * vector_length + i_2
                            )
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

    return Module


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_e2m1_dequantize():
    n = 128

    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)
    num_elem_per_storage = 32 // 4

    # We only test the whether the code can be compiled.
    for func_type, vector_length in product(["shuffle", "scalar"], [1, 2, 4]):
        if func_type == "shuffle" and vector_length == 1:
            # Vectorize is necessary for shuffle.
            continue

        num_blocks = n // (32 * vector_length)

        if func_type == "shuffle":
            mod = _shuffle_reinterpret_module(n, num_blocks, vector_length, num_elem_per_storage)
        else:
            mod = _scalar_reinterpret_module(n, num_blocks, vector_length, num_elem_per_storage)

        tvm.compile(mod, target=target)


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_e2m1_scalar_buffer_offset():
    """Regression test: float4_e2m1fn scalar buffer access uses correct byte offset.

    In CUDA sizeof(__nv_fp4_e2m1) = 1 byte, but fp4 data packs 2 elements per
    byte.  GetBufferRef must emit ``index / 2`` so that the element index is
    converted to the correct byte offset.  Without the fix the index was used
    as-is, producing addresses 2x too large — reading garbage from out-of-bounds
    memory instead of the correct fp4 value.

    We verify by writing known fp4 values, casting each element to float16 on
    the GPU, and checking the results match the expected fp4->fp16 conversion.
    """
    n = 128

    @T.prim_func(s_tir=True)
    def func(A_raw: T.Buffer((n // 2,), "uint8"), B: T.Buffer((n,), "float16")):
        T.func_attr({"tir.noalias": True})
        A = T.decl_buffer((n,), "float4_e2m1fn", data=A_raw.data)
        for i in range(n):
            with T.sblock("B"):
                vi = T.axis.spatial(n, i)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = T.Cast("float16", A[vi])

    sch = tvm.s_tir.Schedule(func)
    block = sch.get_sblock("B")
    loops = sch.get_loops(block)
    bx, tx = sch.split(loops[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    dev = tvm.device(target, 0)
    fadd = tvm.compile(sch.mod, target=target)

    # float4_e2m1fn: 4-bit values 0..15, two packed per byte.
    # Encoding (sign | exp1 | man1 man0):
    #   0→0.0  1→0.5  2→1.0  3→1.5  4→2.0  5→3.0  6→4.0  7→6.0
    #   8→-0.0 9→-0.5 10→-1.0 … 15→-6.0
    fp4_to_fp16 = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float16,
    )

    # Pack DIFFERENT fp4 values in low/high nibbles so the test verifies
    # both byte offset (/2) AND correct nibble extraction (% 2 shift).
    fp4_elements = np.array([i % 16 for i in range(n)], dtype=np.uint8)
    packed = np.zeros(n // 2, dtype=np.uint8)
    for i in range(0, n, 2):
        packed[i // 2] = fp4_elements[i] | (fp4_elements[i + 1] << 4)

    expected = fp4_to_fp16[fp4_elements]

    a = tvm.runtime.empty(shape=(n // 2,), dtype="uint8", device=dev)
    a.copyfrom(packed)
    b = tvm.runtime.empty(shape=(n,), dtype="float16", device=dev)
    fadd(a, b)

    result = b.numpy()
    tvm.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    tvm.testing.main()
