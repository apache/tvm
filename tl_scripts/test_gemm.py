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

import tvm
import tvm.testing
from tvm import tl


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tvm.tl.language as T

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, in_dtype), B: T.Buffer(B_shape, in_dtype), C: T.Buffer((M, N), out_dtype)
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    mod.assert_allclose(ref_program)


def test_gemm_f16f16f16_nn():
    run_gemm(512, 1024, 768, False, False, "float16", "float16", "float16", 128, 256, 32, 2)


def test_gemm_f16f16f32_nn():
    run_gemm(512, 1024, 768, False, False, "float16", "float16", "float32", 128, 128, 32)


def test_gemm_bf16bf16f32_nn():
    run_gemm(512, 1024, 768, False, False, "bfloat16", "bfloat16", "float32", 128, 128, 32)


def test_gemm_f32f32f32_nn():
    run_gemm(512, 1024, 768, False, False, "float32", "float32", "float32", 64, 128, 32)


def test_gemm_f64f64f64_nn():
    run_gemm(512, 1024, 768, False, False, "float64", "float64", "float64", 64, 64, 16)


def test_gemm_i8i8i32_nn():
    run_gemm(512, 1024, 768, False, False, "int8", "int8", "int32", 128, 128, 64)


def test_gemm_f16f16f16_tn():
    run_gemm(512, 1024, 768, True, False, "float16", "float16", "float16", 128, 256, 32, 2)


def test_gemm_f16f16f16_nt():
    run_gemm(512, 1024, 768, False, True, "float16", "float16", "float16", 128, 256, 32, 2)


def test_gemm_i8i8i32_nt():
    run_gemm(512, 1024, 768, False, True, "int8", "int8", "int32", 128, 128, 64)


def test_gemm_i8i8i32_tn():
    run_gemm(512, 1024, 768, True, False, "int8", "int8", "int32", 128, 128, 64)


def test_gemm_f64f64f64_nt():
    run_gemm(512, 1024, 768, False, True, "float64", "float64", "float64", 64, 32, 16)


def test_gemm_f64f64f64_tn():
    run_gemm(512, 1024, 768, True, False, "float64", "float64", "float64", 64, 32, 16)


def test_gemm_f32f32f32_nt():
    run_gemm(512, 1024, 768, False, True, "float32", "float32", "float32", 64, 128, 32)


def test_gemm_f32f32f32_tn():
    run_gemm(512, 1024, 768, True, False, "float32", "float32", "float32", 64, 128, 32)


def test_pad_aligned_f16f16f16_nn():
    run_gemm(
        512 - 8, 1024 - 32, 768 - 24, False, False, "float16", "float16", "float16", 128, 256, 32, 2
    )


def test_pad_f16f16f16_nn():
    run_gemm(
        512 - 9, 1024 - 7, 768 - 5, False, False, "float16", "float16", "float16", 128, 256, 32, 2
    )


def test_pad_f16f16f32_nn():
    run_gemm(
        512 + 19, 1024 + 17, 768 + 15, False, False, "float16", "float16", "float32", 128, 64, 32
    )


if __name__ == "__main__":
    tvm.testing.main()
