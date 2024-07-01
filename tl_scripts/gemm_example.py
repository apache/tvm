import argparse
import torch
from tvm import tl
import tvm.tl.language as T


def matmul(M, N, K, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def ref_program(A, B):
    return A @ B


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=8192, help='M')
    parser.add_argument('--n', type=int, default=8192, help='N')
    parser.add_argument('--k', type=int, default=8192, help='K')
    parser.add_argument('--bm', type=int, default=128, help='blockM')
    parser.add_argument('--bn', type=int, default=128, help='blockN')
    parser.add_argument('--bk', type=int, default=32, help='blockK')
    args = parser.parse_args()
    M, N, K, block_M, block_N, block_K = args.m, args.n, args.k, args.bm, args.bn, args.bk
    total_flops = 2 * M * N * K
    program = matmul(M, N, K, block_M, block_N, block_K)
    mod, params = tl.lower(program)

    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)
    mod.assert_allclose(ref_program)

    latency = mod.do_bench(ref_program, warmup=500)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod.func, warmup=500)
    print("tl: {:.2f} ms".format(latency))
    print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))
