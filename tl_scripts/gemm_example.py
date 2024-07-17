import argparse
import torch
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
import itertools

targetHopper = True
swizzle_M = 4096
swizzle_N = 4096


def ref_program(A, B):
    return A @ B

def get_configs():
    block_M = [128]
    block_N = [128, 256]
    block_K = [64]
    num_stages = [2]
    thread_num = [256]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, thread_num))

    configs = [
        {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3], 'thread_num': c[4]}
        for c in _configs
    ]
    return configs

def matmul(M, N, K):
    
    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Integer, ref_prog=ref_program)
    def kernel(block_M = None, block_N = None, block_K = None, num_stages = None, thread_num = None):
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N), dtype)): # type: ignore
            with T.Kernel(T.ceildiv(M, swizzle_M) * T.ceildiv(N, swizzle_N), T.ceildiv(swizzle_N, block_N), T.ceildiv(swizzle_M, block_M), threads=thread_num) as (bx, by, bz):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[(bx % T.ceildiv(M, swizzle_M)) * swizzle_M + bz * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, T.floordiv(bx, T.ceildiv(N, swizzle_N)) * swizzle_N + by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[(bx % T.ceildiv(M, swizzle_M)) * swizzle_M + bz * block_M, T.floordiv(bx, T.ceildiv(N, swizzle_N)) * swizzle_N + by * block_N])

        @T.prim_func
        def main_hopper(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N), dtype)): # type: ignore
            with T.Kernel(T.ceildiv(M, swizzle_M) * T.ceildiv(N, swizzle_N), T.ceildiv(swizzle_N, block_N), T.ceildiv(swizzle_M, block_M), threads=thread_num) as (bx, by, bz):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[(bx % T.ceildiv(M, swizzle_M)) * swizzle_M + bz * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, T.floordiv(bx, T.ceildiv(N, swizzle_N)) * swizzle_N + by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                if targetHopper:
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[(bx % T.ceildiv(M, swizzle_M)) * swizzle_M + bz * block_M, T.floordiv(bx, T.ceildiv(N, swizzle_N)) * swizzle_N + by * block_N])

        if targetHopper:
            return main_hopper
        else:
            return main
    return kernel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=8192, help='M')
    parser.add_argument('--n', type=int, default=8192, help='N')
    parser.add_argument('--k', type=int, default=8192, help='K')
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    total_flops = 2 * M * N * K
    best_latency, best_config, ref_latency = matmul(M, N, K)
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")
    print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")