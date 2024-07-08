import argparse
import torch
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
import itertools

swizzle_M = 4096
swizzle_N = 4096

def get_configs():
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [32, 64, 128]
    num_stages = [1, 2, 3, 4]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages))

    configs = [
        {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3]}
        for c in _configs
    ]
    return configs

@autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages'], warmup=10, rep=5)
@jit
def matmul(M, N, K, block_M, block_N, block_K, num_stages):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, swizzle_M) * T.ceildiv(N, swizzle_N), T.ceildiv(swizzle_N, block_N), T.ceildiv(swizzle_M, block_M), threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[(bx % T.ceildiv(M, swizzle_M)) * swizzle_M + bz * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, T.floordiv(bx, T.ceildiv(N, swizzle_N)) * swizzle_N + by * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[(bx % T.ceildiv(M, swizzle_M)) * swizzle_M + bz * block_M, T.floordiv(bx, T.ceildiv(N, swizzle_N)) * swizzle_N + by * block_N])

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
    best_latency, best_config = matmul(M, N, K, block_M, block_N, block_K, 0)
    print(f"best_latency: {best_latency}")
    print(f"best_config: {best_config}")
    # program = matmul(M, N, K, block_M, block_N, block_K)
    # mod, params = tl.lower(program)

    # mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)
    # mod.assert_allclose(ref_program)

    # latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=5)
    # # latency = mod.do_bench(ref_program, warmup=500)
    # print("torch: {:.2f} ms".format(latency))
    # print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    # latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=5)
    # latency = mod.do_bench(mod.func, warmup=500)
    # print("tl: {:.2f} ms".format(latency))
    # print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))
