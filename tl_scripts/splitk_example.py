import torch
from tvm import tl
import tvm.tl.language as T


def matmul_splitk(M, N, K, blk_m, blk_n, block_K, split):
    dtype = "float16"
    accum_dtype = "float"
    assert K % (block_K * split) == 0
    KK = K // split

    @T.prim_func
    def main(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N), dtype)):
        SplitC = T.alloc_buffer(
            [split, (M + blk_m - 1) // blk_m * blk_m, (N + blk_n - 1) // blk_n * blk_n], dtype
        )
        with T.Kernel(T.ceildiv(N, blk_n), T.ceildiv(M, blk_m), split) as (bx, by, bz):
            A_shared = T.alloc_shared((blk_m, block_K), dtype)
            B_shared = T.alloc_shared((block_K, blk_n), dtype)
            C_local = T.alloc_fragment((blk_m, blk_n), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(K // (block_K * split), num_stages=3):
                T.copy(A[by * blk_m, KK * bz + k * block_K], A_shared)
                T.copy(B[KK * bz + k * block_K, bx * blk_n], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(
                C_local,
                SplitC[bz, by * blk_m : (by + 1) * blk_m, bx * blk_n : (bx + 1) * blk_n],
            )
        with T.Kernel(T.ceildiv(N, blk_n), T.ceildiv(M, blk_m)) as (bx, by):
            acc = T.alloc_fragment((blk_m, blk_n), dtype)
            T.clear(acc)
            for k in range(split):
                for i, j in T.Parallel(blk_m, blk_n):
                    acc[i, j] += SplitC[k, blk_m * by + i, blk_n * bx + j]
            T.copy(acc, C[by * blk_m, bx * blk_n])

    return main


def ref_program(A, B):
    return A @ B


if __name__ == "__main__":
    M, N, K, blk_m, blk_n, block_K, split = 8192, 8192, 8192, 128, 128, 32, 4
    total_flops = 2 * M * N * K
    program = matmul_splitk(M, N, K, blk_m, blk_n, block_K, split)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)
    mod.assert_allclose(ref_program)

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod.func)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
