import torch
from tvm import tl
import tvm.tl.language as T

def matmul(M, N, K, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype), C: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128 * 2) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_local = T.alloc_fragment((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.annotate_layout({
                A_shared: tl.layout.make_swizzled_layout(A_shared)
            })

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(A_shared, A_local)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_local, B_shared, C_local, policy=T.GemmWarpPolicy.FullCol)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

M, N, K, block_M, block_N, block_K = 8192, 512 * 16, 8192, 64, 512, 64

def ref_program(A, B):
    return A @ B

if __name__ == "__main__":
    total_flops = 2 * M * N * K

    program = matmul(M, N, K, block_M, block_N, block_K)
    mod, params = tl.lower(program)

    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")

    # latency = mod.do_bench(ref_program, warmup=500)
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))