import torch
import tvm.tl.language as T

from tvm.tl.engine import compile
from tvm.tl.utils import ConvertTorch, TensorSupplyType

def matmul(M, N, K, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"
    @T.prim_func
    def main(A: T.Buffer((M, K), dtype), B: T.Buffer((K, N), dtype),
             C: T.Buffer((M, N), dtype)):
        bx, by, _ = T.launch_program(T.ceildiv(N, block_N), T.ceildiv(M, block_M), num_threads=128)

        with T.block():
            A_shared = T.alloc_buffer((block_M, block_K), dtype, scope="shared.dyn")
            B_shared = T.alloc_buffer((block_K, block_N), dtype, scope="shared.dyn")
            C_local = T.alloc_buffer((block_M, block_N), accum_dtype, scope="local.fragment")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by*block_M, k*block_K], A_shared)
                T.copy(B[k*block_K, bx*block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by*block_M, bx*block_N])

    return main

def ref_program(A, B):
    C = torch.matmul(A, B)
    return [C]

if __name__ == "__main__":
    M, N, K, block_M, block_N, block_K = 8192, 8192, 8192, 128, 128, 32
    total_flops = 2 * M * N * K
    program = matmul(M, N, K, block_M, block_N, block_K)
    mod, params = compile(program)

    supply_type = TensorSupplyType.Integer
    mod = ConvertTorch(mod, params, [2], supply_type)
    mod.assert_allclose(ref_program)

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod.func)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
