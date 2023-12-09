import torch
import tvm.tl.language as T

from tvm.tl.engine import lower
from tvm.tl.utils import ConvertTorch, TensorSupplyType


def reduce_sum(M, N, block_M, block_N):
    dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer([M], dtype)):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            A_local = T.alloc_fragment((block_M, block_N), dtype)
            B_local = T.alloc_fragment((block_M,), dtype)
            T.clear(B_local)
            T.clear(A_local)
            for i in T.Pipelined(T.ceildiv(N, block_N), num_stages=0):
                T.copy(A[bx * block_M, i * block_N], A_shared)
                for i, j in T.Parallel(block_M, block_N):
                    A_local[i, j] += A_shared[i, j]
            T.reduce_sum(A_local, B_local, dim=1)
            T.copy(B_local, B[bx * block_M])

    return main


def ref_program(A):
    B = torch.sum(A, dim=1)
    return [B]


if __name__ == "__main__":
    M, N, block_M, block_N = 8192, 8192, 64, 128
    program = reduce_sum(M, N, block_M, block_N)
    mod, params = lower(program)

    supply_type = TensorSupplyType.Integer
    mod = ConvertTorch(mod, params, [1], supply_type)
    print(mod.get_kernel_source())
    mod.assert_allclose(ref_program)

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    latency = mod.do_bench(mod.func)
    print("{:.2f} ms".format(latency))
