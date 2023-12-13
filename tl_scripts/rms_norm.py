import torch
import tvm.tl.language as T

from tvm.tl.engine import lower
from tvm.tl.utils import ConvertTorch, TensorSupplyType


def rms_norm_splitk(M, N, blk_m, blk_k):
    dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, blk_k), dtype)
            A_local = T.alloc_fragment((blk_m, blk_k), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            num_k_step = T.ceildiv(N, blk_k)
            T.clear(A_local)
            for k in range(num_k_step):
                T.copy(A[bx * blk_m, k * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_local[i, j] += A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12

            for k in range(num_k_step):
                # reverse, better cache hit rate
                T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
                for i, j in T.Parallel(blk_m, blk_k):
                    A_shared[i, j] *= A_powsum[i]
                T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return main


def rms_norm(M, N, blk_m):
    dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_shared)
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] = A_shared[i, j] * A_shared[i, j]
            T.reduce_sum(A_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N) + 1e-12
            for i, j in T.Parallel(blk_m, N):
                A_shared[i, j] *= A_powsum[i]
            T.copy(A_shared, B[bx * blk_m : (bx + 1) * blk_m, :])

    return main


def ref_program(x):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)
    return [y]


if __name__ == "__main__":
    M, N, blk_m, blk_k = 8192, 8192, 1, 512
    # program = rms_norm_splitk(M, N, blk_m, blk_k)
    program = rms_norm(M, N, blk_m)
    mod, params = lower(program)

    supply_type = TensorSupplyType.Normal
    mod = ConvertTorch(mod, params, [1], supply_type)
    print(mod.get_kernel_source())
    mod.assert_allclose(ref_program)

    latency = mod.do_bench(ref_program, warmup=500)
    print("Torch: {:.2f} ms".format(latency))
    torch.cuda.synchronize()
    latency = mod.do_bench(mod.func)
    print("TVMTL: {:.2f} ms".format(latency))
    torch.cuda.synchronize()
