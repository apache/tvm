import torch
import tvm.tl.language as T

from tvm.tl.engine import lower
from tvm.tl.utils import ConvertTorch, TensorSupplyType


def transpose(M, N):
    dtype = "float"
    BLK = 64

    @T.prim_func
    def main(ins: T.Buffer((M, N), dtype), outs: T.Buffer((N, M), dtype)):
        with T.Kernel(T.ceildiv(M, BLK), T.ceildiv(N, BLK), threads=256) as (bx, by):
            shared = T.alloc_shared((BLK, BLK), dtype)
            local = T.alloc_fragment((BLK, BLK), dtype)
            local_t = T.alloc_fragment((BLK, BLK), dtype)
            T.annotate_layout(
                {
                    # pad by 4
                    shared: T.Layout(shared.shape, lambda i, j: i * (BLK + 4) + j),
                    # assign 4x4 float tile to each thread
                    local: T.Fragment(local.shape, lambda i, j: j // 4 + 16 * (i // 4)),
                }
            )
            T.copy(ins[BLK * by, BLK * bx], shared)
            T.copy(shared, local)
            for i, j in T.Parallel(BLK, BLK):
                local_t[i, j] = local[j, i]
            T.copy(local_t, shared)
            T.copy(shared, outs[BLK * bx, BLK * by])

    return main


def ref_program(A):
    B = A.T.contiguous()
    return [B]


if __name__ == "__main__":
    M, N = 8192, 8192
    program = transpose(M, N)
    mod, params = lower(program)

    supply_type = TensorSupplyType.Integer
    mod = ConvertTorch(mod, params, [1], supply_type)
    print(mod.get_kernel_source())
    mod.assert_allclose(ref_program)

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    latency = mod.do_bench(mod.func)
    print("{:.2f} ms".format(latency))
