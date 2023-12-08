import torch
import tvm.tl.language as T

from tvm.tl.engine import lower
from tvm.tl.utils import ConvertTorch, TensorSupplyType
from functools import partial


def flashattn(batch_size, num_head, seq_len, dim, is_casual, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch_size, seq_len, num_head, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape, dtype),
        V: T.Buffer(shape, dtype),
        Output: T.Buffer(shape, dtype),
    ):
        bx, by, bz, _ = T.launch_program(
            num_head, T.ceildiv(seq_len, block_M), batch_size, num_threads=128
        )

        with T.block():
            Q_local = T.alloc_fragment([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by * block_M : (by + 1) * block_M, bx, :], Q_local)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            for i, j in T.Parallel(block_M, dim):
                Q_local[i, j] *= scale
            loop_range = (
                T.ceildiv((by + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, bx, :], K_shared)
                if is_casual:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            by * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                        )
                else:
                    T.clear(acc_s)
                T.gemm(Q_local, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, bx, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by * block_M : (by + 1) * block_M, bx, :])

    return main


def ref_program(Q, K, V, casual):
    from flash_attn.flash_attn_interface import flash_attn_func

    out = flash_attn_func(Q, K, V, causal=casual)
    return [out]


if __name__ == "__main__":
    BATCH, H, N_CTX, D_HEAD = 64, 12, 2048, 256
    casual = True
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    program = flashattn(BATCH, H, N_CTX, D_HEAD, casual, 64, 32)
    ref_program = partial(ref_program, casual=casual)
    mod, params = lower(program)
    supply_type = TensorSupplyType.Normal
    mod = ConvertTorch(mod, params, [3], supply_type)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
