import argparse
import torch
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
from functools import partial
import itertools

def get_configs():
    block_M = [32, 64, 128]
    block_N = [32, 64, 128]
    num_stages = [1, 2]
    thread_num = [128, 256]
    _configs = list(itertools.product(block_M, block_N, num_stages, thread_num))

    configs = [
        {'block_M': c[0], 'block_N': c[1], 'num_stages': c[2], 'thread_num': c[3]}
        for c in _configs
    ]
    return configs

def ref_program(Q, K, V, casual):
    from flash_attn.flash_attn_interface import flash_attn_func

    return flash_attn_func(Q, K, V, causal=casual)

def flashattn(batch, heads, seq_len, dim, is_casual):

    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[3], supply_type=tl.TensorSupplyType.Normal, ref_prog=partial(ref_program, casual=is_casual), rtol=0.01, atol=0.01)
    def kernel(block_M = None, block_N = None, num_stages = None, thread_num = None):
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        shape = [batch, seq_len, heads, dim]
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
            Q: T.Buffer(shape, dtype), # type: ignore
            K: T.Buffer(shape, dtype), # type: ignore
            V: T.Buffer(shape, dtype), # type: ignore
            Output: T.Buffer(shape, dtype), # type: ignore
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=thread_num) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
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

                T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
                T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.copy(Q_shared, Q_local)
                for i, j in T.Parallel(block_M, dim):
                    Q_local[i, j] *= scale
                loop_range = (
                    T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
                )
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                    if is_casual:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
                            )
                    else:
                        T.clear(acc_s)
                    T.gemm(Q_local, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
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
                T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

        return main
    return kernel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--h', type=int, default=12, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=2048, help='Context size')
    parser.add_argument('--d_head', type=int, default=256, help='Head dimension')
    parser.add_argument('--casual', type=bool, default=True, help='Casual flag')
    args = parser.parse_args()
    BATCH, H, N_CTX, D_HEAD = args.batch, args.h, args.n_ctx, args.d_head
    casual = args.casual
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5

    best_latency, best_config, ref_latency = flashattn(BATCH, H, N_CTX, D_HEAD, casual)
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")
    print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")