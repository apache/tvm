import argparse
import torch
from tvm import tl
import tvm.tl.language as T
from functools import partial


def retnet(batch, heads, seq_len, dim, is_casual, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape, dtype),
        V: T.Buffer(shape, dtype),
        Output: T.Buffer(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            Q_local = T.alloc_fragment([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            lse = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.fill(scores_max_prev, -T.infinity(accum_dtype))
            T.fill(lse, -T.infinity(accum_dtype))
            T.copy(Q_shared, Q_local)
            loop_range = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_local, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max_prev[i], scores_max[i])
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    lse[i] = scores_max[i] + T.log(T.exp(lse[i] - scores_max[i]) + scores_sum[i])
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] = acc_o[i, j] * T.exp(scores_max_prev[i] - scores_max[i])
                T.copy(scores_max, scores_max_prev)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= T.exp(scores_max[i] - lse[i])
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def ref_program(Q, K, V, casual):
    qk = torch.matmul(Q.permute(0, 2, 1, 3), K.permute(0, 2, 3, 1)) # [B, H, SEQLEN, SEQLEN]
    m = qk.max(dim=-1, keepdim=True).values
    p = torch.exp(qk - m)
    s = p / p.sum(dim=-1, keepdim=True)
    o = torch.matmul(s.to(torch.float16), V.permute(0, 2, 1, 3)) # [B, H, SEQLEN, dim]
    return o.permute(0, 2, 1, 3)


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
    BLOCK_M = 64
    BLOCK_N = 64 if D_HEAD <= 128 else 32
    program = retnet(BATCH, H, N_CTX, D_HEAD, casual, BLOCK_M, BLOCK_N)
    ref_program = partial(ref_program, casual=casual)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [3], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.1, atol=0.1)

    # latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=1)
    # print("torch: {:.2f} ms".format(latency))
    # print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=5)
    print("tl: {:.2f} ms".format(latency))
    print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))
