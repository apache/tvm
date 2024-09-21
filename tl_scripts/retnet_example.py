import argparse
import torch
from tvm import tl
import tvm.tl.language as T
from functools import partial


def retnet(batch, heads, seq_len, dim, block_M, block_N):
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape, dtype),
        V: T.Buffer(shape, dtype),
        mask: T.Buffer([heads, seq_len, seq_len], dtype),
        Output: T.Buffer(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128 * 1) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            mask_shared = T.alloc_shared([block_M, block_N], dtype)
            acc_o_shared = T.alloc_shared([block_M, dim], dtype)
            mask_local = T.alloc_fragment([block_M, block_N], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            abs_sum = T.alloc_fragment([block_M], accum_dtype)
            r_wo_clamp = T.alloc_fragment([block_M], accum_dtype)
            r = T.alloc_fragment([block_M], accum_dtype)
            r_new = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)

            T.fill(r, 0)
            T.fill(r_new, 0)
            T.fill(r_wo_clamp, 0)
            T.fill(acc_o, 0)
            loop_range = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, 
                                 num_stages=1, 
                                 order=[-1,0,-1,-1,1,2], 
                                 stage=[-1,0,-1,-1,0,0], 
                                 group=[[0],[1,2],[3],[4],[5,6,7,8,9,10,11,12,13], [14]]
                                ):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.copy(mask[by, bx * block_M : (bx + 1) * block_M, k * block_N : (k + 1) * block_N], mask_shared)
                T.copy(mask_shared, mask_local)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = acc_s[i, j] * mask_local[i, j]
                T.reduce_abssum(acc_s, abs_sum, dim=1)
                for i in T.Parallel(block_M):
                    r_wo_clamp[i] = r_wo_clamp[i] + abs_sum[i]
                for i in T.Parallel(block_M):
                    r_new[i] = T.max(r_wo_clamp[i], 1)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] = T.if_then_else(k > 0, acc_o[i, j] * r[i] / r_new[i], acc_o[i, j])
                T.copy(r_new, r)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = acc_s[i, j] / r_new[i]
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            T.copy(acc_o, acc_o_shared)
            T.copy(acc_o_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def ref_program(Q, K, V, mask):
    Q = Q.to(dtype=float)
    K = K.to(dtype=float)
    V = V.to(dtype=float)
    mask = mask.to(dtype=float)
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    qkm = qk * mask
    r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o = torch.einsum('bhqk,bkhd->bqhd', qkm/r, V)
    return o.to(dtype=torch.float16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--h', type=int, default=32, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=4096, help='Context size')
    parser.add_argument('--d_head', type=int, default=128, help='Head dimension')
    args = parser.parse_args()
    BATCH, H, N_CTX, D_HEAD = args.batch, args.h, args.n_ctx, args.d_head
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    BLOCK_M = 64
    BLOCK_N = 64
    program = retnet(BATCH, H, N_CTX, D_HEAD, BLOCK_M, BLOCK_N)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [4], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)

    latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=1)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10)
    print("tl: {:.2f} ms".format(latency))
    print("tl: {:.2f} TFlops".format(total_flops / latency * 1e-9))
