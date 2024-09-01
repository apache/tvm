import torch
from tvm import tl
import tvm.tl.language as T
from functools import partial

# This script gives a wrong result when dim=64.
# The error is due to the acc_s_cast tensor reuse the register of Q_local tensor (don't know why).
# It is a strange error because in PTX file, the register of Q_local and acc_s_cast are different.
# To reproduce the error, you can try the following script:
# with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
#     Q_shared = T.alloc_shared([block_M, dim], dtype)
#     Q_local = T.alloc_fragment([block_M, dim], dtype)
#     K_shared = T.alloc_shared([block_N, dim], dtype)
#     V_shared = T.alloc_shared([block_N, dim], dtype)
#     acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
#     acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
#     acc_o = T.alloc_fragment([block_M, dim], accum_dtype)

#     T.annotate_layout({Q_shared: tl.layout.make_swizzled_layout(Q_shared)})
#     T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
#     T.fill(acc_o, 0)
#     T.copy(Q_shared, Q_local)
#     for i, j in T.Parallel(block_M, dim):
#         Q_local[i, j] *= scale
#     loop_range = (
#         T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
#     )
#     for k in T.Pipelined(loop_range, num_stages=1):
#         T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
#         if is_casual:
#             for i, j in T.Parallel(block_M, block_N):
#                 acc_s[i, j] = T.if_then_else(
#                     bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype)
#                 )
#         else:
#             T.clear(acc_s)
#         T.gemm(Q_local, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
#         T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
#         for i, j in T.Parallel(block_M, block_N):
#             acc_s[i, j] = T.exp2(acc_s[i, j] - 32)
#         T.copy(acc_s, acc_s_cast)
#         T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
#     T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

# To fix this, we can either use T.gemm(Q_shared, K_shared, acc_s), like in FlashAttention implementation,
# or use different wgmma instrutcion (like M64N32K16)

def flashattn(batch, heads, seq_len, dim, is_casual, block_M, block_N):
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
            for k in T.Pipelined(loop_range, num_stages=1):
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
                T.fill(scores_max, -T.infinity(accum_dtype))
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


def ref_program(Q, K, V, casual):
    # from flash_attn.flash_attn_interface import flash_attn_func

    # return flash_attn_func(Q, K, V, causal=casual)
    assert casual == False, "casual is not supported"
    batch, seq_len, heads, dim = Q.size()
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    block_M = seq_len
    block_N = 64 if dim <= 128 else 32
    acc_s = torch.empty((batch, heads, block_M, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, heads, block_M, block_N), device="cuda", dtype=torch.float16)
    acc_o = torch.empty((batch, block_M, heads, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
    scores_scale = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
    acc_o.fill_(0)
    logsum.fill_(0)
    scores_max.fill_(float('-inf'))
    Q_scaled = Q * scale

    for i in range(int(seq_len / block_N)):
        acc_s.fill_(0)
        acc_s = torch.einsum('bqhd,bkhd->bhqk', Q_scaled, K[:, i * block_N : (i + 1) * block_N, :, :]) # [batch, seqlen, heads, block_N]
        scores_max_prev = scores_max
        scores_max = acc_s.max(dim=-1, keepdim=False).values # [blockM]
        scores_scale = torch.exp2(scores_max_prev - scores_max)
        acc_o *= scores_scale[:, :, :, None].transpose(1, 2)
        acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
        acc_s_cast = acc_s.to(torch.float16)
        acc_o += torch.einsum('bhqk,bkhd->bqhd', acc_s_cast, V[:, i * block_N : (i + 1) * block_N, :, :])
        scores_sum = acc_s.sum(dim=-1, keepdim=False)
        logsum = logsum * scores_scale + scores_sum
    acc_o /= logsum[:, :, :, None].transpose(1, 2)
    return acc_o.to(torch.float16)

# def ref_program(Q, K, V, casual):
#     dim = Q.size(-1)
#     scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    
#     # Step 2: Scale the scores by the square root of dim
#     scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    
#     # Step 3: Apply softmax to get the attention weights
#     attention_weights = F.softmax(scores, dim=-1)
    
#     # Step 4: Multiply the attention weights by the values (V)
#     # This gives us the final output of shape [batch, seq_len, heads, dim]
#     output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    
#     return output

if __name__ == "__main__":
    BATCH, H, N_CTX, D_HEAD = 1, 1, 64, 64
    casual = False
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    BLOCK_M = 64
    BLOCK_N = 64 if D_HEAD <= 128 else 32
    program = flashattn(BATCH, H, N_CTX, D_HEAD, casual, BLOCK_M, BLOCK_N)
    ref_program = partial(ref_program, casual=casual)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [3], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)

    # latency = mod.do_bench(ref_program, warmup=500)
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    # latency = mod.do_bench(mod)
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))