import torch
from tvm import tl
import tvm.tl.language as T
from functools import partial

# Codegen bug:
#   LoadK should wait for MMA0 done
# @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
# def tvm_callback_cuda_postproc(code, _):
#     code = code.replace("""tl::mbarrier_wait(_mbarrier[1], ((k & 1) ^ 1));""", 
# """tl::mbarrier_wait(_mbarrier[1], ((k & 1))); // replace""")
#     code = code.replace("""tl::gemm_ss<64, 64, 64, 4, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(acc_s[0])));
#     #pragma unroll""", 
# """tl::gemm_ss<64, 64, 64, 4, 1, 0, 1>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(acc_s[0])));
#     tl::mbarrier_arrive(_mbarrier[1]);
#     #pragma unroll // replace""")
#     return code

# loadk(0)
# gemm0(0)
# loadk(1)
# softmax(0)
# loadv(0)

# for i in range(loop_range - 2):
#   gemm0(i+1)
#   gemm1(i+0)
#   loadk(i+2)
#   softmax(i+1)
#   loadv(i+1)

# gemm0(loop_range - 1)
# gemm1(loop_range - 2)
# softmax(loop_range - 1)
# loadv(loop_range - 1)
# gemm1(loop_range - 1)

def flashattn(batch, heads, seq_len, dim, is_casual, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "float16"
    accum_dtype = "float"

    @T.macro
    def MMA0(
        K: T.Buffer(shape, dtype),
        Q_shared: T.Buffer([block_M, dim], dtype),
        K_shared: T.Buffer([block_N, dim], dtype),
        acc_s: T.Buffer([block_M, block_N], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
        T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Buffer(shape, dtype),
        V_shared: T.Buffer([block_M, dim], dtype),
        acc_s_cast: T.Buffer([block_M, block_N], dtype),
        acc_o: T.Buffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.Buffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.Buffer([block_M, block_N], dtype),
        acc_o: T.Buffer([block_M, dim], accum_dtype),
        scores_max: T.Buffer([block_M], accum_dtype),
        scores_max_prev: T.Buffer([block_M], accum_dtype),
        scores_scale: T.Buffer([block_M], accum_dtype),
        scores_sum: T.Buffer([block_M], accum_dtype),
        logsum: T.Buffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_s[i, j] *= scale
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]
        T.copy(acc_s, acc_s_cast)

    @T.prim_func
    def main(
        Q: T.Buffer(shape, dtype),
        K: T.Buffer(shape, dtype),
        V: T.Buffer(shape, dtype),
        Output: T.Buffer(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
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

            MMA0(K, Q_shared, K_shared, acc_s, 0, by, bz)
            Softmax(acc_s, acc_s_cast, acc_o, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)

            loop_range = (
                T.ceildiv((bx + 1) * block_M, block_N) if is_casual else T.ceildiv(seq_len, block_N)
            )
            for k in T.Pipelined(loop_range, num_stages=1):
                if k < loop_range - 1:
                    MMA0(K, Q_shared, K_shared, acc_s, k + 1, by, bz)

                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                
                if k < loop_range - 1:
                    Softmax(acc_s, acc_s_cast, acc_o, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def ref_program(Q, K, V, casual):
    from flash_attn.flash_attn_interface import flash_attn_func

    return flash_attn_func(Q, K, V, causal=casual)


if __name__ == "__main__":
    BATCH, H, N_CTX, D_HEAD = 64, 16, 4096, 64
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

    latency = mod.do_bench(ref_program, warmup=500)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))