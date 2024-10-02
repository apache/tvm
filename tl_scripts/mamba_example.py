import argparse
import torch
import torch.nn.functional as F
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
from functools import partial
from einops import rearrange, repeat
import triton
import itertools

chunk_size = 256

####################################################################################################
# bmm_chunk
####################################################################################################

# def bmm_chunk(batch, seqlen, ngroups, dstate, block_M = None, block_N = None, block_K = None, num_stages = None, thread_num = None):
#     dtype = "float16"
#     accum_dtype = "float"
#     nchunks = T.ceildiv(seqlen, chunk_size)
#     @T.prim_func
#     def main(
#         A: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
#         B: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
#         Output: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype)
#     ):
#         with T.Kernel(T.ceildiv(chunk_size, block_M) * T.ceildiv(chunk_size, block_N), batch, nchunks * ngroups, threads=thread_num) as (bx, by, bz):
#             A_shared = T.alloc_shared((block_M, block_K), dtype)
#             B_shared = T.alloc_shared((block_N, block_K), dtype)
#             acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
#             acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
#             chunk_idx = bz // ngroups
#             group_idx = bz % ngroups
#             m_idx = bx // T.ceildiv(chunk_size, block_N)
#             n_idx = bx % T.ceildiv(chunk_size, block_N)

#             # T.annotate_layout({acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)})

#             loop_range = T.ceildiv(dstate, block_K)
#             T.clear(acc_o)
#             for k in T.Pipelined(loop_range, num_stages=num_stages):
#                 T.copy(A[by, 
#                     chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, 
#                     group_idx, 
#                     k * block_K : (k + 1) * block_K], 
#                     A_shared)
#                 T.copy(B[by, 
#                     chunk_idx * chunk_size + n_idx * block_N : chunk_idx * chunk_size + (n_idx + 1) * block_N, 
#                     group_idx, 
#                     k * block_K : (k + 1) * block_K], 
#                     B_shared)
#                 T.gemm(A_shared, B_shared, acc_o, transpose_B=True)
#             T.copy(acc_o, acc_o_shared)
#             T.copy(acc_o_shared, Output[by, chunk_idx, group_idx, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N])

#     return main

# def bmm_triton(A, B):
#     from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
#     return _bmm_chunk_fwd(A, B, chunk_size)

# def bmm_ref_program(A, B):
#     seqlen = A.shape[1]
#     nchunks = (seqlen + chunk_size - 1) // chunk_size

#     A = rearrange(A, "b (c l) g d -> b c l g d", c=nchunks)
#     B = rearrange(B, "b (c l) g d -> b c l g d", c=nchunks)
#     return torch.einsum("bclgd,bcsgd->bcgls", A, B)

def bmm_chunk(batch, seqlen, ngroups, dstate):

    def bmm_ref_program(A, B):
        seqlen = A.shape[1]
        nchunks = (seqlen + chunk_size - 1) // chunk_size

        A = rearrange(A, "b (c l) g d -> b c l g d", c=nchunks)
        B = rearrange(B, "b (c l) g d -> b c l g d", c=nchunks)

        return torch.einsum("bclgd,bcsgd->bcgls", A, B)

    def bmm_triton(A, B):
        from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
        return _bmm_chunk_fwd(A, B, chunk_size)

    def get_configs():
        block_M = [64, 128]
        block_N = [32, 64, 128]
        block_K = [32, 64]
        num_stages = [1, 2]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages))

        configs = [
            {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3], 'thread_num': c[0] * 2}
            for c in _configs
        ]
        return configs

    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Normal, ref_prog=bmm_triton, rtol=0.01, atol=0.01, profiler="tvm")
    def kernel(block_M = None, block_N = None, block_K = None, num_stages = None, thread_num = None):
        dtype = "float16"
        accum_dtype = "float"
        nchunks = T.ceildiv(seqlen, chunk_size)
        @T.prim_func
        def main(
            A: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
            B: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
            Output: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype)
        ):
            with T.Kernel(T.ceildiv(chunk_size, block_M) * T.ceildiv(chunk_size, block_N), batch, nchunks * ngroups, threads=thread_num) as (bx, by, bz):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
                acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
                chunk_idx = bz // ngroups
                group_idx = bz % ngroups
                m_idx = bx // T.ceildiv(chunk_size, block_N)
                n_idx = bx % T.ceildiv(chunk_size, block_N)

                loop_range = T.ceildiv(dstate, block_K)
                T.clear(acc_o)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(A[by, 
                        chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, 
                        group_idx, 
                        k * block_K : (k + 1) * block_K], 
                        A_shared)
                    T.copy(B[by, 
                        chunk_idx * chunk_size + n_idx * block_N : chunk_idx * chunk_size + (n_idx + 1) * block_N, 
                        group_idx, 
                        k * block_K : (k + 1) * block_K], 
                        B_shared)
                    T.gemm(A_shared, B_shared, acc_o, transpose_B=True)
                T.copy(acc_o, acc_o_shared)
                T.copy(acc_o_shared, Output[by, chunk_idx, group_idx, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N])

        return main
    return kernel()

####################################################################################################
# chunk_state
####################################################################################################

# def chunk_state_triton(B, x, dt, dA_cumsum):
#     from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
#     return _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=False)

# def chunk_state_fwd(batch, seqlen, ngroups, nheads, headdim, dstate, block_M, block_N, block_K):
#     dtype = "float16"
#     accum_dtype = "float"
#     nchunks = T.ceildiv(seqlen, chunk_size)
#     p = 1.44269504
#     @T.prim_func
#     def main(
#         B: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
#         x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
#         dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
#         dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
#         Output: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype)
#     ):
#         with T.Kernel(T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N), batch * nchunks, nheads, threads=128) as (bx, by, bz):
#             x_shared = T.alloc_shared((block_K, block_M), dtype)
#             x_local = T.alloc_fragment((block_K, block_M), dtype)
#             xt_local = T.alloc_fragment((block_M, block_K), dtype)
#             B_shared = T.alloc_shared((block_K, block_N), dtype)
#             dt_shared = T.alloc_shared((block_K), dtype)
#             dA_cumsum_shared = T.alloc_shared((block_K), dtype)
#             acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
#             acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
#             scale = T.alloc_fragment((block_K), accum_dtype)
#             dA_cs_last = T.alloc_fragment((1), accum_dtype)
#             dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
#             dt_local = T.alloc_fragment((block_K), accum_dtype)

#             loop_range = T.ceildiv(chunk_size, block_K)
            
#             batch_idx = by % batch
#             chunk_idx = by // batch
#             m_idx = bx // T.ceildiv(dstate, block_N)
#             n_idx = bx % T.ceildiv(dstate, block_N)

#             T.annotate_layout({
#                 acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
#             })
            
#             dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]
#             T.clear(acc_o)
#             for k in T.Pipelined(
#                 loop_range, 
#                 num_stages=4, 
#                 order=[-1,-1,-1,1,-1,0],
#                 stage=[-1,-1,-1,0,-1,1],
#                 group=[[0],[1],[2],[3,4,5,6,7],[8],[9]],
#             ):
#                 T.copy(x[batch_idx, 
#                     chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, 
#                     bz, 
#                     m_idx * block_M : (m_idx + 1) * block_M], 
#                     x_shared)
#                 T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cumsum_shared)
#                 T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
#                 T.copy(dA_cumsum_shared, dA_cumsum_local)
#                 T.copy(dt_shared, dt_local)
#                 for i in T.Parallel(block_K):
#                     scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
#                 T.copy(x_shared, x_local)
#                 for i, j in T.Parallel(block_M, block_K):
#                     xt_local[i, j] = x_local[j, i] * scale[j]
#                 T.copy(B[batch_idx,
#                     chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
#                     bz // (nheads // ngroups),
#                     n_idx * block_N : (n_idx + 1) * block_N],
#                     B_shared)
#                 T.gemm(xt_local, B_shared, acc_o)
#             T.copy(acc_o, acc_o_shared)
#             T.copy(acc_o_shared, Output[batch_idx, chunk_idx, bz, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N])
#     return main

# def chunk_state_ref(B, x, dt, dA_cumsum):
#     from einops import rearrange, repeat
#     """
#     Argument:
#         B: (batch, seqlen, ngroups, headdim)
#         x: (batch, seqlen, nheads, headdim)
#         dt: (batch, nheads, nchunks, chunk_size)
#         dA_cumsum: (batch, nheads, nchunks, chunk_size)
#     Return:
#         states: (batch, nchunks, nheads, headdim, dstate)
#     """
#     # Check constraints.
#     batch, seqlen, nheads, headdim = x.shape
#     dstate = B.shape[-1]
#     _, _, nchunks, chunk_size = dt.shape
#     assert seqlen <= nchunks * chunk_size
#     assert x.shape == (batch, seqlen, nheads, headdim)
#     assert dt.shape == (batch, nheads, nchunks, chunk_size)
#     ngroups = B.shape[2]
#     assert nheads % ngroups == 0
#     assert B.shape == (batch, seqlen, ngroups, dstate)
#     B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
#     assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
#     if seqlen < nchunks * chunk_size:
#         x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
#         B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
#     x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
#     B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
#     decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
#     return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)

def chunk_state(batch, seqlen, ngroups, nheads, headdim, dstate):
    
    def chunk_state_triton(B, x, dt, dA_cumsum):
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd
        return _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=False)

    def chunk_state_ref(B, x, dt, dA_cumsum):
        """
        Argument:
            B: (batch, seqlen, ngroups, headdim)
            x: (batch, seqlen, nheads, headdim)
            dt: (batch, nheads, nchunks, chunk_size)
            dA_cumsum: (batch, nheads, nchunks, chunk_size)
        Return:
            states: (batch, nchunks, nheads, headdim, dstate)
        """
        # Check constraints.
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[-1]
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        assert x.shape == (batch, seqlen, nheads, headdim)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        ngroups = B.shape[2]
        assert nheads % ngroups == 0
        assert B.shape == (batch, seqlen, ngroups, dstate)
        B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if seqlen < nchunks * chunk_size:
            x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
            B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
        B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
        decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
        return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)
    
    def get_configs():
        # block_M = [64, 128]
        # block_N = [32, 64, 128]
        # block_K = [32, 64]
        # num_stages = [2,3,4,5]
        block_M = [64]
        block_N = [128]
        block_K = [64]
        num_stages = [4]
        _configs = list(itertools.product(block_M, block_N, block_K, num_stages))

        configs = [
            {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3], 'thread_num': c[0] * 2}
            for c in _configs
        ]
        return configs
    
    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'], warmup=10, rep=5)
    @jit(out_idx=[4], supply_type=tl.TensorSupplyType.Normal, ref_prog=chunk_state_triton, check_close=False, rtol=0.01, atol=0.01, profiler="tvm")
    def kernel(block_M = None, block_N = None, block_K = None, num_stages = None, thread_num = None):
        dtype = "float16"
        accum_dtype = "float"
        nchunks = T.ceildiv(seqlen, chunk_size)
        p = 1.44269504
        @T.prim_func
        def main(
            B: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
            x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
            dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
            dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
            Output: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype)
        ):
            with T.Kernel(T.ceildiv(headdim, block_M) * T.ceildiv(dstate, block_N), batch * nchunks, nheads, threads=thread_num) as (bx, by, bz):
                x_shared = T.alloc_shared((block_K, block_M), dtype)
                x_local = T.alloc_fragment((block_K, block_M), dtype)
                xt_local = T.alloc_fragment((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                dt_shared = T.alloc_shared((block_K), dtype)
                dA_cumsum_shared = T.alloc_shared((block_K), dtype)
                acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
                acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
                scale = T.alloc_fragment((block_K), accum_dtype)
                dA_cs_last = T.alloc_fragment((1), accum_dtype)
                dA_cumsum_local = T.alloc_fragment((block_K), accum_dtype)
                dt_local = T.alloc_fragment((block_K), accum_dtype)

                loop_range = T.ceildiv(chunk_size, block_K)
                
                batch_idx = by % batch
                chunk_idx = by // batch
                m_idx = bx // T.ceildiv(dstate, block_N)
                n_idx = bx % T.ceildiv(dstate, block_N)
                dA_cs_last[0] = dA_cumsum[batch_idx, bz, chunk_idx, chunk_size - 1]

                T.annotate_layout({
                    x_shared: tl.layout.make_swizzled_layout(x_shared),
                    acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
                })

                T.clear(acc_o)
                for k in T.Pipelined(
                    loop_range, 
                    num_stages=num_stages, 
                    order=[-1,1,-1,2,-1,3,-1,0],
                    stage=[-1,0,-1,0,-1,0,-1,1],
                    group=[[0],[1],[2],[3],[4],[5,6,7],[8],[9]],
                    # order=[-1,-1,-1,1,-1,0],
                    # stage=[-1,-1,-1,0,-1,1],
                    # group=[[0],[1],[2],[3,4,5,6,7],[8],[9]],
                ):
                    T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cumsum_shared)
                    T.copy(dA_cumsum_shared, dA_cumsum_local)
                    T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                    T.copy(dt_shared, dt_local)
                    T.copy(x[batch_idx, 
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, 
                        bz, 
                        m_idx * block_M : (m_idx + 1) * block_M], 
                        x_shared)
                    T.copy(x_shared, x_local)
                    for i in T.Parallel(block_K):
                        scale[i] = T.exp2(dA_cs_last[0] * p - dA_cumsum_local[i] * p) * dt_local[i]
                    for i, j in T.Parallel(block_M, block_K):
                        xt_local[i, j] = x_local[j, i] * scale[j]
                    T.copy(B[batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz // (nheads // ngroups),
                        n_idx * block_N : (n_idx + 1) * block_N],
                        B_shared)
                    T.gemm(xt_local, B_shared, acc_o)
                T.copy(acc_o, acc_o_shared)
                T.copy(acc_o_shared, Output[batch_idx, chunk_idx, bz, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N])
        return main
    return kernel()

####################################################################################################
# chunk_scan
####################################################################################################

def chunk_scan_triton(cb, x, dt, dA_cumsum, C, states):
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
    out, _ =  _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states)
    return out

def chunk_scan_ref(cb, x, dt, dA_cumsum, C, prev_states):
    from einops import rearrange, repeat
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    _, _, ngroups, _, _ = cb.shape
    batch, seqlen, nheads, headdim = x.shape
    # _, _, ngroups, dstate = B.shape
    # assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    # assert C.shape == B.shape
    # B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    cb = repeat(cb, "b c g l s -> b c (g h) l s", h=nheads // ngroups)
    # CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
    #                   rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = cb * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")

    return out

def chunk_scan_fwd(batch, seqlen, ngroups, nheads, headdim, dstate, block_M, block_N, block_K, block_Dstate):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504
    @T.prim_func
    def main(
        cb: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
        x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
        dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        C: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
        prev_states: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype),
        Output: T.Buffer((batch, seqlen, nheads, headdim), dtype)
    ):
        with T.Kernel(T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, nheads, threads=128) as (bx, by, bz):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            # acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype)
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_M), dtype)
            dA_cs_k_local = T.alloc_fragment((block_M), dtype)
            dA_cs_m_shared = T.alloc_shared((block_M), dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype)
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)


            batch_idx = by % batch
            chunk_idx = by // batch
            # m: chunk_size
            # n : headdim
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            # T.annotate_layout({
            #     acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
            # })
            
            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)
            
            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[batch_idx, 
                  chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                  bz // (nheads // ngroups),
                  0 : block_Dstate
                  ], 
                C_shared
            )
            T.copy(
                prev_states[batch_idx, 
                  chunk_idx,
                  bz,
                  n_idx * block_N : (n_idx + 1) * block_N,
                  0 : block_Dstate
                  ], 
                prev_state_shared
            )
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(
                    cb[batch_idx, 
                       chunk_idx, 
                       bz // (nheads // ngroups), 
                       m_idx * block_M : (m_idx + 1) * block_M, 
                       k * block_K : (k + 1) * block_K], 
                    cb_shared
                )
                T.copy(cb_shared, cb_local)
                T.copy(
                    dA_cumsum[batch_idx, 
                       bz, 
                       chunk_idx,
                       k * block_K : (k + 1) * block_K], 
                    dA_cs_k_shared
                )
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(
                        m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0
                    )
                T.copy(x[batch_idx, chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, bz, n_idx * block_N : (n_idx + 1) * block_N], x_shared)
                T.gemm(cb_local, x_shared, acc_o)
            # T.copy(acc_o, acc_o_shared)
            T.copy(acc_o, Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, bz, n_idx * block_N : (n_idx + 1) * block_N])

    return main

def bmm_chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)

def bmm_chunk_scan_fwd(batch, seqlen, ngroups, nheads, headdim, dstate, block_M, block_N, block_K, block_Dstate):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504
    @T.prim_func
    def main(
        B: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
        x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
        dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
        C: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
        prev_states: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype),
        Output: T.Buffer((batch, seqlen, nheads, headdim), dtype)
    ):
        with T.Kernel(T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, nheads, threads=128) as (bx, by, bz):
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype)
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_M), dtype)
            dA_cs_k_local = T.alloc_fragment((block_M), dtype)
            dA_cs_m_shared = T.alloc_shared((block_M), dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype)
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)


            batch_idx = by % batch
            chunk_idx = by // batch
            # m: chunk_size
            # n : headdim
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            # T.annotate_layout({
            #     acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
            # })
            
            T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)
            
            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[batch_idx, 
                  chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                  bz // (nheads // ngroups),
                  0 : block_Dstate
                  ], 
                C_shared
            )
            T.copy(
                prev_states[batch_idx, 
                  chunk_idx,
                  bz,
                  n_idx * block_N : (n_idx + 1) * block_N,
                  0 : block_Dstate
                  ], 
                prev_state_shared
            )
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

            for k in T.Pipelined(loop_range, num_stages=4):
                T.copy(
                    cb[batch_idx, 
                       chunk_idx, 
                       bz // (nheads // ngroups), 
                       m_idx * block_M : (m_idx + 1) * block_M, 
                       k * block_K : (k + 1) * block_K], 
                    cb_shared
                )
                T.copy(cb_shared, cb_local)
                T.copy(
                    dA_cumsum[batch_idx, 
                       bz, 
                       chunk_idx,
                       k * block_K : (k + 1) * block_K], 
                    dA_cs_k_shared
                )
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(
                        m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0
                    )
                T.copy(x[batch_idx, chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, bz, n_idx * block_N : (n_idx + 1) * block_N], x_shared)
                T.gemm(cb_local, x_shared, acc_o)
            T.copy(acc_o, acc_o_shared)
            T.copy(acc_o_shared, Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, bz, n_idx * block_N : (n_idx + 1) * block_N])

    return main

# def chunk_scan_fwd(batch, seqlen, ngroups, nheads, headdim, dstate):

#     def chunk_scan_triton(cb, x, dt, dA_cumsum, C, states):
#         from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
#         out, _ =  _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states)
#         return out

#     def get_configs():
#         # block_M = [64, 128]
#         # block_N = [32, 64, 128]
#         # block_K = [32, 64]
#         # block_Dstate = [128]
#         # num_stages = [2,3,4,5]
#         block_M = [64]
#         block_N = [64]
#         block_K = [64]
#         block_Dstate = [128]
#         num_stages = [4]
#         _configs = list(itertools.product(block_M, block_N, block_K, block_Dstate, num_stages))

#         configs = [
#             {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'block_Dstate': c[3], 'num_stages': c[4], 'thread_num': c[0] * 2}
#             for c in _configs
#         ]
#         return configs
    
#     @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'block_Dstate', 'num_stages', 'thread_num'], warmup=10, rep=5)
#     @jit(out_idx=[6], supply_type=tl.TensorSupplyType.Normal, ref_prog=chunk_scan_triton, check_close=False, rtol=0.01, atol=0.01, profiler="tvm")
#     def kernel(block_M = None, block_N = None, block_K = None, block_Dstate=None, num_stages = None, thread_num = None):
#         dtype = "float16"
#         accum_dtype = "float"
#         nchunks = T.ceildiv(seqlen, chunk_size)
#         p = 1.44269504
#         @T.prim_func
#         def main(
#             cb: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),
#             x: T.Buffer((batch, seqlen, nheads, headdim), dtype),
#             dt: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
#             dA_cumsum: T.Buffer((batch, nheads, nchunks, chunk_size), dtype),
#             C: T.Buffer((batch, seqlen, ngroups, dstate), dtype),
#             prev_states: T.Buffer((batch, nchunks, nheads, headdim, dstate), dtype),
#             Output: T.Buffer((batch, seqlen, nheads, headdim), dtype)
#         ):
#             with T.Kernel(T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N), batch * nchunks, nheads, threads=thread_num) as (bx, by, bz):
#                 acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
#                 acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
#                 cb_shared = T.alloc_shared((block_M, block_K), dtype)
#                 cb_local = T.alloc_fragment((block_M, block_K), dtype)
#                 dA_cs_k_shared = T.alloc_shared((block_M), dtype)
#                 dA_cs_k_local = T.alloc_fragment((block_M), dtype)
#                 dA_cs_m_shared = T.alloc_shared((block_M), dtype)
#                 dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
#                 dt_shared = T.alloc_shared((block_K), dtype)
#                 dt_local = T.alloc_fragment((block_K), accum_dtype)
#                 x_shared = T.alloc_shared((block_K, block_N), dtype)
#                 scale_m_local = T.alloc_fragment((block_M), accum_dtype)
#                 C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
#                 prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)


#                 batch_idx = by % batch
#                 chunk_idx = by // batch
#                 # m: chunk_size
#                 # n : headdim
#                 m_idx = bx // T.ceildiv(headdim, block_N)
#                 n_idx = bx % T.ceildiv(headdim, block_N)

#                 T.annotate_layout({
#                     acc_o_shared: tl.layout.make_swizzled_layout(acc_o_shared)
#                 })
                
#                 T.copy(dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M], dA_cs_m_shared)
#                 T.copy(dA_cs_m_shared, dA_cs_m_local)
#                 T.clear(acc_o)
                
#                 for i in T.Parallel(block_M):
#                     scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
#                 T.copy(
#                     C[batch_idx, 
#                     chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
#                     bz // (nheads // ngroups),
#                     0 : block_Dstate
#                     ], 
#                     C_shared
#                 )
#                 T.copy(
#                     prev_states[batch_idx, 
#                     chunk_idx,
#                     bz,
#                     n_idx * block_N : (n_idx + 1) * block_N,
#                     0 : block_Dstate
#                     ], 
#                     prev_state_shared
#                 )
#                 T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
#                 for i, j in T.Parallel(block_M, block_N):
#                     acc_o[i, j] *= scale_m_local[i]

#                 loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)

#                 for k in T.Pipelined(loop_range, num_stages=num_stages):
#                     T.copy(
#                         cb[batch_idx, 
#                         chunk_idx, 
#                         bz // (nheads // ngroups), 
#                         m_idx * block_M : (m_idx + 1) * block_M, 
#                         k * block_K : (k + 1) * block_K], 
#                         cb_shared
#                     )
#                     T.copy(cb_shared, cb_local)
#                     T.copy(
#                         dA_cumsum[batch_idx, 
#                         bz, 
#                         chunk_idx,
#                         k * block_K : (k + 1) * block_K], 
#                         dA_cs_k_shared
#                     )
#                     T.copy(dA_cs_k_shared, dA_cs_k_local)
#                     for i, j in T.Parallel(block_M, block_K):
#                         cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
#                     T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
#                     T.copy(dt_shared, dt_local)
#                     for i, j in T.Parallel(block_M, block_K):
#                         cb_local[i, j] *= dt_local[j]
#                     for i, j in T.Parallel(block_M, block_K):
#                         cb_local[i, j] = T.if_then_else(
#                             m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0
#                         )
#                     T.copy(x[batch_idx, chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K, bz, n_idx * block_N : (n_idx + 1) * block_N], x_shared)
#                     T.gemm(cb_local, x_shared, acc_o)
#                 T.copy(acc_o, acc_o_shared)
#                 T.copy(acc_o_shared, Output[batch_idx, chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M, bz, n_idx * block_N : (n_idx + 1) * block_N])

#         return main
#     return kernel()

def state_passing_fwd(batch, seqlen, nheads, headdim, block_M):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504
    @T.prim_func
    def main(
        states: T.Buffer((batch, nchunks, nheads, headdim), dtype),
        dA_chunk_cumsum: T.Buffer((batch, nheads, nchunks), dtype),
        initial_states: T.Buffer((batch, nheads, headdim), dtype),
        Output: T.Buffer((batch, nchunks + 1, nheads, headdim), dtype),
    ):
        with T.Kernel(T.ceildiv(headdim, block_M), batch, nheads, threads=128) as (bx, by, bz):
            # state_shared = T.alloc_shared((block_M), dtype)
            dA_cs_local = T.alloc_fragment((1,1), accum_dtype)
            scale = T.alloc_fragment((1,1), accum_dtype)
            state_local = T.alloc_fragment((block_M), accum_dtype)
            new_state_local = T.alloc_fragment((block_M), accum_dtype)
            
            T.annotate_layout({
                dA_cs_local: tl.layout.make_swizzled_layout(dA_cs_local),
            })

            batch_idx = by
            head_idx = bz
            m_idx = bx

            T.clear(state_local)
            T.copy(initial_states[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M], state_local)
            T.copy(state_local, Output[batch_idx, 0, head_idx, m_idx * block_M : (m_idx + 1) * block_M])
            # T.copy(state_shared, state_local)
            for k in T.Pipelined(nchunks, num_stages=1):
                # T.copy(states[batch_idx, k, head_idx, m_idx * block_M : (m_idx + 1) * block_M], state_shared)
                # T.copy(state_shared, new_state_local)
                for i in T.Parallel(block_M):
                    new_state_local[i] = states[batch_idx, k, head_idx, m_idx * block_M + i]
                dA_cs_local[0,0] = dA_chunk_cumsum[batch_idx, head_idx, k]
                scale[0,0] = T.exp2(dA_cs_local[0,0] * p)
                for i in T.Parallel(block_M):
                    state_local[i] = state_local[i] * scale[0,0] + new_state_local[i]
                T.copy(state_local, Output[batch_idx, k + 1, head_idx, m_idx * block_M : (m_idx + 1) * block_M])

    return main

def state_passing_ref(states, dA_chunk_cumsum, initial_states):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states], dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    # (batch, nheads, nchunks, nchunks)
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :, None] - dA_chunk_cumsum[:, :, None, :]
    # (batch, nheads, nchunks, nchunks)
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(torch.ones(nchunks, nchunks, device=states.device, dtype=bool), diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype), states)
    return out

def selective_scan_update_fwd(batch, seqlen, nheads, ngroups, headdim, dstate, block_M, block_Dstate):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504
    assert dstate == block_Dstate
    @T.prim_func
    def main(
        state: T.Buffer((batch, nheads, headdim, dstate), dtype),
        x: T.Buffer((batch, nheads, headdim), dtype),
        dt: T.Buffer((batch, nheads, headdim), dtype),
        A: T.Buffer((nheads, headdim, dstate), dtype),
        B: T.Buffer((batch, ngroups, dstate), dtype),
        C: T.Buffer((batch, ngroups, dstate), dtype),
        Output: T.Buffer((batch, nheads, headdim), dtype)
    ):
        with T.Kernel(T.ceildiv(headdim, block_M), batch, nheads, threads=128) as (bx, by, bz):
            state_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            state_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            # new_state_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            x_shared = T.alloc_shared((block_M), dtype)
            x_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_M), dtype)
            dt_local = T.alloc_fragment((block_M), accum_dtype)
            A_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            A_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            dA_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            B_shared = T.alloc_shared((block_Dstate), dtype)
            C_shared = T.alloc_shared((block_Dstate), dtype)
            C_local = T.alloc_fragment((block_Dstate), accum_dtype)
            B_local = T.alloc_fragment((block_Dstate), accum_dtype)
            dB_local = T.alloc_fragment((block_M, block_Dstate), accum_dtype)
            state_sum_local = T.alloc_fragment((block_M), accum_dtype)

            batch_idx = by
            head_idx = bz
            m_idx = bx

            # T.annotate_layout({
            #     new_state_local: tl.layout.make_swizzled_layout(state_shared),
            # })

            T.copy(state[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M, :], state_shared)
            T.copy(state_shared, state_local)
            T.copy(x[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M], x_shared)
            T.copy(x_shared, x_local)
            # Not TIE_HDIM
            T.copy(dt[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M], dt_shared)
            T.copy(dt_shared, dt_local)
            T.copy(A[head_idx, m_idx * block_M : (m_idx + 1) * block_M, :], A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(block_M, block_Dstate):
                dA_local[i, j] = T.exp2(A_local[i, j] * dt_local[i] * p)
            T.copy(B[batch_idx, bz // (nheads // ngroups), :], B_shared)
            T.copy(B_shared, B_local)
            T.copy(C[batch_idx, bz // (nheads // ngroups), :], C_shared)
            T.copy(C_shared, C_local)
            for i, j in T.Parallel(block_M, block_Dstate):
                dB_local[i, j] = B_local[j] * dt_local[i]
            for i, j in T.Parallel(block_M, block_Dstate):
                state_local[i, j] *= dA_local[i, j]
            for i, j in T.Parallel(block_M, block_Dstate):
                state_local[i, j] += dB_local[i, j] * x_local[i]
            for i, j in T.Parallel(block_M, block_Dstate):
                state_local[i, j] *= C_local[j]
            T.reduce_sum(state_local, state_sum_local, dim=1)
            T.copy(state_sum_local, Output[batch_idx, head_idx, m_idx * block_M : (m_idx + 1) * block_M])

    return main

def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state_ = state * dA + dB * rearrange(x, "b h d -> b h d 1")  # (batch, dim, dstate
    out = torch.einsum("bhdn,bhn->bhd", state_.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out

if __name__ == "__main__":
    BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 8, 80, 1, 8192, 64, 128
    # BATCH, NHEADS, NGROUPS, SEQLEN, HEADDIM, DSTATE = 1, 1, 1, 256, 64, 128
    block_M, block_N, block_K, block_Dstate = 64, 64, 64, 128
    # chunk_cumsum_fwd

    # state_passing_fwd
    # BATCH, SEQLEN, NHEADS, HEADDIM = 4, 2048, 8, 64
    # block_M = 64
    # program = state_passing_fwd(BATCH, SEQLEN, NHEADS, HEADDIM, block_M)
    # mod, params = tl.lower(program)
    # mod = tl.Profiler(mod, params, [3], tl.TensorSupplyType.Normal)
    # mod.assert_allclose(state_passing_ref, rtol=0.01, atol=0.01)
    

    # chunk_state_fwd
    # total_flops = 2 * BATCH * SEQLEN * NHEADS * HEADDIM * DSTATE
    # best_latency, best_config, ref_latency = chunk_state(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE)
    # print(f"Best latency: {best_latency}")
    # print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    # print(f"Best config: {best_config}")
    # print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")
    # program = chunk_state_fwd(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE, block_M, block_N, block_K) 
    # mod, params = tl.lower(program)
    # mod = tl.Profiler(mod, params, [4], tl.TensorSupplyType.Normal)
    # # mod.assert_allclose(chunk_state_triton, rtol=0.01, atol=0.01)
    # latency = mod.do_bench(chunk_state_triton, n_warmup=10, n_repeat=10, profiler="torch")
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    # latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="tvm")
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))


    # bmm_chunk
    # total_flops = 2 * BATCH * SEQLEN * NGROUPS * DSTATE * chunk_size
    # best_latency, best_config, ref_latency = bmm_chunk(BATCH, SEQLEN, NGROUPS, DSTATE)
    # print(f"Best latency: {best_latency}")
    # print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    # print(f"Best config: {best_config}")
    # print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")
    # program = bmm_chunk(BATCH, SEQLEN, NGROUPS, DSTATE, block_M, block_N, block_K, 2, 128)
    # mod, params = tl.lower(program)
    # mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Normal)
    # mod.assert_allclose(bmm_triton, rtol=0.1, atol=0.1)
    # total_flops = 2 * BATCH * SEQLEN * NGROUPS * DSTATE * chunk_size
    # latency = mod.do_bench(bmm_triton, n_warmup=10, n_repeat=10, profiler="tvm")
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    # latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="tvm")
    # print("{:.2f} ms".format(latency))
    # print("{:.2f} TFlops".format(total_flops / latency * 1e-9))

    # chunk_scan_fwd
    total_flops = 2.0 * BATCH * SEQLEN * chunk_size * NHEADS * HEADDIM * 0.5 + 2.0 * BATCH * SEQLEN * NHEADS * HEADDIM * DSTATE
    # best_latency, best_config, ref_latency = chunk_scan_fwd(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE)
    # print(f"Best latency: {best_latency}")
    # print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    # print(f"Best config: {best_config}")
    # print(f"Ref TFlops: {total_flops / ref_latency * 1e-9}")
    program = chunk_scan_fwd(BATCH, SEQLEN, NGROUPS, NHEADS, HEADDIM, DSTATE, block_M, block_N, block_K, block_Dstate) 
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [6], tl.TensorSupplyType.Normal)
    mod.assert_allclose(chunk_scan_ref, rtol=0.01, atol=0.01)
    latency = mod.do_bench(chunk_scan_ref, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="tvm")
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))

     # selective_state_update_fwd
    # BATCH, SEQLEN, NHEADS, NGROUPS, HEADDIM, DSTATE = 1, 4096, 1, 1, 64, 64
    # block_M, block_Dstate = 64, 64
    # program = selective_scan_update_fwd(BATCH, SEQLEN, NHEADS, NGROUPS, HEADDIM, DSTATE, block_M, block_Dstate)
    # mod, params = tl.lower(program)
    # mod = tl.Profiler(mod, params, [6], tl.TensorSupplyType.Normal)
    # mod.assert_allclose(selective_state_update_ref, rtol=0.1, atol=0.1)