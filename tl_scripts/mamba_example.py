import argparse
import torch
import torch.nn.functional as F
from tvm import tl
import tvm.tl.language as T
from functools import partial

chunk_size = 256
def bmm_chunk(batch, seqlen, ngroups, k, block_M, block_N, block_K):
    dtype = "float16"
    accum_dtype = "float"
    nchunks = T.ceildiv(seqlen, chunk_size)
    @T.prim_func
    def main(
        A: T.Buffer((batch, seqlen, ngroups, k), dtype),
        B: T.Buffer((batch, seqlen, ngroups, k), dtype),
        Output: T.Buffer((batch, nchunks, ngroups, chunk_size, chunk_size), dtype)
    ):
        with T.Kernel(T.ceildiv(chunk_size, block_M) * T.ceildiv(chunk_size, block_N), batch, nchunks * ngroups, threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
            chunk_idx = bz // ngroups
            group_idx = bz % ngroups
            m_idx = bx // T.ceildiv(chunk_size, block_N)
            n_idx = bx % T.ceildiv(chunk_size, block_N)

            loop_range = T.ceildiv(chunk_size, block_K)
            T.clear(acc_o)
            for k in T.Pipelined(loop_range, num_stages=1):
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
            T.copy(acc_o, Output[by, chunk_idx, group_idx, m_idx * block_M : (m_idx + 1) * block_M, n_idx * block_N : (n_idx + 1) * block_N])

    return main

def ref_program(A, B):
    from einops import rearrange, repeat
    seqlen = A.shape[1]
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    A = rearrange(A, "b (c l) g d -> b c l g d", c=nchunks)
    B = rearrange(B, "b (c l) g d -> b c l g d", c=nchunks)

    return torch.einsum("bclgd,bcsgd->bcgls", A, B)

if __name__ == "__main__":
    BATCH, SEQLEN, NGROUPS, DSTATE = 8, 4096, 16, 64
    block_M, block_N, block_K = 64, 64, 64
    program = bmm_chunk(BATCH, SEQLEN, NGROUPS, DSTATE, block_M, block_N, block_K)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Normal)
    mod.assert_allclose(ref_program, rtol=0.1, atol=0.1)