import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# SEQLEN = 1024
# D = 64
# blockM = SEQLEN
# blockN = 64
# Q = torch.randn((SEQLEN, D), dtype=torch.float16)
# K = torch.randn((SEQLEN, D), dtype=torch.float16)
# V = torch.randn((SEQLEN, D), dtype=torch.float16)

# def ref_program(Q, K, V):
#     qk = torch.matmul(Q, K.transpose(-1, -2))
#     m = qk.max(dim=-1, keepdim=True).values
#     p = torch.exp(qk - m)
#     s = p / p.sum(dim=-1, keepdim=True)
#     o = torch.matmul(s, V)
#     return o

# def test_program(Q, K, V):
#     lse = torch.randn((blockM), dtype=float)
#     m = torch.randn((blockM), dtype=float)
#     m_new = torch.randn((blockM), dtype=float)
#     acc_o = torch.randn((blockM, D), dtype=float)
#     m.fill_(float('-inf'))
#     m_new.fill_(float('-inf'))
#     lse.fill_(float('-inf'))
#     acc_o.fill_(float(0))
#     for i in range(int(SEQLEN / blockN)):
#         qk = torch.matmul(Q, (K[i * blockN : (i + 1) * blockN, :]).transpose(-1, -2)) # [blockM, blockN]
#         m_new = torch.max(qk.max(dim=-1, keepdim=False).values, m_new) # [blockM]
#         p = torch.exp(qk - m_new.unsqueeze(dim=1)) # [blockM, blockN]
#         lse = m_new + torch.log(torch.exp(lse - m_new) + p.sum(dim=-1, keepdim=False)) # [blockM]
#         acc_o = acc_o * torch.exp(m - m_new).unsqueeze(1)
#         m = m_new
#         acc_o += torch.matmul(p.to(torch.float16), V[i * blockN : (i + 1) * blockN, :])
#     acc_o = acc_o * torch.exp(m_new - lse).unsqueeze(1)
#     return acc_o.to(torch.float16)

# ref_output = ref_program(Q, K, V)
# test_output = test_program(Q, K, V)
# are_close = torch.allclose(ref_output, test_output, rtol=1e-03, atol=1e-03)
# print(f"Are the outputs close? {are_close}")

# import torch.nn.functional as F

# batch = 1
# seq_len = 128
# heads = 1
# dim = 64
# shape = [batch, seq_len, heads, dim]
# # Q = torch.randn(shape, device="cuda", dtype=torch.float16)
# # K = torch.randn(shape, device="cuda", dtype=torch.float16)
# # V = torch.randn(shape, device="cuda", dtype=torch.float16)
# Q = torch.ones(shape, device="cuda", dtype=torch.float16)
# K = torch.ones(shape, device="cuda", dtype=torch.float16)
# V = torch.ones(shape, device="cuda", dtype=torch.float16)

# # def test_program(Q, K, V):
# #     scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
# #     block_M = seq_len
# #     block_N = 64
# #     acc_s = torch.empty((batch, heads, block_M, block_N), device="cuda", dtype=torch.float)
# #     acc_s_cast = torch.empty((batch, heads, block_M, block_N), device="cuda", dtype=torch.float16)
# #     acc_o = torch.empty((batch, block_M, heads, dim), device="cuda", dtype=torch.float)
# #     scores_max = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
# #     scores_max_prev = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
# #     scores_scale = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
# #     scores_sum = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
# #     logsum = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
# #     acc_o.fill_(0)
# #     logsum.fill_(0)
# #     scores_max.fill_(float('-inf'))
# #     Q *= scale

# #     for i in range(int(seq_len / block_N)):
# #         acc_s.fill_(0)
# #         acc_s = torch.einsum('bqhd,bkhd->bhqk', Q, K[:, i * block_N : (i + 1) * block_N, :, :]) # [batch, seqlen, heads, block_N]
# #         scores_max_prev = scores_max
# #         scores_max = acc_s.max(dim=-1, keepdim=False).values # [blockM]
# #         scores_scale = torch.exp2(scores_max_prev - scores_max)
# #         acc_o *= scores_scale[:, :, :, None].transpose(1, 2)
# #         acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
# #         # print("acc_s:", acc_s)
# #         acc_s_cast = acc_s.to(torch.float16)
# #         acc_o += torch.einsum('bhqk,bkhd->bqhd', acc_s_cast, V[:, i * block_N : (i + 1) * block_N, :, :])
# #         scores_sum = acc_s.sum(dim=-1, keepdim=False)
# #         logsum = logsum * scores_scale + scores_sum
# #         # print("acc_o:", acc_o.size())
# #         # print("logsum:", logsum.size())
# #     acc_o /= logsum[:, :, :, None].transpose(1, 2)
# #     return acc_o.to(torch.float16)

# def test_program(Q, K, V):
#     scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
#     block_M = seq_len
#     block_N = 64
#     acc_s = torch.empty((batch, heads, block_M, block_N), device="cuda", dtype=torch.float)
#     acc_s_cast = torch.empty((batch, heads, block_M, block_N), device="cuda", dtype=torch.float16)
#     acc_o = torch.empty((batch, block_M, heads, dim), device="cuda", dtype=torch.float)
#     scores_max = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
#     scores_max_prev = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
#     scores_scale = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
#     scores_sum = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
#     # logsum = torch.empty((batch, heads, block_M), device="cuda", dtype=torch.float)
#     acc_o.fill_(0)
#     scores_scale.fill_(0)
#     # logsum.fill_(0)
#     scores_max.fill_(float('-inf'))

#     for i in range(int(seq_len / block_N)):
#         acc_s.fill_(0)
#         acc_s = torch.einsum('bqhd,bkhd->bhqk', Q, K[:, i * block_N : (i + 1) * block_N, :, :])

#         if (i == 0):
#             scores_max = acc_s.max(dim=-1, keepdim=False).values
#             acc_s = torch.exp2(acc_s * scale - scores_max[:, :, :, None] * scale)
#             scores_sum = acc_s.sum(dim=-1, keepdim=False)
#             scores_scale.fill_(1)

#         # cute::copy(softmax.template max</*Is_first=*/false>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
#         if (i > 0):
#             scores_max_prev = scores_max
#             scores_max = acc_s.max(dim=-1, keepdim=False).values
#             print("scores_max_prev:", scores_max_prev)
#             print("scores_max:", scores_max)
#             print("scores_max_prev - scores_max:", scores_max_prev - scores_max)
#             scores_scale = torch.exp2((scores_max_prev - scores_max) * scale)
#             scores_sum *= scores_scale

#         # online_softmax
#         if (i > 0):
#             acc_s = torch.exp2(acc_s * scale - scores_max[:, :, :, None] * scale)
#             scores_sum = acc_s.sum(dim=-1, keepdim=False)

#         # rescale_o
#         if (i > 0):
#             acc_o *= scores_scale[:, :, :, None].transpose(1, 2) # softmax.rescale_o(tOrO, scores_scale);

#         acc_s_cast = acc_s.to(torch.float16)
#         print("acc_s_cast:", acc_s_cast)
#         acc_o += torch.einsum('bhqk,bkhd->bqhd', acc_s_cast, V[:, i * block_N : (i + 1) * block_N, :, :])
#         print("acc_o:", acc_o)

#         if (i == int(seq_len / block_N) - 1):
#             acc_o /= scores_sum[:, :, :, None].transpose(1, 2)
        
#         print("scores_scale:", scores_scale)
#         print("scores_sum:", scores_sum)
#     # scores_sum = scores_max * scale * 0.69314718055994530942 + torch.log(scores_sum)
#     # acc_o /= scores_sum[:, :, :, None].transpose(1, 2)
#     acc_o *= scores_scale[:, :, :, None].transpose(1, 2) # softmax.rescale_o(tOrO, scores_scale);
#     return acc_o.to(torch.float16)


# # flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
# # First cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
# # cute::copy(softmax.template max</*Is_first=*/false, /*Check_inf=*/true>(tSrS, mainloop_params.softmax_scale_log2), scores_scale);
# # softmax.template online_softmax</*Is_first=*/true>(tSrS, mainloop_params.softmax_scale_log2);
# # flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
# # softmax.rescale_o(tOrO, scores_scale);
# # finalize

# def ref_program(Q, K, V):
#     dim = Q.size(-1)
#     scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)

      
#     # Step 2: Scale the scores by the square root of dim
#     scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))

    
#     # Step 3: Apply softmax to get the attention weights
#     attention_weights = F.softmax(scores, dim=-1)
    
#     # print("scores:", attention_weights)
#     # Step 4: Multiply the attention weights by the values (V)
#     # This gives us the final output of shape [batch, seq_len, heads, dim]
#     output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    
    
#     return output

# ref_output = ref_program(Q, K, V)
# test_output = test_program(Q, K, V)
# are_close = torch.allclose(ref_output, test_output, rtol=1e-03, atol=1e-03)
# print(f"Are the outputs close? {are_close}")

# print("ref_output:", ref_output)
# print("test_output:", test_output)


# RetNet
batch = 8
head = 16
seq_len = 1024
dim = 64
blockM = seq_len
blockN = 64
# q = torch.randn((BATCH, HEAD, SEQLEN, D), dtype=torch.float16)
# k = torch.randn((BATCH, HEAD, SEQLEN, D), dtype=torch.float16)
# v = torch.randn((BATCH, HEAD, SEQLEN, D), dtype=torch.float16)
# mask = torch.randn((HEAD, SEQLEN, SEQLEN), dtype=torch.float16)

shape = [batch, seq_len, head, dim]
q = torch.randn(shape, dtype=torch.float16)
k = torch.randn(shape, dtype=torch.float16)
v = torch.randn(shape, dtype=torch.float16)
mask = torch.randn((head, seq_len, seq_len), dtype=torch.float16)

def ref_program(q, k, v, mask):
    qk = torch.einsum('bqhd,bkhd->bhqk', q, k)
    qkm = qk * mask.unsqueeze(0)
    r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o = torch.einsum('bhqk,bkhd->bqhd', qkm/r, v)
    return o

def test_program(q, k, v, mask):
    r = torch.zeros((batch, head, seq_len), dtype=torch.float16)
    r_new = torch.zeros((batch, head, seq_len), dtype=torch.float16)
    r_wo_clamp = torch.zeros((batch, head, seq_len), dtype=torch.float16)
    acco = torch.zeros((batch, head, seq_len, dim), dtype=torch.float16)
    for i in range(seq_len // blockN):
        qk = torch.einsum('bqhd,bkhd->bhqk', q, k[:, i * blockN : (i + 1) * blockN, :, :])
        qkm = qk * mask[:, :, i * blockN : (i + 1) * blockN].unsqueeze(0)
        r_wo_clamp += qkm.detach().abs().sum(dim=-1, keepdim=False)
        r_new = torch.max(r_wo_clamp, torch.ones_like(r_wo_clamp))
        if (i != 0):
            acco = acco * r.unsqueeze(-1) / r_new.unsqueeze(-1)
        r = r_new
        acco += torch.einsum('bhqk,bkhd->bhqd', qkm / r_new.unsqueeze(-1), v[:, i * blockN : (i + 1) * blockN, :, :])
    return acco.transpose(1, 2)

ref_output = ref_program(q, k, v, mask)
test_output = test_program(q, k, v, mask)
are_close = torch.allclose(ref_output, test_output, rtol=1e-02, atol=1e-02)
print(f"Are the outputs close? {are_close}")
# print("ref_output:", ref_output)
# print("test_output:", test_output)
# print("ref_output shape:", ref_output.size())
# print("test_output shape:", test_output.size())