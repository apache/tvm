import torch
import fa_test
from flash_attn.flash_attn_interface import flash_attn_func
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
        # scores_max_prev = scores_max
        # scores_max = acc_s.max(dim=-1, keepdim=False).values # [blockM]
        # scores_scale = torch.exp2(scores_max_prev - scores_max)
        # acc_o *= scores_scale[:, :, :, None].transpose(1, 2)
        acc_s = torch.exp2(acc_s - 32)
        acc_s_cast = acc_s.to(torch.float16)
        acc_o += torch.einsum('bhqk,bkhd->bqhd', acc_s_cast, V[:, i * block_N : (i + 1) * block_N, :, :])
        # scores_sum = acc_s.sum(dim=-1, keepdim=False)
    #     logsum = logsum * scores_scale + scores_sum
    # acc_o /= logsum[:, :, :, None].transpose(1, 2)
    return acc_o.to(torch.float16)

set_seed(42)
causal = False
batch, seq_len, heads, dim = 64, 512, 16, 64
shape = [batch, seq_len, heads, dim]
# q = torch.empty(*shape, device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
# k = torch.empty(*shape, device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
q = torch.ones(*shape, device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
k = torch.ones(*shape, device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)
v = torch.empty(*shape, device='cuda', dtype=torch.float16).normal_(-1.0, 1.0)

output = fa_test.kernel_function(q, k, v, causal)
ref_output = flash_attn_func(q, k, v, causal=False)
# ref_output = ref_program(q, k, v, causal)
assert torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
print("Check: PASSED")

warmups = 10
runs = 10
for _ in range(warmups):
    out = fa_test.kernel_function(q, k, v, causal)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

for _ in range(runs):
    out = fa_test.kernel_function(q, k, v, causal)

end_event.record()
torch.cuda.synchronize()

latency = start_event.elapsed_time(end_event)

flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
total_flops = 2 * flops_per_matmul
print(f"total_flops: {total_flops}")
print(f"TFLOPS: {total_flops / latency * runs * 1e-9}")
print(f"Latency: {latency / runs:.2f} ms")