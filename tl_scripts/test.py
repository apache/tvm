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

SEQLEN = 1024
D = 64
blockM = SEQLEN
blockN = 64
Q = torch.randn((SEQLEN, D), dtype=torch.float16)
K = torch.randn((SEQLEN, D), dtype=torch.float16)
V = torch.randn((SEQLEN, D), dtype=torch.float16)

def ref_program(Q, K, V):
    qk = torch.matmul(Q, K.transpose(-1, -2))
    m = qk.max(dim=-1, keepdim=True).values
    p = torch.exp(qk - m)
    s = p / p.sum(dim=-1, keepdim=True)
    o = torch.matmul(s, V)
    return o

def test_program(Q, K, V):
    lse = torch.randn((blockM), dtype=float)
    m = torch.randn((blockM), dtype=float)
    m_new = torch.randn((blockM), dtype=float)
    acc_o = torch.randn((blockM, D), dtype=float)
    m.fill_(float('-inf'))
    m_new.fill_(float('-inf'))
    lse.fill_(float('-inf'))
    acc_o.fill_(float(0))
    for i in range(int(SEQLEN / blockN)):
        qk = torch.matmul(Q, (K[i * blockN : (i + 1) * blockN, :]).transpose(-1, -2)) # [blockM, blockN]
        m_new = torch.max(qk.max(dim=-1, keepdim=False).values, m_new) # [blockM]
        p = torch.exp(qk - m_new.unsqueeze(dim=1)) # [blockM, blockN]
        lse = m_new + torch.log(torch.exp(lse - m_new) + p.sum(dim=-1, keepdim=False)) # [blockM]
        acc_o = acc_o * torch.exp(m - m_new).unsqueeze(1)
        m = m_new
        acc_o += torch.matmul(p.to(torch.float16), V[i * blockN : (i + 1) * blockN, :])
    acc_o = acc_o * torch.exp(m_new - lse).unsqueeze(1)
    return acc_o.to(torch.float16)

ref_output = ref_program(Q, K, V)
test_output = test_program(Q, K, V)
are_close = torch.allclose(ref_output, test_output, rtol=1e-03, atol=1e-03)
print(f"Are the outputs close? {are_close}")