# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test handling of FakeTensor and lifted tensors in from_exported_program"""
import pytest

torch = pytest.importorskip("torch", "2.1")

import math
import torch.nn as nn
from torch.export import export as torch_export

import tvm
from tvm.relax.frontend.torch import from_exported_program


def test_lifted_tensor_with_masked_fill():
    """Test Issue #18407: FakeTensor/lifted tensors from eq+expand+masked_fill_"""

    def get_attn_pad_mask(seq_q, seq_k):
        B, Lq = seq_q.size()
        B2, Lk = seq_k.size()
        assert B == B2
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # (B,1,Lk)
        return pad_attn_mask.expand(B, Lq, Lk)  # (B,Lq,Lk)

    class TinyMHA(nn.Module):
        def __init__(self, d_model=64, d_k=16, n_heads=4, dropout=0.1):
            super().__init__()
            self.h, self.dk = n_heads, d_k
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_V = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.proj = nn.Linear(d_k * n_heads, d_model, bias=False)
            self.ln = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, attn_mask):
            B, L, _ = x.shape
            q = self.W_Q(x).view(B, L, self.h, self.dk).transpose(1, 2)
            k = self.W_K(x).view(B, L, self.h, self.dk).transpose(1, 2)
            v = self.W_V(x).view(B, L, self.h, self.dk).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
            # This masked_fill_ with eq+expand mask triggers lifted_tensor
            scores.masked_fill_(attn_mask.unsqueeze(1), -1e9)
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, self.h * self.dk)
            out = self.drop(self.proj(ctx))
            return self.ln(out + x)

    class MiniModel(nn.Module):
        def __init__(self, vocab=1000, d_model=64):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.mha = TinyMHA(d_model=d_model, d_k=16, n_heads=4, dropout=0.1)
            self.proj = nn.Linear(d_model, vocab, bias=False)

        def forward(self, enc_inputs):
            x = self.emb(enc_inputs)
            mask = get_attn_pad_mask(enc_inputs, enc_inputs)
            y = self.mha(x, mask)
            logits = self.proj(y)
            return logits.reshape(-1, logits.size(-1))

    torch.manual_seed(42)
    model = MiniModel().eval()
    enc = torch.randint(0, 1000, (2, 5))
    enc[0, 0] = 0  # Ensure eq(0) path is taken

    # Export with torch.export (may emit warnings about lifted_tensor)
    ep = torch_export(model, (enc,))

    # This should not crash (Issue #18407)
    mod = from_exported_program(ep)

    # Verify the module was created successfully
    assert isinstance(mod, tvm.IRModule)
    # The module should have a main function
    assert len(mod.functions) > 0


if __name__ == "__main__":
    test_lifted_tensor_with_masked_fill()
    print("Test passed!")
