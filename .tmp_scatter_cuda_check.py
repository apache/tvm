import numpy as np
import torch
from torch import nn
from torch.export import export
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.relax.backend.cuda import get_default_pipeline

class ScatterValue(nn.Module):
    def forward(self, x, index):
        return x.scatter(1, index, 0.5)

torch.manual_seed(0)
x = torch.randn(4, 8, dtype=torch.float32)
idx = torch.randint(0, 8, (4, 2), dtype=torch.int64)

mod = from_exported_program(export(ScatterValue(), args=(x, idx)))
tgt = tvm.target.Target('cuda')
with tgt:
    mod = get_default_pipeline(tgt)(mod)

ex = relax.build(mod, tgt, relax_pipeline=None)
vm = relax.VirtualMachine(ex, tvm.cuda(0))
out = vm['main'](
    tvm.runtime.tensor(x.numpy(), device=tvm.cuda(0)),
    tvm.runtime.tensor(idx.numpy(), device=tvm.cuda(0)),
)
out_np = out.numpy() if hasattr(out, 'numpy') else out[0].numpy()
ref_np = ScatterValue()(x, idx).numpy()

print('shape_match', out_np.shape == ref_np.shape)
print('allclose', np.allclose(out_np, ref_np, rtol=1e-5, atol=1e-6))
print('max_abs_diff', float(np.max(np.abs(out_np - ref_np))))
