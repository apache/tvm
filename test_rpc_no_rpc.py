#!/usr/bin/env python3
"""
Test without RPC first to isolate the issue
"""
import numpy as np
import tvm
from tvm import relax
from tvm.contrib import utils

try:
    import torch
    from torch.export import export
    from tvm.relax.frontend.torch import from_exported_program
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")
    exit(0)

print("=" * 60)
print("Testing PyTorch → Relax WITHOUT RPC")
print("=" * 60)

# Step 1: PyTorch model
class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )
    def forward(self, data):
        return self.net(data)

print("\n[1/5] Exporting PyTorch model...")
torch_model = TorchMLP().eval()
example_args = (torch.randn(1, 1, 28, 28, dtype=torch.float32),)
with torch.no_grad():
    exported_program = export(torch_model, example_args)
print("✓ Exported")

# Step 2: Convert to Relax
print("\n[2/5] Converting to Relax...")
mod = from_exported_program(exported_program, keep_params_as_input=True)
mod, params = relax.frontend.detach_params(mod)
print(f"✓ Converted (parameters: {len(params['main'])})")

# Step 3: Compile
print("\n[3/5] Compiling...")
target = tvm.target.Target("llvm")
pipeline = relax.get_pipeline()
with target:
    built_mod = pipeline(mod)
executable = tvm.compile(built_mod, target=target)
print("✓ Compiled")

# Step 4: Export and reload (like in export_and_load_executable.py)
print("\n[4/5] Exporting and reloading...")
temp = utils.tempdir()
lib_path = temp.relpath("model.so")
executable.export_library(lib_path)

# Reload (NOT via RPC)
loaded_mod = tvm.runtime.load_module(str(lib_path))
dev = tvm.cpu(0)
vm = relax.VirtualMachine(loaded_mod, dev)
print("✓ Loaded")

# Step 5: Run
print("\n[5/5] Running inference...")
input_data = np.random.randn(1, 1, 28, 28).astype("float32")
vm_input = tvm.runtime.tensor(input_data, dev)
vm_params = [tvm.runtime.tensor(p, dev) for p in params["main"]]

output = vm["main"](vm_input, *vm_params)

if hasattr(output, "__len__") and len(output) > 0:
    result = output[0]
else:
    result = output

result_np = result.numpy()
print(f"✓ Inference completed")
print(f"  Output shape: {result_np.shape}")
print(f"  Predicted class: {np.argmax(result_np)}")

print("\n" + "=" * 60)
print("✓ Non-RPC workflow works!")
print("=" * 60)

