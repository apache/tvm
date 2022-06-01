import tvm
from tvm import relay
from tvm.relay.backend.contrib.uma.ultra_trail.backend import UltraTrailBackend

import pytest
import torch
import tarfile
import tempfile
from pathlib import Path


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            16, 24, 9, bias=True, padding=4, stride=1, dilation=1, groups=1
        )
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(
            24, 24, 9, bias=False, padding=4, stride=1, dilation=1, groups=1
        )
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x + 42
        return x


# Target Registration
ut_backend = UltraTrailBackend()
ut_backend.register()

@pytest.mark.parametrize(
    "compound_target", 
    [
        [tvm.target.Target("c"), tvm.target.Target("ultra_trail", host=tvm.target.Target("c"))]
    ]
)
def test_ultra_trail(compound_target):
    torch_mod = TorchModel()
    # Pytorch frontend
    input_shape = (1, 16, 20)
    dummy_input = torch.randn(input_shape)
    scripted_model = torch.jit.trace(torch_mod, dummy_input).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [("input_data", input_shape)])

    # Relay target specific partitioning    
    mod = ut_backend.partition(mod)

    # Relay build (AOT C target)
    RUNTIME = tvm.relay.backend.Runtime("crt")
    EXECUTOR = tvm.relay.backend.Executor("aot", {"unpacked-api": True})

    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
    ):
        module = relay.build(mod, target=compound_target, runtime=RUNTIME, executor=EXECUTOR, params=params)

    with tempfile.TemporaryDirectory() as build_dir:
        build_dir = Path(build_dir)
        model_library_format_tar_path = build_dir / "lib.tar"
        model_library_format_tar_path.unlink(missing_ok=True)
        model_library_format_tar_path.parent.mkdir(parents=True, exist_ok=True)

        tvm.micro.export_model_library_format(module, model_library_format_tar_path)

        print("Built MLF Library: ")
        with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
            print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))
            tar_f.extractall(model_library_format_tar_path.parent)
