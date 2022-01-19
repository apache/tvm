import tvm
import tvm.relay.backend.contrib.generic
from tvm.relay.backend.contrib.generic.ultra_trail.partitioner import UltraTrailPartitioner
from tvm import relay

import torch
import tarfile
from pathlib import Path


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            16, 24, 9, bias=False, padding=4, stride=1, dilation=1, groups=1
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


def main():
    torch_mod = TorchModel()

    # Pytorch frontend
    input_shape = (1, 16, 20)
    dummy_input = torch.randn(input_shape)
    scripted_model = torch.jit.trace(torch_mod, dummy_input).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [("input_data", input_shape)])

    # Relay target specific partitioning
    mod = UltraTrailPartitioner()(mod, params)

    # Relay build (AOT C target)
    TARGET = tvm.target.Target("c")
    RUNTIME = tvm.relay.backend.Runtime("crt")
    EXECUTOR = tvm.relay.backend.Executor("aot", {"unpacked-api": True})

    with tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
    ):
        module = relay.build(mod, target=TARGET, runtime=RUNTIME, executor=EXECUTOR, params=params)

    model_library_format_tar_path = Path("build/lib.tar")
    model_library_format_tar_path.unlink(missing_ok=True)
    model_library_format_tar_path.parent.mkdir(parents=True, exist_ok=True)

    tvm.micro.export_model_library_format(module, model_library_format_tar_path)

    print("Built MLF Library: ")
    with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
        print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))
        tar_f.extractall(model_library_format_tar_path.parent)


if __name__ == "__main__":
    main()
