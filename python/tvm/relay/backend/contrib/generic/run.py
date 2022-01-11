import tvm
import tvm.relay.backend.contrib.generic
from tvm import relay
from tvm.relay.backend.contrib.generic.ultra_trail.pattern import match_ultra_trail

import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            16, 24, 9, bias=False, padding=0, stride=1, dilation=1, groups=1
        )

    def forward(self, x):
        x = self.conv(x)
        return x

def main():
    torch_mod = TorchModel()

    # Pytorch frontend
    input_shape = (1, 16, 20)
    dummy_input = torch.randn(input_shape)
    scripted_model = torch.jit.trace(torch_mod, dummy_input).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [("input_data", input_shape)])

    mod = match_ultra_trail(mod)
    lib = relay.build(mod, tvm.target.Target("c"))
    print(lib)


if __name__ == "__main__":
    main()
