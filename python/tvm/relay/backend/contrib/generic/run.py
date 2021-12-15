import tvm
from tvm import relay
from tvm.relay.backend.contrib import generic

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


custom_target_name = "ultra_trail"
def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, f"target.{custom_target_name}")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.conv1d")


def main():
    torch_mod = TorchModel()

    # Pytorch frontend
    input_shape = (1, 16, 20)
    dummy_input = torch.randn(input_shape)
    scripted_model = torch.jit.trace(torch_mod, dummy_input).eval()
    mod, params = relay.frontend.from_pytorch(scripted_model, [("input_data", input_shape)])

    mod = relay.transform.AnnotateTarget(custom_target_name)(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    print(mod)

    lib = relay.build(mod, tvm.target.Target("c"))
    print(lib)


if __name__ == "__main__":
    main()
