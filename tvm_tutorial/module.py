import torch
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.testing import resnet, squeezenet, mobilenet


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(16, 24, 9, bias=False, padding=0, stride=1, dilation=1, groups=1),
            torch.nn.BatchNorm1d(24),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Sequential(torch.nn.Linear(24, 10), torch.nn.Softmax(dim=0),)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(axis=-1)
        x = self.fc(x)
        return x


class TVMModule:
    def __init__(self, torch_model=True):
        self.torch_model = torch_model
        if torch_model:
            self.input_shape = [1, 16, 100]
            self.input_data = torch.randn(self.input_shape)
            self.input_name = "input0"
        else:
            self.input_shape = [1, 3, 224, 224]
            self.input_data = torch.randn(self.input_shape)
            self.input_name = "data"


    def load(self):
        print("Load module...")
        if self.torch_model:
            scripted_model = torch.jit.trace(TorchModel(), self.input_data).eval()
            shape_list = [(self.input_name, self.input_shape)]
            return relay.frontend.from_pytorch(scripted_model, shape_list)
        else:
            return resnet.get_workload()

    def benchmark(self, lib):
        dev = tvm.cpu(0)
        dtype = "float32"
        m = graph_executor.GraphModule(lib["default"](dev))
        # Set inputs
        m.set_input(self.input_name, tvm.nd.array(self.input_data.numpy().astype(dtype)))
        # Evaluate
        print("Evaluate inference time cost...")
        print(m.benchmark(dev, repeat=10))
