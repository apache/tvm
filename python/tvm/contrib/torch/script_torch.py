# pylint: disable=missing-module-docstring
import torch
import tvm

class TVMScriptModule(torch.nn.Module):
    def __init__(self, module : tvm.runtime.Module, input_shape : tuple, output_shape : tuple, 
    device : str = None, target : str = None ):
        super().__init__()
        self.engine = None

        if device is not None:
            self.to(device)
        
        self.engine = torch.classes.tvm_dsoop.TVMScriptModule(module, input_shape, output_shape, self.device, target)
        # self.mod = tvm.build(ir_module, target=target)
        # self.input_shape = input_shape
        # self.output_shape = output_shape

    def forward(self, torch_input : torch.Tensor):
        r"""Call tvm module to forward"""
        return self.engine.forward(torch_input)
        # tensor_numpy = torch_input.numpy()
        # assert tensor_numpy.shape == self.input_shape
        # tensor_input = tvm.nd.array(tensor_numpy)
        # tensor_output = tvm.nd.empty(self.output_shape)
        # self.mod(tensor_input, tensor_output)
        # torch_output = torch.from_numpy(tensor_output.numpy())
        # return torch_output