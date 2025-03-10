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

import numpy as np
import torch
from torch.export import export

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev):
    """
    This util ensures that a torch module can successfully be exported to TVM
    using torch.export and that the resuling IR program gives the same result
    as PyTorch when ran on CUDA.
    """
    raw_data_for_tvm = raw_data.copy()  # In case the data is modified
    torch_data = torch.from_numpy(raw_data)
    example_args = (torch_data,)

    with torch.no_grad():
        exported_program = export(torch_module, example_args)
        mod_from_torch = from_exported_program(exported_program, keep_params_as_input=True)

    tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)

    relax_pipeline = relax.get_default_pipeline(tvm.target.Target.from_device(tvm.cuda()))
    # TODO try pipeline below?
    # releax_pipeline = relax.backend.cuda.pipeline.get_default_pipeline(target)
    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax_pipeline)
    vm = relax.VirtualMachine(ex, dev)

    gpu_data = tvm.nd.array(raw_data_for_tvm, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params)

    pytorch_out = torch_module(torch_data).detach().numpy()
    actual = gpu_out[0].numpy()
    desired = pytorch_out
    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets("cuda")
def test_tensor_clamp(target, dev):
    class ClampBothTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("min_val", torch.tensor(-1.0))
            self.register_buffer("max_val", torch.tensor(1.0))

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    class ClampBothInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = -1
            self.max_val = 1

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    class ClampMinOnlyTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("min_val", torch.tensor(0.0))

        def forward(self, x):
            return x.clamp(min=self.min_val)

    class ClampMinOnlyInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = 0

        def forward(self, x):
            return x.clamp(min=self.min_val)

    class ClampMaxOnlyTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("max_val", torch.tensor(0.5))

        def forward(self, x):
            return x.clamp(max=self.max_val)

    class ClampMaxOnlyInt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.max_val = 0.5

        def forward(self, x):
            return x.clamp(max=self.max_val)

    class ClampDifferentValues(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.min_val = -2
            self.max_val = 2

        def forward(self, x):
            return x.clamp(min=self.min_val, max=self.max_val)

    # Create random data with values outside our clamp ranges
    raw_data = np.random.uniform(-3.0, 3.0, (2, 3, 4, 5)).astype(np.float32)

    torch_module0 = ClampBothTensor().eval()
    torch_module1 = ClampBothInt().eval()
    torch_module2 = ClampMinOnlyTensor().eval()
    torch_module3 = ClampMinOnlyInt().eval()
    torch_module4 = ClampMaxOnlyTensor().eval()
    torch_module5 = ClampMaxOnlyInt().eval()
    torch_module6 = ClampDifferentValues().eval()

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module1, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module2, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module3, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module4, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module5, target, dev)
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module6, target, dev)


if __name__ == "__main__":
    tvm.testing.main()
