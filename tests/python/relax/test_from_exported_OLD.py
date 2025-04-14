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

import tvm
from tvm import relax
import tvm.testing
import numpy as np
import torch
from torch import nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
from torch.nn import Softmax, Upsample


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
    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax_pipeline)
    vm = relax.VirtualMachine(ex, dev)

    gpu_data = tvm.nd.array(raw_data_for_tvm, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params)

    pytorch_out = torch_module(torch_data)

    if isinstance(pytorch_out, tuple):
        for i in range(len(pytorch_out)):
            actual = gpu_out[i].numpy()
            desired = pytorch_out[i].detach().numpy()
            np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)
    else:
        actual = gpu_out[0].numpy()
        desired = pytorch_out.detach().numpy()
        np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)



@tvm.testing.parametrize_targets("cuda")
def test_full(target, dev):
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.full((2, 3), 3.141592)
            
    torch_module = FullModel().eval()

    raw_data = np.random.rand(3,3).astype("float32")

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


# Test index.Tensor # TODO aggregate into one big tet

@tvm.testing.parametrize_targets("cuda")
def test_index_tensor0(target, dev):
    class IndexModel0(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[torch.tensor([0])]
            
    torch_module = IndexModel0().eval()

    raw_data = np.random.rand(3,3).astype("float32")

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor1(target, dev):
    class IndexModel1(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[torch.tensor([[0]])]
            
    torch_module = IndexModel1().eval()

    raw_data = np.random.rand(2,3).astype("float32")

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor2(target, dev):
    class IndexTensorModel2(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[torch.tensor([0,2])]
            
    torch_module = IndexTensorModel2().eval()

    raw_data = np.random.rand(3,4).astype("float32")

    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor3(target, dev):
    class IndexTensorModel3(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[[0,2],[1,3]]]] 
        
    torch_module = IndexTensorModel3().eval()
    raw_data = np.random.rand(5,5,5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor4(target, dev):
    class IndexTensorModel4(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[1,4]]] 
        
    torch_module = IndexTensorModel4().eval()
    raw_data = np.random.rand(5,5,5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor5(target, dev):
    class IndexTensorModel5(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[[1,2,4]]]] 
        
    torch_module = IndexTensorModel5().eval()
    raw_data = np.random.rand(5,5,5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor6(target, dev):
    class IndexTensorModel6(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[0,1],[0,1]]]
    
    torch_module = IndexTensorModel6().eval()
    raw_data = np.random.rand(5,5,5,5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor7(target, dev):
    class IndexTensorModel7(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[0,1,2,3], [1,2,3,4], [2,3,4,0]]] # both args[0] and indices are expr.Var
        
    torch_module = IndexTensorModel7().eval()
    raw_data = np.random.rand(5,5,5,5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_index_tensor8(target, dev):
    class IndexTensorModel8(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[[[[0,1],[2,3]],[[2,3],[3,4]],[[2,4],[1,2]],[[0,4],[0,3]]]]
        
    torch_module = IndexTensorModel8().eval()
    raw_data = np.random.rand(5,5,5,5).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)



if __name__ == "__main__":
    tvm.testing.main()
