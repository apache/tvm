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


def assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module):
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
    target = tvm.target.Target.from_device(tvm.cuda())

    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)

    gpu_data = tvm.nd.array(raw_data_for_tvm, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params)

    pytorch_out = torch_module(torch_data).detach().numpy()
    actual = gpu_out[0].numpy()
    desired = pytorch_out
    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)


def test_copy_():
    class CopyTester(nn.Module):
        def __init__(self, size):
            super().__init__()
            # self.buffer = torch.zeros(size)
            self.register_buffer("buffer", torch.zeros(size))

        def forward(self, x):
            self.buffer.copy_(x)

            return x * 3 + self.buffer * 5

    size = (2, 2)
    raw_data = np.random.rand(*size).astype(np.float32)
    torch_module = CopyTester(size).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module)


def test_detach_no_change():
    """Most of the time, in TVM, detach() should basically be identity"""

    class DetachTester(nn.Module):
        def forward(self, x):
            detached = x.detach()
            return detached

    raw_data = np.ones((2, 2)).astype(np.float32)
    torch_module = DetachTester().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module)


# TODO test below fails! Is there a way to implement detach such that the
#  memory is shared with the input?
# def test_detach_with_change():
#     """ Testing that detach() shares memory with original tensor"""
#     class DetachTester(nn.Module):
#         def forward(self, x):
#             detached = x.detach()

#             # Test that detached shares same memory as x
#             x[0][0] = 42.0

#             return detached

#     raw_data = np.ones((2,2)).astype(np.float32)
#     torch_module = DetachTester().eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module)


if __name__ == "__main__":
    tvm.testing.main()
