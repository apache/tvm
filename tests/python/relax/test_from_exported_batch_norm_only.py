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

# TODO remove
import sys
sys.path.append('/ssd1/htalendr/tvm/python') # Refer to local TVM build

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

    # mod_from_torch.show() # TODO remove

    tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)

    relax_pipeline = relax.get_default_pipeline(tvm.target.Target.from_device(tvm.cuda()))
    ex = relax.build(tvm_mod, target=target, relax_pipeline=relax_pipeline)
    vm = relax.VirtualMachine(ex, dev)

    gpu_data = tvm.nd.array(raw_data_for_tvm, dev)
    gpu_params = [tvm.nd.array(p, dev) for p in tvm_params["main"]]
    gpu_out = vm["main"](gpu_data, *gpu_params)

    pytorch_out = torch_module(torch_data).detach().numpy()

    print("type of pytorch_out", type(pytorch_out))
    print("pytorch output shape", pytorch_out.shape)

    print("len of gpu_out", len(gpu_out)) # 1 for all tests
    print("type of gpu_out[0]", type(gpu_out[0])) # tvm.ir.container.Array for batch norm, tvm.runtime.ndarray.NDArray for both existing tests 
    print("gpu_out[0] shape", gpu_out[0].shape) # defined for tests that work 

    actual = gpu_out[0].numpy()
    desired = pytorch_out

    np.testing.assert_allclose(actual=actual, desired=desired, rtol=1e-5, atol=1e-5)


@tvm.testing.parametrize_targets("cuda")
def test_detach_no_change(target, dev):
    # In TVM, detach() is just identity
    class DetachTester(nn.Module):
        def forward(self, x):
            detached = x.detach()
            return detached

    raw_data = np.ones((2, 2)).astype(np.float32)
    torch_module = DetachTester().eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


@tvm.testing.parametrize_targets("cuda")
def test_upsample_with_scale_factor(target, dev):
    """
    The Upsample module can be used with the size arugment or the scale
    factor argument but not both. This tests the latter.
    """
    batch_size = 2
    channels = 3
    height, width = 32, 32

    torch_module = Upsample(
        size=None, scale_factor=7, mode="nearest", align_corners=None, recompute_scale_factor=True
    )

    raw_data = np.random.rand(batch_size, channels, height, width).astype("float32")
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


# # TODO in a program! to make sure dimensions work 
# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm_prog(target, dev):
#     # No momentum, eval
#     raw_data = np.random.randn(8, 8, 4, 4).astype(np.float32)

#     class BatchNormWrapper(nn.Module):
#         def __init__(self):
#             super(BatchNormWrapper, self).__init__()
#             self.bn = nn.BatchNorm2d(
#                 8, eps=1e-02, momentum=0.0, affine=False, track_running_stats=True
#             )
#         def forward(self, x):
#             x = self.bn(x)
#             x = x + 1
#             return x    
#     torch_module = BatchNormWrapper().eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module, target, dev)


# # TODO can combine the tests together (they are separete to know which test fails)
# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm0(target, dev):
#     # Eval, no momentum, with affine, without running stats
#     raw_data = np.random.randn(8, 8, 4, 4).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(
#         8, eps=1e-02, momentum=0.0, affine=True, track_running_stats=False, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm1(target, dev):
#     # With momentum, no affine, with running stats
#     raw_data = np.random.randn(1, 4, 2, 2).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(
#         4, eps=1e-05, momentum=0.0, affine=False, track_running_stats=True, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm2(target, dev):
#     # Default args, eval
#     raw_data = np.random.randn(4, 2, 2, 2).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(2).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)


# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm3(target, dev):
#     # No momentum, eval
#     raw_data = np.random.randn(8, 8, 4, 4).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(
#         8, eps=1e-02, momentum=0.0, affine=False, track_running_stats=True, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

#     # With momentum, eval
#     raw_data = np.random.randn(1, 4, 2, 2).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(
#         4, eps=1e-05, momentum=0.0, affine=False, track_running_stats=True, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

#     # Default args, eval
#     raw_data = np.random.randn(4, 2, 2, 2).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(2).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)


# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm4(target, dev):
#     # No momentum, eval
#     raw_data = np.random.randn(8, 8, 4, 4).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(
#         8, eps=1e-02, momentum=0.0, affine=False, track_running_stats=True, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

#     # With momentum, eval
#     raw_data = np.random.randn(1, 4, 2, 2).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(
#         4, eps=1e-05, momentum=0.0, affine=False, track_running_stats=True, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

#     # Default args, eval
#     raw_data = np.random.randn(4, 2, 2, 2).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(2).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm5(target, dev):
#     # No momentum, eval, no running stats
#     raw_data = np.random.randn(8, 8, 4, 4).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d(
#         8, eps=1e-02, momentum=0.0, affine=False, track_running_stats=False, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

# @tvm.testing.parametrize_targets("cuda")
# def test_batch_norm6(target, dev):
#     # Small input
#     raw_data = np.array([[[[ 0.5]]], [[[1.5]]]]).astype(np.float32)
#     torch_module0 = nn.BatchNorm2d( # TODO what does the 8 do? (feature num)
#         8, eps=0.2, momentum=0.0, affine=False, track_running_stats=False, device=None, dtype=None
#     ).eval()
#     assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)

@tvm.testing.parametrize_targets("cuda")
def test_batch_norm7(target, dev):
    # Eval, small input, no momentum, with affine, with running stats
    raw_data = np.array([[[[ 0.5]]], [[[1.5]]]]).astype(np.float32)
    torch_module0 = nn.BatchNorm2d(
        8, eps=1e-02, momentum=0.0, affine=True, track_running_stats=False, device=None, dtype=None
    ).eval()
    assert_torch_output_vs_tvm_from_exported_to_cuda(raw_data, torch_module0, target, dev)




if __name__ == "__main__":
    tvm.testing.main()
