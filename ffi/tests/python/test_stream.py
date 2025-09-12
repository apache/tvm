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


import torch
from torch.utils import cpp_extension
import tvm_ffi
import tvm_ffi.cpp


torch_mod = cpp_extension.load_inline(
    name="add_one_cuda",
    cpp_sources="""
    void add_one_cuda(torch::Tensor x, torch::Tensor y);
    """,
    cuda_sources="""
    #include <c10/cuda/CUDAGuard.h>
    #include <c10/cuda/CUDAStream.h>

    __global__ void AddOneKernel(float* x, float* y, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
        y[idx] = x[idx] + 1;
        }
    }

    void add_one_cuda(torch::Tensor x, torch::Tensor y) {
        int64_t n = x.size(0);
        int64_t nthread_per_block = 256;
        int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
        const c10::cuda::OptionalCUDAGuard device_guard(x.device());
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x.data_ptr()),
                                                                static_cast<float*>(y.data_ptr()), n);
    }
    """,
    functions=["add_one_cuda"],
    extra_cflags=["-O3"],
)

ffi_mod = tvm_ffi.cpp.load_inline(
    name="hello",
    cpp_sources="""
        void add_one_cuda(DLTensor* x, DLTensor* y);
    """,
    cuda_sources="""
        __global__ void AddOneKernel(float* x, float* y, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
            y[idx] = x[idx] + 1;
            }
        }

        void add_one_cuda(DLTensor* x, DLTensor* y) {
            int64_t n = x->shape[0];
            int64_t nthread_per_block = 256;
            int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
            cudaStream_t stream = static_cast<cudaStream_t>(
                TVMFFIEnvGetCurrentStream(x->device.device_type, x->device.device_id));
            AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x->data),
                                                                    static_cast<float*>(y->data), n);
        }
    """,
    functions=["add_one_cuda"],
)


def test_cuda_graph(mod):

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    print(s.device_type, s.device_index, s.cuda_stream)
    with torch.cuda.stream(s), tvm_ffi.stream(str(s.device), s.cuda_stream):
        mod.add_one_cuda(x, y)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        mod.add_one_cuda(x, y)


print("testing torch")
test_cuda_graph(torch_mod)
print()
print("testing ffi")
test_cuda_graph(ffi_mod)
