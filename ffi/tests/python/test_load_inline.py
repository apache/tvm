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

import pytest
import numpy

try:
    import torch
except ImportError:
    torch = None

import tvm_ffi.cpp
from tvm_ffi.module import Module


def test_load_inline_cpp():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_source=r"""
            void AddOne(DLTensor* x, DLTensor* y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";
              for (int i = 0; i < x->shape[0]; ++i) {
                static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 1;
              }
            }
        """,
        cpp_functions={"add_one_cpu": "AddOne"},
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


@pytest.mark.skip(reason="Requires CUDA")
def test_load_inline_cuda():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cuda_source=r"""
            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }

            void AddOneCUDA(DLTensor* x, DLTensor* y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";

              int64_t n = x->shape[0];
              int64_t nthread_per_block = 256;
              int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
              // Obtain the current stream from the environment
              // it will be set to torch.cuda.current_stream() when calling the function
              // with torch.Tensors
              cudaStream_t stream = static_cast<cudaStream_t>(
                  TVMFFIEnvGetCurrentStream(x->device.device_type, x->device.device_id));
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x->data),
                                                                     static_cast<float*>(y->data), n);
            }
        """,
        cuda_functions={"add_one_cuda": "AddOneCUDA"},
    )

    if torch is not None:
        x_cuda = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
        y_cuda = torch.empty_like(x_cuda)
        mod.add_one_cuda(x_cuda, y_cuda)
        torch.testing.assert_close(x_cuda + 1, y_cuda)


@pytest.mark.skip(reason="Requires CUDA")
def test_load_inline_both():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_source=r"""
            void AddOne(DLTensor* x, DLTensor* y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";
              for (int i = 0; i < x->shape[0]; ++i) {
                static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 1;
              }
            }
        """,
        cuda_source=r"""
            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }

            void AddOneCUDA(DLTensor* x, DLTensor* y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";

              int64_t n = x->shape[0];
              int64_t nthread_per_block = 256;
              int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
              // Obtain the current stream from the environment
              // it will be set to torch.cuda.current_stream() when calling the function
              // with torch.Tensors
              cudaStream_t stream = static_cast<cudaStream_t>(
                  TVMFFIEnvGetCurrentStream(x->device.device_type, x->device.device_id));
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x->data),
                                                                     static_cast<float*>(y->data), n);
            }
        """,
        cpp_functions={"add_one_cpu": "AddOne"},
        cuda_functions={"add_one_cuda": "AddOneCUDA"},
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)

    if torch is not None:
        x_cuda = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
        y_cuda = torch.empty_like(x_cuda)
        mod.add_one_cuda(x_cuda, y_cuda)
        torch.testing.assert_close(x_cuda + 1, y_cuda)
