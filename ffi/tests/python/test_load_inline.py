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
import sys

try:
    import torch
except ImportError:
    torch = None

import tvm_ffi.cpp
from tvm_ffi.module import Module


@pytest.mark.xfail(sys.platform.startswith("win"), reason="needs to robustify windows support")
def test_load_inline_cpp():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(DLTensor* x, DLTensor* y) {
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
        functions=["add_one_cpu"],
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


@pytest.mark.xfail(sys.platform.startswith("win"), reason="needs to robustify windows support")
def test_load_inline_cpp_with_docstrings():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(DLTensor* x, DLTensor* y) {
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
        functions={"add_one_cpu": "add two float32 1D tensors element-wise"},
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


@pytest.mark.xfail(sys.platform.startswith("win"), reason="needs to robustify windows support")
def test_load_inline_cpp_multiple_sources():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=[
            r"""
            void add_one_cpu(DLTensor* x, DLTensor* y) {
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
            r"""
            void add_two_cpu(DLTensor* x, DLTensor* y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";
              for (int i = 0; i < x->shape[0]; ++i) {
                static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 2;
              }
            }
        """,
        ],
        functions=["add_one_cpu", "add_two_cpu"],
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


@pytest.mark.xfail(sys.platform.startswith("win"), reason="needs to robustify windows support")
def test_load_inline_cpp_build_dir():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(DLTensor* x, DLTensor* y) {
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
        functions=["add_one_cpu"],
        build_directory="./build_add_one",
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_load_inline_cuda():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cuda_sources=r"""
            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }

            void add_one_cuda(DLTensor* x, DLTensor* y) {
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
                  TVMFFIEnvGetStream(x->device.device_type, x->device.device_id));
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x->data),
                                                                     static_cast<float*>(y->data), n);
            }
        """,
        functions=["add_one_cuda"],
    )

    if torch is not None:
        x_cuda = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
        y_cuda = torch.empty_like(x_cuda)
        mod.add_one_cuda(x_cuda, y_cuda)
        torch.testing.assert_close(x_cuda + 1, y_cuda)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_load_inline_cuda_with_env_tensor_allocator():
    if not hasattr(torch.Tensor, "__c_dlpack_tensor_allocator__"):
        pytest.skip("Torch does not support __c_dlpack_tensor_allocator__")
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            #include <tvm/ffi/container/tensor.h>

            tvm::ffi::Tensor return_add_one(DLTensor* x);
        """,
        cuda_sources=r"""
            #include <tvm/ffi/container/tensor.h>

            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }
            namespace ffi = tvm::ffi;

             ffi::Tensor return_add_one(DLTensor* x) {
              // implementation of a library function
              TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
              // allocate a new tensor with the env tensor allocator
              // it will be redirected to torch.empty when calling the function
              ffi::Tensor y = ffi::Tensor::FromDLPackAlloc(
                TVMFFIEnvGetTensorAllocator(), ffi::Shape({x->shape[0]}), f32_dtype, x->device);
              int64_t n = x->shape[0];
              int64_t nthread_per_block = 256;
              int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
              // Obtain the current stream from the environment
              // it will be set to torch.cuda.current_stream() when calling the function
              // with torch.Tensors
              cudaStream_t stream = static_cast<cudaStream_t>(
                  TVMFFIEnvGetStream(x->device.device_type, x->device.device_id));
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x->data),
                                                                     static_cast<float*>(y->data), n);
              return y;
            }
        """,
        functions=["return_add_one"],
    )

    if torch is not None:
        x_cuda = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
        y_cuda = mod.return_add_one(x_cuda)
        assert isinstance(y_cuda, torch.Tensor)
        assert y_cuda.shape == (5,)
        assert y_cuda.dtype == torch.float32
        torch.testing.assert_close(x_cuda + 1, y_cuda)
        assert y_cuda.is_cuda


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="Requires torch and CUDA"
)
def test_load_inline_both():
    mod: Module = tvm_ffi.cpp.load_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(DLTensor* x, DLTensor* y) {
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

            void add_one_cuda(DLTensor* x, DLTensor* y);
        """,
        cuda_sources=r"""
            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }

            void add_one_cuda(DLTensor* x, DLTensor* y) {
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
                  TVMFFIEnvGetStream(x->device.device_type, x->device.device_id));
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x->data),
                                                                     static_cast<float*>(y->data), n);
            }
        """,
        functions=["add_one_cpu", "add_one_cuda"],
    )

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)

    x_cuda = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
    y_cuda = torch.empty_like(x_cuda)
    mod.add_one_cuda(x_cuda, y_cuda)
    torch.testing.assert_close(x_cuda + 1, y_cuda)
