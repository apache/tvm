import torch
import tvm_ffi.cpp
from tvm_ffi.module import Module


def main():
    mod: Module = tvm_ffi.cpp.load_inline(
        name='hello',
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
        cpp_functions={'add_one_cpu': 'AddOne'},
        cuda_functions={'add_one_cuda': 'AddOneCUDA'},
    )

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y = torch.empty_like(x)
    mod.add_one_cpu(x, y)
    torch.testing.assert_close(x + 1, y)

    x_cuda = x.cuda()
    y_cuda = torch.empty_like(x_cuda)
    mod.add_one_cuda(x_cuda, y_cuda)
    torch.testing.assert_close(x_cuda + 1, y_cuda)


if __name__ == "__main__":
    main()
