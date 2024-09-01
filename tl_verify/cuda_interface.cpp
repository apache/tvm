#include <cuda.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fa_kernel.hpp"

void main_kernel_launcher(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output);
void main_kernel_launcher_no_tma(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output);

at::Tensor kernel_function(at::Tensor Q, at::Tensor K, at::Tensor V) {
  at::Tensor output = torch::empty_like(Q);
  main_kernel_launcher(Q, K, V, output);
  return output;
}

at::Tensor kernel_function_no_tma(at::Tensor Q, at::Tensor K, at::Tensor V) {
  at::Tensor output = torch::empty_like(Q);
  main_kernel_launcher_no_tma(Q, K, V, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel_function", &kernel_function, "FA Kernel Function");
  m.def("kernel_function_no_tma", &kernel_function_no_tma, "FA Kernel Launcher");
}

void main_kernel_launcher(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output) {
  host_function(Flash_fwd_params{Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), Q.size(0), Q.size(1), Q.size(2), Q.size(3), 64, 64});
}

void main_kernel_launcher_no_tma(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output) {
  host_function_no_tma(Flash_fwd_params{Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), Q.size(0), Q.size(1), Q.size(2), Q.size(3), 64, 64});
}