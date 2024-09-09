#include <cuda.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fa_kernel.hpp"

void main_kernel_launcher(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output, bool causal);
void main_kernel_launcher_no_tma(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output);

at::Tensor kernel_function(at::Tensor Q, at::Tensor K, at::Tensor V, bool causal) {
  at::Tensor output = torch::empty_like(Q);
  main_kernel_launcher(Q, K, V, output, causal);
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

void main_kernel_launcher(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output, bool causal) {
  int batch = Q.size(0);
  int seq_len = Q.size(1);
  int heads = Q.size(2);
  int dim = Q.size(3);
  int block_M = 0;
  int block_N = 0;
  int threads = 0;

  if (dim == 64) {
    block_M = 192;
    block_N = 128;
    threads = 16 * 32;
  } else if (dim == 128) {
    block_M = 128;
    block_N = causal ? 128 : 176;
    threads = 12 * 32;
  } else if (dim == 256) {
    block_M = 128;
    block_N = 80;
    threads = 12 * 32;
  } else {
    throw std::invalid_argument("Invalid dimension");
  }
  host_function(Flash_fwd_params{Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), batch, seq_len, heads, dim, block_M, block_N, threads});
}

void main_kernel_launcher_no_tma(at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor output) {
  host_function_no_tma(Flash_fwd_params{Q.data_ptr(), K.data_ptr(), V.data_ptr(), output.data_ptr(), Q.size(0), Q.size(1), Q.size(2), Q.size(3), 64, 64});
}