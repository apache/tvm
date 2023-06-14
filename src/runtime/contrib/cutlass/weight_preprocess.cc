#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../../../3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/cutlass_preprocessors.h"

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("cutlass.ft_preprocess_weight").set_body_typed([](NDArray packed_weight) {
  int rows = packed_weight->shape[0];
  int cols = packed_weight->shape[1];
  std::vector<int8_t> input_cpu(rows * cols);
  std::vector<int8_t> output_cpu(rows * cols);
  packed_weight.CopyToBytes(input_cpu.data(), input_cpu.size());
  fastertransformer::preprocess_weights(output_cpu.data(), input_cpu.data(), rows, cols * 2, true, 80);
  auto out = NDArray::Empty(packed_weight.Shape(), packed_weight->dtype, packed_weight->device);
  out.CopyFromBytes(output_cpu.data(), output_cpu.size());
  return out;
});

}  // namespace runtime
}  // namespace tvm
