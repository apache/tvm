/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../../../3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/cutlass_preprocessors.h"

namespace tvm {
namespace runtime {

// This packed function applies the set of preprocessings on the weight, which are required by
// the FT kernel. They consist of permuting / transposing / interleaving the weight elements,
// and changing the weight dtype to be unsigned by adding a bias. The output has the same size
// as the input.
//
// These processes are not well documented, so we wrap them into a packed function and use it as a
// black box.
//
// The preprocessing functions are defined in C++, so we need to copy the input weight to CPU.
TVM_REGISTER_GLOBAL("cutlass.ft_preprocess_weight")
    .set_body_typed([](NDArray packed_weight, int sm, bool is_int4) {
      int rows = packed_weight->shape[0];
      int cols = packed_weight->shape[1];
      std::vector<int8_t> input_cpu(rows * cols);
      std::vector<int8_t> output_cpu(rows * cols);
      packed_weight.CopyToBytes(input_cpu.data(), input_cpu.size());
      // multiply cols by 2 since the "col" params in preprocess_weights refers to the column of
      // the unpacked weight.
      if (is_int4) {
        cols *= 2;
      }
      fastertransformer::preprocess_weights(output_cpu.data(), input_cpu.data(), rows, cols,
                                            is_int4, sm);
      auto out = NDArray::Empty(packed_weight.Shape(), packed_weight->dtype, packed_weight->device);
      out.CopyFromBytes(output_cpu.data(), output_cpu.size());
      return out;
    });

}  // namespace runtime
}  // namespace tvm
