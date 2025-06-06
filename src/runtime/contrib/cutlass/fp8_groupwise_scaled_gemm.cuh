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

#include <cuda_fp16.h>
#include <float.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/ndarray.h>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

namespace tvm {
namespace runtime {

template <int Arch, typename TileShape, typename ClusterShape, typename ElementA, typename ElementB,
          typename ElementC, typename ElementBlockScale>
struct CutlassFP8GroupwiseGemm;

template <int Arch, typename TileShape, typename ClusterShape>
void tvm_cutlass_fp8_groupwise_scaled_gemm_impl(NDArray a, NDArray b, NDArray scales_a,
                                                NDArray scales_b, NDArray workspace,
                                                int64_t block_size_0, int64_t block_size_1,
                                                NDArray out) {
  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  static tvm::ffi::Function get_stream_func =
      tvm::ffi::Function::GetGlobalRequired("runtime.get_cuda_stream");
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream_func().cast<void*>());

  CHECK_GE(a->ndim, 2);
  CHECK_EQ(scales_a->ndim, a->ndim);
  CHECK_EQ(b->ndim, 2);
  CHECK_EQ(scales_b->ndim, 2);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, a->ndim);
  int64_t m = 1;
  for (int64_t i = 0; i < a->ndim - 1; ++i) {
    m *= a->shape[i];
  }
  int64_t n = b->shape[0];
  CHECK_EQ(a->shape[a->ndim - 1], b->shape[1]) << "Only col-major B is supported now.";
  int64_t k = a->shape[a->ndim - 1];

  // scales_a is col-major of (*a_shape[:-1], k / block_size)
  CHECK_EQ(scales_a->shape[0] * block_size_1, k);
  for (int64_t i = 1; i < scales_a->ndim; ++i) {
    CHECK_EQ(scales_a->shape[i], a->shape[i - 1]);
  }
  // scales_b is col-major of (k / block_size, n / block_size)
  CHECK_EQ((n + block_size_0 - 1) / block_size_0, scales_b->shape[0]);
  CHECK_EQ(scales_b->shape[1] * block_size_1, k);

  using tvm::runtime::DataType;
  CHECK_EQ(DataType(a->dtype), DataType::Float8E4M3FN());
  CHECK_EQ(DataType(b->dtype), DataType::Float8E4M3FN());
  CHECK_EQ(DataType(scales_a->dtype), DataType::Float(32));
  CHECK_EQ(DataType(scales_b->dtype), DataType::Float(32));
  CHECK_EQ(DataType(workspace->dtype), DataType::UInt(8));

  if (DataType(out->dtype) == DataType::Float(16)) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::half_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a->data),
                                        static_cast<cutlass::float_e4m3_t*>(b->data),
                                        static_cast<float*>(scales_a->data),
                                        static_cast<float*>(scales_b->data),
                                        static_cast<cutlass::half_t*>(out->data),
                                        static_cast<uint8_t*>(workspace->data),
                                        workspace->shape[0] * DataType(workspace->dtype).bytes(), m,
                                        n, k, 1, stream);
  } else if (DataType(out->dtype) == DataType::BFloat(16)) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::bfloat16_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a->data),
                                        static_cast<cutlass::float_e4m3_t*>(b->data),
                                        static_cast<float*>(scales_a->data),
                                        static_cast<float*>(scales_b->data),
                                        static_cast<cutlass::bfloat16_t*>(out->data),
                                        static_cast<uint8_t*>(workspace->data),
                                        workspace->shape[0] * DataType(workspace->dtype).bytes(), m,
                                        n, k, 1, stream);
  } else {
    LOG(FATAL) << "Unsupported output dtype: " << DataType(out->dtype);
  }
}

template <int Arch, typename TileShape, typename ClusterShape>
void tvm_cutlass_fp8_groupwise_scaled_bmm_impl(NDArray a, NDArray b, NDArray scales_a,
                                               NDArray scales_b, NDArray workspace,
                                               int64_t block_size_0, int64_t block_size_1,
                                               NDArray out) {
  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  static tvm::ffi::Function get_stream_func =
      tvm::ffi::Function::GetGlobalRequired("runtime.get_cuda_stream");
  cudaStream_t stream = static_cast<cudaStream_t>(get_stream_func().cast<void*>());

  CHECK_EQ(a->ndim, 3);
  CHECK_EQ(scales_a->ndim, 3);
  CHECK_EQ(b->ndim, 3);
  CHECK_EQ(scales_b->ndim, 3);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 3);
  int64_t batch_size = a->shape[0];
  int64_t m = a->shape[1];
  int64_t n = b->shape[1];
  CHECK_EQ(a->shape[2], b->shape[2]) << "Only col-major B is supported now.";
  int64_t k = a->shape[2];
  CHECK_EQ(b->shape[0], batch_size);
  CHECK_EQ(scales_a->shape[0], batch_size);
  CHECK_EQ(scales_b->shape[0], batch_size);
  CHECK_EQ(out->shape[0], batch_size);

  // scales_a is col-major of (batch_size, m, k / block_size)
  CHECK_EQ(scales_a->shape[1] * block_size_1, k);
  CHECK_EQ(scales_a->shape[2], m);
  // scales_b is col-major of (k / block_size, n / block_size)
  CHECK_EQ(scales_b->shape[1] * block_size_0, n);
  CHECK_EQ(scales_b->shape[2] * block_size_1, k);

  using tvm::runtime::DataType;
  CHECK_EQ(DataType(a->dtype), DataType::Float8E4M3FN());
  CHECK_EQ(DataType(b->dtype), DataType::Float8E4M3FN());
  CHECK_EQ(DataType(scales_a->dtype), DataType::Float(32));
  CHECK_EQ(DataType(scales_b->dtype), DataType::Float(32));
  CHECK_EQ(DataType(workspace->dtype), DataType::UInt(8));

  if (DataType(out->dtype) == DataType::Float(16)) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::half_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a->data),
                                        static_cast<cutlass::float_e4m3_t*>(b->data),
                                        static_cast<float*>(scales_a->data),
                                        static_cast<float*>(scales_b->data),
                                        static_cast<cutlass::half_t*>(out->data),
                                        static_cast<uint8_t*>(workspace->data),
                                        workspace->shape[0] * DataType(workspace->dtype).bytes(), m,
                                        n, k, batch_size, stream);
  } else if (DataType(out->dtype) == DataType::BFloat(16)) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::bfloat16_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a->data),
                                        static_cast<cutlass::float_e4m3_t*>(b->data),
                                        static_cast<float*>(scales_a->data),
                                        static_cast<float*>(scales_b->data),
                                        static_cast<cutlass::bfloat16_t*>(out->data),
                                        static_cast<uint8_t*>(workspace->data),
                                        workspace->shape[0] * DataType(workspace->dtype).bytes(), m,
                                        n, k, batch_size, stream);
  } else {
    LOG(FATAL) << "Unsupported output dtype: " << DataType(out->dtype);
  }
}

}  // namespace runtime
}  // namespace tvm
