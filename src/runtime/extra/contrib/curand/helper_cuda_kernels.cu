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

#include "./helper_cuda_kernels.h"

namespace tvm {
namespace runtime {
namespace curand {

__global__ void KernelFp32ToFp16(const float* src, half* dst, int num) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < num) {
    dst[idx] = src[idx];
  }
}

void ConvertFp32toFp16(const void* _src, void* _dst, int64_t num) {
  const float* src = static_cast<const float*>(_src);
  half* dst = static_cast<half*>(_dst);
  KernelFp32ToFp16<<<(num + 255) / 256, 256>>>(src, dst, num);
}

}  // namespace curand
}  // namespace runtime
}  // namespace tvm
