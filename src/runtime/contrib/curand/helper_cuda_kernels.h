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
#ifndef TVM_RUNTIME_CONTRIB_CURAND_HELPER_CUDA_KERNELS_H_
#define TVM_RUNTIME_CONTRIB_CURAND_HELPER_CUDA_KERNELS_H_

#include <curand.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace curand {

/*!
 * \brief An auxiliary function to convert an FP32 array to FP16.
 * \param src The source FP32 array.
 * \param dst The destination FP16 array.
 * \param num The number of elements in the array.
 */
void ConvertFp32toFp16(const void* src, void* dst, int64_t num);

}  // namespace curand
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CURAND_HELPER_CUDA_KERNELS_H_
