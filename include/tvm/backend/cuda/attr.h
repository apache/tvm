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

#ifndef TVM_BACKEND_CUDA_ATTR_H_
#define TVM_BACKEND_CUDA_ATTR_H_

namespace tvm {
namespace backend {
namespace cuda {
namespace attr {
inline constexpr const char* kLaunchFields = "cuda.launch_fields";
inline constexpr const char* kKernelAttrs = "cuda.kernel_attrs";
inline constexpr const char* kSmemRequired = "cuda.smem_required";
inline constexpr const char* kLaunchDimensions = "cuda.launch_dimensions";
}  // namespace attr
}  // namespace cuda
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_CUDA_ATTR_H_
