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
/*!
 * \file tvm/backend/cuda/attr.h
 * \brief CUDA backend attribute keys.
 */
#ifndef TVM_BACKEND_CUDA_ATTR_H_
#define TVM_BACKEND_CUDA_ATTR_H_

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/expr.h>

namespace tvm {
namespace backend {
namespace cuda {
namespace attr {
inline constexpr ffi::reflection::AttrKey<ffi::Array<ffi::String>> kLaunchFields{
    "cuda.launch_fields"};
inline constexpr ffi::reflection::AttrKey<ffi::Map<ffi::String, int64_t>> kKernelAttrs{
    "cuda.kernel_attrs"};
inline constexpr ffi::reflection::AttrKey<int64_t> kSmemRequired{"cuda.smem_required"};
inline constexpr ffi::reflection::AttrKey<ffi::Array<PrimExpr>> kLaunchDimensions{
    "cuda.launch_dimensions"};
inline constexpr ffi::reflection::AttrKey<bool> kIKetEnabled{"cuda.iket.enabled"};
}  // namespace attr
}  // namespace cuda
}  // namespace backend
}  // namespace tvm

#endif  // TVM_BACKEND_CUDA_ATTR_H_
