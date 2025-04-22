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
/*
 * \file src/ffi/ndarray.cc
 * \brief NDArray C API implementation
 */
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/ndarray.h>
#include <tvm/ffi/function.h>

int TVMFFINDArrayFromDLPack(DLManagedTensor* from, int64_t min_alignment,
                            int64_t require_contiguous, TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::NDArray nd =
      tvm::ffi::NDArray::FromDLPack(from, static_cast<size_t>(min_alignment), require_contiguous);
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(nd));
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFINDArrayFromDLPackVersioned(DLManagedTensorVersioned* from, int32_t min_alignment,
                                     int32_t require_contiguous, TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::NDArray nd = tvm::ffi::NDArray::FromDLPackVersioned(
      from, static_cast<size_t>(min_alignment), require_contiguous);
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(nd));
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFINDArrayToDLPack(TVMFFIObjectHandle from, DLManagedTensor** out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = tvm::ffi::details::ObjectUnsafe::RawObjectPtrFromUnowned<tvm::ffi::NDArrayObj>(
             static_cast<TVMFFIObject*>(from))
             ->ToDLPack();
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFINDArrayToDLPackVersioned(TVMFFIObjectHandle from, DLManagedTensorVersioned** out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = tvm::ffi::details::ObjectUnsafe::RawObjectPtrFromUnowned<tvm::ffi::NDArrayObj>(
             static_cast<TVMFFIObject*>(from))
             ->ToDLPackVersioned();
  TVM_FFI_SAFE_CALL_END();
}
