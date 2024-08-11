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
 * \file tvm/ffi/c_ffi_abi.h
 * \brief This file defines the ABI convention of the FFI convention
 *
 * Including global calling conventions
 */
#ifndef TVM_FFI_C_FFI_ABI_H_
#define TVM_FFI_C_FFI_ABI_H_

#include <tvm/ffi/c_ffi_abi.h>

#if !defined(TVM_FFI_DLL) && defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define TVM_FFI_API EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(TVM_FFI_DLL) && defined(_MSC_VER)
#ifdef TVM_FFI_EXPORTS
#define TVM_FFI_DLL __declspec(dllexport)
#else
#define TVM_FFI_DLL __declspec(dllimport)
#endif
#endif
#ifndef TVM_FFI_DLL
#define TVM_FFI_DLL __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
static_assert(
  TVM_FFI_ALLOW_DYN_TYPE,
  "Only include c_ffi_abi when TVM_FFI_ALLOW_DYN_TYPE is set to true"
);
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // TVM_FFI_EXTERN_C
#endif
#endif  // TVM_FFI_C_FFI_ABI_H_
