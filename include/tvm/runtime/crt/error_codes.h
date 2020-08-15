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
 * \file include/tvm/runtime/crt/error_codes.h
 * \brief Defines integral error codes returned by the CRT.
 */
#ifndef TVM_RUNTIME_CRT_ERROR_CODES_H_
#define TVM_RUNTIME_CRT_ERROR_CODES_H_

#ifdef __cplusplus
extern "C" {
#endif

#define TVM_CRT_ERROR_CATEGORY_Pos 8
#define TVM_CRT_ERROR_CATEGORY_Msk (0xff << TVM_CRT_ERROR_CATEGORY_Pos)
#define TVM_CRT_ERROR_CODE_Pos 0
#define TVM_CRT_ERROR_CODE_Msk (0xff << TVM_CRT_ERROR_CODE_Pos)

#define DEFINE_TVM_CRT_ERROR(category, code) \
  (((category) << TVM_CRT_ERROR_CATEGORY_Pos) | ((code) << TVM_CRT_ERROR_CODE_Pos))
typedef enum { kTvmErrorCategoryFunctionRegistry = 1 } tvm_crt_error_category_t;

typedef enum {
  kTvmErrorNoError = 0,

  // Function Registry
  kTvmErrorFunctionNameNotFound = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionRegistry, 0),
  kTvmErrorFunctionIndexInvalid = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionRegistry, 1),
  kTvmErrorFunctionRegistryFull = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionRegistry, 2),
  kTvmErrorFunctionAlreadyDefined = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionRegistry, 3),
  kTvmErrorBufferTooSmall = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionRegistry, 4),
} tvm_crt_error_t;

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_CRT_ERROR_CODES_H_
