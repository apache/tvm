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
 * \file include/tvm/runtime/crt/aot/tvm_error.h
 * \brief Defines a subset of error codes returned by the CRT AOT executor.
 */

#ifndef TVM_RUNTIME_CRT_AOT_TVM_ERROR_H_
#define TVM_RUNTIME_CRT_AOT_TVM_ERROR_H_

#ifdef __cplusplus
extern "C" {
#endif

#define TVM_CRT_ERROR_CATEGORY_Pos 8
#define TVM_CRT_ERROR_CATEGORY_Msk (0xff << TVM_CRT_ERROR_CATEGORY_Pos)
#define TVM_CRT_ERROR_CODE_Pos 0
#define TVM_CRT_ERROR_CODE_Msk (0xff << TVM_CRT_ERROR_CODE_Pos)

#define DEFINE_TVM_CRT_ERROR(category, code) \
  (((category) << TVM_CRT_ERROR_CATEGORY_Pos) | ((code) << TVM_CRT_ERROR_CODE_Pos))
typedef enum {
  kTvmErrorCategoryPlatform = 5,
  kTvmErrorCategoryFunctionCall = 8,
} tvm_crt_error_category_t;

typedef enum {
  kTvmErrorNoError = 0,

  // Platform
  kTvmErrorPlatformCheckFailure = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryPlatform, 0),
  kTvmErrorPlatformMemoryManagerInitialized = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryPlatform, 1),
  kTvmErrorPlatformShutdown = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryPlatform, 2),
  kTvmErrorPlatformNoMemory = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryPlatform, 3),
  kTvmErrorPlatformTimerBadState = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryPlatform, 4),

  // Function Calls - common problems encountered calling functions.
  kTvmErrorFunctionCallNumArguments = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionCall, 0),
  kTvmErrorFunctionCallWrongArgType = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionCall, 1),
  kTvmErrorFunctionCallNotImplemented = DEFINE_TVM_CRT_ERROR(kTvmErrorCategoryFunctionCall, 2),

  // System errors are always negative integers; this mask indicates presence of a system error.
  // Cast tvm_crt_error_t to a signed integer to interpret the negative error code.
  kTvmErrorSystemErrorMask = (1 << (sizeof(int) * 4 - 1)),
} tvm_crt_error_t;

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_CRT_AOT_TVM_ERROR_H_
