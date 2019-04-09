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
 *  Copyright (c) 2018 by Contributors
 * \file Use external cudnn utils function
 */

#ifndef TVM_CONTRIB_CUBLAS_CUBLAS_UTILS_H_
#define TVM_CONTRIB_CUBLAS_CUBLAS_UTILS_H_

#include <dmlc/logging.h>

extern "C" {
#include <cublas_v2.h>
}

namespace tvm {
namespace contrib {

inline const char* GetCublasErrorString(int error) {
  switch (error) {
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unrecognized error";
}

#ifndef CHECK_CUBLAS_ERROR
#define CHECK_CUBLAS_ERROR(fn)                  \
  do {                                          \
    int error = static_cast<int>(fn);                      \
    CHECK_EQ(error, CUBLAS_STATUS_SUCCESS) << "CUBLAS: " << GetCublasErrorString(error); \
  } while (0)  // ; intentionally left off.
#endif  // CHECK_CUBLAS_ERROR


struct CuBlasThreadEntry {
  CuBlasThreadEntry();
  ~CuBlasThreadEntry();
  cublasHandle_t handle{nullptr};
  static CuBlasThreadEntry* ThreadLocal();
};  // CuBlasThreadEntry


}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_CUBLAS_CUBLAS_UTILS_H_
