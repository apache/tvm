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
 * \file common.h
 * \brief TVM SGX common API.
 */
#ifndef TVM_RUNTIME_SGX_COMMON_H_
#define TVM_RUNTIME_SGX_COMMON_H_

#include <sgx_error.h>

namespace tvm {
namespace runtime {
namespace sgx {

#define TVM_SGX_CHECKED_CALL(Function)                                         \
  sgx_status_t TVM_STR_CONCAT(__sgx_status_, __LINE__) = SGX_ERROR_UNEXPECTED; \
  TVM_STR_CONCAT(__sgx_status_, __LINE__) = Function;                          \
  CHECK_EQ(TVM_STR_CONCAT(__sgx_status_, __LINE__), SGX_SUCCESS)               \
    << "SGX Error: " << TVM_STR_CONCAT(__sgx_status_, __LINE__);

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_SGX_COMMON_H_
