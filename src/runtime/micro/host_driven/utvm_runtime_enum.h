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
 * \file utvm_runtime_enum.h
 * \brief Defines constants used both on the host and on device.
 */
#ifndef TVM_RUNTIME_MICRO_HOST_DRIVEN_UTVM_RUNTIME_ENUM_H_
#define TVM_RUNTIME_MICRO_HOST_DRIVEN_UTVM_RUNTIME_ENUM_H_

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief TODO
 */
enum UTVMReturnCode {
  UTVM_ERR_OK = 0,
  UTVM_ERR_NOT_FINISHED = -1,
  UTVM_ERR_TIMER_NOT_IMPLEMENTED = -2,
  UTVM_ERR_TIMER_OVERFLOW = -3,
  UTVM_ERR_WS_DOUBLE_FREE = -4,
  UTVM_ERR_WS_OUT_OF_SPACE = -5,
  UTVM_ERR_WS_TOO_MANY_ALLOCS = -6,
  UTVM_ERR_WS_ZERO_SIZE_ALLOC = -7,
  UTVM_ERR_WS_UNALIGNED_START = -8,
  UTVM_ERR_WS_UNALIGNED_ALLOC_SIZE = -9,
};

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif

#endif  // TVM_RUNTIME_MICRO_HOST_DRIVEN_UTVM_RUNTIME_ENUM_H_
