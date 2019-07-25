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
 *  Copyright (c) 2019 by Contributors
 * \file utvm_runtime.h
 * \brief uTVM runtime headers
 */
#ifndef TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_
#define TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <tvm/runtime/c_runtime_api.h>

/*!
 * \brief Task structure for uTVM
 */
typedef struct {
  /*! \brief Pointer to function to call for this task */
  int32_t (*func)(void*, void*, int32_t);
  /*! \brief Array of argument values */
  TVMValue* arg_values;
  /*! \brief Array of type codes for each argument value */
  int* arg_type_codes;
  /*! \brief Number of arguments */
  int32_t num_args;
} UTVMTask;

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif

#endif  // TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_
