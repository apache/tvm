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
 * \brief utvm runtime headers
 */
#ifndef TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_
#define TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <tvm/runtime/c_runtime_api.h>

/*!
 * \brief POD variant of TVMArgs
 */
typedef struct {
  /*! \brief Array of values */
  TVMValue* values;
  /*! \brief Array of type codes for each value */
  int* type_codes;
  /*! \brief Number of arguments */
  int32_t num_args;
} UTVMArgs;

/*!
 * \brief Task structure for uTVM
 */
typedef struct {
  /*! \brief Pointer to function to call for this task */
  void (*func)(void*, void*, int32_t);
  /*! \brief Arguments for this task's function call */
  UTVMArgs* args;
} UTVMTask;

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocate spaced is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param nbytes The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \param dtype_code_hint The type code of the array elements. Only used in
 * certain backends such as OpenGL.
 * \param dtype_bits_hint The type bits of the array elements. Only used in
 * certain backends such as OpenGL.
 * \return nullptr when error is thrown, a valid ptr if success
 */
void* TVMBackendAllocWorkspace(int device_type,
                               int device_id,
                               uint64_t size,
                               int dtype_code_hint,
                               int dtype_bits_hint);

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 when no error is thrown, -1 when failure happens
 *
 * \sa TVMBackendAllocWorkspace
 */
int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
void TVMAPISetLastError(const char* msg);

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_MICRO_DEVICE_UTVM_RUNTIME_H_
