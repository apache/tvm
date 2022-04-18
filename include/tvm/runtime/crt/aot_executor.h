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
 * \file aot_executor.h
 * \brief AoT Executor
 */
#ifndef TVM_RUNTIME_CRT_AOT_EXECUTOR_H_
#define TVM_RUNTIME_CRT_AOT_EXECUTOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <dlpack/dlpack.h>
#include <tvm/runtime/crt/internal/common/ndarray.h>
#include <tvm/runtime/metadata.h>

typedef struct TVMMetadata TVMMetadata;

typedef struct TVMAotExecutor {
  /*! \brief The top-level metadata structure */
  TVMMetadata* metadata;
  /*! \brief The code module that contains both host and device code */
  TVMModuleHandle module_handle;
  /*! \brief The device type */
  DLDevice device;
  /*! \brief List of allocated arguments, input(s), output(s), and pool(s)*/
  TVMNDArray* args;
  int64_t num_args;
} TVMAotExecutor;

/*!
 * \brief Allocate a new AotExecutor with TVMPlatformMemoryAllocate and initialize it.
 *
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param executor Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
int TVMAotExecutor_Create(TVMModuleHandle module_handle, const DLDevice* devices,
                          TVMAotExecutor** executor);

int TVMAotExecutor_Release(TVMAotExecutor* executor, const DLDevice device);

int TVMAotExecutor_GetNumInputs(TVMAotExecutor* executor);

int TVMAotExecutor_GetNumOutputs(TVMAotExecutor* executor);

int TVMAotExecutor_GetInputIndex(TVMAotExecutor* executor, const char* name);

int TVMAotExecutor_Run(TVMAotExecutor* executor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_AOT_EXECUTOR_H_
