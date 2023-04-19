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
#include <tvm/runtime/metadata_types.h>

typedef struct TVMMetadata TVMMetadata;

typedef struct TVMAotExecutor {
  /*! \brief The top-level metadata structure supplied by the generated code */
  const TVMMetadata* metadata;
  /*! \brief The code module that contains the compiled model */
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
 * \param device Runtime execution device, only supports device type kDLCPU, index 0.
 * \param executor Pointer which receives a pointer to the newly-created instance.
 * \param module_name TVM Module name prefix, typically "default".
 * \return 0 if successful.
 */
int TVMAotExecutor_Create(TVMModuleHandle module_handle, const DLDevice device,
                          TVMAotExecutor** executor, const char* module_name);

/*!
 * \brief Release the AoT executor created by TVMAotExecutor_Create().
 *
 * \param executor Pointer to executor instance, created by TVMAotExecutor_Create().
 * \param device Runtime execution device, only supports device type kDLCPU, index 0.
 * \return 0 if successful.
 */
int TVMAotExecutor_Release(TVMAotExecutor* executor, const DLDevice device);

/*!
 * \brief Return the number of inputs.
 *
 * \param executor Pointer to executor instance, created by TVMAotExecutor_Create().
 * \return Number of inputs.
 */
int TVMAotExecutor_GetNumInputs(TVMAotExecutor* executor);

/*!
 * \brief Return the number of outputs.
 *
 * \param executor Pointer to executor instance, created by TVMAotExecutor_Create().
 * \return Number of outputs.
 */
int TVMAotExecutor_GetNumOutputs(TVMAotExecutor* executor);

/*!
 * \brief Return the input index of the specified input name
 *
 * \param executor Pointer to executor instance, created by TVMAotExecutor_Create().
 * \param name Input name for retrieving index.
 * \return Input index.
 */
int TVMAotExecutor_GetInputIndex(TVMAotExecutor* executor, const char* name);

/*!
 * \brief Return a pointer to name of input with the specified input index
 *
 * \param executor Pointer to executor instance, created by TVMAotExecutor_Create().
 * \param index Input index for retrieving name.
 * \param name Output for retrieving name.
 * \return Pointer to input name in `name`.
 */
int TVMAotExecutor_GetInputName(TVMAotExecutor* executor, int index, const char** name);

/*!
 * \brief Run the generated program.
 *
 * \param executor Pointer to executor instance, created by TVMAotExecutor_Create().
 * \return 0 if successful.
 */
int TVMAotExecutor_Run(TVMAotExecutor* executor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_AOT_EXECUTOR_H_
