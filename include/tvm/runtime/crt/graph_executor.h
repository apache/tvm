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
 * \file graph_executor.h
 * \brief Tiny graph executor that can run graph containing only tvm PackedFunc.
 */
#ifndef TVM_RUNTIME_CRT_GRAPH_EXECUTOR_H_
#define TVM_RUNTIME_CRT_GRAPH_EXECUTOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/packed_func.h>

struct TVMModule;

/*! \brief operator attributes about tvm op */
typedef struct TVMOpParam {
  char func_name[TVM_CRT_MAX_STRLEN_FUNCTION_NAME];
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
} TVMOpParam;

// Graph attribute
typedef struct TVMGraphExecutorGraphAttr {
  uint32_t storage_num_not_alloctaed;
  uint32_t* storage_id;
  uint32_t* device_index;
  char* dltype;  // "int8", "int16", "float32"
  uint32_t dltype_count;
  int64_t* shape;
  uint32_t* ndim;
  uint32_t shape_count;
} TVMGraphExecutorGraphAttr;

typedef struct TVMGraphExecutor TVMGraphExecutor;

// public functions
/*!
 * \brief Allocate a new GraphExecutor with TVMPlatformMemoryAllocate and initialize it.
 *
 * \param sym_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param executor Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
int TVMGraphExecutor_Create(const char* sym_json, TVMModuleHandle module_handle,
                            const DLDevice* devices, TVMGraphExecutor** executor);

int TVMGraphExecutor_GetInputIndex(TVMGraphExecutor* executor, const char* name);

/*!
 * \brief get number of input tensors allocated.
 * \return integer number of tensors available to use.
 */
int TVMGraphExecutor_GetNumInputs();

/*!
 * \brief set input to the graph based on name.
 * \param executor The graph executor.
 * \param name The name of the input.
 * \param data_in The input data.
 */
void TVMGraphExecutor_SetInput(TVMGraphExecutor* executor, const char* name, DLTensor* data_in);

/*!
 * \brief get number of output tensors allocated.
 * \return integer number of output tensors allocated.
 */
int TVMGraphExecutor_GetNumOutputs();

/*!
 * \brief Return NDArray for given output index.
 * \param executor The graph executor.
 * \param index The output index.
 * \param out The DLTensor corresponding to given output node index.
 * \return The result of this function execution.
 */
int TVMGraphExecutor_GetOutput(TVMGraphExecutor* executor, const int32_t index, DLTensor* out);

/*!
 * \brief Load parameters from parameter blob.
 * \param executor The graph executor.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return The result of this function execution.
 */
int TVMGraphExecutor_LoadParams(TVMGraphExecutor* executor, const char* param_blob,
                                const uint32_t param_size);

/*!
 * \brief Execute the graph.
 * \param executor The graph executor.
 */
void TVMGraphExecutor_Run(TVMGraphExecutor* executor);

/*!
 * \brief Release memory associated with the graph executor.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int TVMGraphExecutor_Release(TVMGraphExecutor** executor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_GRAPH_EXECUTOR_H_
