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
 * \file graph_runtime.h
 * \brief Tiny graph runtime that can run graph containing only tvm PackedFunc.
 */
#ifndef TVM_RUNTIME_CRT_GRAPH_RUNTIME_H_
#define TVM_RUNTIME_CRT_GRAPH_RUNTIME_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

struct TVMPackedFunc;
typedef struct TVMPackedFunc TVMPackedFunc;

struct TVMModule;
typedef struct TVMModule TVMModule;

/*! \brief operator attributes about tvm op */
typedef struct TVMOpParam {
  char func_name[120];
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
} TVMOpParam;

// Graph attribute
typedef struct TVMGraphRuntimeGraphAttr {
  uint32_t storage_num_not_alloctaed;
  uint32_t* storage_id;
  uint32_t* device_index;
  char* dltype;  // "int8", "int16", "float32"
  uint32_t dltype_count;
  int64_t* shape;
  uint32_t* ndim;
  uint32_t shape_count;
} TVMGraphRuntimeGraphAttr;

typedef DLTensor* DLTensorPtr;

/*!
 * \brief Tiny graph runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
/* class GraphRuntime : public ModuleNode { */
typedef struct TVMGraphRuntimeAPI {
  void (*Run)(struct TVMGraphRuntimeAPI* runtime);

  /*!
   * \brief Get the input index given the name of input.
   * \param runtime The graph runtime.
   * \param name The name of the input.
   * \return The index of input.
   */
  int (*GetInputIndex)(struct TVMGraphRuntimeAPI* runtime, const char* name);

  /*!
   * \brief set input to the graph based on name.
   * \param runtime The graph runtime.
   * \param name The name of the input.
   * \param data_in The input data.
   */
  void (*SetInput)(struct TVMGraphRuntimeAPI* runtime, const char* name, DLTensor* data_in);

  /*!
   * \brief Return NDArray for given output index.
   * \param runtime The graph runtime.
   * \param index The output index.
   * \param out The DLTensor corresponding to given output node index.
   * \return The result of this function execution.
   */
  int (*GetOutput)(struct TVMGraphRuntimeAPI* runtime, const int32_t index, DLTensor* out);
  /*!
   * \brief Load parameters from parameter blob.
   * \param runtime The graph runtime.
   * \param param_blob A binary blob of parameter.
   * \param param_size The parameter size.
   * \return The result of this function execution.
   */
  int (*LoadParams)(struct TVMGraphRuntimeAPI* runtime, const char* param_blob,
                    const uint32_t param_size);
} TVMGraphRuntimeAPI;

// public functions
TVMGraphRuntimeAPI* TVMGraphRuntimeCreate(const char* sym_json, const TVMModule* m,
                                          const TVMContext* ctxs);
void TVMGraphRuntimeRelease(TVMGraphRuntimeAPI** runtime);

#endif  // TVM_RUNTIME_CRT_GRAPH_RUNTIME_H_
