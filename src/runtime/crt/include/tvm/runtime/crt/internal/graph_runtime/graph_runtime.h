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
 * \file src/runtime/crt/include/tvm/runtime/crt/internal/graph_runtime/graph_runtime.h
 * \brief Tiny graph runtime that can run graph containing only tvm PackedFunc.
 */
#ifndef TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_GRAPH_RUNTIME_GRAPH_RUNTIME_H_
#define TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_GRAPH_RUNTIME_GRAPH_RUNTIME_H_

#include <tvm/runtime/crt/graph_runtime.h>
#include <tvm/runtime/crt/internal/common/ndarray.h>
#include <tvm/runtime/crt/internal/graph_runtime/load_json.h>
#include <tvm/runtime/crt/module.h>

// Memory pool entry.
typedef struct TVMGraphRuntimePoolEntry {
  size_t size;
  int device_type;
} TVMGraphRuntimePoolEntry;

// Node entry
typedef struct TVMGraphRuntimeNodeEntry {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
  // JSON Loader
  void (*Load)(JSONReader* reader);
} TVMGraphRuntimeNodeEntry;

// Node
typedef struct TVMGraphRuntimeNode {
  // operator type in string
  char op_type[16];
  // name of the op
  char name[120];
  // parameters
  TVMOpParam param;
  // inputs
  TVMGraphRuntimeNodeEntry* inputs;
  // number of inputs
  size_t inputs_count;
  // control deps
  uint32_t control_deps[20];
  // JSON Loader
  void (*LoadAttrs)(struct TVMGraphRuntimeNode* node, JSONReader* reader, TVMOpParam* param);
  // JSON Loader
  int (*Load)(struct TVMGraphRuntimeNode* node, JSONReader* reader);
} TVMGraphRuntimeNode;

typedef struct TVMGraphRuntime {
  /*! \brief The graph nodes. */
  TVMGraphRuntimeNode* nodes;
  /*! \brief The graph nodes counter. */
  uint32_t nodes_count;
  /*! \brief The argument nodes. */
  uint32_t* input_nodes;
  uint32_t input_nodes_count;
  /*! \brief Used for quick entry indexing. */
  uint32_t* node_row_ptr;
  uint32_t node_row_ptr_count;
  /*! \brief Output entries. */
  TVMGraphRuntimeNodeEntry* outputs;
  /*! \brief Output entries counter. */
  uint32_t outputs_count;
  /*! \brief Additional graph attributes. */
  TVMGraphRuntimeGraphAttr attrs;
  /*! \brief The code module that contains both host and device code. */
  TVMModuleHandle module_handle;
  /*! \brief Execution context of all devices including the host. */
  TVMContext ctxs[1];
  uint32_t ctxs_count;
  /*! \brief Common storage pool for all devices. */
  TVMNDArray* storage_pool;
  uint32_t storage_pool_count;
  /*! \brief Data entry of each node. */
  TVMNDArray* data_entry;
  uint32_t data_entry_count;
  /*! \brief Operator on each node. */
  TVMPackedFunc* op_execs;
  uint32_t op_execs_count;
} TVMGraphRuntime;

typedef DLTensor* DLTensorPtr;

// private functions
void TVMGraphRuntime_SetInput(TVMGraphRuntime* runtime, const char* name, DLTensor* data_in);
int TVMGraphRuntime_LoadParams(TVMGraphRuntime* runtime, const char* param_blob,
                               const uint32_t param_size);
void TVMGraphRuntime_Run(TVMGraphRuntime* runtime);
int TVMGraphRuntime_GetOutput(TVMGraphRuntime* runtime, const int32_t idx, DLTensor* out);

int32_t TVMGraphRuntime_CreateTVMOp(TVMGraphRuntime* runtime, const TVMOpParam* param,
                                    DLTensorPtr* args, const uint32_t args_count,
                                    uint32_t num_inputs, TVMPackedFunc* pf);

#endif  // TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_GRAPH_RUNTIME_GRAPH_RUNTIME_H_
