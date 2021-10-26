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
 * \file src/runtime/crt/include/tvm/runtime/crt/internal/graph_executor/graph_executor.h
 * \brief Tiny graph executor that can run graph containing only tvm PackedFunc.
 */
#ifndef TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
#define TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/internal/common/ndarray.h>
#include <tvm/runtime/crt/internal/graph_executor/load_json.h>
#include <tvm/runtime/crt/module.h>

// Memory pool entry.
typedef struct TVMGraphExecutorPoolEntry {
  size_t size;
  int device_type;
  int entry_id;
} TVMGraphExecutorPoolEntry;

// Node entry
typedef struct TVMGraphExecutorNodeEntry {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
  // JSON Loader
  void (*Load)(JSONReader* reader);
} TVMGraphExecutorNodeEntry;

// Storage entry.
typedef struct TVMGraphExecutorStorageEntry {
  uint8_t is_linked_param;
  TVMNDArray array;
} TVMGraphExecutorStorageEntry;

// Node
typedef struct TVMGraphExecutorNode {
  // operator type in string
  char op_type[16];
  // name of the op
  char name[TVM_CRT_MAX_STRLEN_FUNCTION_NAME];
  // parameters
  TVMOpParam param;
  // inputs
  TVMGraphExecutorNodeEntry* inputs;
  // number of inputs
  size_t inputs_count;
  // control deps
  uint32_t control_deps[20];
  // JSON Loader
  void (*LoadAttrs)(struct TVMGraphExecutorNode* node, JSONReader* reader, TVMOpParam* param);
  // JSON Loader
  int (*Load)(struct TVMGraphExecutorNode* node, JSONReader* reader);
} TVMGraphExecutorNode;

typedef struct TVMGraphExecutor {
  /*! \brief The graph nodes. */
  TVMGraphExecutorNode* nodes;
  /*! \brief The graph nodes counter. */
  uint32_t nodes_count;
  /*! \brief The argument nodes. */
  uint32_t* input_nodes;
  uint32_t input_nodes_count;
  /*! \brief Used for quick entry indexing. */
  uint32_t* node_row_ptr;
  uint32_t node_row_ptr_count;
  /*! \brief Output entries. */
  TVMGraphExecutorNodeEntry* outputs;
  /*! \brief Output entries counter. */
  uint32_t outputs_count;
  /*! \brief Additional graph attributes. */
  TVMGraphExecutorGraphAttr attrs;
  /*! \brief The code module that contains both host and device code. */
  TVMModuleHandle module_handle;
  /*! \brief Execution context of all devices including the host. */
  DLDevice devices[1];
  uint32_t devices_count;
  /*! \brief Common storage pool for all devices. */
  TVMGraphExecutorStorageEntry* storage_pool;
  uint32_t storage_pool_count;
  /*! \brief Data entry of each node. */
  TVMNDArray* data_entry;
  uint32_t data_entry_count;
  /*! \brief Operator on each node. */
  TVMPackedFunc* op_execs;
  uint32_t op_execs_count;
} TVMGraphExecutor;

typedef DLTensor* DLTensorPtr;

// private functions
uint32_t TVMGraphExecutor_GetEntryId(TVMGraphExecutor* executor, uint32_t nid, uint32_t index);
void TVMGraphExecutor_SetInput(TVMGraphExecutor* executor, const char* name, DLTensor* data_in);
int TVMGraphExecutor_LoadParams(TVMGraphExecutor* executor, const char* param_blob,
                                const uint32_t param_size);
void TVMGraphExecutor_Run(TVMGraphExecutor* executor);
int TVMGraphExecutor_GetOutput(TVMGraphExecutor* executor, const int32_t idx, DLTensor* out);

int32_t TVMGraphExecutor_CreateTVMOp(TVMGraphExecutor* executor, const TVMOpParam* param,
                                     DLTensorPtr* args, const uint32_t args_count,
                                     TVMPackedFunc* pf);
int TVMGraphExecutor_Load(TVMGraphExecutor* executor, JSONReader* reader);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
