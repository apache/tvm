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

#include "load_json.h"
#include "module.h"
#include "ndarray.h"
#include "packed_func.h"

/*! \brief operator attributes about tvm op */
typedef struct TVMOpParam {
  char func_name[120];
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
} TVMOpParam;

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
typedef struct TVMGraphRuntime {
  void (*Run)(struct TVMGraphRuntime* runtime);

  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param runtime The graph runtime.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   */
  void (*Init)(struct TVMGraphRuntime* runtime, const char* graph_json, const TVMModule* module,
               const TVMContext* ctxs);

  /*!
   * \brief Get the input index given the name of input.
   * \param runtime The graph runtime.
   * \param name The name of the input.
   * \return The index of input.
   */
  int (*GetInputIndex)(struct TVMGraphRuntime* runtime, const char* name);

  /*!
   * \brief set input to the graph based on name.
   * \param runtime The graph runtime.
   * \param name The name of the input.
   * \param data_in The input data.
   */
  void (*SetInput)(struct TVMGraphRuntime* runtime, const char* name, DLTensor* data_in);

  /*!
   * \brief Return NDArray for given output index.
   * \param runtime The graph runtime.
   * \param index The output index.
   * \param out The DLTensor corresponding to given output node index.
   * \return The result of this function execution.
   */
  int (*GetOutput)(struct TVMGraphRuntime* runtime, const int32_t index, DLTensor* out);
  /*!
   * \brief Load parameters from parameter blob.
   * \param runtime The graph runtime.
   * \param param_blob A binary blob of parameter.
   * \param param_size The parameter size.
   * \return The result of this function execution.
   */
  int (*LoadParams)(struct TVMGraphRuntime* runtime, const char* param_blob,
                    const uint32_t param_size);

  // The graph attribute fields.
  int (*Load)(struct TVMGraphRuntime* runtime, JSONReader* reader);
  /*! \brief Setup the temporal storage */
  void (*SetupStorage)(struct TVMGraphRuntime* runtime);
  /*! \brief Setup the executors. */
  int (*SetupOpExecs)(struct TVMGraphRuntime* runtime);

  /*!
   * \brief Create an execution function given input.
   * \param runtime The graph runtime.
   * \param attrs The node attributes.
   * \param args The arguments to the functor, including inputs and outputs.
   * \param args_count The total number of arguments.
   * \param num_inputs Number of inputs.
   * \param pf The created executor.
   * \return The result of this function execution.
   */
  int32_t (*CreateTVMOp)(struct TVMGraphRuntime* runtime, const TVMOpParam* attrs,
                         DLTensorPtr* args, const uint32_t args_count, uint32_t num_inputs,
                         TVMPackedFunc* pf);

  // Get node entry index.
  uint32_t (*GetEntryId)(struct TVMGraphRuntime* runtime, uint32_t nid, uint32_t index);

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
  TVMModule module;
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

// public functions
TVMGraphRuntime* TVMGraphRuntimeCreate(const char* sym_json, const TVMModule* m,
                                       const TVMContext* ctxs);
void TVMGraphRuntimeRelease(TVMGraphRuntime** runtime);

// private functions
void TVMGraphRuntime_SetInput(TVMGraphRuntime* runtime, const char* name, DLTensor* data_in);
int TVMGraphRuntime_LoadParams(TVMGraphRuntime* runtime, const char* param_blob,
                               const uint32_t param_size);
void TVMGraphRuntime_Run(TVMGraphRuntime* runtime);
int TVMGraphRuntime_GetOutput(TVMGraphRuntime* runtime, const int32_t idx, DLTensor* out);

#endif  // TVM_RUNTIME_CRT_GRAPH_RUNTIME_H_
