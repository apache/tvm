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
 *  Copyright (c) 2017 by Contributors
 *
 * \brief Tiny graph runtime that can run graph containing only tvm PackedFunc.
 * \file graph_runtime.h
 */
#ifndef TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
#define TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_

#include <dlpack.h>
#include "ndarray.h"
#include "packed_func.h"
#include "module.h"
#include "io.h"
#include "load_json.h"

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                            \
  {                                                                \
    int ret = (func);                                              \
    CHECK_EQ(ret, 0)                                               \
        << TVMGetLastError();                                      \
  }

/*! \brief operator attributes about tvm op */
typedef struct tvm_op_param_t {
  char func_name[120];
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
} TVMOpParam;
  
// Memory pool entry.
typedef struct graph_runtime_pool_entry_t {
  size_t size;
  int device_type;
} PoolEntry;

// Node entry
typedef struct graph_runtime_node_entry_t {
  uint32_t node_id;
  uint32_t index;
  uint32_t version;
  // JSON Loader
  void (*Load)(JSONReader *reader);
} NodeEntry;

static inline int NodeEntry_Load(NodeEntry * entry, JSONReader * reader) {
  int status = TVM_STATUS_SUCCESS;
  reader->BeginArray(reader);
  if (!(reader->NextArrayItem(reader))) { LOGE("invalid json format: failed to parse `node_id`"); }
  reader->ReadUnsignedInteger(reader, &(entry->node_id));
  if (!(reader->NextArrayItem(reader))) { LOGE("invalid json format: failed to parse `index`"); }
  reader->ReadUnsignedInteger(reader, &(entry->index));
  if (reader->NextArrayItem(reader)) {
    reader->ReadUnsignedInteger(reader, &(entry->version));
    if (reader->NextArrayItem(reader)) { LOGE("invalid json format: failed to parse `version`"); }
  } else {
    entry->version = 0;
  }
  return status;
}

// Node
typedef struct graph_runtime_node_t {
  // operator type in string
  char op_type[16];
  // name of the op
  char name[120];
  // parameters
  TVMOpParam param;
  // inputs
  NodeEntry inputs[GRAPH_RUNTIME_NODE_MAX_INPUTS];
  size_t                inputs_count;
  // control deps
  uint32_t control_deps[200];
  // JSON Loader
  void (*LoadAttrs)(struct graph_runtime_node_t * node, JSONReader *reader, TVMOpParam* param);
  // JSON Loader
  int (*Load)(struct graph_runtime_node_t * node, JSONReader *reader);
} GraphRuntimeNode;

static inline void GraphRuntimeNode_LoadAttrs(GraphRuntimeNode * node, JSONReader *reader, TVMOpParam* param) {
  int bitmask = 0;
  char key[20], value[120];
  memset(param, 0, sizeof(TVMOpParam));
  memset(key, 0, sizeof(key));
  memset(value, 0, sizeof(value));
  reader->BeginObject(reader);
  while (reader->NextObjectItem(reader, key)) {
    reader->ReadString(reader, value);
    if (!strcmp(key, "func_name")) {
      /* param->func_name = value; */
      strcpy(param->func_name, value);
      bitmask |= 1;
    } else if (!strcmp(key, "num_inputs")) {
      param->num_inputs = strtoul(value, nullptr, 10);
      bitmask |= 2;
    } else if (!strcmp(key, "num_outputs")) {
      param->num_outputs = strtoul(value, nullptr, 10);
      bitmask |= 4;
    } else if (!strcmp(key, "flatten_data")) {
      param->flatten_data = strtoul(value, nullptr, 10);
      bitmask |= 8;
    } else {
      LOGE("do not support key %s", key);
    }
  }
  if (bitmask != (1|2|4|8)) { LOGE("invalid format"); }
}

static inline int GraphRuntimeNode_Load(GraphRuntimeNode * node, JSONReader *reader) {
  int status = TVM_STATUS_SUCCESS;
  reader->BeginObject(reader);
  int bitmask = 0;
  char key[20];
  while (reader->NextObjectItem(reader, key)) {
    if (!strcmp(key, "op")) {
      reader->ReadString(reader, node->op_type);
      bitmask |= 1;
    } else if (!strcmp(key, "name")) {
      reader->ReadString(reader, node->name);
      bitmask |= 2;
    } else if (!strcmp(key, "inputs")) {
      size_t count = node->inputs_count;
      if (count >= GRAPH_RUNTIME_NODE_MAX_INPUTS) {
        LOGE("The number of inputs in graph runtime node is greater than expected.");
        status = TVM_STATUS_FAILURE;
        break;
      }
      reader->BeginArray(reader);
      while (reader->NextArrayItem(reader)) {
        NodeEntry * inputs = node->inputs + count;
        reader->BeginArray(reader);
        if (!reader->NextArrayItem(reader)) {
          LOGE("invalid json format");
          status = TVM_STATUS_FAILURE;
          break;
        }
        reader->ReadUnsignedInteger(reader, &(inputs->node_id));
        if (!reader->NextArrayItem(reader)) {
          LOGE("invalid json format");
          status = TVM_STATUS_FAILURE;
          break;
        }
        reader->ReadUnsignedInteger(reader, &(inputs->index));
        if (reader->NextArrayItem(reader)) { 
          reader->ReadUnsignedInteger(reader, &(inputs->version));
          if (reader->NextArrayItem(reader)) {
            LOGE("invalid json format");
            status = TVM_STATUS_FAILURE;
            break;
          }
        } else {
          inputs->version = 0;
        }
        count++;
      }
      node->inputs_count = count;
      bitmask |= 4;
    } else if (!strcmp(key, "attr") || !strcmp(key, "attrs")) {
      TVMOpParam param;

      GraphRuntimeNode_LoadAttrs(node, reader, &param);
      memcpy(&node->param, &param, sizeof(param));
    } else if (!strcmp(key, "control_deps")) {
      LOGE("do not support key %s", key);
      status = TVM_STATUS_FAILURE;
    } else {
      LOGE("do not support key %s", key);
      status = TVM_STATUS_FAILURE;
    }
    if (status != TVM_STATUS_SUCCESS) { break; }
  }
  if (bitmask != (1|2|4)) { LOGE("invalid format"); }
  return status;
}

static inline GraphRuntimeNode GraphRuntimeNodeCreate() {
  GraphRuntimeNode node;
  memset(&node, 0, sizeof(GraphRuntimeNode));
  node.LoadAttrs = GraphRuntimeNode_LoadAttrs;
  node.Load = GraphRuntimeNode_Load;
  return node;
}

// Graph attribute
typedef struct graph_runtime_graph_attr_t {
  uint32_t storage_num_not_alloctaed; // {0};
  uint32_t storage_id[GRAPH_RUNTIME_MAX_NODES];
  uint32_t device_index[GRAPH_RUNTIME_MAX_NODES];
  char     dltype[GRAPH_RUNTIME_MAX_NODES][10]; // "int8", "int16", "float32"
  uint32_t dltype_count;
  int64_t  shape[GRAPH_RUNTIME_MAX_NODES][TVM_CRT_MAX_NDIM];
  uint32_t shape_count;
} GraphRuntimeGraphAttr;

static inline int GraphRuntimeGraphAttr_Load(GraphRuntimeGraphAttr * attr, JSONReader *reader) {
  int status = TVM_STATUS_SUCCESS;
  int bitmask = 0;
  char key[16], type[16];
  uint32_t storage_id_count = 0;
  uint32_t dltype_count = 0;
  uint32_t shape_count = 0;
  uint32_t device_index_count = 0;
  reader->BeginObject(reader);
  while (reader->NextObjectItem(reader, key)) {
    if (!strcmp(key, "dltype")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->ReadString(reader, type);
      if (strcmp(type, "list_str")) { LOGE("Invalid json format"); }
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->BeginArray(reader);
      while (reader->NextArrayItem(reader)) {
        reader->ReadString(reader, attr->dltype[dltype_count]);
        dltype_count ++;
      }
      attr->dltype_count = dltype_count;;
      if (reader->NextArrayItem(reader)) { LOGE("Invalid json format"); }
      bitmask |= 1;
    } else if (!strcmp(key, "storage_id")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->ReadString(reader, type);
      if (strcmp(type, "list_int")) { LOGE("Invalid json format"); }
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->BeginArray(reader);
      while (reader->NextArrayItem(reader)) {
        reader->ReadUnsignedInteger(reader, &(attr->storage_id[storage_id_count]));
        storage_id_count++;
      }
      if (reader->NextArrayItem(reader)) { LOGE("Invalid json format"); }
      bitmask |= 2;
    } else if (!strcmp(key, "shape")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->ReadString(reader, type);
      if (strcmp(type, "list_shape")) { LOGE("Invalid json format"); }
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->BeginArray(reader);
      while (reader->NextArrayItem(reader)) {
        reader->BeginArray(reader);
        reader->ReadInteger(reader, &(attr->shape[shape_count][0]));
	      if (reader->NextArrayItem(reader)) { 
          if (reader->NextArrayItem(reader)) { 
            reader->ReadInteger(reader, &(attr->shape[shape_count][1]));
            if (reader->NextArrayItem(reader)) { 
              reader->ReadInteger(reader, &(attr->shape[shape_count][2]));
              if (reader->NextArrayItem(reader)) { 
                reader->ReadInteger(reader, &(attr->shape[shape_count][3]));
                if (reader->NextArrayItem(reader)) { 
                  reader->ReadInteger(reader, &(attr->shape[shape_count][4]));
                  if (reader->NextArrayItem(reader)) { 
                    reader->ReadInteger(reader, &(attr->shape[shape_count][5]));
                    reader->NextArrayItem(reader);
                  }
                }
              }
            }
          }
        }
        shape_count ++;
      }
      attr->shape_count = shape_count;
      if (reader->NextArrayItem(reader)) { LOGE("Invalid json format"); }
      bitmask |= 4;
    } else if (!strcmp(key, "device_index")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->ReadString(reader, type);
      if (strcmp(type, "list_int")) { LOGE("Invalid json format"); }
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      while (reader->NextArrayItem(reader)) {
        reader->ReadUnsignedInteger(reader, &(attr->device_index[device_index_count]));
        device_index_count ++;
      }
      if (reader->NextArrayItem(reader)) { LOGE("Invalid json format"); }
    } else {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
      reader->ReadString(reader, type);
      if (!strcmp(type, "list_int")) {
        if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
        // std::vector<int> temp;
        uint32_t temp[GRAPH_RUNTIME_MAX_NODES];
        uint32_t temp_count = 0;
        reader->BeginArray(reader);
        while (reader->NextArrayItem(reader)) {
          reader->ReadUnsignedInteger(reader, &(temp[temp_count]));
          temp_count ++;
        }
      } else if (!strcmp(type, "size_t")) {
        if (!(reader->NextArrayItem(reader))) { LOGE("Invalid json format"); }
        uint32_t temp;
        reader->ReadUnsignedInteger(reader, &temp);
      } else {
        LOGE("cannot skip graph attr %s", key);
      }
      if (reader->NextArrayItem(reader)) { LOGE("Invalid json format"); }
    }
  }
  if (bitmask != (1|2|4)) { LOGE("invalid format"); }
  return status;
}


typedef DLTensor* DLTensorPtr;

/*!
 * \brief Tiny graph runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
/* class GraphRuntime : public ModuleNode { */
typedef struct graph_runtime_t {
  void (*Run)(struct graph_runtime_t * runtime);

  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   */
  void (*Init)(struct graph_runtime_t * runtime,
               const char * graph_json,
               const Module * module,
               const TVMContext * ctxs);

  /*!
   * \brief Get the input index given the name of input.
   * \param name The name of the input.
   * \return The index of input.
   */
  int (*GetInputIndex)(struct graph_runtime_t * runtime, const char * name);

  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void (*SetInput)(struct graph_runtime_t * runtime, const char * name, DLTensor* data_in);
  /*!
   * \brief Return NDArray for given output index.
   * \param index The output index.
   *
   * \return NDArray corresponding to given output node index.
   */
  int (*GetOutput)(struct graph_runtime_t * runtime, const int32_t index, DLTensor * out);
  /*!
   * \brief Load parameters from parameter blob.
   * \param param_blob A binary blob of parameter.
   */
  int (*LoadParams)(struct graph_runtime_t * runtime, const char * param_blob, const uint32_t param_size);
  
  // The graph attribute fields.
  int (*Load)(struct graph_runtime_t * runtime, JSONReader *reader);
  /*! \brief Setup the temporal storage */
  void (*SetupStorage)(struct graph_runtime_t * runtime);
  /*! \brief Setup the executors. */
  int (*SetupOpExecs)(struct graph_runtime_t * runtime);
  
  /*!
   * \brief Create an execution function given input.
   * \param attrs The node attributes.
   * \param args The arguments to the functor, including inputs and outputs.
   * \param num_inputs Number of inputs.
   * \return The created executor.
   */
  int32_t (*CreateTVMOp)(struct graph_runtime_t * runtime, const TVMOpParam * attrs,
                         DLTensorPtr * args, const uint32_t args_count,
                         uint32_t num_inputs, PackedFunc * pf);
  
  // Get node entry index.
  uint32_t (*GetEntryId)(struct graph_runtime_t * runtime, uint32_t nid, uint32_t index);
  
  // /*! \brief The graph nodes. */
  /* GraphRuntimeNode nodes_[GRAPH_RUNTIME_MAX_NODES]; */
  GraphRuntimeNode nodes[GRAPH_RUNTIME_MAX_NODES];
  uint32_t           nodes_count;
  /*! \brief The argument nodes. */
  uint32_t input_nodes[GRAPH_RUNTIME_MAX_INPUT_NODES];
  uint32_t   input_nodes_count;
  /*! \brief Used for quick entry indexing. */
  uint32_t node_row_ptr[GRAPH_RUNTIME_MAX_NODE_ROW_PTR];
  uint32_t node_row_ptr_count;
  /*! \brief Output entries. */
  NodeEntry outputs[GRAPH_RUNTIME_MAX_OUTPUTS];
  uint32_t              outputs_count;
  /*! \brief Additional graph attributes. */
  GraphRuntimeGraphAttr attrs;
  /*! \brief The code module that contains both host and device code. */
  Module module;
  /*! \brief Execution context of all devices including the host. */
  TVMContext ctxs[GRAPH_RUNTIME_MAX_CONTEXTS];
  uint32_t   ctxs_count;
  /*! \brief Common storage pool for all devices. */
  NDArray  storage_pool[GRAPH_RUNTIME_MAX_NODES];
  uint32_t storage_pool_count;
  /*! \brief Data entry of each node. */
  NDArray  data_entry[GRAPH_RUNTIME_MAX_NODES];
  uint32_t data_entry_count;
  /*! \brief Operator on each node. */
  PackedFunc op_execs[GRAPH_RUNTIME_MAX_NODES];
  uint32_t op_execs_count;
} GraphRuntime;

static inline int GraphRuntime_Load(GraphRuntime * runtime, JSONReader *reader) {
    int status = TVM_STATUS_SUCCESS;
    reader->BeginObject(reader);
    int bitmask = 0;
    /* String key = StringCreate(); */
    char key[20];
    while (reader->NextObjectItem(reader, key)) {
      if (!strcmp(key, "nodes")) {
        reader->BeginArray(reader);
        while (reader->NextArrayItem(reader)) {
          GraphRuntimeNode * node = runtime->nodes + runtime->nodes_count;
          status = GraphRuntimeNode_Load(node, reader);
          if (status != TVM_STATUS_SUCCESS) {
            LOGE("Fail to load an element in `nodes` field in graph runtime node.");
            break;
#if TVM_CRT_DEBUG
          } else {
            LOGI("layer %u: `%s` loaded.", runtime->nodes_count, node->name);
#endif // TVM_CRT_DEBUG
          }
          runtime->nodes_count ++; /* seq push */
        }
        bitmask |= 1;
      } else if (!strcmp(key, "arg_nodes")) {
        reader->BeginArray(reader);
        while (reader->NextArrayItem(reader)) {
          uint32_t * node = runtime->input_nodes + runtime->input_nodes_count;
          reader->ReadUnsignedInteger(reader, node);
          runtime->input_nodes_count ++;
        }
        bitmask |= 2;
      } else if (!strcmp(key, "node_row_ptr")) {
        reader->BeginArray(reader);
        while (reader->NextArrayItem(reader)) {
          uint32_t count = runtime->node_row_ptr_count;
          uint32_t * node = runtime->node_row_ptr + count;
          reader->ReadUnsignedInteger(reader, node);
          runtime->node_row_ptr_count ++;
        }
        bitmask |= 4;
      } else if (!strcmp(key, "heads")) {
        reader->BeginArray(reader);
        while (reader->NextArrayItem(reader)) {
          NodeEntry * entry = runtime->outputs + runtime->outputs_count;
          status = NodeEntry_Load(entry, reader);
          if (status != TVM_STATUS_SUCCESS) {
            LOGE("Fail to load an element in `heads` field in graph runtime node.");
            break;
          }
          runtime->outputs_count ++; /* seq push */
        }        
        bitmask |= 8;
      } else if (!strcmp(key, "attrs")) {
        status = GraphRuntimeGraphAttr_Load(&(runtime->attrs), reader);
        if (status != TVM_STATUS_SUCCESS) {
          LOGE("Fail to load an element in `heads` field in graph runtime node.");
          break;
        }
        bitmask |= 16;
      } else if (!strcmp(key, "metadata")) {
        break;
      } else {
        LOGE("key %s is not supported", key);
        status = TVM_STATUS_FAILURE;
      }
      if (status != TVM_STATUS_SUCCESS) { break; }
    }
    if (!(bitmask == (1|2|4|8|16))) { LOGE("invalid format"); }
    return status;
}

static inline uint32_t GraphRuntime_GetEntryId(GraphRuntime * runtime, uint32_t nid, uint32_t index) {
  return runtime->node_row_ptr[nid] + index;
}

int32_t TVMGraphRuntimeCreate(GraphRuntime * runtime, const char * sym_json, const Module * m, const TVMContext * ctxs);

#endif  // TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
