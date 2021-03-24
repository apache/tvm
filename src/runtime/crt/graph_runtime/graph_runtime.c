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

// LINT_C_FILE

/*!
 * \file graph_runtime.c
 * \brief implement graph runtime in pure C
 */

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/internal/graph_runtime/graph_runtime.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/memory.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/packed_func.h>

#include "crt_config.h"

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif  // MAX

uint32_t Shape_Accumulate(int64_t* shape, uint32_t ndim) {
  int64_t accum = 1;
  uint32_t idx;
  for (idx = 0; idx < ndim; idx++) {
    if (shape[idx] == 0) {
      break;
    }
    accum *= shape[idx];
  }
  return accum;
}

int NodeEntry_Load(TVMGraphRuntimeNodeEntry* entry, JSONReader* reader) {
  int status = 0;
  reader->BeginArray(reader);
  if (!(reader->NextArrayItem(reader))) {
    fprintf(stderr, "invalid json format: failed to parse `node_id`\n");
    status = -1;
  }
  reader->ReadUnsignedInteger(reader, &(entry->node_id));
  if (!(reader->NextArrayItem(reader))) {
    fprintf(stderr, "invalid json format: failed to parse `index`\n");
    status = -1;
  }
  reader->ReadUnsignedInteger(reader, &(entry->index));
  if (reader->NextArrayItem(reader)) {
    reader->ReadUnsignedInteger(reader, &(entry->version));
    if (reader->NextArrayItem(reader)) {
      fprintf(stderr, "invalid json format: failed to parse `version`\n");
      status = -1;
    }
  } else {
    entry->version = 0;
  }
  return status;
}

void TVMGraphRuntimeNode_LoadAttrs(TVMGraphRuntimeNode* node, JSONReader* reader,
                                   TVMOpParam* param) {
  int bitmask = 0;
  char key[20], value[120];
  memset(param, 0, sizeof(TVMOpParam));
  memset(key, 0, sizeof(key));
  memset(value, 0, sizeof(value));
  reader->BeginObject(reader);
  while (reader->NextObjectItem(reader, key, sizeof(key))) {
    int status = reader->ReadString(reader, value, sizeof(value));
    if (status != 0) {
      fprintf(stderr, "error reading value for key: %s\n", key);
      break;
    }
    if (!strcmp(key, "func_name")) {
      snprintf(param->func_name, sizeof(value), "%s", value);
      bitmask |= 1;
    } else if (!strcmp(key, "num_inputs")) {
      param->num_inputs = strtoul(value, 0, 10);
      bitmask |= 2;
    } else if (!strcmp(key, "num_outputs")) {
      param->num_outputs = strtoul(value, 0, 10);
      bitmask |= 4;
    } else if (!strcmp(key, "flatten_data")) {
      param->flatten_data = strtoul(value, 0, 10);
      bitmask |= 8;
    } else {
      fprintf(stderr, "do not support key %s", key);
    }
  }
  if (bitmask != (1 | 2 | 4 | 8)) {
    fprintf(stderr, "invalid format\n");
  }
}

int TVMGraphRuntimeNode_Load(TVMGraphRuntimeNode* node, JSONReader* reader) {
  int status = 0;
  reader->BeginObject(reader);
  int bitmask = 0;
  char key[20];
  while (reader->NextObjectItem(reader, key, sizeof(key))) {
    if (!strcmp(key, "op")) {
      status = reader->ReadString(reader, node->op_type, sizeof(node->op_type));
      if (status != 0) {
        fprintf(stderr, "error reading op\n");
        break;
      }
      bitmask |= 1;
    } else if (!strcmp(key, "name")) {
      status = reader->ReadString(reader, node->name, sizeof(node->name));
      if (status != 0) {
        fprintf(stderr, "error reading name\n");
        break;
      }
      bitmask |= 2;
    } else if (!strcmp(key, "inputs")) {
      size_t count = node->inputs_count;
      reader->BeginArray(reader);
      size_t num_inputs = 0;
      if (reader->ArrayLength(reader, &num_inputs) != 0) {
        fprintf(stderr, "error determining inputs array length\n");
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(TVMGraphRuntimeNodeEntry) * num_inputs,
                                                      ctx, (void**)&node->inputs);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        return -1;
      }
      while (reader->NextArrayItem(reader)) {
        if (count == num_inputs) {
          fprintf(stderr, "too many array elements\n");
          return -1;
        }

        TVMGraphRuntimeNodeEntry* inputs = node->inputs + count;
        reader->BeginArray(reader);
        if (!reader->NextArrayItem(reader)) {
          fprintf(stderr, "invalid json format\n");
          status = -1;
          break;
        }
        reader->ReadUnsignedInteger(reader, &(inputs->node_id));
        if (!reader->NextArrayItem(reader)) {
          fprintf(stderr, "invalid json format\n");
          status = -1;
          break;
        }
        reader->ReadUnsignedInteger(reader, &(inputs->index));
        if (reader->NextArrayItem(reader)) {
          reader->ReadUnsignedInteger(reader, &(inputs->version));
          if (reader->NextArrayItem(reader)) {
            fprintf(stderr, "invalid json format\n");
            status = -1;
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

      TVMGraphRuntimeNode_LoadAttrs(node, reader, &param);
      memcpy(&node->param, &param, sizeof(param));
    } else if (!strcmp(key, "control_deps")) {
      fprintf(stderr, "do not support key %s", key);
      status = -1;
    } else {
      fprintf(stderr, "do not support key %s", key);
      status = -1;
    }
    if (status != 0) {
      break;
    }
  }
  if (bitmask != (1 | 2 | 4)) {
    fprintf(stderr, "invalid format\n");
    status = -1;
  }
  return status;
}

TVMGraphRuntimeNode TVMGraphRuntimeNodeCreate() {
  TVMGraphRuntimeNode node;
  memset(&node, 0, sizeof(TVMGraphRuntimeNode));
  node.LoadAttrs = TVMGraphRuntimeNode_LoadAttrs;
  node.Load = TVMGraphRuntimeNode_Load;
  return node;
}

int TVMGraphRuntimeNodeRelease(TVMGraphRuntimeNode* node) {
  if (!node) {
    return 0;
  }
  if (node->inputs) {
    DLContext ctx = {kDLCPU, 0};
    tvm_crt_error_t err = TVMPlatformMemoryFree(node->inputs, ctx);
    node->inputs = 0;
    if (err != kTvmErrorNoError) {
      return -1;
    }
  }

  return 0;
}

int TVMGraphRuntimeGraphAttr_Load(TVMGraphRuntimeGraphAttr* attr, JSONReader* reader) {
  int status = 0;
  int bitmask = 0;
  char key[16], type[16];
  uint32_t storage_id_count = 0;
  uint32_t dltype_count = 0;
  uint32_t shape_count = 0;
  uint32_t device_index_count = 0;
  reader->BeginObject(reader);
  while (reader->NextObjectItem(reader, key, sizeof(key))) {
    if (!strcmp(key, "dltype")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      status = reader->ReadString(reader, type, sizeof(type));
      if (status != 0) {
        fprintf(stderr, "error reading dltype type\n");
        break;
      }
      if (strcmp(type, "list_str")) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      reader->BeginArray(reader);
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_str length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err =
          TVMPlatformMemoryAllocate(TVM_CRT_STRLEN_DLTYPE * num_items, ctx, (void**)&attr->dltype);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        return -1;
      }
      dltype_count = 0;
      while (reader->NextArrayItem(reader)) {
        if (dltype_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        status = reader->ReadString(reader, attr->dltype + dltype_count * TVM_CRT_STRLEN_DLTYPE,
                                    TVM_CRT_STRLEN_DLTYPE);
        if (status != 0) {
          fprintf(stderr, "error reading dltype array item");
          break;
        }
        dltype_count++;
      }
      attr->dltype_count = dltype_count;

      if (reader->NextArrayItem(reader)) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      bitmask |= 1;
    } else if (!strcmp(key, "storage_id")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      status = reader->ReadString(reader, type, sizeof(type));
      if (status != 0) {
        fprintf(stderr, "error reading device_index array item");
      }
      if (strcmp(type, "list_int")) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      reader->BeginArray(reader);
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_str length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err =
          TVMPlatformMemoryAllocate(sizeof(uint32_t) * num_items, ctx, (void**)&attr->storage_id);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        return -1;
      }
      storage_id_count = 0;
      while (reader->NextArrayItem(reader)) {
        if (storage_id_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        reader->ReadUnsignedInteger(reader, &(attr->storage_id[storage_id_count]));
        storage_id_count++;
      }
      if (reader->NextArrayItem(reader)) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      bitmask |= 2;
    } else if (!strcmp(key, "shape")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      status = reader->ReadString(reader, type, sizeof(type));
      if (status != 0) {
        fprintf(stderr, "error reading shape array item\n");
        break;
      }
      if (strcmp(type, "list_shape")) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      reader->BeginArray(reader);
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_str length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err = TVMPlatformMemoryAllocate(
          sizeof(int64_t) * TVM_CRT_MAX_NDIM * num_items, ctx, (void**)&attr->shape);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        status = -1;
        break;
      }
      err = TVMPlatformMemoryAllocate(sizeof(uint32_t) * num_items, ctx, (void**)&attr->ndim);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        status = -1;
        break;
      }
      shape_count = 0;
      while (reader->NextArrayItem(reader)) {
        if (shape_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        reader->BeginArray(reader);
        int64_t* attr_shape_ptr = attr->shape + shape_count * TVM_CRT_MAX_NDIM;
        reader->ReadInteger(reader, attr_shape_ptr + 0);
        uint32_t ndim = 1;
        if (reader->NextArrayItem(reader)) {
          for (ndim = 1; ndim < TVM_CRT_MAX_NDIM; ndim++) {
            if (reader->NextArrayItem(reader)) {
              reader->ReadInteger(reader, attr_shape_ptr + ndim);
            } else {
              break;
            }
          }
          if (ndim == TVM_CRT_MAX_NDIM) {
            reader->NextArrayItem(reader);
          }
        }
        attr->ndim[shape_count] = ndim;
        shape_count++;
      }
      attr->shape_count = shape_count;
      if (reader->NextArrayItem(reader)) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      bitmask |= 4;
    } else if (!strcmp(key, "device_index")) {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      status = reader->ReadString(reader, type, sizeof(type));
      if (status != 0) {
        fprintf(stderr, "error reading device_index array item");
        break;
      }
      if (strcmp(type, "list_int")) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_int length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err =
          TVMPlatformMemoryAllocate(sizeof(uint32_t) * num_items, ctx, (void**)&attr->device_index);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        status = -1;
        break;
      }
      device_index_count = 0;
      while (reader->NextArrayItem(reader)) {
        if (device_index_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        reader->ReadUnsignedInteger(reader, &(attr->device_index[device_index_count]));
        device_index_count++;
      }
      if (reader->NextArrayItem(reader)) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
    } else {
      reader->BeginArray(reader);
      if (!(reader->NextArrayItem(reader))) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
      reader->ReadString(reader, type, sizeof(type));
      if (!strcmp(type, "list_int")) {
        if (!(reader->NextArrayItem(reader))) {
          fprintf(stderr, "Invalid json format\n");
          status = -1;
          break;
        }
        uint32_t temp_count = 0;
        reader->BeginArray(reader);
        while (reader->NextArrayItem(reader)) {
          uint32_t temp;
          reader->ReadUnsignedInteger(reader, &temp);
          temp_count++;
        }
      } else if (!strcmp(type, "size_t")) {
        if (!(reader->NextArrayItem(reader))) {
          fprintf(stderr, "Invalid json format\n");
          status = -1;
          break;
        }
        uint32_t temp;
        reader->ReadUnsignedInteger(reader, &temp);
      } else {
        fprintf(stderr, "cannot skip graph attr %s", key);
        status = -1;
        break;
      }
      if (reader->NextArrayItem(reader)) {
        fprintf(stderr, "Invalid json format\n");
        status = -1;
        break;
      }
    }
  }
  if (bitmask != (1 | 2 | 4)) {
    fprintf(stderr, "invalid format\n");
    status = -1;
  }
  return status;
}

int TVMGraphRuntimeGraphAttr_Release(TVMGraphRuntimeGraphAttr* attr) {
  if (!attr) {
    return 0;
  }
  if (attr->storage_id) {
    DLContext ctx = {kDLCPU, 0};
    tvm_crt_error_t err = TVMPlatformMemoryFree(attr->storage_id, ctx);
    attr->storage_id = 0;
    if (err != kTvmErrorNoError) {
      return -1;
    }
  }
  if (attr->device_index) {
    DLContext ctx = {kDLCPU, 0};
    tvm_crt_error_t err = TVMPlatformMemoryFree(attr->device_index, ctx);
    attr->device_index = 0;
    if (err != kTvmErrorNoError) {
      return -1;
    }
  }
  if (attr->dltype) {
    DLContext ctx = {kDLCPU, 0};
    tvm_crt_error_t err = TVMPlatformMemoryFree(attr->dltype, ctx);
    attr->dltype = 0;
    if (err != kTvmErrorNoError) {
      return -1;
    }
  }
  if (attr->shape) {
    DLContext ctx = {kDLCPU, 0};
    tvm_crt_error_t err = TVMPlatformMemoryFree(attr->shape, ctx);
    attr->shape = 0;
    if (err != kTvmErrorNoError) {
      return -1;
    }
  }
  if (attr->ndim) {
    DLContext ctx = {kDLCPU, 0};
    tvm_crt_error_t err = TVMPlatformMemoryFree(attr->ndim, ctx);
    attr->ndim = 0;
    if (err != kTvmErrorNoError) {
      return -1;
    }
  }

  return 0;
}

int TVMGraphRuntime_Load(TVMGraphRuntime* runtime, JSONReader* reader) {
  int status = 0;
  reader->BeginObject(reader);
  int bitmask = 0;
  char key[20];
  while (reader->NextObjectItem(reader, key, sizeof(key))) {
    if (!strcmp(key, "nodes")) {
      reader->BeginArray(reader);
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_int length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(TVMGraphRuntimeNode) * num_items, ctx,
                                                      (void**)&runtime->nodes);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        status = -1;
        break;
      }
      while (reader->NextArrayItem(reader)) {
        if (runtime->nodes_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        TVMGraphRuntimeNode* node = runtime->nodes + runtime->nodes_count;
        status = TVMGraphRuntimeNode_Load(node, reader);
        if (status != 0) {
          fprintf(stderr, "failed to load an element in `nodes` field in graph runtime node.\n");
          break;
#if TVM_CRT_DEBUG
        } else {
          printf("loading: node (%u) %s loaded.\n", runtime->nodes_count, node->name);
#endif  // TVM_CRT_DEBUG
        }
        runtime->nodes_count++;
      }
      bitmask |= 1;
    } else if (!strcmp(key, "arg_nodes")) {
      reader->BeginArray(reader);
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_int length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(uint32_t) * num_items, ctx,
                                                      (void**)&runtime->input_nodes);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        status = -1;
        break;
      }
      while (reader->NextArrayItem(reader)) {
        if (runtime->input_nodes_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        uint32_t* node = runtime->input_nodes + runtime->input_nodes_count;
        reader->ReadUnsignedInteger(reader, node);
        runtime->input_nodes_count++;
      }
      bitmask |= 2;
    } else if (!strcmp(key, "node_row_ptr")) {
      reader->BeginArray(reader);
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_int length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(uint32_t) * num_items, ctx,
                                                      (void**)&runtime->node_row_ptr);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        status = -1;
        break;
      }
      while (reader->NextArrayItem(reader)) {
        if (runtime->node_row_ptr_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        uint32_t count = runtime->node_row_ptr_count;
        uint32_t* node = runtime->node_row_ptr + count;
        reader->ReadUnsignedInteger(reader, node);
        runtime->node_row_ptr_count++;
      }
      bitmask |= 4;
    } else if (!strcmp(key, "heads")) {
      reader->BeginArray(reader);
      size_t num_items = 0;
      if (reader->ArrayLength(reader, &num_items) != 0) {
        fprintf(stderr, "error determing list_int length\n");
        status = -1;
        break;
      }
      DLContext ctx = {kDLCPU, 0};
      tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(TVMGraphRuntimeNodeEntry) * num_items,
                                                      ctx, (void**)&runtime->outputs);
      if (err != kTvmErrorNoError) {
        fprintf(stderr, "memory allocate error: %08x", err);
        status = -1;
        break;
      }
      while (reader->NextArrayItem(reader)) {
        if (runtime->outputs_count == num_items) {
          fprintf(stderr, "array too big\n");
          status = -1;
          return status;
        }
        TVMGraphRuntimeNodeEntry* entry = runtime->outputs + runtime->outputs_count;
        status = NodeEntry_Load(entry, reader);
        if (status != 0) {
          fprintf(stderr, "Fail to load an element in `heads` field in graph runtime node.\n");
          break;
        }
        runtime->outputs_count++;
      }
      bitmask |= 8;
    } else if (!strcmp(key, "attrs")) {
      status = TVMGraphRuntimeGraphAttr_Load(&(runtime->attrs), reader);
      if (status != 0) {
        fprintf(stderr, "Fail to load an element in `heads` field in graph runtime node.\n");
        break;
      }
      bitmask |= 16;
    } else if (!strcmp(key, "metadata")) {
      break;
    } else {
      fprintf(stderr, "key %s is not supported\n", key);
      status = -1;
    }
    if (status != 0) {
      break;
    }
  }
  if (!(bitmask == (1 | 2 | 4 | 8 | 16))) {
    fprintf(stderr, "invalid format\n");
    status = -1;
  }
  return status;
}

uint32_t TVMGraphRuntime_GetEntryId(TVMGraphRuntime* runtime, uint32_t nid, uint32_t index) {
  return runtime->node_row_ptr[nid] + index;
}

/*!
 * \brief Get the number of input tensors allocated.
 * \param runtime The graph runtime.
 * \return the number of input tensors allocated.
 */
int TVMGraphRuntime_GetNumInputs(TVMGraphRuntime* runtime) { return runtime->input_nodes_count; }

/*!
 * \brief Get the input index given the name of input.
 * \param runtime The graph runtime.
 * \param name The name of the input.
 * \return The index of input.
 */
int TVMGraphRuntime_GetInputIndex(TVMGraphRuntime* runtime, const char* name) {
  uint32_t i;
  int32_t rv = -1;
  for (i = 0; i < runtime->input_nodes_count; ++i) {
    uint32_t nid = runtime->input_nodes[i];
    if (!strcmp(runtime->nodes[nid].name, name)) {
      rv = i;
      break;
    }
  }
  CHECK_GE(rv, 0, "cannot find '%s' among input.", name);
  return rv;
}

/*!
 * \brief set input to the graph based on name.
 * \param runtime The graph runtime.
 * \param name The name of the input.
 * \param data_in The input data.
 */
void TVMGraphRuntime_SetInput(TVMGraphRuntime* runtime, const char* name, DLTensor* data_in) {
  uint32_t index = TVMGraphRuntime_GetInputIndex(runtime, name);
  if (index >= runtime->input_nodes_count) {
    fprintf(stderr, "given index is greater than num of input nodes.\n");
  }
  uint32_t eid = TVMGraphRuntime_GetEntryId(runtime, runtime->input_nodes[index], 0);
  runtime->data_entry[eid].dl_tensor.data = data_in->data;
}

/*!
 * \brief Load parameters from parameter blob.
 * \param runtime The graph runtime.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return The result of this function execution.
 */
int TVMGraphRuntime_LoadParams(TVMGraphRuntime* runtime, const char* param_blob,
                               const uint32_t param_size) {
  int status = 0;
  const char* bptr = param_blob;
  uint64_t header, reserved;
  memcpy(&header, bptr, sizeof(header));
  bptr += sizeof(header);
  if (header != kTVMNDArrayListMagic) {
    fprintf(stderr, "Invalid parameters file format");
    status = -1;
  }
  memcpy(&reserved, bptr, sizeof(reserved));
  bptr += sizeof(reserved);

  // read names
  char* names = NULL;
  DLContext ctx = {kDLCPU, 0};
  tvm_crt_error_t err =
      TVMPlatformMemoryAllocate(TVM_CRT_STRLEN_NAME * runtime->nodes_count, ctx, (void**)&names);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory allocate error: %08x", err);
    status = -1;
    return status;
  }
  memset(names, 0, TVM_CRT_STRLEN_NAME * runtime->nodes_count);
  uint64_t names_count;
  int idx;
  memcpy(&names_count, bptr, sizeof(names_count));
  bptr += sizeof(names_count);
  for (idx = 0; idx < names_count; idx++) {
    uint64_t name_length;
    memcpy(&name_length, bptr, sizeof(name_length));
    bptr += sizeof(name_length);
    if (name_length >= TVM_CRT_STRLEN_NAME) {
      fprintf(stderr, "Error: function name longer than expected.\n");
      status = -1;
    }
    memcpy(names + TVM_CRT_STRLEN_NAME * idx, bptr, name_length);
    bptr += name_length;
  }

  // read sizes
  uint64_t sz;
  memcpy(&sz, bptr, sizeof(sz));
  bptr += sizeof(sz);
  uint32_t size = sz;
  if (size != names_count) {
    fprintf(stderr, "Invalid parameters file format\n");
    status = -1;
  }

  for (idx = 0; idx < size; idx++) {
    int32_t in_idx = TVMGraphRuntime_GetInputIndex(runtime, names + TVM_CRT_STRLEN_NAME * idx);
    CHECK_GT(in_idx, 0, "Found param for non-existent input: %s\n",
             names + TVM_CRT_STRLEN_NAME * idx);
    uint32_t eid = TVMGraphRuntime_GetEntryId(runtime, runtime->input_nodes[in_idx], 0);
    if (!(eid < runtime->data_entry_count)) {
      fprintf(stderr, "`entry_id`=%d is greater than expected(%d).\n", eid,
              runtime->data_entry_count);
      status = -1;
    }

    if (runtime->data_entry[eid].dl_tensor.shape) {
      err = TVMPlatformMemoryFree(runtime->data_entry[eid].dl_tensor.shape, ctx);
      if (err != kTvmErrorNoError) {
        status = -1;
      }
      runtime->data_entry[eid].dl_tensor.shape = 0;
    }
    if (runtime->data_entry[eid].dl_tensor.data) {
      err = TVMPlatformMemoryFree(runtime->data_entry[eid].dl_tensor.data, ctx);
      if (err != kTvmErrorNoError) {
        status = -1;
      }
      runtime->data_entry[eid].dl_tensor.data = 0;
    }
    status |= TVMNDArray_Load(&(runtime->data_entry[eid]), &bptr);
#if TVM_CRT_DEBUG
    TVMNDArray* entry = &(runtime->data_entry[eid]);
    printf("loading: param %s loaded, in_idx=%d, eid=%d, ndim=%d, data[0]=%f\n",
           names + TVM_CRT_STRLEN_NAME * idx, in_idx, eid, entry->dl_tensor.ndim,
           ((float*)entry->dl_tensor.data)[0]);  // NOLINT(*)
#endif                                           // TVM_CRT_DEBUG
  }

  // Release memory
  err = TVMPlatformMemoryFree(names, ctx);
  if (err != kTvmErrorNoError) {
    status = -1;
    return status;
  }

  return status;
}

/*!
 * \brief Run all the operations one by one.
 * \param runtime The graph runtime.
 */
void TVMGraphRuntime_Run(TVMGraphRuntime* runtime) {
  // setup the array and requirements.
  uint32_t idx;
  for (idx = 0; idx < runtime->op_execs_count; ++idx) {
    if (runtime->op_execs[idx].fexec) {
#if TVM_CRT_DEBUG
      printf("calling: %s (%d)\n", runtime->op_execs[idx].name, idx);
#endif  // TVM_CRT_DEBUG
      runtime->op_execs[idx].Call(&(runtime->op_execs[idx]));
    }
  }
}

/*!
 * \brief Get the number of output tensors allocated.
 * \param runtime The graph runtime.
 * \return the number of output tensors allocated.
 */
int TVMGraphRuntime_GetNumOutputs(TVMGraphRuntime* runtime) { return runtime->outputs_count; }

int TVMGraphRuntime_GetOutput(TVMGraphRuntime* runtime, const int32_t idx, DLTensor* out) {
  int status = 0;
  uint32_t nid = runtime->outputs[idx].node_id;
  uint32_t index = runtime->outputs[idx].index;
  uint32_t eid = TVMGraphRuntime_GetEntryId(runtime, nid, index);

  // copy data section to allocated output tensor
  int32_t elem_bytes = out->dtype.bits / 8;
  int64_t size = Shape_Accumulate(out->shape, out->ndim);
  DLTensor* tensor = &(runtime->data_entry[eid].dl_tensor);
  CHECK(out->ndim == tensor->ndim);
  CHECK(out->dtype.bits == tensor->dtype.bits);
  CHECK(Shape_Accumulate(out->shape, out->ndim) == Shape_Accumulate(tensor->shape, tensor->ndim));
  memcpy(out->data, tensor->data, size * elem_bytes);
  return status;
}

int TVMGraphRuntime_SetupStorage(TVMGraphRuntime* runtime) {
  TVMPackedFunc lookup_linked_param;
  int lookup_linked_param_valid;
  uint32_t idx;

  {
    TVMArgs temp_args;
    temp_args.values[0].v_int64 = 0;
    temp_args.tcodes[0] = kTVMArgInt;
    temp_args.values_count = 1;
    lookup_linked_param_valid =
        (TVMPackedFunc_InitModuleFunc(&lookup_linked_param, runtime->module_handle,
                                      "_lookup_linked_param", &temp_args) == 0);
  }

  // Grab saved optimization plan from graph.
  TVMGraphRuntimeGraphAttr* attrs = &(runtime->attrs);
  DLDataType* vtype = NULL;
  DLContext alloc_ctx = {kDLCPU, 0};
  tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(DLDataType) * attrs->dltype_count,
                                                  alloc_ctx, (void**)&vtype);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory allocate error: %08x", err);
    return -1;
  }
  for (idx = 0; idx < attrs->dltype_count; idx++) {
    vtype[idx] = String2DLDataType(attrs->dltype + idx * TVM_CRT_STRLEN_DLTYPE);
  }

  // Size and device type of each storage pool entry.
  TVMGraphRuntimePoolEntry* pool_entry = NULL;
  err = TVMPlatformMemoryAllocate(sizeof(TVMGraphRuntimePoolEntry) * runtime->nodes_count,
                                  alloc_ctx, (void**)&pool_entry);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory allocate error: %08x", err);
    return -1;
  }
  memset(pool_entry, 0, sizeof(TVMGraphRuntimePoolEntry) * runtime->nodes_count);
  uint32_t pool_entry_count = 0;
  // Find the maximum space size.
  for (idx = 0; idx < attrs->shape_count; idx++) {
    int storage_id = attrs->storage_id[idx];
    // Use the fallback device if no device index is available.
    int device_type = runtime->ctxs[0].device_type;
    uint32_t size = Shape_Accumulate(attrs->shape + idx * TVM_CRT_MAX_NDIM, attrs->ndim[idx]);
    DLDataType t = vtype[idx];
    uint32_t bits = t.bits * t.lanes;
    size_t bytes = ((bits + 7U) / 8U) * size;

    uint32_t sid = storage_id;
    if (sid >= pool_entry_count) {
      pool_entry_count = sid + 1;
    }
    pool_entry[sid].entry_id = idx;
    pool_entry[sid].size = MAX(pool_entry[sid].size, bytes);
    pool_entry[sid].device_type = device_type;
  }

  // Allocate the space.
  err = TVMPlatformMemoryAllocate(sizeof(TVMGraphRuntimeStorageEntry) * pool_entry_count, alloc_ctx,
                                  (void**)&runtime->storage_pool);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory allocate error: %08x", err);
    return -1;
  }
  for (idx = 0; idx < pool_entry_count; idx++) {
    TVMGraphRuntimePoolEntry pit = pool_entry[idx];
    TVMContext ctx = runtime->ctxs[0];
    uint8_t did_find_linked_param = 0;
    if (lookup_linked_param_valid) {
      lookup_linked_param.args.values[0].v_int64 = idx;
      CHECK_EQ(lookup_linked_param.Call(&lookup_linked_param), 0, "lookup_linked_param");

      void* linked_param_data = lookup_linked_param.ret_value.values[0].v_handle;
      if (linked_param_data != NULL) {
        runtime->storage_pool[runtime->storage_pool_count].is_linked_param = 1;
        DLTensor* tensor = &runtime->storage_pool[runtime->storage_pool_count].array.dl_tensor;
        tensor->data = linked_param_data;
        tensor->ctx = ctx;
        tensor->ndim = attrs->ndim[pit.entry_id];
        tensor->shape = attrs->shape + idx * TVM_CRT_MAX_NDIM;
        tensor->strides = NULL;
        tensor->byte_offset = 0;
        did_find_linked_param = 1;
      }
    }
    if (did_find_linked_param == 0) {
      DLDataType dtype = {kDLFloat, 32, 1};
      int64_t shape[TVM_CRT_MAX_NDIM] = {
          0,
      };
      shape[0] = (pit.size + 3) / 4;
      int status = TVMNDArray_Empty(1, shape, dtype, ctx,
                                    &runtime->storage_pool[runtime->storage_pool_count].array);
      CHECK_EQ(status, 0, "fail to create storage_pool with idx=%d\n", idx);
    }
    runtime->storage_pool_count++;
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  runtime->data_entry_count = runtime->node_row_ptr[runtime->node_row_ptr_count - 1];
  err = TVMPlatformMemoryAllocate(sizeof(TVMNDArray) * runtime->data_entry_count, alloc_ctx,
                                  (void**)&runtime->data_entry);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory allocate error: %08x", err);
    return -1;
  }
  for (idx = 0; idx < runtime->data_entry_count; ++idx) {
    uint32_t storage_id = attrs->storage_id[idx];
    CHECK(storage_id < runtime->storage_pool_count);
    int status = TVMNDArray_CreateView(&(runtime->storage_pool[storage_id].array),
                                       attrs->shape + idx * TVM_CRT_MAX_NDIM, attrs->ndim[idx],
                                       vtype[idx], &runtime->data_entry[idx]);
    CHECK_EQ(status, 0, "fail to create for node with idx=%d, storage_id=%u\n", idx, storage_id);
  }

  // Release memory
  err = TVMPlatformMemoryFree(vtype, alloc_ctx);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory free error: %08x", err);
    return err;
  }

  err = TVMPlatformMemoryFree(pool_entry, alloc_ctx);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory free error: %08x", err);
    return -1;
  }

  return 0;
}

int TVMGraphRuntime_SetupOpExecs(TVMGraphRuntime* runtime) {
  int status = 0;
  uint32_t nid, idx;
  runtime->op_execs_count = runtime->nodes_count;
  DLContext ctx = {kDLCPU, 0};
  tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(TVMPackedFunc) * runtime->op_execs_count,
                                                  ctx, (void**)&runtime->op_execs);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory allocate error: %08x", err);
    status = -1;
    return status;
  }
  for (nid = 0; nid < runtime->nodes_count; nid++) {
    const TVMGraphRuntimeNode* inode = runtime->nodes + nid;
    if (strcmp(inode->op_type, "null")) {
      DLTensorPtr args[TVM_CRT_MAX_ARGS];
      uint32_t args_count = 0;
      for (idx = 0; idx < inode->inputs_count; idx++) {
        const TVMGraphRuntimeNodeEntry* entry = inode->inputs + idx;
        uint32_t eid = TVMGraphRuntime_GetEntryId(runtime, entry->node_id, entry->index);
        args[idx] = &(runtime->data_entry[eid].dl_tensor);
        args_count++;
      }
      for (idx = 0; idx < inode->param.num_outputs; idx++) {
        uint32_t eid = TVMGraphRuntime_GetEntryId(runtime, nid, idx);
        args[args_count] = &(runtime->data_entry[eid].dl_tensor);
        args_count++;
      }
      if (strcmp(inode->op_type, "tvm_op")) {
        fprintf(stderr, "Can only take tvm_op as op, but \"%s\" is found.\n", inode->op_type);
        status = -1;
        break;
      }
      if (args_count >= TVM_CRT_MAX_ARGS) {
        fprintf(stderr, "too many arguments: expected less than %d args, but got %d.\n",
                TVM_CRT_MAX_ARGS, args_count);
        status = -1;
        break;
      }
#if TVM_CRT_DEBUG
      printf("tvm_op: creating %s with node_id=%d\n", inode->param.func_name, nid);
#endif  // TVM_CRT_DEBUG
      TVMPackedFunc pf;
      TVMGraphRuntime_CreateTVMOp(runtime, &(inode->param), args, args_count, inode->inputs_count,
                                  &pf);
      runtime->op_execs[nid] = pf;
    }
  }
  return status;
}

typedef struct TVMOpArgs {
  DLTensor args[TVM_CRT_MAX_ARGS];
  uint32_t args_count;
  TVMValue arg_values[TVM_CRT_MAX_ARGS];
  uint32_t arg_values_count;
  uint32_t arg_tcodes[TVM_CRT_MAX_ARGS];
  uint32_t arg_tcodes_count;
  int64_t shape_data[TVM_CRT_MAX_ARGS];
  uint32_t shape_data_count;
} TVMOpArgs;

int32_t TVMGraphRuntime_CreateTVMOp(TVMGraphRuntime* runtime, const TVMOpParam* param,
                                    DLTensorPtr* args, const uint32_t args_count,
                                    uint32_t num_inputs, TVMPackedFunc* pf) {
  int status = 0;
  uint32_t idx;
  TVMOpArgs arg_ptr;
  memset(&arg_ptr, 0, sizeof(TVMOpArgs));
  arg_ptr.args_count = args_count;
  if (param->flatten_data) {
    arg_ptr.shape_data_count = arg_ptr.args_count;
  }
  for (idx = 0; idx < arg_ptr.args_count; ++idx) {
    TVMValue v;
    memset(&v, 0, sizeof(v));
    DLTensor* t = &(arg_ptr.args[idx]);
    /* v.v_handle = &((*args)[idx]); */
    v.v_handle = args[idx];
    arg_ptr.arg_values[idx] = v;
    arg_ptr.arg_values_count++;
    arg_ptr.arg_tcodes[idx] = kTVMNDArrayHandle;
    arg_ptr.arg_tcodes_count++;
    if (param->flatten_data) {
      arg_ptr.shape_data[idx] = Shape_Accumulate(t->shape, t->ndim);
      t->ndim = 1;
      t->shape[0] = arg_ptr.shape_data[idx];
    }
  }
  if (!strcmp(param->func_name, "__nop") || !strcmp(param->func_name, "__copy")) {
    fprintf(stderr, "%s function is not yet supported.", param->func_name);
    status = -1;
  }

  TVMArgs targs = TVMArgs_Create(arg_ptr.arg_values, arg_ptr.arg_tcodes, arg_ptr.arg_values_count);
  status = TVMPackedFunc_InitModuleFunc(pf, runtime->module_handle, param->func_name, &targs);

  return status;
}

/*!
 * \brief Initialize the graph executor with graph and context.
 * \param graph_json The execution graph.
 * \param module_handle The module containing the compiled functions for the host
 * processor.
 * \param ctxs The context of the host and devices where graph nodes will be
 * executed on.
 * \return 0 on success.
 */
int TVMGraphRuntime_Init(TVMGraphRuntime* runtime, const char* graph_json,
                         TVMModuleHandle module_handle, const TVMContext* ctxs) {
  JSONReader reader;
  tvm_crt_error_t err = JSONReader_Create(graph_json, &reader);
  if (err != kTvmErrorNoError) {
    return -1;
  }

  TVMGraphRuntime_Load(runtime, &reader);
  err = JSONReader_Release(&reader);
  if (err != kTvmErrorNoError) {
    return -1;
  }
  runtime->module_handle = module_handle;
  runtime->ctxs[0] = ctxs[0];

  int status;
  status = TVMGraphRuntime_SetupStorage(runtime);
  if (status != 0) {
    return status;
  }
  status = TVMGraphRuntime_SetupOpExecs(runtime);
  if (status != 0) {
    if (status != 0) {
      return status;
    }

    return status;
  }

  return status;
}

int TVMGraphRuntime_Create(const char* sym_json, TVMModuleHandle module_handle,
                           const TVMContext* ctxs, TVMGraphRuntime** runtime) {
  DLContext ctx = {kDLCPU, 0};
  tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(TVMGraphRuntime), ctx, (void**)runtime);
  if (err != kTvmErrorNoError) {
    fprintf(stderr, "memory allocate error: %08x", err);
    return -1;
  }

  memset(*runtime, 0, sizeof(TVMGraphRuntime));
  // init
  return TVMGraphRuntime_Init(*runtime, sym_json, module_handle, ctxs);
}

int TVMGraphRuntime_Release(TVMGraphRuntime** pptr) {
  int status = 0;
  int32_t idx;
  TVMGraphRuntime* runtime = (TVMGraphRuntime*)(*pptr);
  for (idx = 0; idx < runtime->nodes_count; ++idx) {
    status = TVMGraphRuntimeNodeRelease(&(runtime->nodes[idx]));
    if (status != 0) {
      return status;
    }
  }
  DLContext ctx = {kDLCPU, 0};
  status = TVMPlatformMemoryFree(runtime->nodes, ctx);
  if (status != 0) {
    return status;
  }
  status = TVMGraphRuntimeGraphAttr_Release(&(runtime->attrs));
  if (status != 0) {
    return status;
  }
  for (idx = 0; idx < runtime->storage_pool_count; ++idx) {
    if (runtime->storage_pool[idx].is_linked_param == 0) {
      status = TVMNDArray_Release(&(runtime->storage_pool[idx]).array);
      if (status != 0) {
        return status;
      }
    }
  }
  for (idx = 0; idx < runtime->data_entry_count; ++idx) {
    status = TVMPlatformMemoryFree(runtime->data_entry[idx].dl_tensor.shape, ctx);
    if (status != 0) {
      return status;
    }
  }
  status = TVMPlatformMemoryFree(runtime->input_nodes, ctx);
  if (status != 0) {
    return status;
  }
  status = TVMPlatformMemoryFree(runtime->node_row_ptr, ctx);
  if (status != 0) {
    return status;
  }
  status = TVMPlatformMemoryFree(runtime->outputs, ctx);
  if (status != 0) {
    return status;
  }
  status = TVMPlatformMemoryFree(runtime->storage_pool, ctx);
  if (status != 0) {
    return status;
  }
  status = TVMPlatformMemoryFree(runtime->data_entry, ctx);
  if (status != 0) {
    return status;
  }
  status = TVMPlatformMemoryFree(runtime->op_execs, ctx);
  if (status != 0) {
    return status;
  }
  status = TVMPlatformMemoryFree(*pptr, ctx);
  if (status != 0) {
    return status;
  }

  if (g_fexecs) {
    status = TVMPlatformMemoryFree(g_fexecs, ctx);
    g_fexecs = 0;
    if (status != 0) {
      return status;
    }
  }

  return 0;
}
