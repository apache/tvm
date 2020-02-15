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
 * \file graph_runtime.c
 * \brief implement graph runtime in pure C
 */
#include "graph_runtime.h" // <tvm/runtime/crt/graph_runtime.h>
#include <tvm/runtime/crt/vm.h>

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif // MAX

static inline uint32_t Shape_Accumulate(int64_t * shape, uint32_t ndim) {
  int64_t accum = 1;
  uint32_t idx;
  for (idx = 0; idx < ndim; idx++) {
    if (shape[idx] == 0) { break; }
    accum *= shape[idx];
  }
  return accum;
}

/*!
 * \brief Get the input index given the name of input.
 * \param name The name of the input.
 * \return The index of input.
 */
int GraphRuntime_GetInputIndex(GraphRuntime * runtime, const char * name) {
  uint32_t i;
  int32_t rv = -1;
  for (i = 0; i< runtime->input_nodes_count; ++i) {
    uint32_t nid = runtime->input_nodes[i];
    if (!strcmp(runtime->nodes[nid].name, name)) {
      rv = i;
      break;
    }
  }
  if (rv < 0) {
    fprintf(stderr, "cannot find \"%s\" among input", name);
  }
  return rv;
}

/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void GraphRuntime_SetInput(struct graph_runtime_t * runtime, const char * name, DLTensor* data_in) {
  uint32_t index = runtime->GetInputIndex(runtime, name);
  if (index >= runtime->input_nodes_count) {
    fprintf(stderr, "given index is greater than num of input nodes.");
  }
  uint32_t eid = runtime->GetEntryId(runtime, runtime->input_nodes[index], 0);
  runtime->data_entry[eid].dl_tensor = *data_in;
}

int GraphRuntime_LoadParams(struct graph_runtime_t * runtime, const char * param_blob, const uint32_t param_size) {
  int status = 0;
  const char * bptr = param_blob;
  uint64_t header, reserved;
  header = ((uint64_t*)bptr)[0];
  bptr += sizeof(header);
  if (header != kTVMNDArrayListMagic) {
    fprintf(stderr, "Invalid parameters file format");
  }
  reserved = ((uint64_t*)bptr)[0];
  bptr += sizeof(reserved);

  // read names
  char names[GRAPH_RUNTIME_MAX_NODES][80];
  memset(names, 0, sizeof(names));
  uint64_t names_count;
  int idx;
  names_count = ((uint64_t*)bptr)[0];
  bptr += sizeof(names_count);
  for (idx = 0; idx < names_count; idx++) {
    uint64_t name_length;
    name_length = ((uint64_t*)bptr)[0];
    bptr += sizeof(name_length);
    if (name_length >= 80){
      fprintf(stderr, "Error: function name longer than expected.");
    }
    memcpy(names[idx], bptr, name_length);
    bptr += name_length;
  }

  // read sizes
  uint64_t sz;
  sz = ((uint64_t*)bptr)[0];
  bptr += sizeof(sz);
  uint32_t size = sz;
  if (size != names_count) {
    fprintf(stderr, "Invalid parameters file format");
    status = -1;
  }

  for (idx = 0; idx < size; idx++) {
    int32_t in_idx = runtime->GetInputIndex(runtime, names[idx]);
    if (!(in_idx >= 0)) {
      fprintf(stderr, "Found param for non-existent input: %s", names[idx]);
      status = -1;
    }
    uint32_t eid = runtime->GetEntryId(runtime, runtime->input_nodes[in_idx], 0);
    if (!(eid < runtime->data_entry_count)) {
      fprintf(stderr, "`entry_id`=%d is greater than expected(%d).", eid, runtime->data_entry_count);
      status = -1;
    }

    status |= NDArray_Load(&(runtime->data_entry[eid]), &bptr);
#if TVM_CRT_DEBUG
    NDArray * entry = &(runtime->data_entry[eid]);
    printf("param %s loaded, in_idx=%d, eid=%d, ndim=%d, data[0]=%f\n",
         names[idx], in_idx, eid, entry->dl_tensor.ndim,
         ((float*)entry->dl_tensor.data)[0]);
#endif // TVM_CRT_DEBUG
  }

  return status;
}

/*!
 * \brief Run all the operations one by one.
 */
void GraphRuntime_Run(GraphRuntime * runtime) {
  // setup the array and requirements.
  uint32_t idx;
  for (idx = 0; idx < runtime->op_execs_count; ++idx) {
    if (runtime->op_execs[idx].fexec) {
#if TVM_CRT_DEBUG
      printf("calling %s (%d)\n", runtime->op_execs[idx].name, idx);
#endif // TVM_CRT_DEBUG
      runtime->op_execs[idx].Call(&(runtime->op_execs[idx]));
    }
  }
}

int GraphRuntime_GetOutput(GraphRuntime * runtime, const int32_t idx, DLTensor * out) {
  int status = 0;
  uint32_t nid = runtime->outputs[idx].node_id;
  uint32_t index = runtime->outputs[idx].index;
  uint32_t eid = runtime->GetEntryId(runtime, nid, index);

  // copy data section to allocated output tensor
  int32_t elem_bytes = out->dtype.bits / 8;
  int64_t size = Shape_Accumulate(out->shape, out->ndim);
  DLTensor * tensor = &(runtime->data_entry[eid].dl_tensor);
  assert(out->ndim == tensor->ndim);
  assert(out->dtype.bits == tensor->dtype.bits);
  assert(Shape_Accumulate(out->shape, out->ndim) == Shape_Accumulate(tensor->shape, tensor->ndim));
  memcpy(out->data, tensor->data, size * elem_bytes);
  return status;
}

void GraphRuntime_SetupStorage(GraphRuntime * runtime) {
  uint32_t idx, dim;

  // Grab saved optimization plan from graph.
  DLDataType vtype[GRAPH_RUNTIME_MAX_NODES];
  GraphRuntimeGraphAttr * attrs = &(runtime->attrs);
  for (idx = 0; idx < attrs->dltype_count; idx++) {
    vtype[idx] = String2DLDataType(attrs->dltype[idx]);
  }

  // Size and device type of each storage pool entry.
  PoolEntry pool_entry[GRAPH_RUNTIME_MAX_NODES];
  memset(pool_entry, 0, sizeof(pool_entry));
  uint32_t  pool_entry_count = 0;
  // Find the maximum space size.
  for (idx = 0; idx < attrs->shape_count; idx++) {
    int storage_id = attrs->storage_id[idx];
    // Use the fallback device if no device index is available.
    int device_type = runtime->ctxs[0].device_type;
    uint32_t size = Shape_Accumulate(attrs->shape[idx], attrs->ndim[idx]);
    DLDataType t = vtype[idx];
    uint32_t bits = t.bits * t.lanes;
    size_t bytes = ((bits + 7U) / 8U) * size;

    uint32_t sid = storage_id;
    if (sid >= pool_entry_count) {
      pool_entry_count = sid + 1;
    }
    pool_entry[sid].size = MAX(pool_entry[sid].size, bytes);
    pool_entry[sid].device_type = device_type;
  }

  // Allocate the space.
  for (idx = 0; idx < pool_entry_count; idx++) {
    PoolEntry pit = pool_entry[idx];
    int64_t shape[TVM_CRT_MAX_NDIM] = {0,};
    TVMContext ctx = runtime->ctxs[0];
    DLDataType dtype = {kDLFloat, 32, 1};
    shape[0] = (pit.size + 3) / 4;
    runtime->storage_pool[runtime->storage_pool_count] = NDArray_Empty(1, shape, dtype, ctx);
    if (runtime->storage_pool[runtime->storage_pool_count].dl_tensor.data == 0) {
      fprintf(stderr, "fail to create storage_pool with idx=%d\n", idx);
    }
    runtime->storage_pool_count++;
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  runtime->data_entry_count = runtime->node_row_ptr[runtime->node_row_ptr_count - 1];
  for (idx = 0; idx < runtime->data_entry_count; ++idx) {
    size_t storage_id = attrs->storage_id[idx];
    assert(storage_id < runtime->storage_pool_count);
    runtime->data_entry[idx] =
      NDArray_CreateView(&(runtime->storage_pool[storage_id]), attrs->shape[idx], attrs->ndim[idx], vtype[idx]);
    if (runtime->data_entry[idx].dl_tensor.data == 0) {
      fprintf(stderr, "fail to create for node with idx=%d, storage_id=%d\n", idx, storage_id);
    }
  }
}

int GraphRuntime_SetupOpExecs(GraphRuntime * runtime) {
  int status = 0;
  uint32_t nid, idx;
  runtime->op_execs_count = runtime->nodes_count;
  for (nid = 0; nid < runtime->nodes_count; nid++) {
    const GraphRuntimeNode * inode = runtime->nodes + nid;
    if (strcmp(inode->op_type, "null")) {
      DLTensorPtr args[GRAPH_RUNTIME_MAX_NODES];
      uint32_t args_count = 0;
      for (idx = 0; idx < inode->inputs_count; idx++) {
        const NodeEntry * entry = inode->inputs + idx;
        uint32_t eid = runtime->GetEntryId(runtime, entry->node_id, entry->index);
        args[idx] = &(runtime->data_entry[eid].dl_tensor);
        args_count ++;
      }
      for (idx = 0; idx < inode->param.num_outputs; idx++) {
        uint32_t eid = runtime->GetEntryId(runtime, nid, idx);
        args[args_count] = &(runtime->data_entry[eid].dl_tensor);
        args_count ++;
      }
      if (strcmp(inode->op_type, "tvm_op")) {
        fprintf(stderr, "Can only take tvm_op as op\n"); status = -1;
        break;
      }
      if (args_count >= TVM_CRT_MAX_ARGS) {
        fprintf(stderr, "too many arguments: expected less than %d args, but got %d.\n",
                TVM_CRT_MAX_ARGS, args_count);
        status = -1;
        break;
      }
#if TVM_CRT_DEBUG
      printf("creating tvm_op: %s with node_id=%d\n", inode->param.func_name, nid);
#endif // TVM_CRT_DEBUG
      PackedFunc pf;
      runtime->CreateTVMOp(runtime, &(inode->param), args, args_count, inode->inputs_count, &pf);
      runtime->op_execs[nid] = pf;
    }
  }
  return status;
}

typedef struct opargs_t {
  DLTensor args[TVM_CRT_MAX_ARGS];
  uint32_t args_count;
  TVMValue arg_values[TVM_CRT_MAX_ARGS];
  uint32_t arg_values_count;
  uint32_t arg_tcodes[TVM_CRT_MAX_ARGS];
  uint32_t arg_tcodes_count;
  int64_t  shape_data[TVM_CRT_MAX_ARGS];
  uint32_t shape_data_count;
} OpArgs;

int32_t GraphRuntime_CreateTVMOp(GraphRuntime * runtime, const TVMOpParam * param,
                                 DLTensorPtr * args, const uint32_t args_count,
                                 uint32_t num_inputs, PackedFunc * pf) {
  uint32_t idx;
  OpArgs arg_ptr;
  memset(&arg_ptr, 0, sizeof(OpArgs));
  for (idx = 0; idx < args_count; idx++) {
    /* arg_ptr.args[idx] = args[idx]; */
  }
	arg_ptr.args_count = args_count;
  if (param->flatten_data) {
    arg_ptr.shape_data_count = arg_ptr.args_count;
  }
  for (idx = 0; idx < arg_ptr.args_count; ++idx) {
    TVMValue v;
    memset(&v, 0, sizeof(v));
    DLTensor * t = &(arg_ptr.args[idx]);
    /* v.v_handle = &((*args)[idx]); */
    v.v_handle = args[idx];
    arg_ptr.arg_values[idx] = v;
    arg_ptr.arg_values_count ++;
    arg_ptr.arg_tcodes[idx] = kTVMNDArrayHandle;
    arg_ptr.arg_tcodes_count ++;
    if (param->flatten_data) {
      arg_ptr.shape_data[idx] = Shape_Accumulate(t->shape, t->ndim);
      t->ndim = 1;
      t->shape[0] = arg_ptr.shape_data[idx];
    }
  }
  if (!strcmp(param->func_name, "__nop") || !strcmp(param->func_name, "__copy")) {
    fprintf(stderr, "%s function is not yet supported.", param->func_name);
  }

  runtime->module.GetFunction(param->func_name, pf);
  TVMArgs targs = TVMArgs_Create(arg_ptr.arg_values, arg_ptr.arg_tcodes, arg_ptr.arg_values_count);
  pf->SetArgs(pf, &targs);

  return 0;
}

/*!
 * \brief Initialize the graph executor with graph and context.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions for the host
 * processor.
 * \param ctxs The context of the host and devices where graph nodes will be
 * executed on.
 */
void GraphRuntime_Init(GraphRuntime * runtime, const char * graph_json,
                       const Module * module, const TVMContext * ctxs) {
  JSONReader reader = JSONReader_Create(graph_json);
  runtime->Load(runtime, &reader);
  runtime->ctxs[0] = ctxs[0];
  runtime->SetupStorage(runtime);
  PackedFunc_SetupExecs();
  runtime->SetupOpExecs(runtime);
  JSONReader_Release(&reader);
}

GraphRuntime * TVMGraphRuntimeCreate(const char * sym_json,
                              const Module * m, const TVMContext * ctxs) {
  GraphRuntime * runtime = (GraphRuntime*)malloc(sizeof(GraphRuntime));
  memset(runtime, 0, sizeof(GraphRuntime));
  runtime->GetEntryId = GraphRuntime_GetEntryId;
  runtime->GetInputIndex = GraphRuntime_GetInputIndex;
  runtime->Init = GraphRuntime_Init;
  runtime->Load = GraphRuntime_Load;
  runtime->SetInput = GraphRuntime_SetInput;
  runtime->LoadParams = GraphRuntime_LoadParams;
  runtime->Run = GraphRuntime_Run;
  runtime->GetOutput = GraphRuntime_GetOutput;
  runtime->SetupStorage = GraphRuntime_SetupStorage;
  runtime->SetupOpExecs = GraphRuntime_SetupOpExecs;
  runtime->CreateTVMOp = GraphRuntime_CreateTVMOp;
  runtime->module.GetFunction = Module_GetFunction;
  // init
  runtime->Init(runtime, sym_json, m, ctxs);
  return runtime;
}

void TVMGraphRuntimeRelease(GraphRuntime ** pptr) {
  int32_t idx;
  GraphRuntime * runtime = *pptr;
  for (idx = 0; idx < runtime->storage_pool_count; ++idx) {
    free(runtime->storage_pool[idx].dl_tensor.data);
    free(runtime->storage_pool[idx].dl_tensor.shape);
  }
  free(*pptr);
}
