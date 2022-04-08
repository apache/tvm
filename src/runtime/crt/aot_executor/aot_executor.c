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
 * \file aot_executor.c
 * \brief implement AoT executor in C
 */

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/aot_executor.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/packed_func.h>
#include <tvm/runtime/crt/page_allocator.h>

#include <string.h>

static void DumpMetadata(TVMMetadata* md)
{
  fprintf(stderr, "%s:\n", __FUNCTION__);
  fprintf(stderr, "\tmod_name=%s\n", md->mod_name);
  fprintf(stderr, "\tversion=%ld\n", md->version);
  fprintf(stderr, "\tnum_inputs=%ld\n", md->num_inputs);
  fprintf(stderr, "\tnum_outputs=%ld\n", md->num_outputs);
  fprintf(stderr, "\tnum_pools=%ld\n", md->num_pools);

  int i;

  for (i = 0; i < md->num_inputs; ++i) {
    fprintf(stderr, "\tinput[%d]: %s\n", i, md->inputs[i].name);
  }

  for (i = 0; i < md->num_outputs; ++i) {
    fprintf(stderr, "\toutput[%d]: %s\n", i, md->outputs[i].name);
  }

  for (i = 0; i < md->num_pools; ++i) {
    fprintf(stderr, "\tpools[%d]: %s\n", i, md->pools[i].name);
  }
}

int TVMAotExecutor_GetInputIndex(TVMAotExecutor* executor, const char* name) {
  uint32_t i;
  int32_t rv = -1;

  TVMMetadata* md = executor->metadata;
  for (i = 0; i < md->num_inputs; ++i) {
    if (!strcmp(md->inputs[i].name, name)) {
      rv = i;
      break;
    }
  }
  CHECK_GE(rv, 0, "cannot find '%s' among input.", name);
  return rv;
}

int TVMAotExecutor_Init(TVMAotExecutor* executor, TVMModuleHandle module_handle,
                        const DLDevice* device) {

  executor->module_handle = module_handle;
  executor->device = *device;

  // get a pointer to the PackedFunc get_c_metadata() which gives us access to the top-level
  // metadata structure
  TVMPackedFunc get_c_metadata;
  TVMArgs temp_args;
  temp_args.values_count = 0;

  int status = TVMPackedFunc_InitModuleFunc(&get_c_metadata, executor->module_handle, 
                                            "get_c_metadata", &temp_args);
  if (status != 0) {
    return status;
  }

  get_c_metadata.Call(&get_c_metadata);
  // save the returned pointer to the top-level metadata
  executor->metadata = (TVMMetadata *)get_c_metadata.ret_value.values[0].v_handle;

  TVMMetadata* md = executor->metadata;
  
  DumpMetadata(md);
  
  executor->num_args = md->num_inputs + md->num_outputs + md->num_pools;

  TVMPlatformMemoryAllocate(executor->num_args * sizeof(*executor->args),
                            executor->device, (void **)(&executor->args));

  int i;
  int arg_idx = 0;
  for (i = 0; i < md->num_inputs; ++i) {
    fprintf(stderr, "\tinput allocate[%d]: %s\n", i, md->inputs[i].name);

#define FAKE_SHAPE

#ifdef FAKE_SHAPE
    int64_t shape = 2;
    TVMNDArray_Empty(1, &shape, md->inputs[i].dtype,
                     executor->device, &executor->args[arg_idx++]);
#else
    TVMNDArray_Empty(md->inputs[i].num_shape, md->inputs[i].shape, md->inputs[i].dtype,
                     executor->device, &executor->args[arg_idx++]);
#endif

  }

  for (i = 0; i < md->num_outputs; ++i) {
    fprintf(stderr, "\toutput allocate[%d]: %s\n", i, md->outputs[i].name);

    TVMNDArray_Empty(md->outputs[i].num_shape, md->outputs[i].shape, md->outputs[i].dtype,
                     executor->device, &executor->args[arg_idx++]);
  }

  for (i = 0; i < md->num_pools; ++i) {
    fprintf(stderr, "\tpools allocate[%d]: %s\n", i, md->pools[i].name);

    TVMNDArray_Empty(md->pools[i].num_shape, md->pools[i].shape, md->pools[i].dtype,
                     executor->device, &executor->args[arg_idx++]);
  }

  return status;
}

int TVMAotExecutor_Create(TVMModuleHandle module_handle,
                          const DLDevice* device, TVMAotExecutor** executor) {

  tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(**executor), *device, (void**)executor);
  if (err != kTvmErrorNoError) {
    return -1;
  }

  memset(*executor, 0, sizeof(**executor));

  return TVMAotExecutor_Init(*executor, module_handle, device);
}

int TVMAotExecutor_Release(TVMAotExecutor* executor, const DLDevice device) {

  int status;

  if (executor->num_args > 0) {
    // free TVMNDArray data memory for each each argument
    int i;
    for (i = 0; i < executor->num_args; ++i) {
      status = TVMNDArray_Release(&executor->args[i]);
      if (status != 0) {
        return status;
      }
    }

    // free TVMNDArray argument list
    status = TVMPlatformMemoryFree(executor->args, executor->device);
    if (status != 0) {
        return status;
    }
  }

  status = TVMPlatformMemoryFree(executor, device);
  if (status != 0) {
    return status;
  }

  return 0;
}