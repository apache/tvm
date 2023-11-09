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

#include <inttypes.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/aot_executor.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/packed_func.h>
#include <tvm/runtime/crt/page_allocator.h>

static void DumpMetadata(const TVMMetadata* md) {
  LOG_DEBUG("%s:\n", __FUNCTION__);
  LOG_DEBUG("\tmod_name=%s\n", md->mod_name);
  LOG_DEBUG("\tversion=%" PRId64 "\n", md->version);
  LOG_DEBUG("\tnum_inputs=%" PRId64 "\n", md->num_inputs);
  LOG_DEBUG("\tnum_outputs=%" PRId64 "\n", md->num_outputs);
  LOG_DEBUG("\tnum_workspace_pools=%" PRId64 "\n", md->num_workspace_pools);
  LOG_DEBUG("\tnum_constant_pools=%" PRId64 "\n", md->num_constant_pools);

  int i;

  for (i = 0; i < md->num_inputs; ++i) {
    LOG_DEBUG("\tinput[%d]: %s\n", i, md->inputs[i].name);
  }

  for (i = 0; i < md->num_outputs; ++i) {
    LOG_DEBUG("\toutput[%d]: %s\n", i, md->outputs[i].name);
  }

  for (i = 0; i < md->num_workspace_pools; ++i) {
    LOG_DEBUG("\tworkspace_pools[%d]: %s\n", i, md->workspace_pools[i].name);
  }

  for (i = 0; i < md->num_constant_pools; ++i) {
    LOG_DEBUG("\tconstant_pools[%d]: %s\n", i, md->constant_pools[i].name_hint);
  }
}

int TVMAotExecutor_GetNumInputs(TVMAotExecutor* executor) { return executor->metadata->num_inputs; }

int TVMAotExecutor_GetNumOutputs(TVMAotExecutor* executor) {
  return executor->metadata->num_outputs;
}

int TVMAotExecutor_GetInputIndex(TVMAotExecutor* executor, const char* name) {
  int i;
  int rv = -1;

  const TVMMetadata* md = executor->metadata;
  for (i = 0; i < md->num_inputs; ++i) {
    if (!strcmp(md->inputs[i].name, name)) {
      rv = i;
      break;
    }
  }
  CHECK_GE(rv, 0, "cannot find '%s' among input.", name);
  return rv;
}

int TVMAotExecutor_GetInputName(TVMAotExecutor* executor, int index, const char** name) {
  const TVMMetadata* md = executor->metadata;
  *name = md->inputs[index].name;
  return 0;
}

int TVMAotExecutor_Run(TVMAotExecutor* executor) {
  const char* tvm_main_suffix = "_run";
  char tvm_main_name[TVM_CRT_MAX_STRLEN_FUNCTION_NAME];

  {
    const size_t max_strlen = TVM_CRT_MAX_STRLEN_FUNCTION_NAME;
    size_t len = strnlen(executor->metadata->mod_name, max_strlen);
    len += strnlen(tvm_main_suffix, max_strlen);

    CHECK_LT(len, max_strlen, "tvm_main name too long %zu\n", len);
  }

  // create main function name string, e.g. "tvmgen_default___tvm_main__"
  snprintf(tvm_main_name, sizeof(tvm_main_name), "%s%s", executor->metadata->mod_name,
           tvm_main_suffix);

  TVMPackedFunc tvm_main;
  TVMArgs temp_args;

  CHECK_LE(executor->num_args, TVM_CRT_MAX_ARGS, "too many args %" PRId64 "\n", executor->num_args);

  int i;
  for (i = 0; i < executor->num_args; ++i) {
    temp_args.values[i].v_handle = &executor->args[i].dl_tensor;
    temp_args.tcodes[i] = kTVMDLTensorHandle;
  }
  temp_args.values_count = executor->num_args;

  int status =
      TVMPackedFunc_InitModuleFunc(&tvm_main, executor->module_handle, tvm_main_name, &temp_args);

  if (status != 0) {
    return status;
  }

  CHECK_EQ(tvm_main.Call(&tvm_main), 0, "call to %s failed", tvm_main_name);

  return 0;
}

int TVMAotExecutor_Init(TVMAotExecutor* executor, TVMModuleHandle module_handle,
                        const DLDevice device, const char* module_name) {
  executor->module_handle = module_handle;
  executor->device = device;

  // get a pointer to the PackedFunc get_c_metadata() which gives us access to the top-level
  // metadata structure
  TVMPackedFunc get_c_metadata;
  TVMArgs temp_args;
  temp_args.values_count = 0;

  const char* tvmgen_prefix = "tvmgen_";
  const char* get_c_metdata_suffix = "_get_c_metadata";
  char get_c_metdata_name[TVM_CRT_MAX_STRLEN_FUNCTION_NAME];

  {
    size_t max_strlen = TVM_CRT_MAX_STRLEN_FUNCTION_NAME;
    size_t len = strnlen(tvmgen_prefix, max_strlen);
    len += strnlen(module_name, max_strlen);
    len += strnlen(get_c_metdata_suffix, max_strlen);

    CHECK_LT(len, max_strlen, "get_c_metadata name too long %zu\n", len);
  }

  // create get_c_metadata() function name string, e.g. "tvmgen_default_get_c_metadata()"
  snprintf(get_c_metdata_name, sizeof(get_c_metdata_name), "%s%s%s", tvmgen_prefix, module_name,
           get_c_metdata_suffix);

  int status = TVMPackedFunc_InitModuleFunc(&get_c_metadata, executor->module_handle,
                                            get_c_metdata_name, &temp_args);
  if (status != 0) {
    return status;
  }

  CHECK_EQ(get_c_metadata.Call(&get_c_metadata), 0, "get_c_metadata");

  // save the returned pointer to the top-level metadata
  executor->metadata = (TVMMetadata*)get_c_metadata.ret_value.values[0].v_handle;

  const TVMMetadata* md = executor->metadata;

  DumpMetadata(md);

  executor->num_args = md->num_inputs + md->num_outputs + md->num_workspace_pools;

  tvm_crt_error_t err = TVMPlatformMemoryAllocate(executor->num_args * sizeof(*executor->args),
                                                  executor->device, (void**)(&executor->args));
  if (err != kTvmErrorNoError) {
    return -1;
  }

  int i;
  int arg_idx = 0;
  for (i = 0; i < md->num_inputs; ++i) {
    LOG_DEBUG("input allocate[%d]: %s\n", i, md->inputs[i].name);

    TVMNDArray* array = &executor->args[arg_idx++];

    status = TVMNDArray_Empty(md->inputs[i].num_shape, md->inputs[i].shape, md->inputs[i].dtype,
                              executor->device, array);
    if (status != 0) {
      return status;
    }

    TVMNDArray_IncrementReference(array);
  }

  for (i = 0; i < md->num_outputs; ++i) {
    LOG_DEBUG("output allocate[%d]: %s\n", i, md->outputs[i].name);

    TVMNDArray* array = &executor->args[arg_idx++];

    status = TVMNDArray_Empty(md->outputs[i].num_shape, md->outputs[i].shape, md->outputs[i].dtype,
                              executor->device, array);
    if (status != 0) {
      return status;
    }

    TVMNDArray_IncrementReference(array);
  }

  return status;
}

int TVMAotExecutor_Create(TVMModuleHandle module_handle, const DLDevice device,
                          TVMAotExecutor** executor, const char* module_name) {
  tvm_crt_error_t err = TVMPlatformMemoryAllocate(sizeof(**executor), device, (void**)executor);
  if (err != kTvmErrorNoError) {
    return -1;
  }

  memset(*executor, 0, sizeof(**executor));

  return TVMAotExecutor_Init(*executor, module_handle, device, module_name);
}

int TVMAotExecutor_Release(TVMAotExecutor* executor, const DLDevice device) {
  int status;

  if (executor->num_args > 0) {
    // free TVMNDArray data memory for each argument
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
