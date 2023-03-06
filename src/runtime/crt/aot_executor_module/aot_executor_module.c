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
 * \file aot_executor_module.c
 * \brief wrap aot_executor into a TVMModule for use with RPC.
 */

#include <stdio.h>
#include <tvm/runtime/crt/aot_executor.h>
#include <tvm/runtime/crt/aot_executor_module.h>
#include <tvm/runtime/crt/func_registry.h>
#include <tvm/runtime/crt/module.h>

typedef struct {
  TVMModule mod;
  TVMAotExecutor* executor;
} AotExecutorModule;

static AotExecutorModule aot_executor;

int32_t TVMAotExecutorModule_Create(TVMValue* args, int* tcodes, int nargs, TVMValue* ret_values,
                                    int* ret_tcodes, void* resource_handle) {
  if (aot_executor.executor != NULL) {
    return kTvmErrorExecutorModuleAlreadyCreated;
  }

  if (nargs != 3) {
    return kTvmErrorFunctionCallNumArguments;
  }

  if (tcodes[0] != kTVMModuleHandle || tcodes[1] != kDLDevice || tcodes[2] != kTVMStr) {
    return kTvmErrorFunctionCallWrongArgType;
  }

  DLDevice dev = args[1].v_device;

  if (dev.device_type != kDLCPU) {
    return kTvmErrorExecutorModuleBadContext;
  }

  TVMAotExecutor_Create(args[0].v_handle, dev, &aot_executor.executor, args[2].v_str);

  TVMModuleHandle out_mod;
  int status = TVMModCreateFromCModule(&aot_executor.mod, &out_mod);
  if (status != 0) {
    ret_tcodes[0] = kTVMNullptr;
    TVMAotExecutor_Release(aot_executor.executor, dev);
    return status;
  }

  ret_values[0].v_handle = out_mod;
  ret_tcodes[0] = kTVMModuleHandle;
  return kTvmErrorNoError;
}

int32_t TVMAotExecutorModule_NotImplemented(TVMValue* args, int* tcodes, int nargs,
                                            TVMValue* ret_values, int* ret_tcodes,
                                            void* resource_handle) {
  return kTvmErrorFunctionCallNotImplemented;
}

int32_t TVMAotExecutorModule_GetInput(TVMValue* args, int* tcodes, int nargs, TVMValue* ret_values,
                                      int* ret_tcodes, void* resource_handle) {
  int64_t index;

  if (tcodes[0] == kTVMArgInt) {
    if (args[0].v_int64 > TVMAotExecutor_GetNumInputs(aot_executor.executor)) {
      return kTvmErrorFunctionCallInvalidArg;
    }

    index = args[0].v_int64;
  } else {
    index = TVMAotExecutor_GetInputIndex(aot_executor.executor, args[0].v_str);

    if (index < 0) {
      return kTvmErrorExecutorModuleNoSuchInput;
    }
  }

  TVMNDArray* array = &aot_executor.executor->args[index];

  TVMNDArray_IncrementReference(array);

  ret_values[0].v_handle = (void*)(&array->dl_tensor);
  ret_tcodes[0] = kTVMNDArrayHandle;

  return 0;
}

int32_t TVMAotExecutorModule_GetOutput(TVMValue* args, int* tcodes, int nargs, TVMValue* ret_values,
                                       int* ret_tcodes, void* resource_handle) {
  if (nargs != 1) {
    return kTvmErrorFunctionCallNumArguments;
  }

  if (args[0].v_int64 > TVMAotExecutor_GetNumOutputs(aot_executor.executor)) {
    return kTvmErrorFunctionCallInvalidArg;
  }

  // index past the input entries
  int64_t index = args[0].v_int64 + TVMAotExecutor_GetNumInputs(aot_executor.executor);

  TVMNDArray* array = &aot_executor.executor->args[index];

  TVMNDArray_IncrementReference(array);

  ret_values[0].v_handle = (void*)(&array->dl_tensor);
  ret_tcodes[0] = kTVMNDArrayHandle;

  return 0;
}

int32_t TVMAotExecutorModule_GetInputIndex(TVMValue* args, int* tcodes, int nargs,
                                           TVMValue* ret_values, int* ret_tcodes,
                                           void* resource_handle) {
  if (nargs != 1) {
    return kTvmErrorFunctionCallNumArguments;
  }

  int index = TVMAotExecutor_GetInputIndex(aot_executor.executor, args[0].v_str);

  if (index < 0) {
    return kTvmErrorExecutorModuleNoSuchInput;
  }

  ret_values[0].v_int64 = index;
  ret_tcodes[0] = kTVMArgInt;
  return 0;
}

int32_t TVMAotExecutorModule_GetInputName(TVMValue* args, int* tcodes, int nargs,
                                          TVMValue* ret_values, int* ret_tcodes,
                                          void* resource_handle) {
  if (nargs != 1) {
    return kTvmErrorFunctionCallNumArguments;
  }

  char* name;
  int ret = TVMAotExecutor_GetInputName(aot_executor.executor, args[0].v_int64, &name);
  if (ret < 0) {
    return kTvmErrorExecutorModuleNoSuchInput;
  }

  ret_values[0].v_str = name;
  ret_tcodes[0] = kTVMStr;
  return 0;
}

int32_t TVMAotExecutorModule_GetNumInputs(TVMValue* args, int* tcodes, int nargs,
                                          TVMValue* ret_values, int* ret_tcodes,
                                          void* resource_handle) {
  if (nargs != 0) {
    return kTvmErrorFunctionCallNumArguments;
  }

  ret_values[0].v_int64 = TVMAotExecutor_GetNumInputs(aot_executor.executor);
  ret_tcodes[0] = kTVMArgInt;
  return 0;
}

int32_t TVMAotExecutorModule_GetNumOutputs(TVMValue* args, int* tcodes, int nargs,
                                           TVMValue* ret_values, int* ret_tcodes,
                                           void* resource_handle) {
  if (nargs != 0) {
    return kTvmErrorFunctionCallNumArguments;
  }

  ret_values[0].v_int64 = TVMAotExecutor_GetNumOutputs(aot_executor.executor);
  ret_tcodes[0] = kTVMArgInt;
  return 0;
}

int32_t TVMAotExecutorModule_Run(TVMValue* args, int* tcodes, int nargs, TVMValue* ret_values,
                                 int* ret_tcodes, void* resource_handle) {
  if (nargs != 0) {
    return kTvmErrorFunctionCallNumArguments;
  }

  return TVMAotExecutor_Run(aot_executor.executor);
}

static const TVMBackendPackedCFunc aot_executor_registry_funcs[] = {
    &TVMAotExecutorModule_GetInput,        // get_input
    &TVMAotExecutorModule_GetInputIndex,   // get_input_index
    &TVMAotExecutorModule_NotImplemented,  // get_input_info (do not implement)
    &TVMAotExecutorModule_GetNumInputs,    // get_num_inputs
    &TVMAotExecutorModule_GetNumOutputs,   // get_num_outputs
    &TVMAotExecutorModule_GetOutput,       // get_output
    &TVMAotExecutorModule_NotImplemented,  // load_params (do not implement)
    &TVMAotExecutorModule_Run,             // run
    &TVMAotExecutorModule_NotImplemented,  // set_input (implemented via python wrapper)
    &TVMAotExecutorModule_NotImplemented,  // share_params (do not implement)
    &TVMAotExecutorModule_GetInputName,    // get_input_name
};

static const TVMFuncRegistry aot_executor_registry = {
    "\x0b\0get_input\0"
    "get_input_index\0"
    "get_input_info\0"
    "get_num_inputs\0"
    "get_num_outputs\0"
    "get_output\0"
    "load_params\0"
    "run\0"
    "set_input\0"
    "share_params\0"
    "get_input_name\0",
    aot_executor_registry_funcs};

tvm_crt_error_t TVMAotExecutorModule_Register() {
  aot_executor.mod.registry = &aot_executor_registry;
  aot_executor.executor = NULL;

  return TVMFuncRegisterGlobal("tvm.aot_executor.create", &TVMAotExecutorModule_Create, 0);
}
