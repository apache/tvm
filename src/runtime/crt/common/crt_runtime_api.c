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

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/func_registry.h>
#include <tvm/runtime/crt/internal/common/ndarray.h>
#include <tvm/runtime/crt/internal/graph_runtime/graph_runtime.h>
#include <tvm/runtime/crt/internal/memory/memory.h>
#include <tvm/runtime/crt/memory.h>
#include <tvm/runtime/crt/platform.h>

// Handle internal errors

static char g_last_error[1024];

void TVMAPISetLastError(const char* msg) {
  strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
  g_last_error[sizeof(g_last_error) - 1] = 0;
}

__attribute__((format(printf, 1, 2))) int TVMAPIErrorf(const char* msg, ...) {
  va_list args;
  int to_return;

  va_start(args, msg);
  to_return = vsnprintf(g_last_error, sizeof(g_last_error), msg, args);
  va_end(args);

  return to_return;
}

const char* TVMGetLastError(void) { return g_last_error; }

// Manipulate NDArray on target device

int TVMArrayAlloc(const tvm_index_t* shape, int ndim, int dtype_code, int dtype_bits,
                  int dtype_lanes, int device_type, int device_id, TVMArrayHandle* out) {
  DLDataType dtype;
  dtype.code = dtype_code;
  dtype.bits = dtype_bits;
  dtype.lanes = dtype_lanes;
  DLContext ctx;
  ctx.device_type = (DLDeviceType)device_type;
  ctx.device_id = device_id;
  TVMNDArray arr;
  int status = TVMNDArray_Empty(ndim, shape, dtype, ctx, &arr);
  if (status != 0) {
    return status;
  }
  **out = arr.dl_tensor;
  return 0;
}

int TVMArrayFree(TVMArrayHandle handle) {
  TVMNDArray arr;
  arr.dl_tensor = *handle;
  return TVMNDArray_Release(&arr);
}

int TVMDeviceAllocDataSpace(DLContext ctx, size_t nbytes, size_t alignment, DLDataType type_hint,
                            void** out_data) {
  if (alignment != 1) {
    nbytes = (nbytes + alignment - 1) / alignment * alignment;
  }
  return TVMPlatformMemoryAllocate(nbytes, ctx, out_data);
}

int TVMDeviceAllocDataSpaceWithScope(DLContext ctx, int ndim, const int64_t* shape,
                                     DLDataType dtype, const char* mem_scope, void** out_data) {
  size_t nbytes = 1;
  for (int i = 0; i < ndim; ++i) {
    nbytes *= shape[i];
  }
  nbytes *= (dtype.bits * dtype.lanes + 7) / 8;

  int kAllocAlignment = 128;
  size_t align = (dtype.bits / 8) * dtype.lanes;
  if (align < kAllocAlignment) align = kAllocAlignment;
  return TVMDeviceAllocDataSpace(ctx, nbytes, align, dtype, out_data);
}

int TVMDeviceFreeDataSpace(TVMContext ctx, void* ptr) { return TVMPlatformMemoryFree(ptr, ctx); }

static bool IsContiguous(const DLTensor* arr) {
  if (arr->strides == NULL) return true;
  int64_t expected_stride = 1;
  for (int32_t i = arr->ndim; i != 0; --i) {
    int32_t k = i - 1;
    if (arr->strides[k] != expected_stride) return false;
    expected_stride *= arr->shape[k];
  }
  return true;
}

int TVMDeviceCopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  assert(IsContiguous(from) && IsContiguous(to));
  size_t size = 1;
  for (int i = 0; i < from->ndim; ++i) {
    size *= from->shape[i];
  }
  size *= (from->dtype.bits * from->dtype.lanes + 7) / 8;
  memcpy(((uint8_t*)to->data) + to->byte_offset, ((uint8_t*)from->data) + from->byte_offset, size);
  return 0;
}

int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream) { return 0; }

static TVMMutableFuncRegistry global_func_registry;

int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {
  return TVMMutableFuncRegistry_Set(&global_func_registry, name, f, override != 0);
}

static const TVMModule* registered_modules[TVM_CRT_MAX_REGISTERED_MODULES];

/*! \brief Passed as `module_index` to EncodeFunctionHandle. */
static const tvm_module_index_t kGlobalFuncModuleIndex = TVM_CRT_MAX_REGISTERED_MODULES;

/*! \brief Special module handle for retur values from RPCTimeEvaluator. */
static const tvm_module_index_t kTimeEvaluatorModuleIndex = 0x7fff;

static int DecodeModuleHandle(TVMModuleHandle handle, tvm_module_index_t* out_module_index) {
  tvm_module_index_t module_index;

  module_index = ((tvm_module_index_t)((uintptr_t)handle)) & ~0x8000;
  if (module_index > TVM_CRT_MAX_REGISTERED_MODULES || registered_modules[module_index] == NULL) {
    TVMAPIErrorf("invalid module handle: %08x", module_index);
    return -1;
  }

  *out_module_index = module_index;
  return 0;
}

static TVMModuleHandle EncodeModuleHandle(tvm_module_index_t module_index) {
  return (TVMModuleHandle)((uintptr_t)(module_index | 0x8000));
}

int TVMModCreateFromCModule(const TVMModule* mod, TVMModuleHandle* out_handle) {
  tvm_module_index_t idx;

  for (idx = 0; idx < TVM_CRT_MAX_REGISTERED_MODULES; idx++) {
    if (registered_modules[idx] == NULL) {
      registered_modules[idx] = mod;
      *out_handle = EncodeModuleHandle(idx);
      return 0;
    }
  }

  return -1;
}

int TVMModFree(TVMModuleHandle mod) {
  tvm_module_index_t module_index;
  if (DecodeModuleHandle(mod, &module_index) != 0) {
    return -1;
  }

  registered_modules[module_index] = NULL;
  return 0;
}

static const TVMModuleHandle kTVMModuleHandleUninitialized = (TVMModuleHandle)(~0UL);

static TVMModuleHandle system_lib_handle;

int SystemLibraryCreate(TVMValue* args, int* type_codes, int num_args, TVMValue* ret_val,
                        int* ret_type_codes) {
  const TVMModule* system_lib;

  if (system_lib_handle == kTVMModuleHandleUninitialized) {
    system_lib = TVMSystemLibEntryPoint();
    if (TVMModCreateFromCModule(system_lib, &system_lib_handle) != 0) {
      TVMAPIErrorf("error registering system lib");
      return -1;
    }
  }

  ret_val[0].v_handle = system_lib_handle;
  ret_type_codes[0] = kTVMModuleHandle;
  return 0;
}

static TVMFunctionHandle EncodeFunctionHandle(tvm_module_index_t module_index,
                                              tvm_function_index_t function_index) {
  return (TVMFunctionHandle)((uintptr_t)(
      ((module_index | 0x8000) << (sizeof(tvm_function_index_t) * 8)) | (function_index | 0x8000)));
}

static int DecodeFunctionHandle(TVMFunctionHandle handle, tvm_module_index_t* module_index,
                                tvm_function_index_t* function_index) {
  tvm_module_index_t unvalidated_module_index;
  unvalidated_module_index =
      (tvm_module_index_t)(((uintptr_t)handle) >> (sizeof(tvm_function_index_t) * 8));
  unvalidated_module_index &= ~0x8000;

  if (unvalidated_module_index != kTimeEvaluatorModuleIndex) {
    if (unvalidated_module_index > kGlobalFuncModuleIndex) {
      TVMAPIErrorf("invalid module handle: index=%08x", unvalidated_module_index);
      return -1;
    } else if (unvalidated_module_index < kGlobalFuncModuleIndex &&
               registered_modules[unvalidated_module_index] == NULL) {
      TVMAPIErrorf("unregistered module: index=%08x", unvalidated_module_index);
      return -1;
    }
  }

  *function_index = ((uint32_t)((uintptr_t)handle)) & ~0x8000;
  *module_index = unvalidated_module_index;
  return 0;
}

int TVMByteArrayFree(TVMByteArray* arr) {
  DLContext ctx = {kDLCPU, 0};
  int to_return = TVMPlatformMemoryFree((void*)arr->data, ctx);
  if (to_return != 0) {
    return to_return;
  }

  return TVMPlatformMemoryFree((void*)arr, ctx);
}

tvm_crt_error_t RunTimeEvaluator(tvm_function_index_t function_index, TVMValue* args,
                                 int* type_codes, int num_args, TVMValue* ret_val,
                                 int* ret_type_code);

int TVMFuncCall(TVMFunctionHandle func_handle, TVMValue* arg_values, int* type_codes, int num_args,
                TVMValue* ret_val, int* ret_type_code) {
  tvm_module_index_t module_index;
  tvm_function_index_t function_index;
  void* resource_handle;
  const TVMFuncRegistry* registry;
  TVMBackendPackedCFunc func;

  if (DecodeFunctionHandle(func_handle, &module_index, &function_index) != 0) {
    return -1;
  }

  if (module_index == kTimeEvaluatorModuleIndex) {
    return RunTimeEvaluator(function_index, arg_values, type_codes, num_args, ret_val,
                            ret_type_code);
  } else if (module_index == kGlobalFuncModuleIndex) {
    resource_handle = NULL;
    registry = &global_func_registry.registry;
  } else {
    resource_handle = (void*)registered_modules[module_index]->registry;
    registry = registered_modules[module_index]->registry;
  }

  if (TVMFuncRegistry_GetByIndex(registry, function_index, &func) != 0) {
    TVMAPIErrorf("invalid function index: %04" PRIx16, function_index);
    return -1;
  }

  ret_type_code[0] = kTVMNullptr;
  ret_val[0].v_handle = NULL;
  return func(arg_values, type_codes, num_args, ret_val, ret_type_code, resource_handle);
}

static tvm_crt_error_t FindFunctionOrSetAPIError(tvm_module_index_t module_index,
                                                 const TVMFuncRegistry* registry, const char* name,
                                                 TVMFunctionHandle* out) {
  tvm_function_index_t function_index;
  tvm_crt_error_t err = TVMFuncRegistry_Lookup(registry, name, &function_index);
  if (err != kTvmErrorNoError) {
    return err;
  }

  *out = EncodeFunctionHandle(module_index, function_index);
  return kTvmErrorNoError;
}

int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {
  return FindFunctionOrSetAPIError(kGlobalFuncModuleIndex, &global_func_registry.registry, name,
                                   out);
}

int TVMModGetFunction(TVMModuleHandle mod, const char* func_name, int query_imports,
                      TVMFunctionHandle* out) {
  tvm_module_index_t module_index;
  if (DecodeModuleHandle(mod, &module_index) != 0) {
    return -1;
  }

  return FindFunctionOrSetAPIError(module_index, registered_modules[module_index]->registry,
                                   func_name, out);
}

int ModuleGetFunction(TVMValue* args, int* type_codes, int num_args, TVMValue* ret_value,
                      int* ret_type_codes) {
  TVMModuleHandle mod;
  const char* name;
  int to_return;
  int query_imports;

  ret_value[0].v_handle = NULL;
  ret_type_codes[0] = kTVMNullptr;
  if (num_args != 3 || type_codes[0] != kTVMModuleHandle || type_codes[1] != kTVMStr ||
      type_codes[2] != kDLInt) {
    return 0;
  }

  mod = (TVMModuleHandle)args[0].v_handle;
  name = args[1].v_str;
  query_imports = args[2].v_int64 != 0;
  to_return = TVMModGetFunction(mod, name, query_imports, &ret_value->v_handle);

  if (to_return == 0) {
    ret_type_codes[0] = kTVMPackedFuncHandle;
  } else {
    ret_value->v_handle = NULL;
  }

  // NOTE: For compatibility with C++ runtime API, return no error (but NULL function) when the
  // function lookup failed.
  if (to_return == kTvmErrorFunctionNameNotFound) {
    to_return = kTvmErrorNoError;
  }

  return to_return;
}

typedef struct TVMCReturnValue {
  TVMValue* ret_val;
  int* ret_type_code;
} TVMCReturnValue;

int TVMCFuncSetReturn(TVMRetValueHandle ret, TVMValue* value, int* type_code, int num_ret) {
  TVMCReturnValue* ret_val;
  int idx;

  ret_val = (TVMCReturnValue*)ret;
  for (idx = 0; idx < num_ret; idx++) {
    ret_val->ret_val[idx] = value[idx];
    ret_val->ret_type_code[idx] = type_code[idx];
  }

  return 0;
}

int TVMFuncFree(TVMFunctionHandle func) {
  // A no-op, since we don't actually allocate anything in GetFunction
  return 0;
}

int RPCTimeEvaluator(TVMValue* args, int* type_codes, int num_args, TVMValue* ret_val,
                     int* ret_type_code);
tvm_crt_error_t TVMInitializeRuntime() {
  int idx = 0;
  tvm_crt_error_t error = kTvmErrorNoError;
  void* func_registry_memory = NULL;

  DLContext ctx = {kDLCPU, 0};
  error = TVMPlatformMemoryAllocate(TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES, ctx,
                                    &func_registry_memory);
  if (error != kTvmErrorNoError) {
    return error;
  }

  void* registry_backing_memory;
  error = TVMPlatformMemoryAllocate(TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES, ctx,
                                    &registry_backing_memory);
  if (error != kTvmErrorNoError) {
    TVMPlatformMemoryFree(func_registry_memory, ctx);
    return error;
  }

  system_lib_handle = kTVMModuleHandleUninitialized;

  error = TVMMutableFuncRegistry_Create(&global_func_registry, registry_backing_memory,
                                        TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES);
  for (idx = 0; idx < TVM_CRT_MAX_REGISTERED_MODULES; idx++) {
    registered_modules[idx] = NULL;
  }

  if (error == kTvmErrorNoError) {
    error = TVMFuncRegisterGlobal("runtime.SystemLib", &SystemLibraryCreate, 0);
  }

  if (error == kTvmErrorNoError) {
    error = TVMFuncRegisterGlobal("tvm.rpc.server.ModuleGetFunction", &ModuleGetFunction, 0);
  }

  if (error == kTvmErrorNoError) {
    error = TVMFuncRegisterGlobal("runtime.RPCTimeEvaluator", &RPCTimeEvaluator, 0);
  }

  if (error != kTvmErrorNoError) {
    TVMPlatformMemoryFree(registry_backing_memory, ctx);
    TVMPlatformMemoryFree(func_registry_memory, ctx);
  }

  return error;
}

typedef struct {
  uint16_t function_index;
  TVMFunctionHandle func_to_time;
  TVMContext ctx;
  int number;
  int repeat;
  int min_repeat_ms;
} time_evaluator_state_t;

static time_evaluator_state_t g_time_evaluator_state;

int RPCTimeEvaluator(TVMValue* args, int* type_codes, int num_args, TVMValue* ret_val,
                     int* ret_type_code) {
  ret_val[0].v_handle = NULL;
  ret_type_code[0] = kTVMNullptr;
  if (num_args < 8) {
    TVMAPIErrorf("not enough args");
    return kTvmErrorFunctionCallNumArguments;
  }
  if (type_codes[0] != kTVMModuleHandle || type_codes[1] != kTVMStr ||
      type_codes[2] != kTVMArgInt || type_codes[3] != kTVMArgInt || type_codes[4] != kTVMArgInt ||
      type_codes[5] != kTVMArgInt || type_codes[6] != kTVMArgInt || type_codes[7] != kTVMStr) {
    TVMAPIErrorf("one or more invalid arg types");
    return kTvmErrorFunctionCallWrongArgType;
  }

  TVMModuleHandle mod = (TVMModuleHandle)args[0].v_handle;
  const char* name = args[1].v_str;
  g_time_evaluator_state.ctx.device_type = args[2].v_int64;
  g_time_evaluator_state.ctx.device_id = args[3].v_int64;
  g_time_evaluator_state.number = args[4].v_int64;
  g_time_evaluator_state.repeat = args[5].v_int64;
  g_time_evaluator_state.min_repeat_ms = args[6].v_int64;

  int ret_code =
      TVMModGetFunction(mod, name, /* query_imports */ 0, &g_time_evaluator_state.func_to_time);
  if (ret_code != 0) {
    return ret_code;
  }

  g_time_evaluator_state.function_index++;
  ret_val[0].v_handle =
      EncodeFunctionHandle(kTimeEvaluatorModuleIndex, g_time_evaluator_state.function_index);
  ret_type_code[0] = kTVMPackedFuncHandle;
  return kTvmErrorNoError;
}

tvm_crt_error_t RunTimeEvaluator(tvm_function_index_t function_index, TVMValue* args,
                                 int* type_codes, int num_args, TVMValue* ret_val,
                                 int* ret_type_code) {
  if (function_index != g_time_evaluator_state.function_index) {
    return kTvmErrorTimeEvaluatorBadHandle;
  }

  // TODO(areusch): should *really* rethink needing to return doubles
  DLContext result_byte_ctx = {kDLCPU, 0};
  TVMByteArray* result_byte_arr = NULL;
  tvm_crt_error_t err =
      TVMPlatformMemoryAllocate(sizeof(TVMByteArray), result_byte_ctx, (void*)&result_byte_arr);
  if (err != kTvmErrorNoError) {
    goto release_and_return;
  }
  result_byte_arr->data = NULL;
  size_t data_size = sizeof(double) * g_time_evaluator_state.repeat;
  err = TVMPlatformMemoryAllocate(data_size, result_byte_ctx, (void*)&result_byte_arr->data);
  if (err != kTvmErrorNoError) {
    goto release_and_return;
  }
  result_byte_arr->size = data_size;
  double min_repeat_seconds = ((double)g_time_evaluator_state.min_repeat_ms) / 1000;
  double* iter = (double*)result_byte_arr->data;
  for (int i = 0; i < g_time_evaluator_state.repeat; i++) {
    double repeat_res_seconds = 0.0;
    int exec_count = 0;
    // do-while structure ensures we run even when `min_repeat_ms` isn't set (i.e., is 0).
    do {
      err = TVMPlatformTimerStart();
      if (err != kTvmErrorNoError) {
        goto release_and_return;
      }

      for (int j = 0; j < g_time_evaluator_state.number; j++) {
        err = TVMFuncCall(g_time_evaluator_state.func_to_time, args, type_codes, num_args, ret_val,
                          ret_type_code);
        if (err != kTvmErrorNoError) {
          goto release_and_return;
        }
      }
      exec_count += g_time_evaluator_state.number;

      double curr_res_seconds;
      err = TVMPlatformTimerStop(&curr_res_seconds);
      if (err != kTvmErrorNoError) {
        goto release_and_return;
      }
      repeat_res_seconds += curr_res_seconds;
    } while (repeat_res_seconds < min_repeat_seconds);
    double mean_exec_seconds = repeat_res_seconds / exec_count;
    *iter = mean_exec_seconds;
    iter++;
  }

  *ret_type_code = kTVMBytes;
  ret_val->v_handle = result_byte_arr;
  return err;

release_and_return : {
  tvm_crt_error_t release_err =
      TVMPlatformMemoryFree((void*)&result_byte_arr->data, result_byte_ctx);
  if (release_err != kTvmErrorNoError) {
    release_err = TVMPlatformMemoryFree((void*)&result_byte_arr, result_byte_ctx);
  }

  if (err == kTvmErrorNoError && release_err != kTvmErrorNoError) {
    err = release_err;
  }
}
  return err;
}

// Default implementation, overridden by the platform runtime.
__attribute__((weak)) tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  return kTvmErrorFunctionCallNotImplemented;
}
