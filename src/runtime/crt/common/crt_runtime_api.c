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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/func_registry.h>
#include <tvm/runtime/crt/internal/common/memory.h>
#include <tvm/runtime/crt/internal/common/ndarray.h>
#include <tvm/runtime/crt/internal/graph_runtime/graph_runtime.h>
#include <tvm/runtime/crt/memory.h>
#include <tvm/runtime/crt/platform.h>

// Handle internal errors

static char g_last_error[1024];

void TVMAPISetLastError(const char* msg) { strncpy(g_last_error, msg, sizeof(g_last_error)); }

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
  TVMNDArray arr = TVMNDArray_Empty(ndim, shape, dtype, ctx);
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

  *out_data = vmalloc(nbytes);
  return 0;
}

int TVMDeviceFreeDataSpace(TVMContext ctx, void* ptr) {
  vfree(ptr);
  return 0;
}

int TVMDeviceCopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                            size_t num_bytes, TVMContext ctx_from, TVMContext ctx_to,
                            DLDataType type_hint, TVMStreamHandle stream) {
  memcpy(((uint8_t*)to) + to_offset, ((uint8_t*)from) + from_offset, num_bytes);
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

static int TVMModCreateFromCModule(const TVMModule* mod, TVMModuleHandle* out_handle) {
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

  if (unvalidated_module_index > kGlobalFuncModuleIndex) {
    TVMAPIErrorf("invalid module handle: index=%08x", unvalidated_module_index);
    return -1;
  } else if (unvalidated_module_index < kGlobalFuncModuleIndex &&
             registered_modules[unvalidated_module_index] == NULL) {
    TVMAPIErrorf("unregistered module: index=%08x", unvalidated_module_index);
    return -1;
  }

  *function_index = ((uint32_t)((uintptr_t)handle)) & ~0x8000;
  *module_index = unvalidated_module_index;
  return 0;
}

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

  if (module_index == kGlobalFuncModuleIndex) {
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

static int FindFunctionOrSetAPIError(tvm_module_index_t module_index,
                                     const TVMFuncRegistry* registry, const char* name,
                                     TVMFunctionHandle* out) {
  tvm_function_index_t function_index;
  if (TVMFuncRegistry_Lookup(registry, name, &function_index) != 0) {
    TVMAPIErrorf("failed to get function: mod_index=%04" PRIx16 ", name=%s", module_index, name);
    return -1;
  }

  *out = EncodeFunctionHandle(module_index, function_index);
  return 0;
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

tvm_crt_error_t TVMInitializeRuntime(uint8_t* memory_pool, size_t memory_pool_size_bytes,
                                     size_t page_size_bytes_log2) {
  int idx;
  tvm_crt_error_t error;

  error =
      TVMInitializeGlobalMemoryManager(memory_pool, memory_pool_size_bytes, page_size_bytes_log2);
  if (error != kTvmErrorNoError) {
    return error;
  }

  system_lib_handle = kTVMModuleHandleUninitialized;

  TVMMutableFuncRegistry_Create(&global_func_registry,
                                vmalloc(TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES),
                                TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES);
  for (idx = 0; idx < TVM_CRT_MAX_REGISTERED_MODULES; idx++) {
    registered_modules[idx] = NULL;
  }

  error = TVMFuncRegisterGlobal("runtime.SystemLib", &SystemLibraryCreate, 0);
  if (error != kTvmErrorNoError) {
    return error;
  }

  error = TVMFuncRegisterGlobal("tvm.rpc.server.ModuleGetFunction", &ModuleGetFunction, 0);
  if (error != kTvmErrorNoError) {
    return error;
  }

  return kTvmErrorNoError;
}
