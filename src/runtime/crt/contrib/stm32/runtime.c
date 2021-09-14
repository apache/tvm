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
 * \file runtime.c
 * \brief A minimal "C" runtime support required by the TVM 
 *        generated C code. Declared in "runtime/c_backend_api.h" 
 *        and "runtime/c_runtime_api.h"
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <malloc.h>

#define _USE_TVM_BACKEND_API_

#ifdef _USE_TVM_BACKEND_API_
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/crt/error_codes.h>
#else
#include "tvm/runtime/c_runtime_api.h"
#endif

static char * g_last_error = NULL;

#ifdef _USE_TVM_BACKEND_API_

// ====================================================
//   TVMPlatformMemoryAllocate
// ====================================================
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
#ifdef __arm__
  *out_ptr = malloc(num_bytes);
#else  // _x86_
  *out_ptr = malloc(num_bytes);
#endif
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// ====================================================
//   TVMPlatformMemoryFree
// ====================================================
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  free(ptr);
  return kTvmErrorNoError;
}

// ====================================================
//   TVMFuncRegisterGlobal
// ====================================================
int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {
  return 0;
}

// ====================================================
//   TVMPlatformAbort
// ====================================================
void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code) {
}

// ====================================================
//   TVMLogf
// ====================================================
void TVMLogf(const char* msg, ...) {
  return;
}

#else

// ====================================================
//   TVMBackendAllocWorkspace
// ====================================================
void *
TVMBackendAllocWorkspace(
  int device_type,
  int device_id,
  uint64_t nbytes,
  int dtype_code_hint,
  int dtype_bits_hint
) {
  void * ptr = NULL;
  assert(nbytes > 0);

#ifdef __arm__
  ptr = malloc(nbytes);
#else  // _x86_
  ptr = malloc(nbytes);
#endif

  return ptr;
}

// ====================================================
//   TVMBackendFreeWorkspace
// ====================================================
int
TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  free(ptr);
  return 0;
}
#endif  // _USE_TVM_BACKEND_API_

// ====================================================
//   TVMAPISetLastError
// ====================================================
void TVMAPISetLastError(const char * msg) {
  if (g_last_error) {
    free(g_last_error);
  }
  uint32_t nbytes = strlen(msg)+1;
  g_last_error = malloc(nbytes);
  snprintf(g_last_error, nbytes, "%s", msg);
}

// ====================================================
//   TVMGetLastError
// ====================================================
const char * TVMGetLastError(void) {
  assert(g_last_error);
  return g_last_error;
}
