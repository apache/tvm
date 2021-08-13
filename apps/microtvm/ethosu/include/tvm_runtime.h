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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/stack_allocator.h>

#define WORKSPACE_SIZE (16384 * 1024)
__attribute__((section("ethosu_scratch"))) static uint8_t g_aot_memory[WORKSPACE_SIZE];

tvm_workspace_t app_workspace;

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
  printf("TVMPlatformAbort: %d\n", error_code);
  printf("EXITTHESIM\n");
  exit(-1);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  vfprintf(stdout, msg, args);
  va_end(args);
}

TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) { return 0; }