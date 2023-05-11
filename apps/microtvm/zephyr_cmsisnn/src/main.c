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
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "tvmgen_default.h"

extern char* labels[12];
extern float input_storage[490];
extern float output_storage[12];

extern const size_t output_len;

static uint8_t __attribute__((aligned(TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES)))
g_crt_workspace[TVMGEN_DEFAULT_WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  vfprintf(stderr, msg, args);
  va_end(args);
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t error_code) {
  fprintf(stderr, "TVMPlatformAbort: %d\n", error_code);
  exit(-1);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  uintptr_t ret = StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
  return ret;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

void main(void) {
  StackMemoryManager_Init(&app_workspace, g_crt_workspace, TVMGEN_DEFAULT_WORKSPACE_SIZE);

  struct tvmgen_default_inputs inputs = {.input = input_storage};
  struct tvmgen_default_outputs outputs = {.Identity = output_storage};

  if (tvmgen_default_run(&inputs, &outputs) != 0) {
    printk("Model run failed\n");
    exit(-1);
  }

  // Calculate index of max value
  float max_value = 0.0;
  size_t max_index = -1;
  for (unsigned int i = 0; i < output_len; ++i) {
    if (output_storage[i] > max_value) {
      max_value = output_storage[i];
      max_index = i;
    }
  }
  printk("The word is '%s'!\n", labels[max_index]);
  exit(0);
}
