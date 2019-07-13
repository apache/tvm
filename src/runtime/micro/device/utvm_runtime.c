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
 *  Copyright (c) 2019 by Contributors
 * \file utvm_runtime.cc
 * \brief uTVM runtime
 *
 * All function calls go through `UTVMMain`, which reads from the current
 * `UTVMTask` and calls the appropriate function with the arguments from the
 * task.
 *
 * Additionally included in this file are definitions for some of the most
 * common functions used in the C runtime API.
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "utvm_runtime.h"

// Task pointers must be patched before calling a function.
UTVMTask task;

// These pointers are patched at load time to point to the workspace section.
char* utvm_workspace_begin = NULL;  // NOLINT(*)
char* utvm_workspace_end = NULL;  // NOLINT(*)
char* utvm_workspace_curr = NULL;  // NOLINT(*)
// Keep track of how many active allocations there are on the workspace.
size_t utvm_num_active_allocs = 0;

const char* utvm_last_error = NULL;  // NOLINT(*)
int32_t utvm_return_code = 0;  // NOLINT(*)

// We use a dummy function to signal execution is finished for device
// backends which require breakpoints.
void UTVMDone() { }

void UTVMMain() {
  utvm_workspace_curr = utvm_workspace_begin;
  utvm_num_active_allocs = 0;
  utvm_last_error = NULL;  // NOLINT(*)
  utvm_return_code = 0;
  utvm_return_code = task.func((void*) task.arg_values, (void*) task.arg_type_codes,  // NOLINT(*)
                               task.num_args);
  UTVMDone();
}

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
                               int dtype_code_hint, int dtype_bits_hint) {
  // Align up to 8 bytes.
  utvm_workspace_curr += (8 - ((uintptr_t) utvm_workspace_curr % 8)) % 8;  // NOLINT(*)
  if (utvm_workspace_curr + size > utvm_workspace_end) {
    // Out of space in workspace.
    return NULL;
  }
  void* ret_ptr = (void*) utvm_workspace_curr;  // NOLINT(*)
  utvm_workspace_curr += size;
  utvm_num_active_allocs++;
  return ret_ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  utvm_num_active_allocs--;
  if (utvm_num_active_allocs < 0) {
    TVMAPISetLastError("free called with no active workspace allocations");
    // Reset allocations and workspace (for future task executions).
    utvm_num_active_allocs = 0;
    utvm_workspace_curr = utvm_workspace_begin;
    return -1;
  } else if (utvm_num_active_allocs == 0) {
    // No more allocations.  Reset workspace.
    utvm_workspace_curr = utvm_workspace_begin;
    return 0;
  } else {
    return 0;
  }
}

void TVMAPISetLastError(const char* msg) {
  utvm_last_error = msg;
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
