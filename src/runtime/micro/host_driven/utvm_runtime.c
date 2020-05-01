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
 * \file utvm_runtime.cc
 * \brief uTVM runtime
 *
 * All function calls go through the externally defined `UTVMInit`, which
 * performs device-specific setup, then calls `UTVMMain`.  `UTVMMain` then
 * calls the function in `utvm_task` with the arguments from the task.
 *
 * Additionally included in this file are definitions for some of the most
 * common functions used in the C runtime API.
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "utvm_runtime.h"

// TODO(weberlo, areusch): move defines into header
// TODO(weberlo, areusch): unify TASK_QUEUE_SIZE and MicroSession::kTaskQueueCapacity.
#define TASK_QUEUE_SIZE 20
volatile UTVMTask utvm_tasks[TASK_QUEUE_SIZE] = { };
volatile uint32_t utvm_num_tasks = 0;
volatile uint32_t utvm_task_times[TASK_QUEUE_SIZE] = { };

// These pointers are patched at load time to point to the workspace section.
volatile char* utvm_workspace_start = NULL;  // NOLINT(*)
volatile char* utvm_workspace_end = NULL;    // NOLINT(*)
volatile char* utvm_workspace_curr = NULL;   // NOLINT(*)
#define MAX_WS_ALLOCS 10
volatile char* utvm_alloc_ends[MAX_WS_ALLOCS] = {};  // NOLINT(*)
volatile uint32_t utvm_alloc_idx = 0;
// Keep track of how many active allocations there are on the workspace.
volatile uint32_t utvm_num_active_allocs = 0;

volatile uint32_t utvm_word_size = 0;

volatile int32_t utvm_last_error = 0;  // NOLINT(*)

volatile uint32_t utvm_done = 0;

// Gets called by UTVMInit, after device-specific initialization is finished.
void UTVMMain() {
  utvm_done = 0;
  // loss of precision should be fine here, since we only care about the lower bits
  if (((uint32_t) utvm_workspace_start) % utvm_word_size) {
    utvm_last_error = UTVM_ERR_WS_UNALIGNED_START;
    UTVMDone();
    return;
  }
  utvm_workspace_curr = utvm_workspace_start;
  utvm_num_active_allocs = 0;
  utvm_alloc_idx = 0;
  utvm_last_error = UTVM_ERR_NOT_FINISHED;
  for (uint32_t i = 0; i < utvm_num_tasks; i++) {
    int32_t err = UTVM_ERR_OK;
    utvm_task_times[i] = 0;
    err = UTVMTimerStart();
    if (err < 0) {
      utvm_last_error = err;
      UTVMDone();
      return;
    }
    err = utvm_tasks[i].func(
        (void*) utvm_tasks[i].arg_values,      // NOLINT(*)
        (void*) utvm_tasks[i].arg_type_codes,  // NOLINT(*)
        utvm_tasks[i].num_args);
    if (err < 0) {
      UTVMDone();
      return;
    }
    utvm_task_times[i] = UTVMTimerStop(&err);
    if (err < 0) {
      utvm_last_error = err;
      UTVMDone();
      return;
    }
  }
  if (utvm_last_error == UTVM_ERR_NOT_FINISHED) {
    utvm_last_error = UTVM_ERR_OK;
  }
  UTVMDone();
}

// We use a dummy function to signal execution is finished for device
// backends which require breakpoints.
void __attribute__((noinline)) UTVMDone() {
  utvm_done = 1;
}

#define ALIGNED_UP(x, word_size) \
  ((((word_size) - (((uintptr_t) (x)) % (word_size))) % (word_size)) + (x))

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
                               int dtype_code_hint, int dtype_bits_hint) {
  if (size == 0) {
    utvm_last_error = UTVM_ERR_WS_ZERO_SIZE_ALLOC;
    return NULL;
  }
  size_t alloc_requested_bytes = size;
  size_t alloc_size_words = (alloc_requested_bytes + utvm_word_size - 1) / utvm_word_size;
  size_t alloc_size_bytes = alloc_size_words * utvm_word_size;

  // Align up to the target word size.
  if (utvm_workspace_curr + alloc_size_bytes > utvm_workspace_end) {
    // Out of space in workspace.
    utvm_last_error = UTVM_ERR_WS_OUT_OF_SPACE;
    return NULL;
  }
  if (utvm_alloc_idx == MAX_WS_ALLOCS - 1) {
    // Exceeded number of allocs we can keep track of.
    utvm_last_error = UTVM_ERR_WS_TOO_MANY_ALLOCS;
    return NULL;
  }
  void* ret_ptr = (void*) utvm_workspace_curr;  // NOLINT(*)
  utvm_workspace_curr = utvm_workspace_curr + alloc_size_bytes;
  // store the *end* of the alloc, so we can restore the WS pointer when freeing
  utvm_alloc_ends[utvm_alloc_idx] = utvm_workspace_curr;
  utvm_alloc_idx++;
  utvm_num_active_allocs++;
  return ret_ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  // TODO(weberlo, areusch): add dev type check
  if (utvm_num_active_allocs == 0) {
    TVMAPISetLastError("free called with no active workspace allocations");
    // Reset allocations and workspace (for future task executions).
    utvm_num_active_allocs = 0;
    utvm_workspace_curr = utvm_workspace_start;
    utvm_last_error = UTVM_ERR_WS_DOUBLE_FREE;
    return -1;
  } else {
    utvm_num_active_allocs--;
    if (ptr == utvm_workspace_start) {
      // it's the first allocation
      utvm_alloc_ends[0] = NULL;
    } else {
      for (uint32_t i = utvm_alloc_idx - 1; i >= 0; i--) {
        if (utvm_alloc_ends[i] == ptr) {
          utvm_alloc_ends[i + 1] = NULL;
          break;
        }
      }
    }
    while (utvm_alloc_idx > 0 && utvm_alloc_ends[utvm_alloc_idx - 1] == NULL) {
      utvm_alloc_idx--;
    }
    if (utvm_alloc_idx == 0) {
      utvm_workspace_curr = utvm_workspace_start;
    } else {
      // TODO(weberlo, areusch): could you possibly have utvm_alloc_idx pointing to a NULL entry in
      // this branch?
      utvm_workspace_curr = utvm_alloc_ends[utvm_alloc_idx - 1];
    }
    return 0;
  }
}

void TVMAPISetLastError(const char* msg) { }

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
