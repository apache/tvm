/*!
 *  Copyright (c) 2019 by Contributors
 * \file utvm_runtime.cc
 * \brief micro device init stub
 */
#include "utvm_runtime.h"

// Task pointers must be patched before calling a function.
UTVMTask task;

// We use a dummy function to signal execution is finished for device
// backends which require breakpoints.
void UTVMDone() {}

void UTVMMain() {
  task.func((void*) task.args->values, (void*) task.args->type_codes, task.args->num_args);
  UTVMDone();
}

// TODO(weberlo): Writes fail to pointer variables if they're initialized to
// `NULL`.  Why?

// These pointers are patched at load time to point to the workspace section.
char *utvm_workspace_begin = (char*) 1;
char *utvm_workspace_curr = (char*) 1;

const char *last_error = (char*) 1;

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
                               int dtype_code_hint, int dtype_bits_hint) {
  // Align up to 8 bytes.
  utvm_workspace_curr += (8 - ((uintptr_t) utvm_workspace_curr % 8)) % 8;
  void* ret_ptr = (void*) utvm_workspace_curr;
  utvm_workspace_curr += size;
  return ret_ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  // TODO(weberlo): Actually free memory.
  return 0;
}

void TVMAPISetLastError(const char* msg) {
  last_error = msg;
}
