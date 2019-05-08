/*!
 *  Copyright (c) 2019 by Contributors
 * \file utvm_runtime.cc
 * \brief micro device init stub
 */
#include "utvm_runtime.h"

// task pointers must be patched before calling a function
UTVMTask task;

// dummy function to signal execution is finished
void UTVMDone() {}

// init stub
uint64_t UTVMMain() {
  // TODO(weberlo): Change codegen so we don't need these casts.
  return task.func((void*) task.args->values, (void*) task.args->type_codes, task.args->num_args);
  // UTVMDone();
}

// These pointers are patched at load time to point to the workspace section.
// char *workspace_start = NULL;
// char *workspace_curr = NULL;
char *workspace_start = (char *) 1;
char *workspace_curr = (char *) 1;

const char *last_error = NULL;

// TODO(weberlo): Remove duplicate docs.

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocate spaced is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param nbytes The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \param dtype_code_hint The type code of the array elements. Only used in
 * certain backends such as OpenGL.
 * \param dtype_bits_hint The type bits of the array elements. Only used in
 * certain backends such as OpenGL.
 * \return nullptr when error is thrown, a valid ptr if success
 */
void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
                               int dtype_code_hint, int dtype_bits_hint) {
  // Align up to 8 bytes.
  workspace_curr += (8 - ((uintptr_t) workspace_curr % 8)) % 8;
  void* ret_ptr = (void*) workspace_curr;
  workspace_curr += size;
  return ret_ptr;
}

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 when no error is thrown, -1 when failure happens
 *
 * \sa TVMBackendAllocWorkspace
 */
int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  // We don't actually free memory in the current allocation scheme.
  return 0;
}

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
void TVMAPISetLastError(const char* msg) {
  last_error = msg;
}
