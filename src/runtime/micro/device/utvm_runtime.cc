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
int UTVMMain() {
  // TODO(weberlo): Change codegen so we don't need these casts.
  task.func((void*) task.args->values, (void*) task.args->type_codes, task.args->num_args);
  UTVMDone();
  return 0;
}
