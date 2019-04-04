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
  task.func(task.args, task.arg_type_ids, *task.num_args);
  UTVMDone();
  return 0;
}
