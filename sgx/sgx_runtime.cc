/*!
 *  Copyright (c) 2018 by Contributors
 * \file sgx_runtime.cc
 */
#include "../../src/runtime/c_runtime_api.cc"
#include "../../src/runtime/cpu_device_api.cc"
#include "../../src/runtime/workspace_pool.cc"
#include "../../src/runtime/module_util.cc"
#include "../../src/runtime/module.cc"
#include "../../src/runtime/registry.cc"
#include "../../src/runtime/system_lib_module.cc"

// dummy parallel runtime
int TVMBackendParallelLaunch(
    FTVMParallelLambda flambda,
    void* cdata,
    int num_task) {
  TVMAPISetLastError("Parallel is not (yet) supported in SGX runtime");
  return -1;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) {
  return 0;
}

