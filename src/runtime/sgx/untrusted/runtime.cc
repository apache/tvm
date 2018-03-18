/*!
 *  Copyright (c) 2018 by Contributors
 * \file runtime.cc
 * \brief SGX untrusted runtime.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/threading_backend.h>
#include <sgx_edger8r.h>
#include <memory>
#include "runtime.h"

namespace tvm {
namespace runtime {
namespace sgx {

static std::unique_ptr<tvm::runtime::threading::ThreadGroup> sgx_thread_group;
extern thread_local sgx_enclave_id_t last_eid;

extern "C" {

void ocall_tvm_thread_group_launch(int num_tasks, void* cb) {
  sgx_enclave_id_t cb_eid = last_eid;
  std::function<void(int)> runner = [cb, cb_eid](int _worker_id) {
    sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
    sgx_status = ecall_tvm_run_worker(cb_eid, cb);
    CHECK(sgx_status == SGX_SUCCESS) << "SGX Error: " << sgx_status;
  };
  sgx_thread_group.reset(new tvm::runtime::threading::ThreadGroup(
        num_tasks, runner, false /* include_main_thread */));
}

void ocall_tvm_thread_group_join() {
  sgx_thread_group->Join();
}

void ocall_tvm_api_set_last_error(const char* err) {
  TVMAPISetLastError(err);
}

}  // extern "C"

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm
