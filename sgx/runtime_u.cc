#include <tvm/runtime/threading_backend.h>
#include "../../src/runtime/threading_backend.cc"
#include <iostream>

extern sgx_enclave_id_t tvm_sgx_eid;
extern "C" {
sgx_status_t tvm_ecall_run_worker(sgx_enclave_id_t eid, const void* cb);
}

namespace tvm {
namespace runtime {
namespace sgx {

static std::unique_ptr<tvm::runtime::threading::ThreadGroup> sgx_thread_group;

extern "C" {

void tvm_ocall_thread_group_launch(int num_tasks, void* cb) {
  std::function<void(int)> runner = [cb](int _worker_id) {
    sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
    sgx_status = tvm_ecall_run_worker(tvm_sgx_eid, cb);
    CHECK(sgx_status == SGX_SUCCESS) << "SGX Error: " << sgx_status;
  };
  sgx_thread_group.reset(new tvm::runtime::threading::ThreadGroup(
        num_tasks, runner, false /* include_main_thread */));
}

void tvm_ocall_thread_group_join() {
  sgx_thread_group->Join();
}

}

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm
