#include <tvm/runtime/threading_backend.h>
#include "../../src/runtime/threading_backend.cc"
#include <iostream>

extern "C" {
sgx_enclave_id_t tvm_sgx_eid;
sgx_status_t tvm_ecall_run_worker(sgx_enclave_id_t eid);
}

namespace tvm {
namespace runtime {
namespace sgx {

using tvm::runtime::threading::ThreadGroup;

static ThreadGroup sgx_thread_group;

extern "C" {
void tvm_ocall_thread_pool_launch(int num_tasks) {
  std::function<void()> runner = [] {
    sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
    sgx_status = tvm_ecall_run_worker(tvm_sgx_eid);
    CHECK(sgx_status == SGX_SUCCESS) << "SGX Error: " << sgx_status;
  };
  std::vector<std::function<void()>> task_callbacks(num_tasks, runner);
  sgx_thread_group.Launch(task_callbacks);
}
}

void Shutdown() {
  sgx_thread_group.Join();
}

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm
