/*!
 *  Copyright (c) 2018 by Contributors
 * \file sgx/threading_backend.cc
 * \brief SGX threading backend
 */
#include <tvm/runtime/threading_backend.h>
#include <dmlc/logging.h>
#include <sgx_edger8r.h>
#include <sgx_trts.h>
#include <atomic>

extern "C" {
sgx_status_t SGX_CDECL tvm_ocall_thread_group_launch(int num_workers, void* cb);
sgx_status_t SGX_CDECL tvm_ocall_thread_group_join();
}

#ifndef TVM_SGX_MAX_CONCURRENCY
#define TVM_SGX_MAX_CONCURRENCY 1
#endif

namespace tvm {
namespace runtime {
namespace threading {

class ThreadGroup::Impl {
 public:
  Impl(int num_workers, std::function<void(int)> worker_callback,
       bool exclude_worker0)
      : num_workers_(num_workers),
        worker_callback_(worker_callback),
        next_task_id_(exclude_worker0) {
    CHECK(num_workers <= TVM_SGX_MAX_CONCURRENCY)
      << "Tried spawning more threads than allowed by TVM_SGX_MAX_CONCURRENCY.";
    sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
    sgx_status = tvm_ocall_thread_group_launch(num_workers, this);
    CHECK(sgx_status == SGX_SUCCESS) << "SGX Error: " << sgx_status;
  }

  ~Impl() {
    tvm_ocall_thread_group_join();
  }

  void RunTask() {
    int task_id = next_task_id_++;
    CHECK(task_id < num_workers_)
      << "More workers entered enclave than allowed by TVM_SGX_MAX_CONCURRENCY";
    worker_callback_(task_id);
  }

 private:
  int num_workers_;
  std::function<void(int)> worker_callback_;
  std::atomic<int> next_task_id_;
};

ThreadGroup::ThreadGroup(int num_workers,
                         std::function<void(int)> worker_callback,
                         bool exclude_worker0)
  : impl_(new ThreadGroup::Impl(num_workers, worker_callback, exclude_worker0)) {}
void ThreadGroup::Join() {}
ThreadGroup::~ThreadGroup() { delete impl_; }

void Yield() {}

int MaxConcurrency() { return TVM_SGX_MAX_CONCURRENCY; }

extern "C" {
void tvm_ecall_run_worker(const void* impl) {
  if (!sgx_is_within_enclave(impl, sizeof(ThreadGroup::Impl))) return;
  ((ThreadGroup::Impl*)impl)->RunTask();
}
}

}  // namespace threading
}  // namespace runtime
}  // namespace tvm
