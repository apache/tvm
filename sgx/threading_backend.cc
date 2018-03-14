/*!
 *  Copyright (c) 2018 by Contributors
 * \file sgx/threading_backend.cc
 * \brief SGX threading backend
 */
#include <tvm/runtime/threading_backend.h>
#include <dmlc/logging.h>
#include <mutex>
#include <queue>

namespace tvm {
namespace runtime {
namespace threading {

class ThreadGroup::ThreadGroupImpl {
 public:
  void Launch(std::vector<std::function<void()>> task_callbacks) {
    std::lock_guard<std::mutex> lock(qmut_);
    CHECK(Size() + task_callbacks.size() <= MaxConcurrency())
      << "Tried spawning more threads than allowed by max concurrency.";
    for (std::function<void()> cb : task_callbacks) {
      pending_tasks_.push(cb);
    }
    num_tasks_ += task_callbacks.size();
    sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
    sgx_status = ocall_thread_pool_launch(task_callbacks.size());
    CHECK(sgx_status == SGX_SUCCESS) << "SGX Error: " << sgx_status;
  }

  size_t Size() { return num_tasks_; }

  void RunTask() {
    std::function<void()> task = GetPendingTask();
    if (task == nullptr) return;
    task();
    std::lock_guard<std::mutex> lock(qmut_);
    --num_tasks_;
  }

  static ThreadGroupImpl* Global() {
    static ThreadGroupImpl inst;
    return &inst;
  }

 private:
  std::function<void()> GetPendingTask() {
    std::lock_guard<std::mutex> lock(qmut_);
    if (pending_tasks_.size() == 0) return nullptr;
    std::function<void()> task = pending_tasks_.front();
    pending_tasks_.pop();
    return task;
  }

  size_t num_tasks_;
  std::mutex qmut_;
  std::queue<std::function<void()>> pending_tasks_;
};

ThreadGroup::ThreadGroup(): impl_(ThreadGroup::ThreadGroupImpl::Global()) {}
ThreadGroup::~ThreadGroup() {}
void ThreadGroup::Launch(std::vector<std::function<void()>> task_callbacks) {
  return impl_->Launch(task_callbacks);
}
size_t ThreadGroup::Size() { return impl_->Size(); }
void ThreadGroup::RunTask() { return impl_->RunTask(); }

void Yield() {}

int MaxConcurrency() { return std::max(TVM_SGX_MAX_CONCURRENCY, 1); }

extern "C" {
void ecall_run_worker() {
  (new ThreadGroup())->RunTask();
}
}

}  // namespace threading
}  // namespace runtime
}  // namespace tvm
