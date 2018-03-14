/*!
 *  Copyright (c) 2018 by Contributors
 * \file threading_backend.h
 * \brief Utilities for manipulating thread pool threads.
 */
#ifndef TVM_RUNTIME_THREADING_BACKEND_H_
#define TVM_RUNTIME_THREADING_BACKEND_H_

#include <functional>
#include <memory>
#include <vector>

namespace tvm {
namespace runtime {
namespace threading {

class ThreadGroup {
 public:
  ThreadGroup();
  void Launch(std::vector<std::function<void()>> task_callbacks);
  size_t Size();
  void Join();
#ifdef _LIBCPP_SGX_CONFIG
  void RunTask();
#endif
  ~ThreadGroup();

 private:
  class ThreadGroupImpl;
  ThreadGroupImpl* impl_;
};

int MaxConcurrency();
void Yield();

}  // namespace threading
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREADING_BACKEND_H_
