/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
#include "runtime.h"

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
    sgx::OCallPackedFunc("__sgx_thread_group_launch__",
        num_workers_, reinterpret_cast<void*>(this));
  }

  ~Impl() {
    sgx::OCallPackedFunc("__sgx_thread_group_join__");
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
int ThreadGroup::Configure(AffinityMode mode, int nthreads, bool exclude_worker0) {
  int max_conc = MaxConcurrency();
  if (!nthreads || ntheads > max_conc) {
    return max_conc;
  }
  return nthreads;
}
ThreadGroup::~ThreadGroup() { delete impl_; }

void Yield() {}

int MaxConcurrency() { return TVM_SGX_MAX_CONCURRENCY; }

TVM_REGISTER_ENCLAVE_FUNC("__tvm_run_worker__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    void* tg = args[0];
    if (!sgx_is_within_enclave(tg, sizeof(ThreadGroup::Impl))) return;
    reinterpret_cast<ThreadGroup::Impl*>(tg)->RunTask();
  });

}  // namespace threading
}  // namespace runtime
}  // namespace tvm
