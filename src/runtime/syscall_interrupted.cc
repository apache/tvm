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
 * \file syscall_interrupted.cc
 * \brief Defines functions that decide when interrupted system calls should continue.
 */

#include "syscall_interrupted.h"

#include <tvm/runtime/registry.h>
#include <dmlc/logging.h>

namespace tvm {
namespace runtime {

static PackedFunc syscall_interrupted_callback{nullptr};

bool ShouldContinueAfterInterruptedSyscall() {
  // LOG(INFO) << "should continue cb: " << (syscall_interrupted_callback != nullptr);
  if (syscall_interrupted_callback != nullptr) {
    TVMRetValue rv = syscall_interrupted_callback();
    LOG(INFO) << "Ret val " << rv.type_code() << " " << kTVMNullptr << ": " << rv.ptr<void*>();
    if (rv.type_code() == kTVMNullptr) {
      // NOTE: Python functions that return nothing implicitly return true here.
      return true;
    } else {
      return bool(rv);
    }
  } else {
    return true;
  }
}

/*!
 * \brief Set a callback function invoked by the runtime when a system call is interrupted.
 *
 * Should the frontend need to perform additional checking (e.g. run Python signal handlers),
 * this callback can implement that checking and potentially throw errors when execution should
 * not continue.
 *
 */
TVM_REGISTER_GLOBAL("tvm.runtime.SetSyscallInterruptedCallback").set_body(
  [](TVMArgs args, TVMRetValue* rv) {
    syscall_interrupted_callback = args[0];
  });

}  // namespace runtime
}  // namespace tvm
