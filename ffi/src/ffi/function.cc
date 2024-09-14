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
/*
 * \file src/ffi/function.cc
 * \brief Function call registry and safecall context
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {

class SafeCallContext {
 public:
  void SetLastError(const TVMFFIAny* error_view) {
    last_error_ = Any(AnyView::CopyFromTVMFFIAny(error_view[0]));
    // turn string into formal error.
    if (std::optional<String> opt_str = last_error_.TryAs<String>()) {
      last_error_ = ::tvm::ffi::Error("RuntimeError", *opt_str, "");
    }
  }

  void MoveFromLastError(TVMFFIAny* result) { last_error_.MoveToTVMFFIAny(result); }

  static SafeCallContext* ThreadLocal() {
    static thread_local SafeCallContext ctx;
    return &ctx;
  }

 private:
  Any last_error_;
};

}  // namespace ffi
}  // namespace tvm

extern "C" {
void TVMFFISetLastError(const TVMFFIAny* error_view) {
  tvm::ffi::SafeCallContext::ThreadLocal()->SetLastError(error_view);
}

void TVMFFIMoveFromLastError(TVMFFIAny* result) {
  tvm::ffi::SafeCallContext::ThreadLocal()->MoveFromLastError(result);
}
}
