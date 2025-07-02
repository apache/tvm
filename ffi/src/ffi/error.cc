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
 * \file src/ffi/error.cc
 * \brief Error handling implementation
 */
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

namespace tvm {
namespace ffi {

class SafeCallContext {
 public:
  void SetRaised(TVMFFIObjectHandle error) {
    last_error_ =
        details::ObjectUnsafe::ObjectPtrFromUnowned<ErrorObj>(static_cast<TVMFFIObject*>(error));
  }

  void SetRaisedByCstr(const char* kind, const char* message, const TVMFFIByteArray* traceback) {
    Error error(kind, message, traceback);
    last_error_ = details::ObjectUnsafe::ObjectPtrFromObjectRef<ErrorObj>(std::move(error));
  }

  void MoveFromRaised(TVMFFIObjectHandle* result) {
    result[0] = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(last_error_));
  }

  static SafeCallContext* ThreadLocal() {
    static thread_local SafeCallContext ctx;
    return &ctx;
  }

 private:
  ObjectPtr<ErrorObj> last_error_;
};

}  // namespace ffi
}  // namespace tvm

void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message) {
  // NOTE: run traceback here to simplify the depth of tracekback
  tvm::ffi::SafeCallContext::ThreadLocal()->SetRaisedByCstr(kind, message, TVM_FFI_TRACEBACK_HERE);
}

void TVMFFIErrorSetRaised(TVMFFIObjectHandle error) {
  tvm::ffi::SafeCallContext::ThreadLocal()->SetRaised(error);
}

void TVMFFIErrorMoveFromRaised(TVMFFIObjectHandle* result) {
  tvm::ffi::SafeCallContext::ThreadLocal()->MoveFromRaised(result);
}

TVMFFIObjectHandle TVMFFIErrorCreate(const TVMFFIByteArray* kind, const TVMFFIByteArray* message,
                                     const TVMFFIByteArray* traceback) {
  TVM_FFI_LOG_EXCEPTION_CALL_BEGIN();
  tvm::ffi::Error error(std::string(kind->data, kind->size),
                        std::string(message->data, message->size),
                        std::string(traceback->data, traceback->size));
  TVMFFIObjectHandle out =
      tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(error));
  return out;
  TVM_FFI_LOG_EXCEPTION_CALL_END(TVMFFIErrorCreate);
}
