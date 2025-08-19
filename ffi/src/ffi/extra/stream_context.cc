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
 * \file src/ffi/extra/stream_context.cc
 *
 * \brief A minimalistic stream context based on ffi values.
 */

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <vector>

namespace tvm {
namespace ffi {

class StreamContext {
 public:
  void SetStream(int32_t device_type, int32_t device_id, TVMFFIStreamHandle stream,
                 TVMFFIStreamHandle* out_original_stream) {
    if (static_cast<size_t>(device_type) >= stream_table_.size()) {
      stream_table_.resize(device_type + 1);
    }
    if (static_cast<size_t>(device_id) >= stream_table_[device_type].size()) {
      stream_table_[device_type].resize(device_id + 1, nullptr);
    }
    if (out_original_stream != nullptr) {
      *out_original_stream = stream_table_[device_type][device_id];
    }
    stream_table_[device_type][device_id] = stream;
  }

  TVMFFIStreamHandle GetStream(int32_t device_type, int32_t device_id) {
    if (static_cast<size_t>(device_type) < stream_table_.size() &&
        static_cast<size_t>(device_id) < stream_table_[device_type].size()) {
      return stream_table_[device_type][device_id];
    }
    return nullptr;
  }

  static StreamContext* ThreadLocal() {
    static thread_local StreamContext inst;
    return &inst;
  }

 private:
  std::vector<std::vector<TVMFFIStreamHandle>> stream_table_;
};

}  // namespace ffi
}  // namespace tvm

int TVMFFIEnvSetStream(int32_t device_type, int32_t device_id, TVMFFIStreamHandle stream,
                       TVMFFIStreamHandle* out_original_stream) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::StreamContext::ThreadLocal()->SetStream(device_type, device_id, stream,
                                                    out_original_stream);
  TVM_FFI_SAFE_CALL_END();
}

TVMFFIStreamHandle TVMFFIEnvGetCurrentStream(int32_t device_type, int32_t device_id) {
  TVM_FFI_LOG_EXCEPTION_CALL_BEGIN();
  return tvm::ffi::StreamContext::ThreadLocal()->GetStream(device_type, device_id);
  TVM_FFI_LOG_EXCEPTION_CALL_END(TVMFFIEnvGetCurrentStream);
}
