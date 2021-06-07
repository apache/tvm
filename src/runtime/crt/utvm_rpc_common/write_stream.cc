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
 * \file framing.h
 * \brief Framing for RPC.
 */
#include <tvm/runtime/crt/rpc_common/write_stream.h>

namespace tvm {
namespace runtime {
namespace micro_rpc {

WriteStream::~WriteStream() {}

tvm_crt_error_t WriteStream::WriteAll(uint8_t* data, size_t data_size_bytes,
                                      size_t* bytes_consumed) {
  *bytes_consumed = 0;
  while (data_size_bytes > 0) {
    ssize_t to_return = Write(data, data_size_bytes);
    if (to_return == 0) {
      return kTvmErrorWriteStreamShortWrite;
    } else if (to_return < 0) {
      return (tvm_crt_error_t)to_return;
    } else if (to_return > 0 && (static_cast<size_t>(to_return)) > data_size_bytes) {
      return kTvmErrorWriteStreamLongWrite;
    }

    data += to_return;
    data_size_bytes -= to_return;
    *bytes_consumed += to_return;
  }

  return kTvmErrorNoError;
}

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm
