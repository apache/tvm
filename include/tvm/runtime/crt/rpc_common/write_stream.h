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

#ifndef TVM_RUNTIME_CRT_RPC_COMMON_WRITE_STREAM_H_
#define TVM_RUNTIME_CRT_RPC_COMMON_WRITE_STREAM_H_

#include <inttypes.h>
#include <stddef.h>
#include <sys/types.h>
#include <tvm/runtime/crt/error_codes.h>

#include "../../../../../src/support/ssize.h"

namespace tvm {
namespace runtime {
namespace micro_rpc {

class WriteStream {
 public:
  virtual ~WriteStream();
  virtual ssize_t Write(const uint8_t* data, size_t data_size_bytes) = 0;
  virtual void PacketDone(bool is_valid) = 0;

  tvm_crt_error_t WriteAll(uint8_t* data, size_t data_size_bytes, size_t* bytes_consumed);
};

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CRT_RPC_COMMON_WRITE_STREAM_H_
