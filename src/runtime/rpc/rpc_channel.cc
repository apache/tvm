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
 * \file rpc_channel.cc
 */
#include "rpc_channel.h"

#include <string>

namespace tvm {
namespace runtime {

size_t CallbackChannel::Send(const void* data, size_t size) {
  TVMByteArray bytes;
  bytes.data = static_cast<const char*>(data);
  bytes.size = size;
  int64_t n = fsend_(bytes);
  if (n == -1) {
    LOG(FATAL) << "CallbackChannel::Send";
  }
  return static_cast<size_t>(n);
}

size_t CallbackChannel::Recv(void* data, size_t size) {
  TVMRetValue ret = frecv_(size);

  if (ret.type_code() != kTVMBytes) {
    LOG(FATAL) << "CallbackChannel::Recv";
  }
  std::string* bytes = ret.ptr<std::string>();
  memcpy(static_cast<char*>(data), bytes->c_str(), bytes->length());
  return bytes->length();
}

}  // namespace runtime
}  // namespace tvm
