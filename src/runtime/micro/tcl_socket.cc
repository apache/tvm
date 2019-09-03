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
 *  Copyright (c) 2019 by Contributors
 * \file tcl_socket.cc
 */
#include <string>

#include "tcl_socket.h"

namespace tvm {
namespace runtime {

TclSocket::TclSocket() {
  tcp_socket_.Create();
  tcp_socket_.SetKeepAlive(true);
  reply_buf_.reserve(kReplyBufSize);
}

TclSocket::~TclSocket() {
  tcp_socket_.Close();
}

void TclSocket::Connect(tvm::common::SockAddr addr) {
  CHECK(tcp_socket_.Connect(addr)) << "failed to connect";
}

void TclSocket::SendCommand() {
  cmd_builder_ << kCommandTerminateToken;
  std::string full_cmd = cmd_builder_.str();
  CHECK(tcp_socket_.Send(full_cmd.data(), full_cmd.length()) != -1)
    << "failed to send command";
  cmd_builder_.str(std::string());

  reply_builder_.str(std::string());
  char last_read = '\0';
  // Receive from the socket until we reach a command terminator.
  do {
    ssize_t bytes_read;
    // Recieve from the socket until it's drained.
    do {
      // Leave room at the end of `reply_buf` to tack on a null terminator.
      bytes_read = tcp_socket_.Recv(reply_buf_.data(), kReplyBufSize - 1);
      reply_buf_[bytes_read] = '\0';
      reply_builder_ << reply_buf_.data();
      // Update last read character.
      last_read = reply_buf_[bytes_read - 1];
    } while (bytes_read == kReplyBufSize - 1);
    CHECK(bytes_read != -1) << "failed to read command reply";
  } while (last_read != kCommandTerminateToken);
  last_reply_ = reply_builder_.str();
  CHECK_EQ(last_reply_[last_reply_.length()-1], kCommandTerminateToken)
    << "missing command terminator";
}

}  // namespace runtime
}  // namespace tvm
