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
 * \file tcl_socket.h
 * \brief TCP socket wrapper for communicating using Tcl commands
 */
#ifndef TVM_RUNTIME_MICRO_TCL_SOCKET_H_
#define TVM_RUNTIME_MICRO_TCL_SOCKET_H_

#include <string>
#include <vector>

#include "../../common/socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief TCP socket wrapper for communicating using Tcl commands
 *
 * Usage generally involves building a command using the `cmd_builder` stream
 * interface, then sending the command with `SendCommand`, and if necessary,
 * reading the reply.
 */
class TclSocket {
 public:
  /*!
   * \brief constructor to create the socket
   */
  TclSocket();

  /*!
   * \brief destructor to close the socket connection
   */
  ~TclSocket();

  /*!
   * \brief open connection with server
   * \param addr server address
   */
  void Connect(tvm::common::SockAddr addr);

  /*
   * \brief send the built command to the server and await a reply
   *
   * \return the reply
   */
  void SendCommand();

  /*
   * \return string stream for current command being built
  */
  std::ostringstream& cmd_builder() { return cmd_builder_; }

  /*
   * \return reply from most recently sent command
  */
  const std::string& last_reply() { return last_reply_; }

 private:
  /*! \brief underlying TCP socket being wrapped */
  tvm::common::TCPSocket tcp_socket_;
  /*! \brief buffer used to receive messages from the socket */
  std::vector<uint8_t> reply_buf_;
  /*! \brief string stream used to build current command */
  std::ostringstream cmd_builder_;
  /*! \brief string stream used to receive replies from sent commands */
  std::ostringstream reply_builder_;
  /*! \brief reply from most recently sent command */
  std::string last_reply_;

  /*! \brief character denoting the end of a Tcl command */
  static const constexpr char kCommandTerminateToken = '\x1a';
  /*! \brief size of the buffer used to receive messages (in bytes) */
  static const constexpr size_t kReplyBufSize = 4096;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_TCL_SOCKET_H_
