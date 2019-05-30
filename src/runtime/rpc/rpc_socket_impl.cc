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
 *  Copyright (c) 2017 by Contributors
 * \file rpc_socket_impl.cc
 * \brief Socket based RPC implementation.
 */
#include <tvm/runtime/registry.h>
#include <memory>
#include "rpc_session.h"
#include "../../common/socket.h"

namespace tvm {
namespace runtime {

class SockChannel final : public RPCChannel {
 public:
  explicit SockChannel(common::TCPSocket sock)
      : sock_(sock) {}
  ~SockChannel() {
    if (!sock_.BadSocket()) {
        sock_.Close();
    }
  }
  size_t Send(const void* data, size_t size) final {
    ssize_t n = sock_.Send(data, size);
    if (n == -1) {
      common::Socket::Error("SockChannel::Send");
    }
    return static_cast<size_t>(n);
  }
  size_t Recv(void* data, size_t size) final {
    ssize_t n = sock_.Recv(data, size);
    if (n == -1) {
      common::Socket::Error("SockChannel::Recv");
    }
    return static_cast<size_t>(n);
  }

 private:
  common::TCPSocket sock_;
};

std::shared_ptr<RPCSession>
RPCConnect(std::string url, int port, std::string key) {
  common::TCPSocket sock;
  common::SockAddr addr(url.c_str(), port);
  sock.Create(addr.ss_family());
  CHECK(sock.Connect(addr))
      << "Connect to " << addr.AsString() << " failed";
  // hand shake
  std::ostringstream os;
  int code = kRPCMagic;
  int keylen = static_cast<int>(key.length());
  CHECK_EQ(sock.SendAll(&code, sizeof(code)), sizeof(code));
  CHECK_EQ(sock.SendAll(&keylen, sizeof(keylen)), sizeof(keylen));
  if (keylen != 0) {
    CHECK_EQ(sock.SendAll(key.c_str(), keylen), keylen);
  }
  CHECK_EQ(sock.RecvAll(&code, sizeof(code)), sizeof(code));
  if (code == kRPCMagic + 2) {
    sock.Close();
    LOG(FATAL) << "URL " << url << ":" << port
               << " cannot find server that matches key=" << key;
  } else if (code == kRPCMagic + 1) {
    sock.Close();
    LOG(FATAL) << "URL " << url << ":" << port
               << " server already have key=" << key;
  } else if (code != kRPCMagic) {
    sock.Close();
    LOG(FATAL) << "URL " << url << ":" << port << " is not TVM RPC server";
  }
  CHECK_EQ(sock.RecvAll(&keylen, sizeof(keylen)), sizeof(keylen));
  std::string remote_key;
  if (keylen != 0) {
    remote_key.resize(keylen);
    CHECK_EQ(sock.RecvAll(&remote_key[0], keylen), keylen);
  }
  return RPCSession::Create(
      std::unique_ptr<SockChannel>(new SockChannel(sock)), key, remote_key);
}

Module RPCClientConnect(std::string url, int port, std::string key) {
  return CreateRPCModule(RPCConnect(url, port, "client:" + key));
}

void RPCServerLoop(int sockfd) {
  common::TCPSocket sock(
      static_cast<common::TCPSocket::SockType>(sockfd));
  RPCSession::Create(
      std::unique_ptr<SockChannel>(new SockChannel(sock)),
      "SockServerLoop", "")->ServerLoop();
}

TVM_REGISTER_GLOBAL("rpc._Connect")
.set_body_typed(RPCClientConnect);

TVM_REGISTER_GLOBAL("rpc._ServerLoop")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    RPCServerLoop(args[0]);
  });
}  // namespace runtime
}  // namespace tvm
