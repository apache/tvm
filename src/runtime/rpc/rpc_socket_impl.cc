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
 * \file rpc_socket_impl.cc
 * \brief Socket based RPC implementation.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <memory>
#include "rpc_endpoint.h"
#include "rpc_session.h"
#include "rpc_local_session.h"
#include "../../support/socket.h"

namespace tvm {
namespace runtime {

class SockChannel final : public RPCChannel {
 public:
  explicit SockChannel(support::TCPSocket sock)
      : sock_(sock) {}
  ~SockChannel() {
    try {
      // BadSocket can throw
      if (!sock_.BadSocket()) {
        sock_.Close();
      }
    } catch (...) {
    }
  }
  size_t Send(const void* data, size_t size) final {
    ssize_t n = sock_.Send(data, size);
    if (n == -1) {
      support::Socket::Error("SockChannel::Send");
    }
    return static_cast<size_t>(n);
  }
  size_t Recv(void* data, size_t size) final {
    ssize_t n = sock_.Recv(data, size);
    if (n == -1) {
      support::Socket::Error("SockChannel::Recv");
    }
    return static_cast<size_t>(n);
  }

 private:
  support::TCPSocket sock_;
};

std::shared_ptr<RPCEndpoint>
RPCConnect(std::string url, int port, std::string key, TVMArgs init_seq) {
  support::TCPSocket sock;
  support::SockAddr addr(url.c_str(), port);
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
  auto endpt = RPCEndpoint::Create(
      std::unique_ptr<SockChannel>(new SockChannel(sock)), key, remote_key);
  endpt->InitRemoteSession(init_seq);
  return endpt;
}

Module RPCClientConnect(std::string url,
                        int port,
                        std::string key,
                        TVMArgs init_seq) {
  auto endpt = RPCConnect(url, port, "client:" + key, init_seq);
  return CreateRPCSessionModule(CreateClientSession(endpt));
}

// TVM_DLL needed for MSVC
TVM_DLL void RPCServerLoop(int sockfd) {
  support::TCPSocket sock(
      static_cast<support::TCPSocket::SockType>(sockfd));
  RPCEndpoint::Create(
      std::unique_ptr<SockChannel>(new SockChannel(sock)),
      "SockServerLoop", "")->ServerLoop();
}

void RPCServerLoop(PackedFunc fsend,
                   PackedFunc frecv) {
  RPCEndpoint::Create(
      std::unique_ptr<CallbackChannel>(new CallbackChannel(fsend, frecv)),
      "SockServerLoop", "")->ServerLoop();
}

TVM_REGISTER_GLOBAL("rpc.Connect")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  std::string url = args[0];
  int port = args[1];
  std::string key = args[2];
  *rv = RPCClientConnect(
      url, port, key,
      TVMArgs(args.values + 3, args.type_codes + 3, args.size() - 3));
});

TVM_REGISTER_GLOBAL("rpc.ServerLoop")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args[0].type_code() == kDLInt) {
    RPCServerLoop(args[0]);
  } else {
    RPCServerLoop(
        args[0].operator tvm::runtime::PackedFunc(),
        args[1].operator tvm::runtime::PackedFunc());
  }
});

}  // namespace runtime
}  // namespace tvm
