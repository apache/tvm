/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_socket_impl.cc
 * \brief Socket based RPC implementation.
 */
#include <tvm/runtime/registry.h>
#include <memory>
#include "./rpc_session.h"
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

Module RPCConnect(std::string url, int port, std::string key) {
  common::TCPSocket sock;
  common::SockAddr addr(url.c_str(), port);
  sock.Create();
  CHECK(sock.Connect(addr))
      << "Connect to " << addr.AsString() << " failed";
  // hand shake
  std::ostringstream os;
  os << "client:" << key;
  key = os.str();
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
               << " server already have client key=" << key;
  } else if (code != kRPCMagic) {
    sock.Close();
    LOG(FATAL) << "URL " << url << ":" << port << " is not TVM RPC server";
  }
  return CreateRPCModule(
      RPCSession::Create(
          std::unique_ptr<SockChannel>(new SockChannel(sock)),
          "SockClient"));
}

void RPCServerLoop(int sockfd) {
  common::TCPSocket sock(
      static_cast<common::TCPSocket::SockType>(sockfd));
  RPCSession::Create(
      std::unique_ptr<SockChannel>(new SockChannel(sock)),
                     "SockServerLoop")->ServerLoop();
}

TVM_REGISTER_GLOBAL("contrib.rpc._Connect")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = RPCConnect(args[0], args[1], args[2]);
  });

TVM_REGISTER_GLOBAL("contrib.rpc._ServerLoop")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    RPCServerLoop(args[0]);
  });
}  // namespace runtime
}  // namespace tvm
