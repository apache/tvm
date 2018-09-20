/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_base.cc
 * \brief Base definitions for RPC.
 */
#include <iostream>
#include <set>
#include <chrono>
#include <thread>

#include "../../common/socket.h"
#include "rpc_base.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Send the data to remote.
 * \param sock The socket.
 * \param data The data to be sent.
 */
void SendData(common::TCPSocket sock, std::string data) {
  int datalen = data.length();
  CHECK_EQ(sock.SendAll(&datalen, sizeof(datalen)), sizeof(datalen));
  CHECK_EQ(sock.SendAll(data.c_str(), datalen), datalen);
}

/*!
 * \brief Receive the data to remote.
 * \param sock The socket.
 * \return The data received.
 */
std::string RecvData(common::TCPSocket sock) {
  int datalen = 0;
  CHECK_EQ(sock.RecvAll(&datalen, sizeof(datalen)), sizeof(datalen));
  std::string data;
  data.resize(datalen);
  CHECK_EQ(sock.RecvAll(&data[0], datalen), datalen);
  return data;
}

/*!
 * \brief Generate a random key.
 * \param prefix The string prefix.
 * \return cmap The conflict map set.
 */
std::string RandomKey(std::string prefix, std::set <std::string> cmap) {
  float r;
  if (!cmap.empty()) {
    r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    while(1) {
      std::string key = prefix + std::to_string(r);
      if (cmap.find(key) == cmap.end()) {
        return key;
      }
    }
  }
  r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  return prefix + std::to_string(r);
}


/*!
 * \brief Get the socket address from url.
 * \param url The url containing the ip and port number. Format is ('192.169.1.100', 9090)
 * \return SockAddr parsed from url.
 */
common::SockAddr GetSockAddr(std::string url) {
  size_t sep = url.find(",");
  std::string host = url.substr(2, sep - 3);
  std::string port = url.substr(sep + 1, url.length() - 1);
  if (host == "localhost") {
    host = "127.0.0.1";
  }
  return common::SockAddr(host.c_str(), std::stoi(port));
}

/*!
 * \brief Connect to a TPC address with retry.
          This function is only reliable to short period of server restart.
 * \param url The ipadress and port number
 * \param timeout Timeout during retry
 * \param retry_period Number of seconds before we retry again.
 * \return TCPSocket The socket information if connect is success.
 */
common::TCPSocket ConnectWithRetry(std::string url, int timeout, int retry_period) {
  auto tbegin = std::chrono::system_clock::now();
  while (1) {
    common::SockAddr addr = GetSockAddr(url);
    common::TCPSocket sock;
    sock.Create();
    LOG(INFO) << "Tracker connecting to " << addr.AsString();
    if (sock.Connect(addr)) {
      return sock;
    }

    auto period = (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - tbegin)).count();
    CHECK(period < timeout) << "Failed to connect to server" << addr.AsString();
    LOG(WARNING) << "Cannot connect to tracker " << addr.AsString()
                 << " retry in " << retry_period << " seconds.";
    std::this_thread::sleep_for(std::chrono::seconds(retry_period));
   }
}

}  // namespace runtime
}  // namespace tvm
