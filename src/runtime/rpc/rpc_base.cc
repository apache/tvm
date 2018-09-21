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
 * \brief IsNumber check whether string is a number.
 * \param str input string
 * \return result of operation.
 */
bool IsNumber(const std::string& str) {
  return !str.empty() &&
    (str.find_first_not_of("[0123456789]") == std::string::npos);
}

/*!
 * \brief split SplitString the string based on delimiter
 * \param str Input string
 * \param delim The delimiter.
 * \return vector of strings which are splitted.
 */
std::vector<std::string> SplitString(const std::string& str, char delim) {
    auto i = 0;
    std::vector<std::string> list;
    auto pos = str.find(delim);
    while (pos != std::string::npos) {
      list.push_back(str.substr(i, pos - i));
      i = ++pos;
      pos = str.find(delim, pos);
    }
    list.push_back(str.substr(i, str.length()));
    return list;
}

/*!
 * \brief ValidateIP validates an ip address.
 * \param ip The ip address in string format
 * \return result of operation.
 */
bool ValidateIP(std::string ip) {
    std::vector<std::string> list = SplitString(ip, '.');
    if (list.size() != 4)
        return false;
    for (std::string str : list) {
      if (!IsNumber(str) || std::stoi(str) > 255 || std::stoi(str) < 0)
        return false;
    }
    return true;
}

/*!
 * \brief ValidateTracker Check the tracker address format is correct and changes the format.
 * \param tracker The tracker input.
 * \return result of operation.
 */
bool ValidateTracker(std::string &tracker) {
  std::vector<std::string> list = SplitString(tracker, ':');
  if ((list.size() != 2) || (!ValidateIP(list[0])) || (!IsNumber(list[1]))) {
    return false;
  }
  std::ostringstream ss;
  ss << "('" << list[0] << "', " << list[1] << ")";
  tracker = ss.str();
  return true;
}

/*!
 * \brief GetSockAddr Get the socket address from tracker.
 * \param url The url containing the ip and port number. Format is ('192.169.1.100', 9090)
 * \return SockAddr parsed from url.
 */
common::SockAddr GetSockAddr(std::string tracker) {
  CHECK(ValidateTracker(tracker)) << "Tracker url is not valid";
  size_t sep = tracker.find(",");
  std::string host = tracker.substr(2, sep - 3);
  std::string port = tracker.substr(sep + 1, tracker.length() - 1);
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
