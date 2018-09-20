/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_base.h
 * \brief Base definitions for RPC.
 */
#ifndef TVM_RUNTIME_RPC_RPC_BASE_H_
#define TVM_RUNTIME_RPC_RPC_BASE_H_

#include <iostream>
#include <set>

#include "../../common/socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Send the data to remote.
 * \param sock The socket.
 * \param data The data to be sent.
 */
void SendData(common::TCPSocket sock, std::string data);

/*!
 * \brief Receive the data to remote.
 * \param sock The socket.
 * \return The data received.
 */
std::string RecvData(common::TCPSocket sock);

/*!
 * \brief Generate a random key.
 * \param prefix The string prefix.
 * \return cmap The conflict map set.
 */
std::string RandomKey(std::string prefix, std::set <std::string> cmap);

/*!
 * \brief Get the socket address from url.
 * \param url The url containing the ip and port number. Format is ('192.169.1.100', 9090)
 * \return SockAddr parsed from url.
 */
common::SockAddr GetSockAddr(std::string url);

/*!
 * \brief Connect to a TPC address with retry.
          This function is only reliable to short period of server restart.
 * \param url The ipadress and port number
 * \param timeout Timeout during retry
 * \param retry_period Number of seconds before we retry again.
 * \return TCPSocket The socket information if connect is success.
 */
common::TCPSocket ConnectWithRetry(std::string url, int timeout=60, int retry_period=5);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_BASE_H_