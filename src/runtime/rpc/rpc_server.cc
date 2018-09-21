/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_server.cc
 * \brief RPC Server implementation.
 */

#include <tvm/runtime/registry.h>

#include <sys/select.h>
#include <set>
#include <iostream>
#include <future>
#include <thread>

#include "rpc_server.h"
#include "rpc_session.h"
#include "rpc_server_env.h"
#include "rpc_socket_impl.h"
#include "rpc_base.h"
#include "../../common/socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief acceptConnection Accepts the RPC Server connection.
 * \param listen_sock The listen socket information.
 * \param port The port of the RPC.
 * \param tracker_conn The tracker connection information.
 * \param rpc_key The key used to identify the device type in tracker.
 * \param custom_addr Custom IP Address to Report to RPC Tracker.
 * \param conn New connection information.
 * \param addr New connection address information.
 * \param opts Parsed options for socket
 * \param ping_period Timeout for select call waiting
 */
void acceptConnection(common::TCPSocket listen_sock,
                      int port,
                      common::TCPSocket tracker_conn,
                      std::string rpc_key,
                      std::string custom_addr,
                      common::TCPSocket *conn_sock,
                      common::SockAddr *addr,
                      std::string *opts,
                      int ping_period = 2) {
  std::set <std::string> old_keyset;

  std::string matchkey;

  // Report resource to tracker
  if (tracker_conn.sockfd != common::TCPSocket::INVALID_SOCKET) {
    matchkey = RandomKey(rpc_key + ":", old_keyset);
    if (custom_addr.empty()) {
      custom_addr = "null";
    }

    std::ostringstream ss;
    ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << rpc_key << "\", ["
       << port << ", \"" << matchkey << "\"], " << custom_addr << "]";

    SendData(tracker_conn, ss.str());

    // Receive status and validate
    std::string remote_status = RecvData(tracker_conn);
    CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
  } else {
      matchkey = rpc_key;
  }

  int unmatch_period_count = 0;
  int unmatch_timeout = 4;
  while (1) {
    if (tracker_conn.sockfd != common::TCPSocket::INVALID_SOCKET) {
      fd_set readfds;
      FD_ZERO(&readfds);
      FD_SET(listen_sock.sockfd, &readfds);
      int maxfd = listen_sock.sockfd + 1;

      struct timeval tv = {ping_period, 0};

      int ready = select(maxfd, &readfds, nullptr, nullptr, &tv);
      if ((ready <= 0) || (!FD_ISSET(listen_sock.sockfd, &readfds))) {
        std::ostringstream ss;
        ss << "[" << int(TrackerCode::kGetPendingMatchKeys) << "]";
        SendData(tracker_conn, ss.str());

        // Receive status and validate
        std::string pending_keys = RecvData(tracker_conn);
        old_keyset.insert(matchkey);

        // if match key not in pending key set
        // it means the key is acquired by a client but not used.
        if (pending_keys.find(matchkey) == std::string::npos) {
            unmatch_period_count += 1;
        } else {
            unmatch_period_count = 0;
        }
        // regenerate match key if key is acquired but not used for a while
        if (unmatch_period_count * ping_period > unmatch_timeout + ping_period) {
          LOG(INFO) << "no incoming connections, regenerate key ...";

          matchkey = RandomKey(rpc_key + ":", old_keyset);

          std::ostringstream ss;
          ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << rpc_key << "\", ["
             << port << ", \"" << matchkey << "\"], " << custom_addr << "]";
          SendData(tracker_conn, ss.str());

          std::string remote_status = RecvData(tracker_conn);
          CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
          unmatch_period_count = 0;
        }
        continue;
      }
    }

    common::TCPSocket conn = listen_sock.Accept(addr);

    int code = kRPCMagic;
    CHECK_EQ(conn.RecvAll(&code, sizeof(code)), sizeof(code));
    if (code != kRPCMagic) {
      conn.Close();
      LOG(FATAL) << "Client connected is not TVM RPC server";
      continue;
    }

    int keylen = 0;
    CHECK_EQ(conn.RecvAll(&keylen, sizeof(keylen)), sizeof(keylen));

    #define CLIENT_HEADER "client:"
    #define SERVER_HEADER "server:"

    std::string expect_header = CLIENT_HEADER + matchkey;
    std::string server_key = SERVER_HEADER + rpc_key;
    if (size_t(keylen) < expect_header.length()) {
      conn.Close();
      LOG(FATAL) << "Wrong client header length";
      continue;
    }

    std::string remote_key;
    remote_key.resize(keylen);
    CHECK_EQ(conn.RecvAll(&remote_key[0], keylen), keylen);

    std::stringstream ssin(remote_key);
    std::string arg0;
    ssin >> arg0;
    if (arg0 != expect_header) {
        code = kRPCMismatch;
        CHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
        conn.Close();
        LOG(WARNING) << "Mismatch key from" << addr->AsString();
        continue;
    } else {
      code = kRPCSuccess;
      CHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
      keylen = server_key.length();
      CHECK_EQ(conn.SendAll(&keylen, sizeof(keylen)), sizeof(keylen));
      CHECK_EQ(conn.SendAll(server_key.c_str(), keylen), keylen);
      LOG(INFO) << "Connection success " << addr->AsString();
      ssin >> *opts;
      *conn_sock = conn;
      return;
    }
  }
}

/*!
 * \brief serverLoopProc The Server loop process.
 * \param sock The socket information
 * \param addr The socket address information
 */
void serverLoopProc(common::TCPSocket sock, common::SockAddr addr) {
    // Server loop
    auto env = RPCEnv();
    RPCServerLoop(sock.sockfd);
    LOG(INFO) << "Finish serving " << addr.AsString();
    env.Remove();
}

/*!
 * \brief getTimeOutFromOpts Parse and get the timeout option.
 * \param opts The option string
 * \param timeout value after parsing.
 */
int getTimeOutFromOpts(std::string opts) {
  std::string cmd;
  std::string option = "--timeout=";

  if (opts.find(option) == 0) {
    cmd = opts.substr(opts.find_last_of(option) + 1);
    CHECK(IsNumber(cmd)) << "Timeout is not valid";
    return std::stoi(cmd);
  }
  return 0;
}

/*!
 * \brief listenLoopProc The listen process.
 * \param sock The socket information
 * \param port The port of the RPC
 * \param tracker The address of RPC tracker in host:port format.
 * \param key The key used to identify the device type in tracker
 * \param custom_addr Custom IP Address to Report to RPC Tracker.
 */
void listenLoopProc(common::TCPSocket sock,
                    int port,
                    std::string tracker_addr,
                    std::string rpc_key,
                    std::string custom_addr) {
  common::TCPSocket tracker_conn;
  while (1) {
    common::TCPSocket conn;
    common::SockAddr addr("0.0.0.0", 0);
    std::string opts;
    try {
      // step 1: setup tracker and report to tracker
      if (!tracker_addr.empty() &&
        (tracker_conn.sockfd == common::TCPSocket::INVALID_SOCKET)) {
        tracker_conn = ConnectWithRetry(tracker_addr);

        int code = kRPCTrackerMagic;
        CHECK_EQ(tracker_conn.SendAll(&code, sizeof(code)), sizeof(code));
        CHECK_EQ(tracker_conn.RecvAll(&code, sizeof(code)), sizeof(code));
        CHECK_EQ(code, kRPCTrackerMagic) << tracker_addr.c_str() << " is not RPC Tracker";

        std::ostringstream ss;
        ss << "[" << static_cast<int>(TrackerCode::kUpdateInfo)
           << ", {\"key\": \"server:"<< rpc_key << "\"}]";
        SendData(tracker_conn, ss.str());

        // Receive status and validate
        std::string remote_status = RecvData(tracker_conn);
        CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
      }
      // step 2: wait for in-coming connections
      acceptConnection(sock, port, tracker_conn, rpc_key, custom_addr, &conn, &addr, &opts);
    }
    catch (const char* msg) {  // (socket.error, IOError):
        // retry when tracker is dropped
        if (tracker_conn.sockfd != common::TCPSocket::INVALID_SOCKET) {
          tracker_conn.Close();
        }
        continue;
    }
    catch (std::exception& e) {
          // Other errors
          LOG(FATAL) << "Exception standard: " << e.what();
    }

    // step 3: serving
    std::future<void> server_proc(std::async(std::launch::async, serverLoopProc, conn, addr));
    // wait until server process finish or timeout
    int timeout = getTimeOutFromOpts(opts);
    if (timeout) {
      // Autoterminate after timeout
      server_proc.wait_for(std::chrono::seconds(timeout));
    } else {
      // Wait for the result
      server_proc.get();
    }
    // close from our side.
    conn.Close();
  }
}

/*!
 * \brief RPCServerCreate Creates the RPC Server.
 * \param host The hostname of the server, Default=0.0.0.0
 * \param port The port of the RPC, Default=9090
 * \param port_end The end search port of the RPC, Default=9199
 * \param tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \param key The key used to identify the device type in tracker. Default=""
 * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 * \param silent Whether run in silent mode. Default=True
 * \param isProxy Whether to run in proxy mode. Default=False
 */
void RPCServerCreate(std::string host,
                     int port,
                     int port_end,
                     std::string tracker_addr,
                     std::string key,
                     std::string custom_addr,
                     bool silent,
                     bool is_proxy) {
    common::TCPSocket sock;
    common::SockAddr addr(host.c_str(), port);
    sock.Create();
    int my_port = sock.TryBindHost(host, port, port_end);
    LOG(INFO) << "bind to " << host << ":" << my_port;
    sock.Listen(1);
    std::future<void> proc(std::async(std::launch::async,
                                      listenLoopProc,
                                      sock,
                                      my_port,
                                      tracker_addr,
                                      key,
                                      custom_addr));
    proc.get();
    sock.Close();
}
}  // namespace runtime
}  // namespace tvm

