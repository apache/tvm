/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_server.cc
 * \brief RPC Server implementation.
 */

#include <tvm/runtime/registry.h>

#if defined(__linux__)
#include <sys/select.h>
#include <sys/wait.h>
#endif
#include <set>
#include <iostream>
#include <future>
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <string>

#include "rpc_server.h"
#include "rpc_server_env.h"
#include "../../src/runtime/rpc/rpc_session.h"
#include "../../src/runtime/rpc/rpc_socket_impl.h"
#include "../../src/common/socket.h"

namespace tvm {
namespace runtime {

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
class RPCServer {
  public:
  RPCServer(std::string host, int port, int port_end, std::string tracker_addr, std::string key,
            std::string custom_addr, bool silent, bool is_proxy) {
    host_ = host;
    port_ = port;
    port_end_ = port_end;
    tracker_addr_ = tracker_addr;
    key_ = key;
    custom_addr_ = custom_addr;
    silent_ = silent;
    is_proxy_ = is_proxy;
  }

  /*!
   * \brief Start Creates the RPC Server.
  */
  void Start() {
      //common::SockAddr addr(host_.c_str(), port_);
      sock_.Create();
      my_port_ = sock_.TryBindHost(host_, port_, port_end_);
      LOG(INFO) << "bind to " << host_ << ":" << my_port_;
      sock_.Listen(1);
      std::future<void> proc(std::async(std::launch::async, &RPCServer::ListenLoopProc, this));
      proc.get();
      sock_.Close();
  }

  private:
  /*!
   * \brief ListenLoopProc The listen process.
   */
  void ListenLoopProc() {
    common::TCPSocket tracker_conn;
    while (1) {
      common::TCPSocket conn;
      common::SockAddr addr("0.0.0.0", 0);
      std::string opts;
      try {
        // step 1: setup tracker and report to tracker
        if (!tracker_addr_.empty() && (tracker_conn.IsClosed())) {
          tracker_conn = ConnectWithRetry();

          int code = kRPCTrackerMagic;
          CHECK_EQ(tracker_conn.SendAll(&code, sizeof(code)), sizeof(code));
          CHECK_EQ(tracker_conn.RecvAll(&code, sizeof(code)), sizeof(code));
          CHECK_EQ(code, kRPCTrackerMagic) << tracker_addr_.c_str() << " is not RPC Tracker";

          std::ostringstream ss;
          ss << "[" << static_cast<int>(TrackerCode::kUpdateInfo)
             << ", {\"key\": \"server:"<< key_ << "\"}]";
          tracker_conn.SendBytes(ss.str());

          // Receive status and validate
          std::string remote_status = tracker_conn.RecvBytes();
          CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
        }
        // step 2: wait for in-coming connections
        AcceptConnection(tracker_conn, &conn, &addr, &opts);
      }
      catch (const char* msg) {
        // retry when tracker is dropped
        if (!tracker_conn.IsClosed()) {
          tracker_conn.Close();
        }
        continue;
      }
      catch (std::exception& e) {
        // Other errors
        LOG(FATAL) << "Exception standard: " << e.what();
      }

      int timeout = GetTimeOutFromOpts(opts);
      #if defined(__linux__) || defined(__ANDROID__)
        // step 3: serving
        auto pid = fork();
        if (pid == 0) {
          ServerLoopProc(conn, addr);
          return;
        }
        // wait until server process finish or timeout
        if (timeout) {
          sleep(timeout);
          kill(pid, SIGTERM);  // Terminate after timeout
        } else {
          // Wait for the result
          int status = 0;
          wait(&status);
          LOG(INFO) << "Child pid=" << pid << " exited, Process status =" << status;
        }
      #else
        // step 3: serving
        std::future<void> proc(std::async(std::launch::async,
                                          &RPCServer::ServerLoopProc, this, conn, addr));
        // wait until server process finish or timeout
        if (timeout) {
          // Autoterminate after timeout
          proc.wait_for(std::chrono::seconds(timeout));
        } else {
          // Wait for the result
          proc.get();
        }
      #endif
      // close from our side.
      LOG(INFO) << "Socket Connection Closed";
      conn.Close();
    }
  }


  /*!
   * \brief AcceptConnection Accepts the RPC Server connection.
   * \param tracker_conn The tracker connection information.
   * \param conn New connection information.
   * \param addr New connection address information.
   * \param opts Parsed options for socket
   * \param ping_period Timeout for select call waiting
   */
  void AcceptConnection(common::TCPSocket tracker_conn,
                        common::TCPSocket *conn_sock,
                        common::SockAddr *addr,
                        std::string *opts,
                        int ping_period = 2) {
    std::set <std::string> old_keyset;

    std::string matchkey;

    // Report resource to tracker
    if (!tracker_conn.IsClosed()) {
      matchkey = RandomKey(key_ + ":", old_keyset);
      if (custom_addr_.empty()) {
        custom_addr_ = "null";
      }

      std::ostringstream ss;
      ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << key_ << "\", ["
         << my_port_ << ", \"" << matchkey << "\"], " << custom_addr_ << "]";

      tracker_conn.SendBytes(ss.str());

      // Receive status and validate
      std::string remote_status = tracker_conn.RecvBytes();
      CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
    } else {
        matchkey = key_;
    }

    int unmatch_period_count = 0;
    int unmatch_timeout = 4;
    while (1) {
      if (!tracker_conn.IsClosed()) {
        common::SelectHelper selecter;
        selecter.WatchRead(sock_.sockfd);

        int ready = selecter.Select(ping_period * 1000);
        if ((ready <= 0) || (!selecter.CheckRead(sock_.sockfd))) {
          std::ostringstream ss;
          ss << "[" << int(TrackerCode::kGetPendingMatchKeys) << "]";
          tracker_conn.SendBytes(ss.str());

          // Receive status and validate
          std::string pending_keys = tracker_conn.RecvBytes();
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

            matchkey = RandomKey(key_ + ":", old_keyset);

            std::ostringstream ss;
            ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << key_ << "\", ["
               << my_port_ << ", \"" << matchkey << "\"], " << custom_addr_ << "]";
            tracker_conn.SendBytes(ss.str());

            std::string remote_status = tracker_conn.RecvBytes();
            CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
            unmatch_period_count = 0;
          }
          continue;
        }
      }

      common::TCPSocket conn = sock_.Accept(addr);

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
      std::string server_key = SERVER_HEADER + key_;
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
   * \brief ServerLoopProc The Server loop process.
   * \param sock The socket information
   * \param addr The socket address information
   */
  void ServerLoopProc(common::TCPSocket sock, common::SockAddr addr) {
      // Server loop
      auto env = RPCEnv();
      RPCServerLoop(sock.sockfd);
      LOG(INFO) << "Finish serving " << addr.AsString();
      env.Remove();
  }

  /*!
   * \brief GetTimeOutFromOpts Parse and get the timeout option.
   * \param opts The option string
   * \param timeout value after parsing.
   */
  int GetTimeOutFromOpts(std::string opts) {
    std::string cmd;
    std::string option = "--timeout=";

    if (opts.find(option) == 0) {
      cmd = opts.substr(opts.find_last_of(option) + 1);
      CHECK(common::IsNumber(cmd)) << "Timeout is not valid";
      return std::stoi(cmd);
    }
    return 0;
  }

  /*!
   * \brief Connect to a TPC address with retry.
            This function is only reliable to short period of server restart.
   * \param url The ipadress and port number
   * \param timeout Timeout during retry
   * \param retry_period Number of seconds before we retry again.
   * \return TCPSocket The socket information if connect is success.
   */
  common::TCPSocket ConnectWithRetry(int timeout = 60, int retry_period = 5) {
    auto tbegin = std::chrono::system_clock::now();
    while (1) {
      common::SockAddr addr(tracker_addr_);
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

  /*!
  * \brief Random Generate a random number between 0 and 1.
  * \return random float value.
  */
  float Random() {
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
  }

  /*!
   * \brief Generate a random key.
   * \param prefix The string prefix.
   * \return cmap The conflict map set.
   */
  std::string RandomKey(std::string prefix, const std::set <std::string> &cmap) {
    if (!cmap.empty()) {
      while (1) {
        std::string key = prefix + std::to_string(Random());
        if (cmap.find(key) == cmap.end()) {
          return key;
        }
      }
    }
    return prefix + std::to_string(Random());
  }

  std::string host_;
  int port_;
  int my_port_;
  int port_end_;
  std::string tracker_addr_;
  std::string key_;
  std::string custom_addr_;
  bool silent_;
  bool is_proxy_;
  common::TCPSocket sock_;
};

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

    RPCServer rpc(host, port, port_end, tracker_addr, key, custom_addr, silent, is_proxy);
    rpc.Start();
}

TVM_REGISTER_GLOBAL("rpc._ServerCreate")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    RPCServerCreate(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
  });
}  // namespace runtime
}  // namespace tvm
