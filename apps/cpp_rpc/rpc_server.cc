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
#include "rpc_env.h"
#include "rpc_tracker_client.h"
#include "../../src/runtime/rpc/rpc_session.h"
#include "../../src/runtime/rpc/rpc_socket_impl.h"
#include "../../src/common/socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief RPCServer RPC Server class.
 * \param host The hostname of the server, Default=0.0.0.0
 * \param port The port of the RPC, Default=9090
 * \param port_end The end search port of the RPC, Default=9199
 * \param tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \param key The key used to identify the device type in tracker. Default=""
 * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 * \param isProxy Whether to run in proxy mode. Default=False
 */
class RPCServer {
 public:
  /*!
   * \brief Constructor.
  */
  RPCServer(const std::string &host,
            int port,
            int port_end,
            const std::string &tracker_addr,
            const std::string &key,
            const std::string &custom_addr,
            bool is_proxy) {
    // Init the values
    host_ = host;
    port_ = port;
    port_end_ = port_end;
    tracker_addr_ = tracker_addr;
    key_ = key;
    custom_addr_ = custom_addr;
    is_proxy_ = is_proxy;
  }

  /*!
   * \brief Destructor.
  */
  ~RPCServer() {
    // Free the resources
    tracker_sock_.Close();
    listen_sock_.Close();
  }

  /*!
   * \brief Start Creates the RPC listen process and execution.
  */
  void Start() {
      listen_sock_.Create();
      my_port_ = listen_sock_.TryBindHost(host_, port_, port_end_);
      LOG(INFO) << "bind to " << host_ << ":" << my_port_;
      listen_sock_.Listen(1);
      std::future<void> proc(std::async(std::launch::async, &RPCServer::ListenLoopProc, this));
      proc.get();
      //Close the listen socket
      listen_sock_.Close();
  }

 private:
  /*!
   * \brief ListenLoopProc The listen process.
   */
  void ListenLoopProc() {
    TrackerClient tracker(tracker_addr_, key_, custom_addr_);
    while (1) {
      common::TCPSocket conn;
      common::SockAddr addr("0.0.0.0", 0);
      std::string opts;
      try {
        // step 1: setup tracker and report to tracker
        tracker.TryConnect();
        // step 2: wait for in-coming connections
        AcceptConnection(&tracker, &conn, &addr, &opts);
      }
      catch (const char* msg) {
        // close tracker resource
        tracker.Close();
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
    tracker.Close();
  }


  /*!
   * \brief AcceptConnection Accepts the RPC Server connection.
   * \param tracker Tracker details.
   * \param conn New connection information.
   * \param addr New connection address information.
   * \param opts Parsed options for socket
   * \param ping_period Timeout for select call waiting
   */
  void AcceptConnection(TrackerClient *tracker,
                        common::TCPSocket *conn_sock,
                        common::SockAddr *addr,
                        std::string *opts,
                        int ping_period = 2) {
    std::set <std::string> old_keyset;
    std::string matchkey;

    // Report resource to tracker and get key
    tracker->ReportResourceAndGetKey(my_port_, matchkey);

    while (1) {
      tracker->WaitConnectionAndUpdateKey(listen_sock_, my_port_, ping_period, matchkey);
      common::TCPSocket conn = listen_sock_.Accept(addr);

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

  std::string host_;
  int port_;
  int my_port_;
  int port_end_;
  std::string tracker_addr_;
  std::string key_;
  std::string custom_addr_;
  bool is_proxy_;
  common::TCPSocket listen_sock_;
  common::TCPSocket tracker_sock_;
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
  if (silent) {
    // Only errors and fatal is logged
    dmlc::InitLogging("--minloglevel=2");
  }
  // Start the rpc server
  RPCServer rpc(host, port, port_end, tracker_addr, key, custom_addr, is_proxy);
  rpc.Start();
}

TVM_REGISTER_GLOBAL("rpc._ServerCreate")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    RPCServerCreate(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
  });
}  // namespace runtime
}  // namespace tvm
