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
 * \file rpc_server.cc
 * \brief RPC Server implementation.
 */
#include <tvm/runtime/registry.h>
#if defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)
#include <signal.h>
#include <sys/select.h>
#include <sys/wait.h>
#endif
#include <chrono>
#include <future>
#include <iostream>
#include <set>
#include <string>
#include <thread>

#include "../../src/runtime/rpc/rpc_endpoint.h"
#include "../../src/runtime/rpc/rpc_socket_impl.h"
#include "../../src/support/socket.h"
#include "rpc_env.h"
#include "rpc_server.h"
#include "rpc_tracker_client.h"
#if defined(_WIN32)
#include "win32_process.h"
#endif

using namespace std::chrono;

namespace tvm {
namespace runtime {

/*!
 * \brief wait the child process end.
 * \param status status value
 */
#if defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)
static pid_t waitPidEintr(int* status) {
  pid_t pid = 0;
  while ((pid = waitpid(-1, status, 0)) == -1) {
    if (errno == EINTR) {
      continue;
    } else {
      perror("waitpid");
      abort();
    }
  }
  return pid;
}
#endif

#ifdef __ANDROID__
static std::string getNextString(std::stringstream* iss) {
  std::string str = iss->str();
  size_t start = iss->tellg();
  size_t len = str.size();
  // Skip leading spaces.
  while (start < len && isspace(str[start])) start++;

  size_t end = start;
  while (end < len && !isspace(str[end])) end++;

  iss->seekg(end);
  return str.substr(start, end - start);
}
#endif

/*!
 * \brief RPCServer RPC Server class.
 *
 * \param host The hostname of the server, Default=0.0.0.0
 *
 * \param port_search_start The low end of the search range for an
 *     available port for the RPC, Default=9090
 *
 * \param port_search_end The high search the search range for an
 *     available port for the RPC, Default=9099
 *
 * \param tracker The address of RPC tracker in host:port format
 *     (e.g. "10.77.1.234:9190")
 *
 * \param key The key used to identify the device type in tracker.
 *
 * \param custom_addr Custom IP Address to Report to RPC Tracker.
 */
class RPCServer {
 public:
  /*!
   * \brief Constructor.
   */
  RPCServer(std::string host, int port_search_start, int port_search_end, std::string tracker_addr,
            std::string key, std::string custom_addr, std::string work_dir)
      : host_(std::move(host)),
        port_search_start_(port_search_start),
        my_port_(0),
        port_search_end_(port_search_end),
        tracker_addr_(std::move(tracker_addr)),
        key_(std::move(key)),
        custom_addr_(std::move(custom_addr)),
        work_dir_(std::move(work_dir)) {}

  /*!
   * \brief Destructor.
   */
  ~RPCServer() {
    try {
      // Free the resources
      tracker_sock_.Close();
      listen_sock_.Close();
    } catch (...) {
    }
  }

  /*!
   * \brief Start Creates the RPC listen process and execution.
   */
  void Start() {
    listen_sock_.Create();
    my_port_ = listen_sock_.TryBindHost(host_, port_search_start_, port_search_end_);
    LOG(INFO) << "bind to " << host_ << ":" << my_port_;
    listen_sock_.Listen(1);
    std::future<void> proc(std::async(std::launch::async, &RPCServer::ListenLoopProc, this));
    proc.get();
    // Close the listen socket
    listen_sock_.Close();
  }

 private:
  /*!
   * \brief ListenLoopProc The listen process.
   */
  void ListenLoopProc() {
    TrackerClient tracker(tracker_addr_, key_, custom_addr_, my_port_);
    while (true) {
      support::TCPSocket conn;
      support::SockAddr addr("0.0.0.0", 0);
      std::string opts;
      try {
        // step 1: setup tracker and report to tracker
        tracker.TryConnect();
        // step 2: wait for in-coming connections
        AcceptConnection(&tracker, &conn, &addr, &opts);
      } catch (const char* msg) {
        LOG(WARNING) << "Socket exception: " << msg;
        // close tracker resource
        tracker.Close();
        continue;
      } catch (const std::exception& e) {
        // close tracker resource
        tracker.Close();
        LOG(WARNING) << "Exception standard: " << e.what();
        continue;
      }

      int timeout = GetTimeOutFromOpts(opts);
#if defined(__linux__) || defined(__ANDROID__) || defined(__APPLE__)
      // step 3: serving
      if (timeout != 0) {
        const pid_t timer_pid = fork();
        if (timer_pid == 0) {
          // Timer process
          sleep(timeout);
          _exit(0);
        }

        const pid_t worker_pid = fork();
        if (worker_pid == 0) {
          // Worker process
          ServerLoopProc(conn, addr, work_dir_);
          _exit(0);
        }

        int status = 0;
        const pid_t finished_first = waitPidEintr(&status);
        if (finished_first == timer_pid) {
          kill(worker_pid, SIGTERM);
        } else if (finished_first == worker_pid) {
          kill(timer_pid, SIGTERM);
        } else {
          LOG(INFO) << "Child pid=" << finished_first << " unexpected, but still continue.";
        }

        int status_second = 0;
        waitPidEintr(&status_second);

        // Logging.
        if (finished_first == timer_pid) {
          LOG(INFO) << "Child pid=" << worker_pid << " killed (timeout = " << timeout
                    << "), Process status = " << status_second;
        } else if (finished_first == worker_pid) {
          LOG(INFO) << "Child pid=" << timer_pid << " killed, Process status = " << status_second;
        }
      } else {
        auto pid = fork();
        if (pid == 0) {
          ServerLoopProc(conn, addr, work_dir_);
          _exit(0);
        }
        // Wait for the result
        int status = 0;
        wait(&status);
        LOG(INFO) << "Child pid=" << pid << " exited, Process status =" << status;
      }
#elif defined(WIN32)
      auto start_time = high_resolution_clock::now();
      try {
        SpawnRPCChild(conn.sockfd, seconds(timeout));
      } catch (const std::exception&) {
      }
      auto dur = high_resolution_clock::now() - start_time;

      LOG(INFO) << "Serve Time " << duration_cast<milliseconds>(dur).count() << "ms";
#else
      LOG(WARNING) << "Unknown platform. It is not known how to bring up the subprocess."
                   << " RPC will be launched in the main thread.";
      ServerLoopProc(conn, addr, work_dir_);
#endif
      // close from our side.
      LOG(INFO) << "Socket Connection Closed";
      conn.Close();
    }
  }

  /*!
   * \brief AcceptConnection Accepts the RPC Server connection.
   * \param tracker Tracker details.
   * \param conn_sock New connection information.
   * \param addr New connection address information.
   * \param opts Parsed options for socket
   * \param ping_period Timeout for select call waiting
   */
  void AcceptConnection(TrackerClient* tracker, support::TCPSocket* conn_sock,
                        support::SockAddr* addr, std::string* opts, int ping_period = 2) {
    std::set<std::string> old_keyset;
    std::string matchkey;

    // Report resource to tracker and get key
    tracker->ReportResourceAndGetKey(my_port_, &matchkey);

    while (true) {
      tracker->WaitConnectionAndUpdateKey(listen_sock_, my_port_, ping_period, &matchkey);
      support::TCPSocket conn = listen_sock_.Accept(addr);

      int code = kRPCMagic;
      ICHECK_EQ(conn.RecvAll(&code, sizeof(code)), sizeof(code));
      if (code != kRPCMagic) {
        conn.Close();
        LOG(FATAL) << "Client connected is not TVM RPC server";
        continue;
      }

      int keylen = 0;
      ICHECK_EQ(conn.RecvAll(&keylen, sizeof(keylen)), sizeof(keylen));

      const char* CLIENT_HEADER = "client:";
      const char* SERVER_HEADER = "server:";
      std::string expect_header = CLIENT_HEADER + matchkey;
      std::string server_key = SERVER_HEADER + key_;
      if (size_t(keylen) < expect_header.length()) {
        conn.Close();
        LOG(INFO) << "Wrong client header length";
        continue;
      }

      ICHECK_NE(keylen, 0);
      std::string remote_key;
      remote_key.resize(keylen);
      ICHECK_EQ(conn.RecvAll(&remote_key[0], keylen), keylen);

      std::stringstream ssin(remote_key);
      std::string arg0;
#ifndef __ANDROID__
      ssin >> arg0;
#else
      arg0 = getNextString(&ssin);
#endif

      if (arg0 != expect_header) {
        code = kRPCMismatch;
        ICHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
        conn.Close();
        LOG(WARNING) << "Mismatch key from" << addr->AsString();
        continue;
      } else {
        code = kRPCSuccess;
        ICHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
        keylen = int(server_key.length());
        ICHECK_EQ(conn.SendAll(&keylen, sizeof(keylen)), sizeof(keylen));
        ICHECK_EQ(conn.SendAll(server_key.c_str(), keylen), keylen);
        LOG(INFO) << "Connection success " << addr->AsString();
#ifndef __ANDROID__
        ssin >> *opts;
#else
        *opts = getNextString(&ssin);
#endif
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
  static void ServerLoopProc(support::TCPSocket sock, support::SockAddr addr,
                             std::string work_dir) {
    // Server loop
    const auto env = RPCEnv(work_dir);
    RPCServerLoop(int(sock.sockfd));
    LOG(INFO) << "Finish serving " << addr.AsString();
    env.CleanUp();
  }

  /*!
   * \brief GetTimeOutFromOpts Parse and get the timeout option.
   * \param opts The option string
   */
  int GetTimeOutFromOpts(const std::string& opts) const {
    const std::string option = "-timeout=";

    size_t pos = opts.rfind(option);
    if (pos != std::string::npos) {
      const std::string cmd = opts.substr(pos + option.size());
      ICHECK(support::IsNumber(cmd)) << "Timeout is not valid";
      return std::stoi(cmd);
    }
    return 0;
  }

  std::string host_;
  int port_search_start_;
  int my_port_;
  int port_search_end_;
  std::string tracker_addr_;
  std::string key_;
  std::string custom_addr_;
  std::string work_dir_;
  support::TCPSocket listen_sock_;
  support::TCPSocket tracker_sock_;
};

#if defined(WIN32)
/*!
 * \brief ServerLoopFromChild The Server loop process.
 * \param socket The socket information
 */
void ServerLoopFromChild(SOCKET socket) {
  // Server loop
  tvm::support::TCPSocket sock(socket);
  const auto env = RPCEnv();
  RPCServerLoop(int(sock.sockfd));

  sock.Close();
  env.CleanUp();
}
#endif

/*!
 * \brief RPCServerCreate Creates the RPC Server.
 * \param host The hostname of the server, Default=0.0.0.0
 * \param port The port of the RPC, Default=9090
 * \param port_end The end search port of the RPC, Default=9099
 * \param tracker_addr The address of RPC tracker in host:port format e.g. 10.77.1.234:9190
 * Default="" \param key The key used to identify the device type in tracker. Default="" \param
 * custom_addr Custom IP Address to Report to RPC Tracker. Default="" \param silent Whether run in
 * silent mode. Default=True
 */
void RPCServerCreate(std::string host, int port, int port_end, std::string tracker_addr,
                     std::string key, std::string custom_addr, std::string work_dir, bool silent) {
  if (silent) {
    // Only errors and fatal is logged
    dmlc::InitLogging("--minloglevel=2");
  }
  // Start the rpc server
  RPCServer rpc(std::move(host), port, port_end, std::move(tracker_addr), std::move(key),
                std::move(custom_addr), std::move(work_dir));
  rpc.Start();
}

TVM_REGISTER_GLOBAL("rpc.ServerCreate").set_body([](TVMArgs args, TVMRetValue* rv) {
  RPCServerCreate(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
});
}  // namespace runtime
}  // namespace tvm
