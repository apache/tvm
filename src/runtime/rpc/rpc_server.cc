/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_server.cc
 * \brief RPC Server implementation.
 */
#include <future>
#include <iostream>
#include <thread>
#include <sys/select.h>

#include <tvm/runtime/registry.h>
#include "../../common/socket.h"
#include "rpc_session.h"
#include "rpc_server_env.h"
#include "rpc_socket_impl.h"
#include "rpc_base.h"

namespace tvm {
namespace runtime {


void AcceptConnection(common::TCPSocket listen_sock,
                      int port,
                      common::TCPSocket tracker_conn,
                      std::string rpc_key,
                      std::string custom_addr,
                      common::TCPSocket &conn,
                      common::SockAddr &addr,
                      std::string &opts,
                      int ping_period=2) {

  /*
  Accept connection from the other places.
  listen_sock: Socket
      The socket used by listening process.
  tracker_conn : connnection to tracker
      Tracker connection
  ping_period : float, optional
      ping tracker every k seconds if no connection is accepted.
  */
  std::set <std::string> old_keyset;

  std::string matchkey;

  // Report resource to tracker
  if (tracker_conn.sockfd != common::TCPSocket::INVALID_SOCKET) {
    matchkey = RandomKey(rpc_key + ":", old_keyset);
    if (custom_addr.empty()) {
      custom_addr = "null";
    }

    std::ostringstream ss;
    ss << "[" << int(TrackerCode::kPut) << ", \"" << rpc_key << "\", [" << port << ", \""
       << matchkey << "\"], " << custom_addr << "]" ;

    SendData(tracker_conn, ss.str());

    //Receive status and validate
    std::string remote_status = RecvData(tracker_conn);
    CHECK_EQ(std::stoi(remote_status), int(TrackerCode::kSuccess));

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

      //FD_ISSET(listen_sock.sockfd, &readfds)
      if ((ready <= 0) || (!FD_ISSET(listen_sock.sockfd, &readfds))) {

        std::ostringstream ss;
        ss << "[" << int(TrackerCode::kGetPendingMatchKeys) << "]";
        SendData(tracker_conn, ss.str());

        //Receive status and validate
        std::string pending_keys = RecvData(tracker_conn);
        old_keyset.insert(matchkey);

        //if match key not in pending key set
        //it means the key is acquired by a client but not used.
        if (pending_keys.find(matchkey) == std::string::npos) {
            unmatch_period_count += 1;
        } else {
            unmatch_period_count = 0;
        }
        //regenerate match key if key is acquired but not used for a while
        if (unmatch_period_count * ping_period > unmatch_timeout + ping_period) {
          LOG(INFO) << "no incoming connections, regenerate key ...";

          matchkey = RandomKey(rpc_key + ":", old_keyset);

          std::ostringstream ss;
          ss << "[" << int(TrackerCode::kPut) << ", \"" << rpc_key << "\", [" << port << ", \""
             << matchkey << "\"], " << custom_addr << "]" ;
          SendData(tracker_conn, ss.str());

          std::string remote_status = RecvData(tracker_conn);
          CHECK_EQ(std::stoi(remote_status), int(TrackerCode::kSuccess));
          unmatch_period_count = 0;
        }
        continue;
      }
    }
    conn = listen_sock.Accept(addr);

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
        LOG(WARNING) << "Mismatch key from" << addr.AsString();
        //continue;
    }
    else {
      code = kRPCSuccess;
      CHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
      keylen = server_key.length();
      CHECK_EQ(conn.SendAll(&keylen, sizeof(keylen)), sizeof(keylen));
      CHECK_EQ(conn.SendAll(server_key.c_str(), keylen), keylen);
      LOG(INFO) << "Connection success " << addr.AsString();
      ssin >> opts;
      return;
    }
  }
}

void ServerLoopProc(common::TCPSocket sock, common::SockAddr addr){
    //Server loop
    auto env = RPCEnv();
    RPCServerLoop(sock.sockfd);
    LOG(INFO) << "Finish serving " << addr.AsString();
    env.Remove();
}

void ListenLoopProc(common::TCPSocket sock,
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
            ss << "[" << int(TrackerCode::kUpdateInfo)
               << ", {\"key\": \"server:"<< rpc_key << "\"}]";
            SendData(tracker_conn, ss.str());

            //Receive status and validate
            std::string remote_status = RecvData(tracker_conn);
            CHECK_EQ(std::stoi(remote_status), int(TrackerCode::kSuccess));
          }
          // step 2: wait for in-coming connections
          AcceptConnection(sock, port, tracker_conn, rpc_key, custom_addr, conn, addr, opts);
        }
        catch (const char* msg) {//(socket.error, IOError):
            // retry when tracker is dropped
            if (tracker_conn.sockfd != common::TCPSocket::INVALID_SOCKET) {
              tracker_conn.Close();
            }
            continue;
        }
        catch (std::exception& e) {
              // Other errors
              std::cerr << e.what();
              printf("\nException standard\n");
        }

        //step 3: serving
        std::future<void> server_proc(std::async(std::launch::async, ServerLoopProc, conn, addr));
        server_proc.get();

        //# close from our side.
        conn.Close();
        //TODO status = future.wait_for
        //# wait until server process finish or timeout

        /*
        server_proc.join(opts.get("timeout", None))
        if server_proc.is_alive():
            logger.info("Timeout in RPC session, kill..")
            server_proc.terminate()
            print("step 5: ait until server process finish or timeout ")
        */
    }
}

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
    sock.Listen(1);
    std::future<void> proc(std::async(std::launch::async,
                                      ListenLoopProc,
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

