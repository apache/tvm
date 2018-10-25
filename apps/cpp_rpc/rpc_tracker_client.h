/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_tracker_client.h
 * \brief RPC Tracker client to report resources.
 */
#ifndef TVM_APPS_CPP_RPC_TRACKER_CLIENT_H_
#define TVM_APPS_CPP_RPC_TRACKER_CLIENT_H_

#include <set>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <string>

#include "../../src/runtime/rpc/rpc_session.h"
#include "../../src/common/socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief TrackerClient Tracker client class.
 * \param tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \param key The key used to identify the device type in tracker. Default=""
 * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 */
class TrackerClient {
 public:
  /*!
   * \brief Constructor.
  */
  TrackerClient(const std::string &tracker_addr,
                const std::string &key,
                const std::string &custom_addr) {
    tracker_addr_ = tracker_addr;
    key_ = key;
    custom_addr_ = custom_addr;
  }
  /*!
   * \brief Destructor.
  */
  ~TrackerClient() {
    // Free the resources
    Close();
  }
  /*!
   * \brief IsValid Check tracker is valid.
  */
  bool IsValid() {
    return (!tracker_addr_.empty() && !tracker_sock_.IsClosed());
  }
  /*!
   * \brief TryConnect Connect to tracker if the tracker address is valid.
  */
  void TryConnect() {
    if (!tracker_addr_.empty() && (tracker_sock_.IsClosed())) {
      tracker_sock_ = ConnectWithRetry();

      int code = kRPCTrackerMagic;
      CHECK_EQ(tracker_sock_.SendAll(&code, sizeof(code)), sizeof(code));
      CHECK_EQ(tracker_sock_.RecvAll(&code, sizeof(code)), sizeof(code));
      CHECK_EQ(code, kRPCTrackerMagic) << tracker_addr_.c_str() << " is not RPC Tracker";

      std::ostringstream ss;
      ss << "[" << static_cast<int>(TrackerCode::kUpdateInfo)
         << ", {\"key\": \"server:"<< key_ << "\"}]";
      tracker_sock_.SendBytes(ss.str());

      // Receive status and validate
      std::string remote_status = tracker_sock_.RecvBytes();
      CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
    }
  }
  /*!
   * \brief Close Clean up tracker resources.
  */
  void Close() {
    // close tracker resource
    if (!tracker_sock_.IsClosed()) {
      tracker_sock_.Close();
    }
  }
 /*!
  * \brief ReportResourceAndGetKey Report resource to tracker.
  * \param port listening port.
  * \param matchkey Random match key output.
 */
  void ReportResourceAndGetKey(int port,
                               std::string *matchkey) {
    if (!tracker_sock_.IsClosed()) {
      *matchkey = RandomKey(key_ + ":", old_keyset_);
      if (custom_addr_.empty()) {
        custom_addr_ = "null";
      }

      std::ostringstream ss;
      ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << key_ << "\", ["
         << port << ", \"" << *matchkey << "\"], " << custom_addr_ << "]";

      tracker_sock_.SendBytes(ss.str());

      // Receive status and validate
      std::string remote_status = tracker_sock_.RecvBytes();
      CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
    } else {
        *matchkey = key_;
    }
  }

  /*!
   * \brief ReportResourceAndGetKey Report resource to tracker.
   * \param listen_sock Listen socket details for select.
   * \param port listening port.
   * \param ping_period Select wait time.
   * \param matchkey Random match key output.
  */
  void WaitConnectionAndUpdateKey(common::TCPSocket listen_sock,
                                  int port,
                                  int ping_period,
                                  std::string *matchkey) {
    int unmatch_period_count = 0;
    int unmatch_timeout = 4;
    while (1) {
      if (!tracker_sock_.IsClosed()) {
        common::SelectHelper selecter;
        selecter.WatchRead(listen_sock.sockfd);

        int ready = selecter.Select(ping_period * 1000);
        if ((ready <= 0) || (!selecter.CheckRead(listen_sock.sockfd))) {
          std::ostringstream ss;
          ss << "[" << int(TrackerCode::kGetPendingMatchKeys) << "]";
          tracker_sock_.SendBytes(ss.str());

          // Receive status and validate
          std::string pending_keys = tracker_sock_.RecvBytes();
          old_keyset_.insert(*matchkey);

          // if match key not in pending key set
          // it means the key is acquired by a client but not used.
          if (pending_keys.find(*matchkey) == std::string::npos) {
              unmatch_period_count += 1;
          } else {
              unmatch_period_count = 0;
          }
          // regenerate match key if key is acquired but not used for a while
          if (unmatch_period_count * ping_period > unmatch_timeout + ping_period) {
            LOG(INFO) << "no incoming connections, regenerate key ...";

            *matchkey = RandomKey(key_ + ":", old_keyset_);

            std::ostringstream ss;
            ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << key_ << "\", ["
               << port << ", \"" << *matchkey << "\"], " << custom_addr_ << "]";
            tracker_sock_.SendBytes(ss.str());

            std::string remote_status = tracker_sock_.RecvBytes();
            CHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
            unmatch_period_count = 0;
          }
          continue;
        }
      }
      break;
    }
  }

 private:
  /*!
   * \brief Connect to a RPC address with retry.
            This function is only reliable to short period of server restart.
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

  std::string tracker_addr_;
  std::string key_;
  std::string custom_addr_;
  common::TCPSocket tracker_sock_;
  std::set <std::string> old_keyset_;
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_CPP_RPC_TRACKER_CLIENT_H_
