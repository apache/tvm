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
 * \file rpc_tracker_server.h
 * \brief RPC Tracker implementation.
 */
#ifndef TVM_APPS_CPP_RPC_TRACKER_SERVER_H_
#define TVM_APPS_CPP_RPC_TRACKER_SERVER_H_

#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../../src/runtime/rpc/rpc_endpoint.h"
#include "../../src/runtime/rpc/rpc_socket_impl.h"
#include "../../src/support/socket.h"

using namespace tvm::support;

namespace tvm {
namespace runtime {

/*! \brief Server resource entry for active TVM RPC Server node */
struct ServerInfo {
  std::string addr;
  int port;
  std::string matchkey;
  std::string key;
  double last_ping_time;
  TCPSocket control_sock;
};

/*! \brief Client request entry for server node availablility */
struct RequestInfo {
  std::string key;
  std::string user;
  int priority;
  TCPSocket client_sock;
};

using ResourceData = std::tuple<std::string, std::string, std::string>;
using ResourceCallback = std::function<bool(const ResourceData&)>;

/*!
 * \brief Scheduler base class.
 */
class Scheduler {
 public:
  virtual ~Scheduler() = default;
  virtual void put(const ResourceData& value) = 0;
  virtual void request(const std::string& user, int priority, ResourceCallback callback) = 0;
  virtual void remove(const ResourceData& value) = 0;
  virtual std::string summary() = 0;
};

/*!
 * \brief PriorityScheduler class.
 * \param key The key used to identify the device type in tracker.
 */
class PriorityScheduler : public Scheduler {
 public:
  explicit PriorityScheduler(const std::string& key);

  void put(const ResourceData& value) override;
  void request(const std::string& user, int priority, ResourceCallback callback) override;
  void remove(const ResourceData& value) override;
  std::string summary() override;

 private:
  struct Request {
    int priority;
    int order;
    ResourceCallback callback;

    bool operator<(const Request& other) const {
      if (priority != other.priority) return priority < other.priority;
      return order > other.order;
    }
  };

  void _schedule();

  std::string _key;
  std::vector<ResourceData> _values;
  std::priority_queue<Request> _requests;

  std::mutex _lock;
  int _request_cnt;
};

/*!
 * \brief The main rpc tracker server class.
 * Manages resource registration, routing, and distribution schedules.
 */
class RPCTracker {
 public:
  RPCTracker(const std::string& host, int port, int port_end, bool silent);
  ~RPCTracker();

  void Start();
  void Stop();

 private:
  std::string host_;
  int port_start_;
  int port_end_;
  int listen_port_;
  bool silent_;

  bool is_running_{false};
  TCPSocket server_sock_;

  std::mutex mutex_;
  std::thread schedule_thread_;
  std::condition_variable scheduler_cv_;

  std::unordered_map<int, std::unique_ptr<TCPSocket>> active_connections_;
  std::map<std::string, std::vector<RequestInfo>> client_queues_;
  std::map<std::string, std::vector<ServerInfo>> server_resources_;

  void ListenLoop();
  void ScheduleLoop();
  void HandleConnection(TCPSocket client_sock, SockAddr addr);

  bool ReadMessage(TCPSocket& sock, std::string* out_data);
  bool WriteMessage(TCPSocket& sock, const std::string& data);
  bool WriteCode(TCPSocket& sock, TrackerCode code);

  void ProcessPut(TCPSocket& sock, const std::string& client_ip,
                  const std::vector<std::string>& args);
  void ProcessRequest(TCPSocket& sock, const std::vector<std::string>& args);
  void ProcessUpdateInfo(TCPSocket& sock, const std::string& client_ip,
                         const std::vector<std::string>& args);
  void ProcessGetPendingMatches(TCPSocket& sock);
  void ProcessSummary(TCPSocket& sock);

  std::string _addr;
  std::unordered_map<std::string, std::string> _info;
  std::set<std::string> _connections;

  std::set<std::string> pending_matchkeys;
  std::vector<ResourceData> put_values;
  std::unordered_map<std::string, std::shared_ptr<Scheduler>> _scheduler_map;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_APPS_CPP_RPC_TRACKER_SERVER_H_
