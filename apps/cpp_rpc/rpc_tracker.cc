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
 * \file rpc_tracker.cc
 * \brief RPC Tracker implementation.
 */
#include "../../apps/cpp_rpc/rpc_tracker.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <any>
#include <array>
#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../apps/cpp_rpc/rpc_tracker_server.h"

namespace tvm {
namespace runtime {

static std::string setToString(const std::set<std::string>& inputSet) {
  std::string result = "[";
  bool first = true;
  for (const auto& str : inputSet) {
    if (!first) result += ", ";
    result += "\"" + str + "\"";
    first = false;
  }

  return result + "]";
}

static std::string stripQuotes(const std::string& Str, char quote = '"') {
  if (Str.size() >= 2 && Str[0] == quote && Str[Str.size() - 1] == quote) {
    return Str.substr(1, Str.size() - 2);
  }

  return Str;
}

static std::vector<std::string> ParseArray(std::string Raw) {
  std::vector<std::string> elements;

  if (Raw.size() >= 2 && Raw.front() == '[' && Raw.back() == ']') {
    Raw = Raw.substr(1, Raw.size() - 2);
  }

  std::string current;
  int brace_level = 0;
  int bracket_level = 0;
  bool in_quotes = false;

  for (size_t i = 0; i < Raw.size(); ++i) {
    char c = Raw[i];

    if (c == '"') {
      in_quotes = !in_quotes;
    }

    if (!in_quotes) {
      if (c == '[')
        bracket_level++;
      else if (c == ']')
        bracket_level--;
      else if (c == '{')
        brace_level++;
      else if (c == '}')
        brace_level--;

      if (c == ',' && bracket_level == 0 && brace_level == 0) {
        if (!current.empty()) elements.push_back(current);
        current.clear();
        if (i + 1 < Raw.size() && Raw[i + 1] == ' ') i++;
        continue;
      }
    }
    current += c;
  }

  if (!current.empty()) elements.push_back(current);

  return elements;
}

static std::map<std::string, std::string> ParseJSONDict(std::string input) {
  std::map<std::string, std::string> result;
  size_t pos = 0;

  while ((pos = input.find("\"", pos)) != std::string::npos) {
    size_t key_start = ++pos;
    size_t key_end = input.find("\"", key_start);
    if (key_end == std::string::npos) break;

    std::string key = input.substr(key_start, key_end - key_start);
    pos = key_end + 1;

    pos = input.find(':', pos);
    if (pos == std::string::npos) break;
    pos++;

    while (pos < input.length() && (input[pos] == ' ' || input[pos] == '\t' || input[pos] == '\n'))
      pos++;

    size_t val_start = pos;
    size_t val_end = pos;

    if (input[pos] == '"') {
      val_start++;
      val_end = input.find('"', val_start);
      pos = val_end + 1;
    } else if (input[pos] == '[') {
      int depth = 1;
      val_end = val_start + 1;
      while (val_end < input.length() && depth > 0) {
        if (input[val_end] == '[')
          depth++;
        else if (input[val_end] == ']')
          depth--;
        val_end++;
      }
      pos = val_end;
    } else {
      val_end = input.find_first_of(",}", val_start);
      pos = val_end;
    }

    if (val_end != std::string::npos) {
      result[key] = input.substr(val_start, val_end - val_start);
    }
  }

  return result;
}

PriorityScheduler::PriorityScheduler(const std::string& key) : _key(key), _request_cnt(0) {}

void PriorityScheduler::_schedule() {
  std::lock_guard<std::mutex> lock(_lock);
  while (!_requests.empty() && !_values.empty()) {
    Request item = _requests.top();
    ResourceData value = _values.front();
    if (item.callback(value)) {
      _values.erase(_values.begin());
      _requests.pop();
    } else {
      break;
    }
  }
}

void PriorityScheduler::put(const ResourceData& value) {
  {
    std::lock_guard<std::mutex> lock(_lock);
    _values.push_back(value);
  }
  _schedule();
}

void PriorityScheduler::request(const std::string& user, int priority, ResourceCallback callback) {
  {
    std::lock_guard<std::mutex> lock(_lock);
    _requests.push({priority, _request_cnt++, callback});
  }
  _schedule();
}

void PriorityScheduler::remove(const ResourceData& value) {
  std::lock_guard<std::mutex> lock(_lock);
  auto it = std::find(_values.begin(), _values.end(), value);
  if (it != _values.end()) {
    _values.erase(it);
  }
}

std::string PriorityScheduler::summary() {
  std::lock_guard<std::mutex> lock(_lock);
  return "{\"free\": " + std::to_string(_values.size()) +
         ", \"pending\": " + std::to_string(_requests.size()) + "}";
}

RPCTracker::RPCTracker(const std::string& host, int port, int port_end, bool silent)
    : host_(host), port_start_(port), port_end_(port_end), silent_(silent) {
  _addr = host;
}

RPCTracker::~RPCTracker() { Stop(); }

void RPCTracker::Start() {
  is_running_ = true;
  server_sock_.Create();
  listen_port_ = server_sock_.TryBindHost(host_, port_start_, port_end_);
  if (listen_port_ < 0) {
    LOG(FATAL) << "Failed to bind to any port in range " << port_start_ << "-" << port_end_;
  }
  LOG(INFO) << "Bind to " << host_ << ":" << listen_port_;
  server_sock_.Listen(128);
  schedule_thread_ = std::thread(&RPCTracker::ScheduleLoop, this);
  ListenLoop();
  server_sock_.Close();
  is_running_ = false;
}

void RPCTracker::Stop() {
  if (!is_running_) return;
  is_running_ = false;
  if (!server_sock_.IsClosed()) {
    server_sock_.Close();
  }
  scheduler_cv_.notify_all();
  if (schedule_thread_.joinable()) schedule_thread_.join();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : active_connections_) {
      if (pair.second) {
        pair.second->Close();
      }
    }
    active_connections_.clear();
  }
  LOG(INFO) << "Tracker server stopped.";
}

void RPCTracker::ListenLoop() {
  while (is_running_) {
    SockAddr client_addr;
    TCPSocket client_sock = server_sock_.Accept(&client_addr);
    if (client_sock.IsClosed()) {
      if (!is_running_) break;
      continue;
    }
    LOG(INFO) << "New session from " << client_addr.AsString();
    std::thread([this, client_sock = std::move(client_sock), client_addr]() mutable {
      this->HandleConnection(std::move(client_sock), client_addr);
    }).detach();
  }
}

bool RPCTracker::ReadMessage(TCPSocket& sock, std::string* out_data) {
  int32_t msg_len = 0;
  if (sock.RecvAll(&msg_len, sizeof(msg_len)) != sizeof(msg_len)) return false;
  out_data->resize(msg_len);
  if (sock.RecvAll(&(*out_data)[0], msg_len) != static_cast<size_t>(msg_len)) return false;
  return true;
}

bool RPCTracker::WriteMessage(TCPSocket& sock, const std::string& data) {
  int32_t msg_len = static_cast<int32_t>(data.size());
  if (sock.SendAll(&msg_len, sizeof(msg_len)) != sizeof(msg_len)) return false;
  if (sock.SendAll(data.data(), msg_len) != static_cast<size_t>(msg_len)) return false;
  return true;
}

bool RPCTracker::WriteCode(TCPSocket& sock, TrackerCode code) {
  return WriteMessage(sock, std::to_string(static_cast<int>(code)));
}

void RPCTracker::HandleConnection(TCPSocket client_sock, SockAddr addr) {
  const auto remote_ip = addr.ipaddr();
  const auto remote_port = std::to_string(addr.port());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    int fd = client_sock.sockfd;
    auto socket_ptr = std::make_unique<TCPSocket>(std::move(client_sock));
    active_connections_[fd] = std::move(socket_ptr);
  }

  try {
    int32_t magic = 0;
    int n = client_sock.RecvAll(&magic, sizeof(magic));
    if (n != sizeof(magic) || magic != kRPCTrackerMagic) {
      LOG(INFO) << "Handshake with " << remote_ip << ":" << remote_port << " failed - Magic: 0x"
                << magic;
      client_sock.Close();
      return;
    }
    LOG(INFO) << "Handshake with " << remote_ip << ":" << remote_port << " successful";
    client_sock.SendAll(&magic, sizeof(magic));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      _connections.insert(remote_ip);
    }
  } catch (const std::exception& e) {
    TVM_FFI_THROW(InternalError) << "Connection handler error";
  } catch (...) {
    TVM_FFI_THROW(InternalError) << "Unknown error in connection handler";
  }

  while (is_running_) {
    std::string raw_message;
    if (!ReadMessage(client_sock, &raw_message)) break;

    const auto args = ParseArray(raw_message);
    if (args.empty()) break;

    const int code = std::stoi(args[0]);
    TrackerCode opcode = static_cast<TrackerCode>(code);

    if (opcode == TrackerCode::kPing) {
      WriteCode(client_sock, TrackerCode::kSuccess);
    } else if (opcode == TrackerCode::kPut) {
      ProcessPut(client_sock, remote_ip, args);
    } else if (opcode == TrackerCode::kRequest) {
      LOG(INFO) << "Request using key '" << stripQuotes(args[1]) << "'"
                << " from " << remote_ip << ":" << remote_port;
      ProcessRequest(client_sock, args);
    } else if (opcode == TrackerCode::kUpdateInfo) {
      LOG(INFO) << "Received key '" << ParseJSONDict(args[1])["key"] << "'"
                << " from " << remote_ip << ":" << remote_port;
      ProcessUpdateInfo(client_sock, remote_ip, args);
    } else if (opcode == TrackerCode::kGetPendingMatchKeys) {
      ProcessGetPendingMatches(client_sock);
    } else if (opcode == TrackerCode::kSummary) {
      LOG(INFO) << "Summary requested from " << remote_ip << ":" << remote_port;
      ProcessSummary(client_sock);
    }
  }

  LOG(INFO) << "End session with " << remote_ip << ":" << remote_port;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_connections_.erase(client_sock.sockfd);
    client_sock.Close();
  }
}

void RPCTracker::ProcessPut(TCPSocket& sock, const std::string& remote_ip,
                            const std::vector<std::string>& args) {
  if (args.size() < 4) {
    TVM_FFI_THROW(InternalError) << "Invalid ProcessPut arguments received";
  }

  const auto key = stripQuotes(args[1]);
  const auto data = ParseArray(args[2]);
  const auto ipaddr = stripQuotes(args[3]);
  const auto port = stripQuotes(data[0]);
  const auto matchkey = stripQuotes(data[1]);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_matchkeys.insert(matchkey);
  }

  ResourceData value;
  if ((args.size() >= 4) && (args[3] != "null")) {
    value = ResourceData{ipaddr, port, matchkey};
  } else {
    value = ResourceData{remote_ip, port, matchkey};
  }

  if (!_scheduler_map.count(key)) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!_scheduler_map.count(key)) {
      _scheduler_map[key] = std::make_shared<PriorityScheduler>(key);
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    _scheduler_map[key]->put(value);
    put_values.push_back(value);
  }

  WriteCode(sock, TrackerCode::kSuccess);
  scheduler_cv_.notify_all();
}

void RPCTracker::ProcessRequest(TCPSocket& sock, const std::vector<std::string>& args) {
  if (args.size() < 4) {
    TVM_FFI_THROW(InternalError) << "Invalid ProcessRequest arguments received";
  }

  const auto key = stripQuotes(args[1]);
  const auto user = stripQuotes(args[2]);
  const int prio = std::stoi(args[3]);

  ResourceCallback cb = [this, sock](const ResourceData& data) mutable -> bool {
    if (sock.GetSockError() || sock.BadSocket()) return false;

    const auto [ipaddr, port, matchkey] = data;
    std::string response = "[" + std::to_string(static_cast<int>(TrackerCode::kSuccess)) + ", [\"" +
                           ipaddr + "\", " + port + ", \"" + matchkey + "\"]]";

    auto remote_addr = sock.GetRemoteAddr();
    if (!remote_addr.has_value()) {
      TVM_FFI_THROW(InternalError) << "Socket has no valid remote address.";
    }
    LOG(INFO) << "Offering matchkey '" << matchkey << "@" << ipaddr << ":" << port << "'"
              << " to " << remote_addr.value().AsString();

    return this->WriteMessage(sock, response);
  };

  if (!_scheduler_map.count(key)) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!_scheduler_map.count(key)) {
      _scheduler_map[key] = std::make_shared<PriorityScheduler>(key);
    }
  }
  _scheduler_map[key]->request(user, prio, cb);

  scheduler_cv_.notify_all();
}

void RPCTracker::ProcessUpdateInfo(TCPSocket& sock, const std::string& remote_ip,
                                   const std::vector<std::string>& args) {
  if (args.size() != 2) {
    TVM_FFI_THROW(InternalError) << "Invalid ProcessUpdateInfo arguments received";
  }

  auto dict = ParseJSONDict(args[1]);
  const auto addr = ParseArray(dict["addr"]);

  auto ipaddr = remote_ip;
  if (addr[0] != "null") ipaddr = stripQuotes(addr[0]);
  std::string port = stripQuotes(addr[1]);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    _info[dict["key"]] =
        "{\"key\": \"" + dict["key"] + "\", \"addr\": [\"" + ipaddr + "\", \"" + port + "\"]}";
  }

  WriteCode(sock, TrackerCode::kSuccess);
  scheduler_cv_.notify_all();
}

void RPCTracker::ProcessGetPendingMatches(TCPSocket& sock) {
  std::string response = setToString(pending_matchkeys);
  WriteMessage(sock, response);
  scheduler_cv_.notify_all();
}

void RPCTracker::ProcessSummary(TCPSocket& sock) {
  std::string info =
      "[" + std::to_string(static_cast<int>(TrackerCode::kSuccess)) + ", {\"queue_info\": {";
  bool first = true;
  for (const auto& [key, sch_ptr] : _scheduler_map) {
    if (sch_ptr) {
      if (!first) info += ",";
      const auto summary = sch_ptr->summary();
      info += " \"" + key + "\": " + summary;
      first = false;
    }
  }

  info += "}, \"server_info\": [";

  first = true;
  for (const auto& [key, data] : _info) {
    if (!first) info += ", ";
    info += data;
    first = false;
  }

  info += "]}]";

  WriteMessage(sock, info);
  scheduler_cv_.notify_all();
}

void RPCTracker::ScheduleLoop() {
  while (is_running_) {
    std::unique_lock<std::mutex> lock(mutex_);
    scheduler_cv_.wait_for(lock, std::chrono::seconds(1));
    if (!is_running_) break;

    std::vector<std::string> keys;
    for (const auto& kv : client_queues_) keys.push_back(kv.first);

    for (const std::string& key : keys) {
      auto& clients = client_queues_[key];
      auto& servers = server_resources_[key];

      while (!clients.empty() && !servers.empty()) {
        RequestInfo client_req = clients.front();
        clients.erase(clients.begin());
        ServerInfo server_res = servers.front();
        servers.erase(servers.begin());

        bool notify_success = true;
        if (!server_res.control_sock.IsClosed()) {
          std::string payload = "[" + std::to_string(static_cast<int>(TrackerCode::kRequest)) +
                                ", \"" + server_res.matchkey + "\"]";
          if (!WriteMessage(server_res.control_sock, payload)) notify_success = false;
        }

        if (notify_success) {
          std::string payload = "[" + std::to_string(static_cast<int>(TrackerCode::kRequest)) +
                                ", [\"" + server_res.addr + "\", " +
                                std::to_string(server_res.port) + ", \"" + server_res.matchkey +
                                "\"]]";
          if (!WriteMessage(client_req.client_sock, payload)) {
            servers.insert(servers.begin(), server_res);
          }
        } else {
          clients.insert(clients.begin(), client_req);
        }
      }
    }
  }
}

/*!
 * \brief RPCTrackerCreate Creates the RPC Tracker.
 * \param host The listen address of the tracker, Default=0.0.0.0
 * \param port The port of the RPC tracker, Default=9190
 * \param port_end The end search port of the RPC tracker, Default=9199
 * silent mode. Default=True
 */
void RPCTrackerCreate(std::string host, int port, int port_end, bool silent) {
  RPCTracker tracker(std::move(host), port, port_end, silent);
  tracker.Start();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("rpc.TrackerCreate", RPCTrackerCreate);
}

}  // namespace runtime
}  // namespace tvm
