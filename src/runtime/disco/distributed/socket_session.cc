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
#include <tvm/runtime/registry.h>

#include <numeric>

#include "../../../support/socket.h"
#include "../bcast_session.h"
#include "../message_queue.h"

namespace tvm {
namespace runtime {

using namespace tvm::support;

enum class DiscoSocketAction {
  kShutdown = static_cast<int>(DiscoAction::kShutDown),
  kSend,
  kReceive,
};

class DiscoSocketChannel : public DiscoChannel {
 public:
  explicit DiscoSocketChannel(const TCPSocket& socket)
      : socket_(socket), message_queue_(&socket_) {}

  DiscoSocketChannel(DiscoSocketChannel&& other) = delete;
  DiscoSocketChannel(const DiscoSocketChannel& other) = delete;
  void Send(const TVMArgs& args) { message_queue_.Send(args); }
  TVMArgs Recv() { return message_queue_.Recv(); }
  void Reply(const TVMArgs& args) { message_queue_.Send(args); }
  TVMArgs RecvReply() { return message_queue_.Recv(); }

 private:
  TCPSocket socket_;
  DiscoStreamMessageQueue message_queue_;
};

class SocketSessionObj : public BcastSessionObj {
 public:
  explicit SocketSessionObj(int num_nodes, int num_workers_per_node, int num_groups,
                            const String& host, int port)
      : num_nodes_(num_nodes), num_workers_per_node_(num_workers_per_node) {
    const PackedFunc* f_create_local_session =
        Registry::Get("runtime.disco.create_socket_session_local_workers");
    ICHECK(f_create_local_session != nullptr)
        << "Cannot find function runtime.disco.create_socket_session_local_workers";
    local_session_ = ((*f_create_local_session)(num_workers_per_node)).AsObjectRef<BcastSession>();
    DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.socket_session_init_workers");
    local_session_->CallPacked(f_init_workers, num_nodes_, /*node_id=*/0, num_groups,
                               num_workers_per_node_);

    Socket::Startup();
    socket_.Create();
    socket_.SetKeepAlive(true);
    socket_.Bind(SockAddr(host.c_str(), port));
    socket_.Listen();
    LOG(INFO) << "SocketSession controller listening on " << host << ":" << port;

    TVMValue values[4];
    int type_codes[4];
    TVMArgsSetter setter(values, type_codes);
    setter(0, num_nodes);
    setter(1, num_workers_per_node);
    setter(2, num_groups);

    for (int i = 0; i + 1 < num_nodes; ++i) {
      SockAddr addr;
      remote_sockets_.push_back(socket_.Accept(&addr));
      remote_channels_.emplace_back(std::make_unique<DiscoSocketChannel>(remote_sockets_.back()));
      setter(3, i + 1);
      // Send metadata to each remote node:
      //  - num_nodes
      //  - num_workers_per_node
      //  - num_groups
      //  - node_id
      remote_channels_.back()->Send(TVMArgs(values, type_codes, 4));
      LOG(INFO) << "Remote node " << addr.AsString() << " connected";
    }
  }

  int64_t GetNumWorkers() final { return num_nodes_ * num_workers_per_node_; }

  TVMRetValue DebugGetFromRemote(int64_t reg_id, int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      return local_session_->DebugGetFromRemote(reg_id, worker_id);
    } else {
      std::vector<TVMValue> values(5);
      std::vector<int> type_codes(5);
      PackArgs(values.data(), type_codes.data(), static_cast<int>(DiscoSocketAction::kSend),
               worker_id, static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id, worker_id);

      remote_channels_[node_id - 1]->Send(TVMArgs(values.data(), type_codes.data(), values.size()));
      TVMArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 2);
      ICHECK(static_cast<DiscoAction>(args[0].operator int()) == DiscoAction::kDebugGetFromRemote);
      TVMRetValue result;
      result = args[1];
      return result;
    }
  }

  void DebugSetRegister(int64_t reg_id, TVMArgValue value, int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      local_session_->DebugSetRegister(reg_id, value, worker_id);
    } else {
      ObjectRef wrapped{nullptr};
      if (value.type_code() == kTVMNDArrayHandle || value.type_code() == kTVMObjectHandle) {
        wrapped = DiscoDebugObject::Wrap(value);
        TVMValue tvm_value;
        int type_code = kTVMObjectHandle;
        tvm_value.v_handle = const_cast<Object*>(wrapped.get());
        value = TVMArgValue(tvm_value, type_code);
      }
      {
        TVMValue values[6];
        int type_codes[6];
        PackArgs(values, type_codes, static_cast<int>(DiscoSocketAction::kSend), worker_id,
                 static_cast<int>(DiscoAction::kDebugSetRegister), reg_id, worker_id, value);
        remote_channels_[node_id - 1]->Send(TVMArgs(values, type_codes, 6));
      }
      TVMRetValue result;
      TVMArgs args = this->RecvReplyPacked(worker_id);
      ICHECK_EQ(args.size(), 1);
      ICHECK(static_cast<DiscoAction>(args[0].operator int()) == DiscoAction::kDebugSetRegister);
    }
  }

  void BroadcastPacked(const TVMArgs& args) final {
    local_session_->BroadcastPacked(args);
    std::vector<TVMValue> values(args.size() + 2);
    std::vector<int> type_codes(args.size() + 2);
    PackArgs(values.data(), type_codes.data(), static_cast<int>(DiscoSocketAction::kSend), -1);
    std::copy(args.values, args.values + args.size(), values.begin() + 2);
    std::copy(args.type_codes, args.type_codes + args.size(), type_codes.begin() + 2);
    for (auto& channel : remote_channels_) {
      channel->Send(TVMArgs(values.data(), type_codes.data(), values.size()));
    }
  }

  void SendPacked(int worker_id, const TVMArgs& args) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      local_session_->SendPacked(worker_id, args);
      return;
    }
    std::vector<TVMValue> values(args.size() + 2);
    std::vector<int> type_codes(args.size() + 2);
    PackArgs(values.data(), type_codes.data(), static_cast<int>(DiscoSocketAction::kSend),
             worker_id);
    std::copy(args.values, args.values + args.size(), values.begin() + 2);
    std::copy(args.type_codes, args.type_codes + args.size(), type_codes.begin() + 2);
    remote_channels_[node_id - 1]->Send(TVMArgs(values.data(), type_codes.data(), values.size()));
  }

  TVMArgs RecvReplyPacked(int worker_id) final {
    int node_id = worker_id / num_workers_per_node_;
    if (node_id == 0) {
      return local_session_->RecvReplyPacked(worker_id);
    }
    TVMValue values[2];
    int type_codes[2];
    PackArgs(values, type_codes, static_cast<int>(DiscoSocketAction::kReceive), worker_id);
    remote_channels_[node_id - 1]->Send(TVMArgs(values, type_codes, 2));
    return remote_channels_[node_id - 1]->Recv();
  }

  void AppendHostNDArray(const NDArray& host_array) final {
    local_session_->AppendHostNDArray(host_array);
  }

  void Shutdown() final {
    // local session will be implicitly shutdown by its destructor
    TVMValue values[2];
    int type_codes[2];
    PackArgs(values, type_codes, static_cast<int>(DiscoSocketAction::kShutdown), -1);
    for (auto& channel : remote_channels_) {
      channel->Send(TVMArgs(values, type_codes, 2));
    }
    for (auto& socket : remote_sockets_) {
      socket.Close();
    }
    remote_sockets_.clear();
    remote_channels_.clear();
    if (!socket_.IsClosed()) {
      socket_.Close();
    }
    Socket::Finalize();
  }

  ~SocketSessionObj() { Shutdown(); }

  static constexpr const char* _type_key = "runtime.disco.SocketSession";
  TVM_DECLARE_FINAL_OBJECT_INFO(SocketSessionObj, BcastSessionObj);
  int num_nodes_;
  int num_workers_per_node_;
  TCPSocket socket_;
  std::vector<TCPSocket> remote_sockets_;
  std::vector<std::unique_ptr<DiscoSocketChannel>> remote_channels_;
  BcastSession local_session_{nullptr};
};

TVM_REGISTER_OBJECT_TYPE(SocketSessionObj);

class RemoteSocketSession {
 public:
  explicit RemoteSocketSession(const String& server_host, int server_port, int num_local_workers) {
    socket_.Create();
    socket_.SetKeepAlive(true);
    SockAddr server_addr{server_host.c_str(), server_port};
    Socket::Startup();
    if (!socket_.Connect(server_addr)) {
      LOG(FATAL) << "Failed to connect to server " << server_addr.AsString()
                 << ", errno = " << Socket::GetLastErrorCode();
    }
    channel_ = std::make_unique<DiscoSocketChannel>(socket_);
    TVMArgs metadata = channel_->Recv();
    ICHECK_EQ(metadata.size(), 4);
    num_nodes_ = metadata[0].operator int();
    num_workers_per_node_ = metadata[1].operator int();
    num_groups_ = metadata[2].operator int();
    node_id_ = metadata[3].operator int();
    CHECK_GE(num_local_workers, num_workers_per_node_);
    InitLocalSession();
  }

  void MainLoop() {
    while (true) {
      TVMArgs args = channel_->Recv();
      DiscoSocketAction action = static_cast<DiscoSocketAction>(args[0].operator int());
      int worker_id = args[1].operator int();
      int local_worker_id = worker_id - node_id_ * num_workers_per_node_;
      switch (action) {
        case DiscoSocketAction::kSend: {
          args = TVMArgs(args.values + 2, args.type_codes + 2, args.size() - 2);
          if (worker_id == -1) {
            local_session_->BroadcastPacked(args);
          } else {
            local_session_->SendPacked(local_worker_id, args);
          }
          break;
        }
        case DiscoSocketAction::kReceive: {
          args = local_session_->RecvReplyPacked(local_worker_id);
          channel_->Reply(args);
          break;
        }
        case DiscoSocketAction::kShutdown: {
          local_session_->Shutdown();
          LOG(INFO) << "Connection closed by remote controller.";
          return;
        }
        default:
          LOG(FATAL) << "Invalid action " << static_cast<int>(action);
      }
    }
  }

  ~RemoteSocketSession() {
    socket_.Close();
    Socket::Finalize();
  }

 private:
  void InitLocalSession() {
    const PackedFunc* f_create_local_session =
        Registry::Get("runtime.disco.create_socket_session_local_workers");
    local_session_ = ((*f_create_local_session)(num_workers_per_node_)).AsObjectRef<BcastSession>();

    DRef f_init_workers =
        local_session_->GetGlobalFunc("runtime.disco.socket_session_init_workers");
    local_session_->CallPacked(f_init_workers, num_nodes_, node_id_, num_groups_,
                               num_workers_per_node_);
  }

  TCPSocket socket_;
  BcastSession local_session_{nullptr};
  std::unique_ptr<DiscoSocketChannel> channel_;
  int num_nodes_{-1};
  int node_id_{-1};
  int num_groups_{-1};
  int num_workers_per_node_{-1};
};

void RemoteSocketSessionEntryPoint(const String& server_host, int server_port,
                                   int num_local_workers) {
  RemoteSocketSession proxy(server_host, server_port, num_local_workers);
  proxy.MainLoop();
}

TVM_REGISTER_GLOBAL("runtime.disco.RemoteSocketSession")
    .set_body_typed(RemoteSocketSessionEntryPoint);

Session SocketSession(int num_nodes, int num_workers_per_node, int num_groups, const String& host,
                      int port) {
  auto n = make_object<SocketSessionObj>(num_nodes, num_workers_per_node, num_groups, host, port);
  return Session(n);
}

TVM_REGISTER_GLOBAL("runtime.disco.SocketSession").set_body_typed(SocketSession);

TVM_REGISTER_GLOBAL("runtime.disco.socket_session_init_workers")
    .set_body_typed([](int num_nodes, int node_id, int num_groups, int num_workers_per_node) {
      LOG(INFO) << "Initializing worker group with " << num_nodes << " nodes, "
                << num_workers_per_node << " workers per node, and " << num_groups << " groups.";
      DiscoWorker* worker = DiscoWorker::ThreadLocal();
      worker->num_groups = num_groups;
      worker->worker_id = worker->worker_id + node_id * num_workers_per_node;
      worker->num_workers = num_nodes * num_workers_per_node;
    });

}  // namespace runtime
}  // namespace tvm
