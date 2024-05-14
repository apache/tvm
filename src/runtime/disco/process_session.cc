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
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "../../support/pipe.h"
#include "../minrpc/rpc_reference.h"
#include "./bcast_session.h"
#include "./disco_worker_thread.h"
#include "./protocol.h"

namespace tvm {
namespace runtime {

class DiscoPipeMessageQueue : private dmlc::Stream, private DiscoProtocol<DiscoPipeMessageQueue> {
 public:
  explicit DiscoPipeMessageQueue(int64_t handle) : pipe_(handle) {}

  ~DiscoPipeMessageQueue() = default;

  void Send(const TVMArgs& args) {
    RPCReference::ReturnPackedSeq(args.values, args.type_codes, args.num_args, this);
    CommitSendAndNotifyEnqueue();
  }

  TVMArgs Recv() {
    bool is_implicit_shutdown = DequeueNextPacket();
    TVMValue* values = nullptr;
    int* type_codes = nullptr;
    int num_args = 0;

    if (is_implicit_shutdown) {
      num_args = 2;
      values = ArenaAlloc<TVMValue>(num_args);
      type_codes = ArenaAlloc<int>(num_args);
      TVMArgsSetter setter(values, type_codes);
      setter(0, static_cast<int>(DiscoAction::kShutDown));
      setter(1, 0);
    } else {
      RPCReference::RecvPackedSeq(&values, &type_codes, &num_args, this);
    }
    return TVMArgs(values, type_codes, num_args);
  }

 protected:
  void CommitSendAndNotifyEnqueue() {
    pipe_.Write(write_buffer_.data(), write_buffer_.size());
    write_buffer_.clear();
  }

  /* \brief Read next packet and reset unpacker
   *
   * Read the next packet into `read_buffer_`, releasing all arena
   * allocations performed by the unpacker and resetting the unpacker
   * to its initial state.
   *
   * \return A boolean value.  If true, this packet should be treated
   *    equivalently to a `DiscoAction::kShutdown` event.  If false,
   *    this packet should be unpacked.
   */
  bool DequeueNextPacket() {
    uint64_t packet_nbytes = 0;
    int read_size = pipe_.Read(&packet_nbytes, sizeof(packet_nbytes));
    if (read_size == 0) {
      // Special case, connection dropped between packets.  Treat as a
      // request to shutdown.
      return true;
    }

    ICHECK_EQ(read_size, sizeof(packet_nbytes))
        << "Pipe closed without proper shutdown. Please make sure to explicitly call "
           "`Session::Shutdown`";
    read_buffer_.resize(packet_nbytes);
    read_size = pipe_.Read(read_buffer_.data(), packet_nbytes);
    ICHECK_EQ(read_size, packet_nbytes)
        << "Pipe closed without proper shutdown. Please make sure to explicitly call "
           "`Session::Shutdown`";
    read_offset_ = 0;
    this->RecycleAll();
    RPCCode code = RPCCode::kReturn;
    this->Read(&code);
    return false;
  }

  size_t Read(void* data, size_t size) final {
    std::memcpy(data, read_buffer_.data() + read_offset_, size);
    read_offset_ += size;
    ICHECK_LE(read_offset_, read_buffer_.size());
    return size;
  }

  size_t Write(const void* data, size_t size) final {
    size_t cur_size = write_buffer_.size();
    write_buffer_.resize(cur_size + size);
    std::memcpy(write_buffer_.data() + cur_size, data, size);
    return size;
  }

  using dmlc::Stream::Read;
  using dmlc::Stream::ReadArray;
  using dmlc::Stream::Write;
  using dmlc::Stream::WriteArray;
  friend struct RPCReference;
  friend struct DiscoProtocol<DiscoPipeMessageQueue>;

  // The read/write buffer will only be accessed by the producer thread.
  std::string write_buffer_;
  std::string read_buffer_;
  size_t read_offset_ = 0;
  support::Pipe pipe_;
};

class DiscoProcessChannel final : public DiscoChannel {
 public:
  DiscoProcessChannel(int64_t controler_to_worker_fd, int64_t worker_to_controler_fd)
      : controler_to_worker_(controler_to_worker_fd),
        worker_to_controler_(worker_to_controler_fd) {}

  DiscoProcessChannel(DiscoProcessChannel&& other) = delete;
  DiscoProcessChannel(const DiscoProcessChannel& other) = delete;

  void Send(const TVMArgs& args) { controler_to_worker_.Send(args); }
  TVMArgs Recv() { return controler_to_worker_.Recv(); }
  void Reply(const TVMArgs& args) { worker_to_controler_.Send(args); }
  TVMArgs RecvReply() { return worker_to_controler_.Recv(); }

  DiscoPipeMessageQueue controler_to_worker_;
  DiscoPipeMessageQueue worker_to_controler_;
};

class ProcessSessionObj final : public BcastSessionObj {
 public:
  explicit ProcessSessionObj(int num_workers, PackedFunc process_pool)
      : process_pool_(process_pool),
        worker_0_(std::make_unique<DiscoWorkerThread>(0, num_workers, &worker_zero_data_)) {
    std::vector<int64_t> read_fds;
    std::vector<int64_t> write_fds;
    read_fds.reserve(num_workers - 1);
    write_fds.reserve(num_workers - 1);
    for (int i = 1; i < num_workers; ++i) {
      IntTuple fds = process_pool(i);
      CHECK_EQ(fds.size(), 2) << "ValueError: process_pool(" << i << ") should return a tuple of "
                              << "size 2, but got a tuple of size " << fds.size() << ".";
      read_fds.push_back(fds[0]);
      write_fds.push_back(fds[1]);
    }
    for (int i = 0; i < num_workers - 1; ++i) {
      workers_.emplace_back(std::make_unique<DiscoProcessChannel>(write_fds[i], read_fds[i]));
    }
  }

  void Kill() {
    if (this->worker_0_ != nullptr) {
      this->Shutdown();
      this->worker_0_.reset();
      this->workers_.clear();
      this->process_pool_(0);
    }
  }

  ~ProcessSessionObj() { Kill(); }

  int64_t GetNumWorkers() { return workers_.size() + 1; }

  TVMRetValue DebugGetFromRemote(int64_t reg_id, int worker_id) {
    if (worker_id == 0) {
      this->SyncWorker(worker_id);
      return worker_0_->worker->register_file.at(reg_id);
    }
    {
      TVMValue values[3];
      int type_codes[3];
      PackArgs(values, type_codes, static_cast<int>(DiscoAction::kDebugGetFromRemote), reg_id,
               worker_id);
      workers_[worker_id - 1]->Send(TVMArgs(values, type_codes, 3));
    }
    TVMArgs args = this->RecvReplyPacked(worker_id);
    ICHECK_EQ(args.size(), 2);
    ICHECK(static_cast<DiscoAction>(args[0].operator int()) == DiscoAction::kDebugGetFromRemote);
    TVMRetValue result;
    result = args[1];
    return result;
  }

  void DebugSetRegister(int64_t reg_id, TVMArgValue value, int worker_id) {
    if (worker_id == 0) {
      this->SyncWorker(worker_id);
      worker_0_->worker->SetRegister(reg_id, value);
      return;
    }
    ObjectRef wrapped{nullptr};
    if (value.type_code() == kTVMNDArrayHandle || value.type_code() == kTVMObjectHandle) {
      wrapped = DiscoDebugObject::Wrap(value);
      TVMValue tvm_value;
      int type_code = kTVMObjectHandle;
      tvm_value.v_handle = const_cast<Object*>(wrapped.get());
      value = TVMArgValue(tvm_value, type_code);
    }
    {
      TVMValue values[4];
      int type_codes[4];
      PackArgs(values, type_codes, static_cast<int>(DiscoAction::kDebugSetRegister), reg_id,
               worker_id, value);
      workers_[worker_id - 1]->Send(TVMArgs(values, type_codes, 4));
    }
    TVMRetValue result;
    TVMArgs args = this->RecvReplyPacked(worker_id);
    ICHECK_EQ(args.size(), 1);
    ICHECK(static_cast<DiscoAction>(args[0].operator int()) == DiscoAction::kDebugSetRegister);
  }

  void BroadcastPacked(const TVMArgs& args) final {
    worker_0_->channel->Send(args);
    for (std::unique_ptr<DiscoProcessChannel>& channel : workers_) {
      channel->Send(args);
    }
  }

  TVMArgs RecvReplyPacked(int worker_id) final {
    if (worker_id == 0) {
      return worker_0_->channel->RecvReply();
    }
    return this->workers_.at(worker_id - 1)->RecvReply();
  }

  PackedFunc process_pool_;
  std::unique_ptr<DiscoWorkerThread> worker_0_;
  std::vector<std::unique_ptr<DiscoProcessChannel>> workers_;

  static constexpr const char* _type_key = "runtime.disco.ProcessSession";
  TVM_DECLARE_FINAL_OBJECT_INFO(ProcessSessionObj, SessionObj);
};

TVM_REGISTER_OBJECT_TYPE(DiscoDebugObject);
TVM_REGISTER_OBJECT_TYPE(ProcessSessionObj);

Session Session::ProcessSession(int num_workers, String process_pool_creator, String entrypoint) {
  const PackedFunc* pf = Registry::Get(process_pool_creator);
  CHECK(pf) << "ValueError: Cannot find function " << process_pool_creator
            << " in the registry. Please check if it is registered.";
  PackedFunc process_pool = (*pf)(num_workers, entrypoint);
  auto n = make_object<ProcessSessionObj>(num_workers, process_pool);
  return Session(n);
}

void WorkerProcess(int worker_id, int num_workers, int64_t read_fd, int64_t write_fd) {
  DiscoProcessChannel channel(read_fd, write_fd);
  DiscoWorker worker(worker_id, num_workers, nullptr, &channel);
  worker.MainLoop();
}

TVM_REGISTER_GLOBAL("runtime.disco.SessionProcess").set_body_typed(Session::ProcessSession);
TVM_REGISTER_GLOBAL("runtime.disco.WorkerProcess").set_body_typed(WorkerProcess);

}  // namespace runtime
}  // namespace tvm
