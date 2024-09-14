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
#include <dmlc/io.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/object.h>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "../../support/ring_buffer.h"
#include "../minrpc/rpc_reference.h"
#include "./bcast_session.h"
#include "./disco_worker_thread.h"
#include "./protocol.h"

namespace tvm {
namespace runtime {

class DiscoThreadedMessageQueue : private dmlc::Stream,
                                  private DiscoProtocol<DiscoThreadedMessageQueue> {
 public:
  void Send(const TVMArgs& args) {
    RPCReference::ReturnPackedSeq(args.values, args.type_codes, args.num_args, this);
    CommitSendAndNotifyEnqueue();
  }

  TVMArgs Recv() {
    DequeueNextPacket();
    TVMValue* values = nullptr;
    int* type_codes = nullptr;
    int num_args = 0;
    RPCReference::RecvPackedSeq(&values, &type_codes, &num_args, this);
    return TVMArgs(values, type_codes, num_args);
  }

 protected:
  void CommitSendAndNotifyEnqueue() {
    bool need_notify = false;
    {
      std::lock_guard<std::mutex> lock{mutex_};
      ++msg_cnt_;
      ring_buffer_.Write(write_buffer_.data(), write_buffer_.size());
      need_notify = dequeue_waiting_;
    }
    if (need_notify) {
      condition_.notify_one();
    }
    write_buffer_.clear();
  }

  void DequeueNextPacket() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      dequeue_waiting_ = true;
      condition_.wait(lock, [this] { return msg_cnt_.load() > 0; });
      dequeue_waiting_ = false;
      --msg_cnt_;
      uint64_t packet_nbytes = 0;
      ring_buffer_.Read(&packet_nbytes, sizeof(packet_nbytes));
      read_buffer_.resize(packet_nbytes);
      ring_buffer_.Read(read_buffer_.data(), packet_nbytes);
      read_offset_ = 0;
    }
    this->RecycleAll();
    RPCCode code = RPCCode::kReturn;
    this->Read(&code);
  }

  void MessageStart(uint64_t packet_nbytes) {}

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
  friend struct DiscoProtocol<DiscoThreadedMessageQueue>;

  // The read/write buffer will only be accessed by the producer thread.
  std::string write_buffer_;
  std::string read_buffer_;
  size_t read_offset_ = 0;
  bool dequeue_waiting_ = false;

  std::mutex mutex_;
  std::atomic<int> msg_cnt_{0};
  std::condition_variable condition_;
  support::RingBuffer ring_buffer_;
};

class DiscoThreadChannel final : public DiscoChannel {
 public:
  void Send(const TVMArgs& args) { controler_to_worker_.Send(args); }
  TVMArgs Recv() { return controler_to_worker_.Recv(); }
  void Reply(const TVMArgs& args) { worker_to_controler_.Send(args); }
  TVMArgs RecvReply() { return worker_to_controler_.Recv(); }

  DiscoThreadedMessageQueue controler_to_worker_;
  DiscoThreadedMessageQueue worker_to_controler_;
};

DiscoWorkerThread::DiscoWorkerThread(int worker_id, int num_workers, int num_groups,
                                     WorkerZeroData* worker_zero_data_)
    : channel(std::make_unique<DiscoThreadChannel>()),
      worker(std::make_unique<DiscoWorker>(worker_id, num_workers, num_groups, worker_zero_data_,
                                           channel.get())),
      thread(std::make_unique<std::thread>([worker = this->worker.get()] { worker->MainLoop(); })) {
}

class ThreadedSessionObj final : public BcastSessionObj {
 public:
  explicit ThreadedSessionObj(int num_workers, int num_groups) {
    for (int i = 0; i < num_workers; ++i) {
      WorkerZeroData* data = (i == 0) ? &worker_zero_data_ : nullptr;
      workers_.emplace_back(i, num_workers, num_groups, data);
    }
  }

  ~ThreadedSessionObj() {
    this->Shutdown();
    workers_.clear();
  }

  int64_t GetNumWorkers() { return workers_.size(); }

  TVMRetValue DebugGetFromRemote(int64_t reg_id, int worker_id) {
    this->SyncWorker(worker_id);
    return this->workers_.at(worker_id).worker->register_file.at(reg_id);
  }

  void DebugSetRegister(int64_t reg_id, TVMArgValue value, int worker_id) {
    this->SyncWorker(worker_id);
    this->workers_.at(worker_id).worker->SetRegister(reg_id, value);
  }

  void BroadcastPacked(const TVMArgs& args) final {
    for (const DiscoWorkerThread& worker : this->workers_) {
      worker.channel->Send(args);
    }
  }

  void SendPacked(int worker_id, const TVMArgs& args) final {
    this->workers_.at(worker_id).channel->Send(args);
  }

  TVMArgs RecvReplyPacked(int worker_id) final {
    return this->workers_.at(worker_id).channel->RecvReply();
  }

  static constexpr const char* _type_key = "runtime.disco.ThreadedSession";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThreadedSessionObj, SessionObj);

  std::vector<DiscoWorkerThread> workers_;
};

TVM_REGISTER_OBJECT_TYPE(ThreadedSessionObj);

Session Session::ThreadedSession(int num_workers, int num_group) {
  CHECK_EQ(num_workers % num_group, 0)
      << "The number of workers should be divisible by the number of worker group.";
  ObjectPtr<ThreadedSessionObj> n = make_object<ThreadedSessionObj>(num_workers, num_group);
  return Session(std::move(n));
}

}  // namespace runtime
}  // namespace tvm
