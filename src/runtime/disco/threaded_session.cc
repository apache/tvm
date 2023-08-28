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
#include <tvm/runtime/object.h>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "../../support/arena.h"
#include "../../support/ring_buffer.h"
#include "../minrpc/rpc_reference.h"
#include "./bcast_session.h"
#include "./worker.h"

namespace tvm {
namespace runtime {

class DiscoThreadedMessageQueue : public dmlc::Stream {
 public:
  void Send(const TVMArgs& args) {
    RPCReference::ReturnPackedSeq(args.values, args.type_codes, args.num_args, this);
    NotifyEnqueue();
  }

  TVMArgs Recv() {
    WaitDequeue();
    TVMValue* values = nullptr;
    int* type_codes = nullptr;
    int num_args = 0;
    RPCReference::RecvPackedSeq(&values, &type_codes, &num_args, this);
    return TVMArgs(values, type_codes, num_args);
  }

 protected:
  void NotifyEnqueue() {
    {
      std::lock_guard<std::mutex> lock{mutex_};
      ++msg_cnt_;
    }
    condition_.notify_one();
  }

  void WaitDequeue() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this] { return msg_cnt_.load() > 0; });
      --msg_cnt_;
    }
    {
      this->arena_.RecycleAll();
      this->object_arena_.clear();
    }
    uint64_t packet_nbytes = 0;
    RPCCode code = RPCCode::kReturn;
    this->Read(&packet_nbytes);
    this->Read(&code);
  }

  void MessageStart(uint64_t packet_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = ring_buffer_.bytes_available();
    n += packet_nbytes + sizeof(uint64_t);
    this->ring_buffer_.Reserve(n);
  }

  void MessageDone() {}

  void ThrowError(RPCServerStatus status) {
    LOG(FATAL) << "InternalError: Unexpected error in RPC: " << RPCServerStatusToString(status);
  }

  template <typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return arena_.template allocate_<T>(count);
  }

  size_t Read(void* data, size_t size) final {
    std::lock_guard<std::mutex> lock(mutex_);
    ring_buffer_.Read(data, size);
    return size;
  }

  void Write(const void* data, size_t size) final {
    std::lock_guard<std::mutex> lock(mutex_);
    ring_buffer_.Write(data, size);
  }

  uint64_t GetObjectBytes(Object* obj) {
    if (obj->IsInstance<DRefObj>()) {
      return sizeof(uint32_t) + sizeof(int64_t);
    } else if (obj->IsInstance<StringObj>()) {
      uint64_t size = static_cast<StringObj*>(obj)->size;
      return sizeof(uint32_t) + sizeof(uint64_t) + size * sizeof(char);
    } else if (obj->IsInstance<ShapeTupleObj>()) {
      uint64_t ndim = static_cast<ShapeTupleObj*>(obj)->size;
      return sizeof(uint32_t) + sizeof(uint64_t) + ndim * sizeof(ShapeTupleObj::index_type);
    } else {
      LOG(FATAL) << "ValueError: Object type is not supported in Disco calling convention: "
                 << obj->GetTypeKey() << " (type_index = " << obj->type_index() << ")";
    }
  }

  void WriteObject(Object* obj) {
    if (obj->IsInstance<DRefObj>()) {
      int64_t reg_id = static_cast<DRefObj*>(obj)->reg_id;
      this->Write<uint32_t>(TypeIndex::kRuntimeDiscoDRef);
      this->Write<int64_t>(reg_id);
    } else if (obj->IsInstance<StringObj>()) {
      StringObj* str = static_cast<StringObj*>(obj);
      this->Write<uint32_t>(TypeIndex::kRuntimeString);
      this->Write<uint64_t>(str->size);
      this->WriteArray<char>(str->data, str->size);
    } else if (obj->IsInstance<ShapeTupleObj>()) {
      ShapeTupleObj* shape = static_cast<ShapeTupleObj*>(obj);
      this->Write<uint32_t>(TypeIndex::kRuntimeShapeTuple);
      this->Write<uint64_t>(shape->size);
      this->WriteArray<ShapeTupleObj::index_type>(shape->data, shape->size);
    } else {
      LOG(FATAL) << "ValueError: Object type is not supported in Disco calling convention: "
                 << obj->GetTypeKey() << " (type_index = " << obj->type_index() << ")";
    }
  }

  void ReadObject(int* tcode, TVMValue* value) {
    ObjectRef result{nullptr};
    uint32_t type_index;
    this->Read<uint32_t>(&type_index);
    if (type_index == TypeIndex::kRuntimeDiscoDRef) {
      ObjectPtr<DRefObj> dref = make_object<DRefObj>();
      this->Read<int64_t>(&dref->reg_id);
      dref->session = Session{nullptr};
      result = ObjectRef(std::move(dref));
    } else if (type_index == TypeIndex::kRuntimeString) {
      uint64_t size = 0;
      this->Read<uint64_t>(&size);
      std::string data(size, '\0');
      this->ReadArray<char>(data.data(), size);
      result = String(std::move(data));
    } else if (type_index == TypeIndex::kRuntimeShapeTuple) {
      uint64_t ndim = 0;
      this->Read<uint64_t>(&ndim);
      std::vector<ShapeTupleObj::index_type> data(ndim);
      this->ReadArray<ShapeTupleObj::index_type>(data.data(), ndim);
      result = ShapeTuple(std::move(data));
    } else {
      LOG(FATAL) << "ValueError: Object type is not supported in Disco calling convention: "
                 << Object::TypeIndex2Key(type_index) << " (type_index = " << type_index << ")";
    }
    *tcode = kTVMObjectHandle;
    value->v_handle = const_cast<Object*>(result.get());
    object_arena_.push_back(result);
  }

  using dmlc::Stream::Read;
  using dmlc::Stream::ReadArray;
  using dmlc::Stream::Write;
  using dmlc::Stream::WriteArray;
  friend struct RPCReference;

  std::mutex mutex_;
  std::atomic<int> msg_cnt_{0};
  std::condition_variable condition_;

  support::RingBuffer ring_buffer_;
  support::Arena arena_;
  std::vector<ObjectRef> object_arena_;
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

class ThreadedSessionObj final : public BcastSessionObj {
 public:
  explicit ThreadedSessionObj(int num_workers) {
    for (int i = 0; i < num_workers; ++i) {
      std::unique_ptr<DiscoThreadChannel> channel = std::make_unique<DiscoThreadChannel>();
      WorkerZeroData* data = (i == 0) ? &worker_zero_data_ : nullptr;
      workers_.emplace_back(std::make_unique<DiscoWorker>(i, num_workers, data, channel.get()));
      channels_.emplace_back(std::move(channel));
      worker_threads_.emplace_back([worker = workers_.back().get()] { worker->MainLoop(); });
    }
  }

  ~ThreadedSessionObj() {
    this->Shutdown();
    for (std::thread& worker : this->worker_threads_) {
      worker.join();
    }
  }

  TVMRetValue DebugGetFromRemote(int64_t reg_id, int worker_id) {
    this->SyncWorker(worker_id);
    return this->workers_.at(worker_id)->register_file.at(reg_id);
  }

  void BroadcastPacked(const TVMArgs& args) final {
    for (const std::unique_ptr<DiscoThreadChannel>& channel : this->channels_) {
      channel->Send(args);
    }
  }

  TVMArgs RecvReplyPacked(int worker_id) final { return channels_[worker_id]->RecvReply(); }

  static constexpr const char* _type_key = "runtime.disco.ThreadedSession";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThreadedSessionObj, SessionObj);

  std::vector<std::unique_ptr<DiscoThreadChannel>> channels_;
  std::vector<std::unique_ptr<DiscoWorker>> workers_;
  std::vector<std::thread> worker_threads_;
};

TVM_REGISTER_OBJECT_TYPE(ThreadedSessionObj);

Session Session::ThreadedSession(int num_workers) {
  ObjectPtr<ThreadedSessionObj> n = make_object<ThreadedSessionObj>(num_workers);
  return Session(std::move(n));
}

}  // namespace runtime
}  // namespace tvm
