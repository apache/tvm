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
#ifndef TVM_RUNTIME_DISCO_MESSAGE_QUEUE_H_
#define TVM_RUNTIME_DISCO_MESSAGE_QUEUE_H_

#include <dmlc/io.h>

#include <string>

#include "./protocol.h"

namespace tvm {
namespace runtime {

class DiscoStreamMessageQueue : private dmlc::Stream,
                                private DiscoProtocol<DiscoStreamMessageQueue> {
 public:
  explicit DiscoStreamMessageQueue(Stream* stream) : stream_(stream) {}

  ~DiscoStreamMessageQueue() = default;

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
    stream_->Write(write_buffer_.data(), write_buffer_.size());
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
    int read_size = stream_->Read(&packet_nbytes, sizeof(packet_nbytes));
    if (read_size == 0) {
      // Special case, connection dropped between packets.  Treat as a
      // request to shutdown.
      return true;
    }

    ICHECK_EQ(read_size, sizeof(packet_nbytes))
        << "Stream closed without proper shutdown. Please make sure to explicitly call "
           "`Session::Shutdown`";
    read_buffer_.resize(packet_nbytes);
    read_size = stream_->Read(read_buffer_.data(), packet_nbytes);
    ICHECK_EQ(read_size, packet_nbytes)
        << "Stream closed without proper shutdown. Please make sure to explicitly call "
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
  friend struct DiscoProtocol<DiscoStreamMessageQueue>;

  // The read/write buffer will only be accessed by the producer thread.
  std::string write_buffer_;
  std::string read_buffer_;
  size_t read_offset_ = 0;
  dmlc::Stream* stream_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_MESSAGE_QUEUE_H_
