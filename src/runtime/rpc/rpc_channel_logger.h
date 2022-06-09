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
 * \file rpc_channel_logger.h
 * \brief A wrapper for RPCChannel with a NanoRPCListener for logging the commands.
 */
#ifndef TVM_RUNTIME_RPC_RPC_CHANNEL_LOGGER_H_
#define TVM_RUNTIME_RPC_RPC_CHANNEL_LOGGER_H_

#include <tvm/runtime/c_runtime_api.h>

#include <memory>
#include <utility>

#include "../../support/ssize.h"
#include "../minrpc/minrpc_server_logging.h"
#include "rpc_channel.h"

#define RX_BUFFER_SIZE 65536

namespace tvm {
namespace runtime {

class Buffer {
 public:
  Buffer(uint8_t* data, size_t data_size_bytes)
      : data_{data}, capacity_{data_size_bytes}, num_valid_bytes_{0}, read_cursor_{0} {}

  size_t Write(const uint8_t* data, size_t data_size_bytes) {
    size_t num_bytes_available = capacity_ - num_valid_bytes_;
    size_t num_bytes_to_copy = data_size_bytes;
    if (num_bytes_available < num_bytes_to_copy) {
      num_bytes_to_copy = num_bytes_available;
    }

    memcpy(&data_[num_valid_bytes_], data, num_bytes_to_copy);
    num_valid_bytes_ += num_bytes_to_copy;
    return num_bytes_to_copy;
  }

  size_t Read(uint8_t* data, size_t data_size_bytes) {
    size_t num_bytes_to_copy = data_size_bytes;
    size_t num_bytes_available = num_valid_bytes_ - read_cursor_;
    if (num_bytes_available < num_bytes_to_copy) {
      num_bytes_to_copy = num_bytes_available;
    }

    memcpy(data, &data_[read_cursor_], num_bytes_to_copy);
    read_cursor_ += num_bytes_to_copy;
    return num_bytes_to_copy;
  }

  void Clear() {
    num_valid_bytes_ = 0;
    read_cursor_ = 0;
  }

  size_t Size() const { return num_valid_bytes_; }

 private:
  /*! \brief pointer to data buffer. */
  uint8_t* data_;

  /*! \brief The total number of bytes available in data_.*/
  size_t capacity_;

  /*! \brief number of valid bytes in the buffer. */
  size_t num_valid_bytes_;

  /*! \brief Read cursor position. */
  size_t read_cursor_;
};

/*!
 * \brief A simple IO handler for MinRPCSniffer.
 *
 * \tparam Buffer* buffer to store received data.
 */
class SnifferIOHandler {
 public:
  explicit SnifferIOHandler(Buffer* receive_buffer) : receive_buffer_(receive_buffer) {}

  void MessageStart(size_t message_size_bytes) {}

  ssize_t PosixWrite(const uint8_t* buf, size_t buf_size_bytes) { return 0; }

  void MessageDone() {}

  ssize_t PosixRead(uint8_t* buf, size_t buf_size_bytes) {
    return receive_buffer_->Read(buf, buf_size_bytes);
  }

  void Close() {}

  void Exit(int code) {}

 private:
  Buffer* receive_buffer_;
};

/*!
 * \brief A simple rpc session that logs the received commands.
 */
class NanoRPCListener {
 public:
  NanoRPCListener()
      : receive_buffer_(receive_storage_, receive_storage_size_bytes_),
        io_(&receive_buffer_),
        rpc_server_(&io_) {}

  void Listen(const uint8_t* data, size_t size) { receive_buffer_.Write(data, size); }

  void ProcessTxPacket() {
    rpc_server_.ProcessOnePacket();
    ClearBuffer();
  }

  void ProcessRxPacket() {
    rpc_server_.ProcessOneResponse();
    ClearBuffer();
  }

 private:
  void ClearBuffer() { receive_buffer_.Clear(); }

 private:
  size_t receive_storage_size_bytes_ = RX_BUFFER_SIZE;
  uint8_t receive_storage_[RX_BUFFER_SIZE];
  Buffer receive_buffer_;
  SnifferIOHandler io_;
  MinRPCSniffer<SnifferIOHandler> rpc_server_;

  void HandleCompleteMessage() { rpc_server_.ProcessOnePacket(); }

  static void HandleCompleteMessageCb(void* context) {
    static_cast<NanoRPCListener*>(context)->HandleCompleteMessage();
  }
};

/*!
 * \brief A wrapper for RPCChannel, that also logs the commands sent.
 *
 * \tparam std::unique_ptr<RPCChannel>&& underlying RPCChannel unique_ptr.
 */
class RPCChannelLogging : public RPCChannel {
 public:
  explicit RPCChannelLogging(std::unique_ptr<RPCChannel>&& next) { next_ = std::move(next); }

  size_t Send(const void* data, size_t size) {
    listener_.ProcessRxPacket();
    listener_.Listen((const uint8_t*)data, size);
    listener_.ProcessTxPacket();
    return next_->Send(data, size);
  }

  size_t Recv(void* data, size_t size) {
    size_t ret = next_->Recv(data, size);
    listener_.Listen((const uint8_t*)data, size);
    return ret;
  }

 private:
  std::unique_ptr<RPCChannel> next_;
  NanoRPCListener listener_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_CHANNEL_LOGGER_H_
