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
 * \file micro_session.cc
 */

#include "micro_session.h"

#include <dmlc/logging.h>
#include <tvm/runtime/crt/rpc_common/framing.h>
#include <tvm/runtime/crt/rpc_common/session.h>
#include <tvm/runtime/registry.h>

#include <cstdarg>
#include <memory>
#include <string>
#include <utility>

#include "../../support/str_escape.h"
#include "../crt/host/crt_config.h"
#include "../rpc/rpc_channel.h"
#include "../rpc/rpc_endpoint.h"
#include "../rpc/rpc_session.h"

namespace tvm {
namespace runtime {
namespace micro_rpc {

class CallbackWriteStream : public WriteStream {
 public:
  explicit CallbackWriteStream(PackedFunc fsend) : fsend_{fsend} {}

  ssize_t Write(const uint8_t* data, size_t data_size_bytes) override {
    TVMByteArray bytes;
    bytes.data = (const char*)data;
    bytes.size = data_size_bytes;
    int64_t n = fsend_(bytes);
    return n;
  }

  void PacketDone(bool is_valid) override {}

 private:
  PackedFunc fsend_;
};

class MicroTransportChannel : public RPCChannel {
 public:
  MicroTransportChannel(PackedFunc fsend, PackedFunc frecv)
      : write_stream_{fsend},
        framer_{&write_stream_},
        receive_buffer_{new uint8_t[TVM_CRT_MAX_PACKET_SIZE_BYTES], TVM_CRT_MAX_PACKET_SIZE_BYTES},
        session_{0x5b, &framer_, &receive_buffer_, &HandleMessageReceivedCb, this},
        unframer_{session_.Receiver()},
        did_receive_message_{false},
        frecv_{frecv},
        message_buffer_{nullptr} {}

  size_t ReceiveUntil(TypedPackedFunc<bool(void)> pf) {
    size_t bytes_received = 0;
    if (pf()) {
      return 0;
    }

    for (;;) {
      while (pending_chunk_.size() > 0) {
        size_t bytes_consumed = 0;
        int unframer_error = unframer_.Write((const uint8_t*)pending_chunk_.data(),
                                             pending_chunk_.size(), &bytes_consumed);

        CHECK(bytes_consumed <= pending_chunk_.size());
        pending_chunk_ = pending_chunk_.substr(bytes_consumed);
        bytes_received += bytes_consumed;
        if (unframer_error < 0) {
          LOG(ERROR) << "unframer got error code: " << unframer_error;
        } else {
          if (pf()) {
            return bytes_received;
          }
        }
      }

      std::string chunk = frecv_(128);
      pending_chunk_ = chunk;
      CHECK(pending_chunk_.size() != 0) << "zero-size chunk encountered";
      CHECK_GT(pending_chunk_.size(), 0);
    }
  }

  void StartSession() {
    CHECK_EQ(kTvmErrorNoError, session_.Initialize());
    CHECK_EQ(kTvmErrorNoError, session_.StartSession());
    ReceiveUntil([this]() -> bool { return session_.IsEstablished(); });
  }

  size_t Send(const void* data, size_t size) override {
    const uint8_t* data_bytes = static_cast<const uint8_t*>(data);
    ssize_t ret = session_.SendMessage(MessageType::kNormal, data_bytes, size);
    CHECK(ret == 0) << "SendMessage returned " << ret;

    return size;
  }

  size_t Recv(void* data, size_t size) override {
    size_t num_bytes_recv = 0;
    while (num_bytes_recv < size) {
      if (message_buffer_ != nullptr) {
        num_bytes_recv += message_buffer_->Read(static_cast<uint8_t*>(data), size);
        if (message_buffer_->ReadAvailable() == 0) {
          message_buffer_ = nullptr;
          session_.ClearReceiveBuffer();
        }
        if (num_bytes_recv == size) {
          CHECK(message_buffer_ == nullptr || message_buffer_->ReadAvailable() > 0);
          return num_bytes_recv;
        }
      }

      did_receive_message_ = false;
      ReceiveUntil([this]() -> bool { return did_receive_message_; });
    }

    return num_bytes_recv;
  }

  FrameBuffer* GetReceivedMessage() {
    if (did_receive_message_) {
      did_receive_message_ = false;
      return message_buffer_;
    }

    return nullptr;
  }

 private:
  static void HandleMessageReceivedCb(void* context, MessageType message_type, FrameBuffer* buf) {
    static_cast<MicroTransportChannel*>(context)->HandleMessageReceived(message_type, buf);
  }

  void HandleMessageReceived(MessageType message_type, FrameBuffer* buf) {
    size_t message_size_bytes;
    switch (message_type) {
      case MessageType::kStartSessionInit:
      case MessageType::kStartSessionReply:
        break;

      case MessageType::kTerminateSession:
        LOG(FATAL) << "SessionTerminatedError: remote side has probably reset";
        break;

      case MessageType::kLog:
        uint8_t message[1024];
        message_size_bytes = buf->ReadAvailable();
        if (message_size_bytes == 0) {
          return;
        } else if (message_size_bytes > sizeof(message) - 1) {
          LOG(ERROR) << "Remote log message is too long to display: " << message_size_bytes
                     << " bytes";
          return;
        }

        CHECK_EQ(buf->Read(message, sizeof(message) - 1), message_size_bytes);
        message[message_size_bytes] = 0;
        LOG(INFO) << "remote: " << message;
        session_.ClearReceiveBuffer();
        return;

      case MessageType::kNormal:
        did_receive_message_ = true;
        message_buffer_ = buf;
        break;
    }
  }

  CallbackWriteStream write_stream_;
  Framer framer_;
  FrameBuffer receive_buffer_;
  Session session_;
  Unframer unframer_;
  bool did_receive_message_;
  PackedFunc frecv_;
  FrameBuffer* message_buffer_;
  std::string pending_chunk_;
};

TVM_REGISTER_GLOBAL("micro._rpc_connect").set_body([](TVMArgs args, TVMRetValue* rv) {
  MicroTransportChannel* micro_channel = new MicroTransportChannel(args[1], args[2]);
  micro_channel->StartSession();
  std::unique_ptr<RPCChannel> channel(micro_channel);
  auto ep = RPCEndpoint::Create(std::move(channel), args[0], "");
  auto sess = CreateClientSession(ep);
  *rv = CreateRPCSessionModule(sess);
});

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

extern "C" {

void TVMLogf(const char* fmt, ...) {
  va_list args;
  char msg_buf[256];
  va_start(args, fmt);
  vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
  va_end(args);
  LOG(INFO) << msg_buf;
}

void TVMPlatformAbort(int error_code) { CHECK(false) << "TVMPlatformAbort: " << error_code; }
}
