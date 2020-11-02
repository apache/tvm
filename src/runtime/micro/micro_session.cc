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

#include <tvm/runtime/crt/rpc_common/framing.h>
#include <tvm/runtime/crt/rpc_common/session.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <memory>
#include <sstream>
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
  explicit CallbackWriteStream(PackedFunc fsend, ::std::chrono::microseconds write_timeout)
      : fsend_{fsend}, write_timeout_{write_timeout} {}

  ssize_t Write(const uint8_t* data, size_t data_size_bytes) override {
    TVMByteArray bytes;
    bytes.data = (const char*)data;
    bytes.size = data_size_bytes;
    int64_t n = fsend_(bytes, write_timeout_.count());
    return n;
  }

  void PacketDone(bool is_valid) override {}

  void SetWriteTimeout(::std::chrono::microseconds timeout) { write_timeout_ = timeout; }

 private:
  PackedFunc fsend_;
  ::std::chrono::microseconds write_timeout_;
};

class MicroTransportChannel : public RPCChannel {
 public:
  enum class State : uint8_t {
    kReset = 0,               // state entered before the transport has been read or written to.
    kSessionTerminated = 1,   // session is terminated, but transport is alive.
    kSessionEstablished = 2,  // session is alive.
  };

  MicroTransportChannel(PackedFunc fsend, PackedFunc frecv,
                        ::std::chrono::microseconds session_start_retry_timeout,
                        ::std::chrono::microseconds session_start_timeout,
                        ::std::chrono::microseconds session_established_timeout)
      : state_{State::kReset},
        session_start_retry_timeout_{session_start_retry_timeout},
        session_start_timeout_{session_start_timeout},
        session_established_timeout_{session_established_timeout},
        write_stream_{fsend, session_start_timeout},
        framer_{&write_stream_},
        receive_buffer_{new uint8_t[TVM_CRT_MAX_PACKET_SIZE_BYTES], TVM_CRT_MAX_PACKET_SIZE_BYTES},
        session_{0x5c, &framer_, &receive_buffer_, &HandleMessageReceivedCb, this},
        unframer_{session_.Receiver()},
        did_receive_message_{false},
        frecv_{frecv},
        message_buffer_{nullptr} {}

  bool ReceiveUntil(TypedPackedFunc<bool(void)> pf, ::std::chrono::microseconds timeout) {
    size_t bytes_received = 0;
    if (pf()) {
      return true;
    }

    auto end_time = ::std::chrono::steady_clock::now() + timeout;
    for (;;) {
      while (pending_chunk_.size() > 0) {
        size_t bytes_consumed = 0;
        int unframer_error = unframer_.Write((const uint8_t*)pending_chunk_.data(),
                                             pending_chunk_.size(), &bytes_consumed);

        ICHECK(bytes_consumed <= pending_chunk_.size())
            << "consumed " << bytes_consumed << " want <= " << pending_chunk_.size();
        pending_chunk_ = pending_chunk_.substr(bytes_consumed);
        bytes_received += bytes_consumed;
        if (unframer_error < 0) {
          LOG(ERROR) << "unframer got error code: " << unframer_error;
        } else {
          if (pf()) {
            return true;
          }
        }
      }

      ::std::string chunk;
      if (timeout != ::std::chrono::microseconds::zero()) {
        ::std::chrono::microseconds iter_timeout{
            ::std::max(::std::chrono::microseconds{0},
                       ::std::chrono::duration_cast<::std::chrono::microseconds>(
                           end_time - ::std::chrono::steady_clock::now()))};
        chunk = frecv_(128, iter_timeout.count()).operator std::string();
      } else {
        chunk = frecv_(128, nullptr).operator std::string();
      }
      pending_chunk_ = chunk;
      if (pending_chunk_.size() == 0) {
        // Timeout occurred
        return false;
      }
    }
  }

  bool StartSession() {
    ICHECK(state_ == State::kReset)
        << "MicroSession: state_: expected kReset, got " << uint8_t(state_);

    ::std::chrono::steady_clock::time_point start_time = ::std::chrono::steady_clock::now();
    auto session_start_end_time = start_time + session_start_timeout_;

    ::std::chrono::steady_clock::time_point end_time;
    if (session_start_retry_timeout_ != ::std::chrono::microseconds::zero()) {
      end_time = start_time + session_start_retry_timeout_;
    } else {
      end_time = session_start_end_time;
    }
    while (!session_.IsEstablished()) {
      ICHECK_EQ(kTvmErrorNoError, session_.Initialize());
      ICHECK_EQ(kTvmErrorNoError, session_.StartSession());

      ::std::chrono::microseconds time_remaining = ::std::max(
          ::std::chrono::microseconds{0}, ::std::chrono::duration_cast<::std::chrono::microseconds>(
                                              end_time - ::std::chrono::steady_clock::now()));

      if (!ReceiveUntil([this]() -> bool { return session_.IsEstablished(); }, time_remaining)) {
        if (session_start_timeout_ != ::std::chrono::microseconds::zero() &&
            end_time >= session_start_end_time) {
          break;
        }
        end_time += session_start_retry_timeout_;
      }
    }

    if (session_.IsEstablished()) {
      write_stream_.SetWriteTimeout(session_established_timeout_);
    }

    return session_.IsEstablished();
  }

  size_t Send(const void* data, size_t size) override {
    const uint8_t* data_bytes = static_cast<const uint8_t*>(data);
    tvm_crt_error_t err = session_.SendMessage(MessageType::kNormal, data_bytes, size);
    ICHECK(err == kTvmErrorNoError) << "SendMessage returned " << err;

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
          ICHECK(message_buffer_ == nullptr || message_buffer_->ReadAvailable() > 0);
          return num_bytes_recv;
        }
      }

      did_receive_message_ = false;
      if (!ReceiveUntil([this]() -> bool { return did_receive_message_; },
                        session_established_timeout_)) {
        if (session_established_timeout_ != ::std::chrono::microseconds::zero()) {
          std::stringstream ss;
          ss << "MicroSessionTimeoutError: failed to read reply message after timeout "
             << session_established_timeout_.count() / 1e6 << "s";

          throw std::runtime_error(ss.str());
        }
      }
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
        break;

      case MessageType::kStartSessionReply:
        state_ = State::kSessionEstablished;
        break;

      case MessageType::kTerminateSession:
        if (state_ == State::kReset) {
          state_ = State::kSessionTerminated;
        } else if (state_ == State::kSessionTerminated) {
          LOG(FATAL) << "SessionTerminatedError: multiple session-terminated messages received; "
                        "device in reboot loop?";
        } else if (state_ == State::kSessionEstablished) {
          LOG(FATAL) << "SessionTerminatedError: remote device terminated connection";
        }
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

        ICHECK_EQ(buf->Read(message, sizeof(message) - 1), message_size_bytes);
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

  State state_;
  ::std::chrono::microseconds session_start_retry_timeout_;
  ::std::chrono::microseconds session_start_timeout_;
  ::std::chrono::microseconds session_established_timeout_;
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
  MicroTransportChannel* micro_channel =
      new MicroTransportChannel(args[1], args[2], ::std::chrono::microseconds(uint64_t(args[3])),
                                ::std::chrono::microseconds(uint64_t(args[4])),
                                ::std::chrono::microseconds(uint64_t(args[5])));
  if (!micro_channel->StartSession()) {
    std::stringstream ss;
    ss << "MicroSessionTimeoutError: session start handshake failed after " << double(args[4]) / 1e6
       << "s";
    throw std::runtime_error(ss.str());
  }
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

void TVMPlatformAbort(int error_code) { ICHECK(false) << "TVMPlatformAbort: " << error_code; }
}
