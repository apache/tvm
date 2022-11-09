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
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "../../support/str_escape.h"
#include "../rpc/rpc_channel.h"
#include "../rpc/rpc_channel_logger.h"
#include "../rpc/rpc_endpoint.h"
#include "../rpc/rpc_session.h"
#include "crt_config.h"

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
    if (write_timeout_ == ::std::chrono::microseconds::zero()) {
      fsend_(bytes, nullptr);
    } else {
      fsend_(bytes, write_timeout_.count());
    }

    return static_cast<ssize_t>(data_size_bytes);
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

  /*!
   * \brief Construct a new MicroTransportChannel.
   * \param fsend A PackedFunc accepting (data_bytes, timeout_usec) and returning the number of
   *  bytes sent. If a timeout_usec elapses before all data is sent, it should return 0.
   * \param frecv A PackedFunc accepting (num_bytes, timeout_usec) and returning a string containing
   *  the received data. Must not return an empty string, except to indicate a timeout.
   * \param session_start_retry_timeout During session initialization, the session start message is
   *  re-sent after this many microseconds elapse without a reply. If 0, the session start message
   *  is sent only once.
   * \param session_start_timeout Session initialization is considered "timed out" if no reply is
   *  received this many microseconds after the session start is sent. If 0, a session start never
   *  times out.
   * \param session_established_timeout Timeout used for the Recv() function. This is used for
   *  messages sent after a session is already established. If 0, Recv() never times out.
   */
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
        session_{&framer_, &receive_buffer_, &HandleMessageReceivedCb, this},
        unframer_{session_.Receiver()},
        did_receive_message_{false},
        frecv_{frecv},
        message_buffer_{nullptr} {}

 private:
  static constexpr const size_t kReceiveBufferSizeBytes = 128;

  /*
   * \brief Receive data until either pf() returns true or a timeout occurs.
   *
   * The condition function is called first, so this function may return without performing a read.
   * Following this call, received data is consumed and frecv_ is invoked until the timeout occurs
   * or the condition function passes.
   *
   * \param pf A condition function that returns true when enough data has been received for the
   *  caller to proceed.
   * \param timeout Pointer to number of microseconds to wait before timing out. If nullptr, no
   *  timeout ever occurs in this function, so it may block forever. If 0, a single non-blocking
   *  read is performed, and any data returned is processed.
   * \return true if the condition passed, false if the timeout expired.
   */
  bool ReceiveUntil(TypedPackedFunc<bool(void)> pf, ::std::chrono::microseconds* timeout) {
    if (pf()) {
      return true;
    }

    auto end_time = ::std::chrono::steady_clock::now();
    if (timeout != nullptr) {
      end_time += *timeout;
    }
    for (;;) {
      if (ConsumeReceivedPayload(pf)) {
        return true;
      }

      ::std::string chunk;
      size_t bytes_needed = unframer_.BytesNeeded();
      CHECK_GT(bytes_needed, 0) << "unframer unexpectedly needs no data";
      if (timeout != nullptr) {
        ::std::chrono::microseconds iter_timeout{
            ::std::max(::std::chrono::microseconds{0},
                       ::std::chrono::duration_cast<::std::chrono::microseconds>(
                           end_time - ::std::chrono::steady_clock::now()))};
        chunk = frecv_(bytes_needed, iter_timeout.count()).operator std::string();
      } else {
        chunk = frecv_(bytes_needed, nullptr).operator std::string();
      }
      pending_chunk_ = chunk;
      if (pending_chunk_.size() == 0) {
        // Timeout occurred
        return false;
      }
    }
  }

  static constexpr const int kNumRandRetries = 10;
  static std::atomic<unsigned int> random_seed;

  inline uint8_t GenerateRandomNonce() {
    // NOTE: this is bad concurrent programming but in practice we don't really expect race
    // conditions here, and even if they occur we don't particularly care whether a competing
    // process computes a different random seed. This value is just chosen pseudo-randomly to
    // form an initial distinct session id. Here we just want to protect against bad loads causing
    // confusion.
    unsigned int seed = random_seed.load();
    if (seed == 0) {
#if defined(_MSC_VER)
      seed = (unsigned int)time(nullptr);
      srand(seed);
#else
      seed = (unsigned int)time(nullptr);
#endif
    }
    uint8_t initial_nonce = 0;
    for (int i = 0; i < kNumRandRetries && initial_nonce == 0; ++i) {
#if defined(_MSC_VER)
      initial_nonce = rand();  // NOLINT(runtime/threadsafe_fn)
#else
      initial_nonce = rand_r(&seed);
#endif
    }
    random_seed.store(seed);
    ICHECK_NE(initial_nonce, 0) << "rand() does not seem to be producing random values";
    return initial_nonce;
  }

  bool StartSessionInternal() {
    using ::std::chrono::duration_cast;
    using ::std::chrono::microseconds;
    using ::std::chrono::steady_clock;

    steady_clock::time_point start_time = steady_clock::now();
    ICHECK_EQ(kTvmErrorNoError, session_.Initialize(GenerateRandomNonce()));
    ICHECK_EQ(kTvmErrorNoError, session_.StartSession());

    if (session_start_timeout_ == microseconds::zero() &&
        session_start_retry_timeout_ == microseconds::zero()) {
      ICHECK(ReceiveUntil([this]() -> bool { return session_.IsEstablished(); }, nullptr))
          << "ReceiveUntil indicated timeout expired, but no timeout set!";
      ICHECK(session_.IsEstablished()) << "Session not established, but should be";
      return true;
    }

    auto session_start_end_time = start_time + session_start_timeout_;
    steady_clock::time_point end_time;
    if (session_start_retry_timeout_ != ::std::chrono::microseconds::zero()) {
      end_time = start_time + session_start_retry_timeout_;
    } else {
      end_time = session_start_end_time;
    }

    while (!session_.IsEstablished()) {
      microseconds time_remaining =
          ::std::max(microseconds{0}, duration_cast<microseconds>(end_time - steady_clock::now()));
      if (ReceiveUntil([this]() -> bool { return session_.IsEstablished(); }, &time_remaining)) {
        break;
      }

      if (session_start_timeout_ != microseconds::zero() && end_time >= session_start_end_time) {
        return false;
      }
      end_time += session_start_retry_timeout_;

      ICHECK_EQ(kTvmErrorNoError, session_.Initialize(GenerateRandomNonce()));
      ICHECK_EQ(kTvmErrorNoError, session_.StartSession());
    }

    return true;
  }

 public:
  bool StartSession() {
    ICHECK(state_ == State::kReset)
        << "MicroSession: state_: expected kReset, got " << uint8_t(state_);

    bool to_return = StartSessionInternal();
    if (to_return) {
      write_stream_.SetWriteTimeout(session_established_timeout_);
    }

    return to_return;
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
      if (session_established_timeout_ == ::std::chrono::microseconds::zero()) {
        ICHECK(ReceiveUntil([this]() -> bool { return did_receive_message_; }, nullptr))
            << "ReceiveUntil timeout expired, but no timeout configured!";
      } else {
        if (!ReceiveUntil([this]() -> bool { return did_receive_message_; },
                          &session_established_timeout_)) {
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
  /*!
   * \brief Consume the entire received payload, unless the pf condition is met halfway through.
   *
   * This function expects pending_chunk_ to contain a chunk of unprocessed packet data. It
   * repeatedly writes the chunk to the Unframer until either a) pf() returns True or b) no more
   * data remains to be written.
   *
   * \param pf A PackedFunc which returns true when ReceiveUntil should return.
   * \returns true if pf() returned true during processing; false otherwise.
   */
  bool ConsumeReceivedPayload(TypedPackedFunc<bool(void)> pf) {
    while (pending_chunk_.size() > 0) {
      size_t bytes_consumed = 0;
      int unframer_error = unframer_.Write((const uint8_t*)pending_chunk_.data(),
                                           pending_chunk_.size(), &bytes_consumed);

      ICHECK(bytes_consumed <= pending_chunk_.size())
          << "consumed " << bytes_consumed << " want <= " << pending_chunk_.size();
      pending_chunk_ = pending_chunk_.substr(bytes_consumed);
      if (unframer_error < 0) {
        LOG(ERROR) << "unframer got error code: " << unframer_error;
      } else {
        if (pf()) {
          return true;
        }
      }
    }

    return false;
  }

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

std::atomic<unsigned int> MicroTransportChannel::random_seed{0};

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
  bool enable_logging = false;
  if (args.num_args > 7) {
    enable_logging = args[7];
  }
  if (enable_logging) {
    channel.reset(new RPCChannelLogging(std::move(channel)));
  }
  auto ep = RPCEndpoint::Create(std::move(channel), args[0], "", args[6]);
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
