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
 * \file session.h
 * \brief RPC Session
 */

#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/rpc_common/session.h>

#include "crt_config.h"

namespace tvm {
namespace runtime {
namespace micro_rpc {

struct microtvm_session_start_payload_t {
  uint8_t version;
};

void Session::RegenerateNonce() {
  local_nonce_ = (((local_nonce_ << 5) | (local_nonce_ >> 5)) + 1);

  if (local_nonce_ == kInvalidNonce) {
    local_nonce_++;
  }
}

tvm_crt_error_t Session::SendInternal(MessageType message_type, const uint8_t* message_data,
                                      size_t message_size_bytes) {
  tvm_crt_error_t to_return = StartMessage(message_type, message_size_bytes);
  if (to_return != kTvmErrorNoError) {
    return to_return;
  }

  if (message_size_bytes > 0) {
    to_return = SendBodyChunk(message_data, message_size_bytes);
    if (to_return != kTvmErrorNoError) {
      return to_return;
    }
  }

  return framer_->FinishPacket();
}

tvm_crt_error_t Session::StartMessage(MessageType message_type, size_t message_size_bytes) {
  SessionHeader header{session_id_, message_type};
  if (message_type == MessageType::kLog) {
    header.session_id = 0;
  }

  tvm_crt_error_t to_return = framer_->StartPacket(message_size_bytes + sizeof(SessionHeader));
  if (to_return != 0) {
    return to_return;
  }

  return framer_->WritePayloadChunk(reinterpret_cast<uint8_t*>(&header), sizeof(SessionHeader));
}

tvm_crt_error_t Session::SendBodyChunk(const uint8_t* chunk, size_t chunk_size_bytes) {
  return framer_->WritePayloadChunk(chunk, chunk_size_bytes);
}

tvm_crt_error_t Session::FinishMessage() { return framer_->FinishPacket(); }

tvm_crt_error_t Session::StartSession() {
  CHECK_NE(state_, State::kReset, "must call Initialize");

  RegenerateNonce();
  SetSessionId(local_nonce_, 0);
  microtvm_session_start_payload_t payload = {Session::kVersion};
  tvm_crt_error_t to_return = SendInternal(MessageType::kStartSessionInit,
                                           reinterpret_cast<uint8_t*>(&payload), sizeof(payload));
  if (to_return == 0) {
    state_ = State::kStartSessionSent;
  }

  return to_return;
}

tvm_crt_error_t Session::Initialize(uint8_t initial_session_nonce) {
  local_nonce_ = initial_session_nonce;
  return TerminateSession();
}

tvm_crt_error_t Session::TerminateSession() {
  SetSessionId(0, 0);
  state_ = State::kNoSessionEstablished;
  return SendInternal(MessageType::kTerminateSession, nullptr, 0);
}

tvm_crt_error_t Session::SendMessage(MessageType message_type, const uint8_t* message_data,
                                     size_t message_size_bytes) {
  if (state_ != State::kSessionEstablished && message_type != MessageType::kLog) {
    return kTvmErrorSessionInvalidState;
  }

  return SendInternal(message_type, message_data, message_size_bytes);
}

ssize_t Session::SessionReceiver::Write(const uint8_t* data, size_t data_size_bytes) {
  if (session_->receive_buffer_has_complete_message_) {
    return kTvmErrorSessionReceiveBufferBusy;
  }

  size_t bytes_written = session_->receive_buffer_->Write(data, data_size_bytes);
  if (bytes_written != data_size_bytes) {
    return kTvmErrorSessionReceiveBufferShortWrite;
  }

  return bytes_written;
}

void Session::SessionReceiver::PacketDone(bool is_valid) {
  if (!is_valid) {
    return;
  }

  SessionHeader header;
  int bytes_read =
      session_->receive_buffer_->Read(reinterpret_cast<uint8_t*>(&header), sizeof(header));
  if (bytes_read != sizeof(header)) {
    return;
  }
  session_->receive_buffer_has_complete_message_ = true;

  switch (header.message_type) {
    case MessageType::kStartSessionInit:
      session_->ProcessStartSessionInit(header);
      session_->receive_buffer_has_complete_message_ = false;
      break;
    case MessageType::kStartSessionReply:
      session_->ProcessStartSessionReply(header);
      session_->receive_buffer_has_complete_message_ = false;
      break;
    case MessageType::kTerminateSession:
      if (session_->state_ == State::kSessionEstablished) {
        session_->state_ = State::kNoSessionEstablished;
        session_->OnSessionTerminatedMessage();
      }
      session_->receive_buffer_has_complete_message_ = false;
      break;
    case MessageType::kLog:
      if (header.session_id == 0 || header.session_id == session_->session_id_) {
        // Special case for log messages: session id can be 0.
        session_->message_received_func_(session_->message_received_func_context_,
                                         header.message_type, session_->receive_buffer_);
      }
      break;
    default:
      if (session_->state_ == State::kSessionEstablished &&
          header.session_id == session_->session_id_) {
        session_->message_received_func_(session_->message_received_func_context_,
                                         header.message_type, session_->receive_buffer_);
      }
      break;
  }
}

void Session::ClearReceiveBuffer() {
  receive_buffer_has_complete_message_ = false;
  receive_buffer_->Clear();
}

void Session::SendSessionStartReply(const SessionHeader& header) {
  RegenerateNonce();
  SetSessionId(InitiatorNonce(header.session_id), local_nonce_);
  microtvm_session_start_payload_t payload = {Session::kVersion};
  tvm_crt_error_t to_return = SendInternal(MessageType::kStartSessionReply,
                                           reinterpret_cast<uint8_t*>(&payload), sizeof(payload));
  state_ = State::kSessionEstablished;
  CHECK_EQ(to_return, kTvmErrorNoError, "SendSessionStartReply");
  OnSessionEstablishedMessage();
}

void Session::ProcessStartSessionInit(const SessionHeader& header) {
  if (InitiatorNonce(header.session_id) == kInvalidNonce) {
    return;
  }

  microtvm_session_start_payload_t payload;
  int bytes_read = receive_buffer_->Read(reinterpret_cast<uint8_t*>(&payload), sizeof(payload));
  if (bytes_read != sizeof(payload)) {
    return;
  }

  switch (state_) {
    case State::kReset:
    case State::kNoSessionEstablished:
      // Normal case: received a StartSession packet from reset.
      SendSessionStartReply(header);
      break;

    case State::kStartSessionSent:
      // When two StartSessionInit packets sent simultaneously: lowest nonce wins; ties retry.
      if (InitiatorNonce(header.session_id) < local_nonce_) {
        if (payload.version == Session::kVersion) {
          SendSessionStartReply(header);
        }
      } else if (InitiatorNonce(header.session_id) == local_nonce_) {
        StartSession();
      }

      break;

    case State::kSessionEstablished:
      SendSessionStartReply(header);
      OnSessionEstablishedMessage();
      break;

    default:
      state_ = State::kReset;
  }
}

void Session::ProcessStartSessionReply(const SessionHeader& header) {
  if (ResponderNonce(header.session_id) == kInvalidNonce) {
    return;
  }

  microtvm_session_start_payload_t payload;
  int bytes_read = receive_buffer_->Read(reinterpret_cast<uint8_t*>(&payload), sizeof(payload));
  if (bytes_read != sizeof(payload)) {
    return;
  }

  switch (state_) {
    case State::kReset:
    case State::kNoSessionEstablished:
      break;
    case State::kStartSessionSent:
      if (InitiatorNonce(header.session_id) == local_nonce_ &&
          payload.version == Session::kVersion) {
        SetSessionId(local_nonce_, ResponderNonce(header.session_id));
        state_ = State::kSessionEstablished;
        OnSessionEstablishedMessage();
      }
      break;
    case State::kSessionEstablished:
      if (InitiatorNonce(header.session_id) != kInvalidNonce &&
          ResponderNonce(header.session_id) == kInvalidNonce) {
        if (payload.version == Session::kVersion) {
          SendSessionStartReply(header);
        } else {
          SetSessionId(local_nonce_, 0);
          state_ = State::kReset;
        }
      } else {
        state_ = State::kReset;
      }
      break;
  }
}

void Session::OnSessionEstablishedMessage() {
  message_received_func_(message_received_func_context_, MessageType::kStartSessionReply, NULL);
}

void Session::OnSessionTerminatedMessage() {
  message_received_func_(message_received_func_context_, MessageType::kTerminateSession, NULL);
}

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm
