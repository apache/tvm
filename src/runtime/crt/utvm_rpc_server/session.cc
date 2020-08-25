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

#include "session.h"

#include <tvm/runtime/crt/logging.h>

#include "crt_config.h"

namespace tvm {
namespace runtime {

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
  SessionHeader header{outbound_session_id(), message_type};
  if (state_ != State::kSessionEstablished && message_type == MessageType::kLogMessage) {
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
  RegenerateNonce();
  remote_nonce_ = kInvalidNonce;
  tvm_crt_error_t to_return = SendInternal(MessageType::kStartSessionMessage, nullptr, 0);
  if (to_return == 0) {
    state_ = State::kStartSessionSent;
  }

  return to_return;
}

tvm_crt_error_t Session::SendMessage(MessageType message_type, const uint8_t* message_data,
                                     size_t message_size_bytes) {
  if (state_ != State::kSessionEstablished && message_type != MessageType::kLogMessage) {
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
  // LOG_DEBUG("MessageDone: %zu/%zu",
  //           session_->receive_buffer_->ReadAvailable(),
  //           session_->receive_buffer_->Size());

  switch (header.message_type) {
    case MessageType::kStartSessionMessage:
      session_->ProcessStartSession(header);
      session_->receive_buffer_has_complete_message_ = false;
      break;
    case MessageType::kLogMessage:
      if (header.session_id == 0 || header.session_id == session_->inbound_session_id()) {
        // Special case for log messages: session id can be 0.
        session_->message_received_func_(session_->message_received_func_context_,
                                         header.message_type, session_->receive_buffer_);
      }
      break;
    default:
      if (session_->state_ == State::kSessionEstablished &&
          header.session_id == session_->inbound_session_id()) {
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
  remote_nonce_ = sender_nonce(header.session_id);
  tvm_crt_error_t to_return = SendInternal(MessageType::kStartSessionMessage, nullptr, 0);
  CHECK_EQ(to_return, kTvmErrorNoError, "SendSessionStartReply");
}

void Session::ProcessStartSession(const SessionHeader& header) {
  if (header.session_id == 0) {
    return;
  }

  uint8_t remote_nonce = sender_nonce(header.session_id);
  uint8_t my_nonce = receiver_nonce(header.session_id);
  switch (state_) {
    case State::kReset:
      if (remote_nonce != 0 && my_nonce == 0) {
        // Normal case: received a StartSession packet from reset.
        SendSessionStartReply(header);
        state_ = State::kSessionEstablished;
        OnSessionEstablishedMessage();
      }
      // NOTE: don't issue any StartSession packets to try and rescue other cases. This prevents
      // runaway livelock.
      break;

    case State::kStartSessionSent:
      if (my_nonce == local_nonce_ && remote_nonce != kInvalidNonce) {
        remote_nonce_ = remote_nonce;
        state_ = State::kSessionEstablished;
        OnSessionEstablishedMessage();
      } else if (my_nonce == kInvalidNonce) {
        // if both sides sent StartSession simultaneously, the lowest nonce sent is the initiator.
        // if both sides choose the same non-zero nonce as the initiating StartSession packet,
        // retry.
        if (remote_nonce == local_nonce_) {
          CHECK_EQ(StartSession(), kTvmErrorNoError, "StartSession");
          state_ = State::kStartSessionSent;
        } else if (remote_nonce < local_nonce_) {
          SendSessionStartReply(header);
          state_ = State::kSessionEstablished;
          OnSessionEstablishedMessage();
        }
      }
      break;

    case State::kSessionEstablished:
      if (remote_nonce != kInvalidNonce && my_nonce == kInvalidNonce) {
        SendSessionStartReply(header);
        OnSessionEstablishedMessage();
      } else {
        state_ = State::kReset;
      }
      break;

    default:
      state_ = State::kReset;
  }
}

void Session::OnSessionEstablishedMessage() {
  message_received_func_(message_received_func_context_, MessageType::kStartSessionMessage, NULL);
}

}  // namespace runtime
}  // namespace tvm
