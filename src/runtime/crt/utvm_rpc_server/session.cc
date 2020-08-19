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
  nonce_ = (((nonce_ << 5) | (nonce_ >> 5)) + 1);

  if (nonce_ == 0) {
    nonce_++;
  }
}

int Session::SendInternal(MessageType message_type, const uint8_t* message_data, size_t message_size_bytes) {
  int to_return = StartMessage(message_type, message_size_bytes);
  if (to_return != 0) {
    return to_return;
  }

  to_return = SendBodyChunk(message_data, message_size_bytes);
  if (to_return != 0) {
    return to_return;
  }

  return framer_->FinishPacket();
}

int Session::StartMessage(MessageType message_type, size_t message_size_bytes) {
  SessionHeader header{session_id_, message_type};
  if (state_ != State::kSessionEstablished && message_type == MessageType::kLogMessage) {
    header.session_id = 0;
  }

  int to_return = framer_->StartPacket(message_size_bytes + sizeof(SessionHeader));
  if (to_return != 0) {
    return to_return;
  }

  return framer_->WritePayloadChunk(reinterpret_cast<uint8_t*>(&header), sizeof(SessionHeader));
}

int Session::SendBodyChunk(const uint8_t* chunk, size_t chunk_size_bytes) {
  return framer_->WritePayloadChunk(chunk, chunk_size_bytes);
}

int Session::FinishMessage() {
  return framer_->FinishPacket();
}

int Session::StartSession() {
  RegenerateNonce();
  session_id_ = nonce_;
  int to_return = SendInternal(MessageType::kStartSessionMessage, nullptr, 0);
  if (to_return == 0) {
    state_ = State::kStartSessionSent;
  }

  return to_return;
}

int Session::SendMessage(MessageType message_type, const uint8_t* message_data, size_t message_size_bytes) {
  if (state_ != State::kSessionEstablished && message_type != MessageType::kLogMessage) {
    return -1;
  }

  return SendInternal(message_type, message_data, message_size_bytes);
}

ssize_t Session::SessionReceiver::Write(const uint8_t* data, size_t data_size_bytes) {
  if (session_->receive_buffer_has_complete_message_) {
    return -1;
  }

  size_t bytes_written = session_->receive_buffer_->Write(data, data_size_bytes);
  if (bytes_written != data_size_bytes) {
    return -1;
  }

  return bytes_written;
}

void Session::SessionReceiver::PacketDone(bool is_valid) {
  if (!is_valid) {
    return;
  }

  SessionHeader header;
  int bytes_read = session_->receive_buffer_->Read(
    reinterpret_cast<uint8_t*>(&header), sizeof(header));
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
    if (header.session_id == 0 || header.session_id == session_->session_id_) {
      // Special case for log messages: session id can be 0.
      session_->message_received_func_(
        session_->message_received_func_context_, header.message_type, session_->receive_buffer_);
    }
    break;
  default:
    if (session_->state_ == State::kSessionEstablished &&
        header.session_id == session_->session_id_) {
      session_->message_received_func_(
        session_->message_received_func_context_, header.message_type, session_->receive_buffer_);
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
  session_id_ = (header.session_id & 0xff) | (nonce_ << 8);
  int to_return = SendInternal(MessageType::kStartSessionMessage, nullptr, 0);
  CHECK(to_return == 0);
}

void Session::ProcessStartSession(const SessionHeader& header) {
  switch (state_) {
  case State::kReset:
    if ((header.session_id & 0xff) != 0 &&
        ((header.session_id >> 8) & 0xff) == 0) {
      SendSessionStartReply(header);
      state_ = State::kSessionEstablished;
    } else {
      CHECK(StartSession() == 0);
      state_ = State::kStartSessionSent;
    }
    break;

  case State::kStartSessionSent:
    if ((header.session_id & 0xff) == nonce_) {
      session_id_ = header.session_id;
      state_ = State::kSessionEstablished;
    } else {
      CHECK(StartSession() == 0);
      state_ = State::kStartSessionSent;
    }
    break;

  case State::kSessionEstablished:
    if (header.session_id != session_id_ &&
        ((header.session_id >> 8) & 0xff) == 0) {
      SendSessionStartReply(header);
    } else {
      state_ = State::kReset;
    }
    break;

  default:
    state_ = State::kReset;
  }
}

}  // namespace runtime
}  // namespace tvm
