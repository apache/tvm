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
 * \file framing.cc
 * \brief Framing for RPC.
 */

#include <checksum.h>
#include <string.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/rpc_common/framing.h>

#include "crt_config.h"

// For debugging purposes, Framer logs can be enabled, but this should only be done when
// running from the host. This is done differently from TVMLogf() because TVMLogf() uses the
// framer in its implementation.
#ifdef TVM_CRT_FRAMER_ENABLE_LOGS
#include <cstdio>
#define TVM_FRAMER_DEBUG_LOG(msg, ...) fprintf(stderr, "microTVM framer: " msg " \n", ##__VA_ARGS__)
#define TVM_UNFRAMER_DEBUG_LOG(msg, ...) \
  fprintf(stderr, "microTVM unframer: " msg " \n", ##__VA_ARGS__)
#else
#define TVM_FRAMER_DEBUG_LOG(msg, ...)
#define TVM_UNFRAMER_DEBUG_LOG(msg, ...)
#endif

namespace tvm {
namespace runtime {
namespace micro_rpc {

uint16_t crc16_compute(const uint8_t* data, size_t data_size_bytes, uint16_t* previous_crc) {
  uint16_t crc = (previous_crc != nullptr ? *previous_crc : 0xffff);
  for (size_t i = 0; i < data_size_bytes; ++i) {
    crc = update_crc_ccitt(crc, data[i]);
  }

  return crc;
}

template <typename E>
static constexpr uint8_t to_integral(E e) {
  return static_cast<uint8_t>(e);
}

void Unframer::Reset() {
  state_ = State::kFindPacketStart;
  saw_escape_start_ = false;
  num_buffer_bytes_valid_ = 0;
}

size_t Unframer::BytesNeeded() {
  size_t bytes_needed = 0;
  switch (state_) {
    case State::kFindPacketStart:
      return 1;
    case State::kFindPacketLength:
      bytes_needed = PacketFieldSizeBytes::kPayloadLength;
      break;
    case State::kFindPacketCrc:
      return num_payload_bytes_remaining_;
    case State::kFindCrcEnd:
      bytes_needed = PacketFieldSizeBytes::kCrc;
      break;
    default:
      CHECK(false);
  }

  return bytes_needed > num_buffer_bytes_valid_ ? bytes_needed - num_buffer_bytes_valid_ : 0;
}

tvm_crt_error_t Unframer::Write(const uint8_t* data, size_t data_size_bytes,
                                size_t* bytes_consumed) {
  tvm_crt_error_t return_code = kTvmErrorNoError;
  input_ = data;
  input_size_bytes_ = data_size_bytes;

  while (return_code == kTvmErrorNoError && input_size_bytes_ > 0) {
    TVM_UNFRAMER_DEBUG_LOG("state: %02x size 0x%02zx", to_integral(state_), input_size_bytes_);
    switch (state_) {
      case State::kFindPacketStart:
        return_code = FindPacketStart();
        break;
      case State::kFindPacketLength:
        return_code = FindPacketLength();
        break;
      case State::kFindPacketCrc:
        return_code = FindPacketCrc();
        break;
      case State::kFindCrcEnd:
        return_code = FindCrcEnd();
        break;
      default:
        return_code = kTvmErrorFramingInvalidState;
        break;
    }
  }

  *bytes_consumed = data_size_bytes - input_size_bytes_;
  input_ = nullptr;
  input_size_bytes_ = 0;

  if (return_code != kTvmErrorNoError) {
    state_ = State::kFindPacketStart;
    ClearBuffer();
  }

  return return_code;
}

tvm_crt_error_t Unframer::FindPacketStart() {
  size_t i;
  for (i = 0; i < input_size_bytes_; ++i) {
    if (input_[i] == to_integral(Escape::kEscapeStart)) {
      saw_escape_start_ = true;
    } else if (input_[i] == to_integral(Escape::kPacketStart) && saw_escape_start_) {
      uint8_t packet_start_sequence[2]{to_integral(Escape::kEscapeStart),
                                       to_integral(Escape::kPacketStart)};
      crc_ = crc16_compute(packet_start_sequence, sizeof(packet_start_sequence), nullptr);
      saw_escape_start_ = false;
      state_ = State::kFindPacketLength;
      i++;
      break;
    } else {
      saw_escape_start_ = false;
    }
  }

  input_ += i;
  input_size_bytes_ -= i;
  return kTvmErrorNoError;
}

tvm_crt_error_t Unframer::ConsumeInput(uint8_t* buffer, size_t buffer_size_bytes,
                                       size_t* bytes_filled, bool update_crc) {
  CHECK(*bytes_filled < buffer_size_bytes);
  tvm_crt_error_t to_return = kTvmErrorNoError;
  size_t i;
  for (i = 0; i < input_size_bytes_; ++i) {
    uint8_t c = input_[i];
    if (saw_escape_start_) {
      saw_escape_start_ = false;
      if (c == to_integral(Escape::kPacketStart)) {
        // When the start packet sequence is seen, abort unframing the current packet. Since the
        // escape byte has already been parsed, update the CRC include only the escape byte. This
        // readies the unframer to consume the kPacketStart byte on the next Write() call.
        uint8_t escape_start = to_integral(Escape::kEscapeStart);
        crc_ = crc16_compute(&escape_start, 1, nullptr);
        to_return = kTvmErrorFramingShortPacket;
        saw_escape_start_ = true;

        break;
      } else if (c == to_integral(Escape::kEscapeNop)) {
        continue;
      } else if (c == to_integral(Escape::kEscapeStart)) {
        // do nothing (allow character to be printed)
      } else {
        // Invalid escape sequence.
        to_return = kTvmErrorFramingInvalidEscape;
        i++;
        break;
      }
    } else if (c == to_integral(Escape::kEscapeStart)) {
      saw_escape_start_ = true;
      continue;
    } else {
      saw_escape_start_ = false;
    }

    buffer[*bytes_filled] = c;
    (*bytes_filled)++;
    if (*bytes_filled == buffer_size_bytes) {
      i++;
      break;
    }
  }

  if (update_crc) {
    crc_ = crc16_compute(input_, i, &crc_);
  }

  input_ += i;
  input_size_bytes_ -= i;
  return to_return;
}

tvm_crt_error_t Unframer::AddToBuffer(size_t buffer_full_bytes, bool update_crc) {
  CHECK(!IsBufferFull(buffer_full_bytes));
  return ConsumeInput(buffer_, buffer_full_bytes, &num_buffer_bytes_valid_, update_crc);
}

void Unframer::ClearBuffer() { num_buffer_bytes_valid_ = 0; }

tvm_crt_error_t Unframer::FindPacketLength() {
  tvm_crt_error_t to_return = AddToBuffer(PacketFieldSizeBytes::kPayloadLength, true);
  if (to_return != kTvmErrorNoError) {
    return to_return;
  }

  if (!IsBufferFull(PacketFieldSizeBytes::kPayloadLength)) {
    return to_return;
  }

  num_payload_bytes_remaining_ = *reinterpret_cast<uint32_t*>(buffer_);
  TVM_UNFRAMER_DEBUG_LOG("payload length: 0x%zx", num_payload_bytes_remaining_);
  ClearBuffer();
  state_ = State::kFindPacketCrc;
  return to_return;
}

tvm_crt_error_t Unframer::FindPacketCrc() {
  //  CHECK(num_buffer_bytes_valid_ == 0);
  while (num_payload_bytes_remaining_ > 0) {
    size_t num_bytes_to_buffer = num_payload_bytes_remaining_;
    if (num_bytes_to_buffer > sizeof(buffer_)) {
      num_bytes_to_buffer = sizeof(buffer_);
    }

    // remember in case we need to rewind due to WriteAll() error.
    size_t prev_input_size_bytes = input_size_bytes_;
    size_t prev_num_buffer_bytes_valid = num_buffer_bytes_valid_;
    {
      tvm_crt_error_t to_return = AddToBuffer(num_bytes_to_buffer, true);
      if (to_return != kTvmErrorNoError) {
        return to_return;
      }
    }

    if (prev_num_buffer_bytes_valid == num_buffer_bytes_valid_) {
      // Return if no bytes were consumed from the input.
      return kTvmErrorNoError;
    }

    {
      size_t bytes_consumed;
      tvm_crt_error_t to_return =
          stream_->WriteAll(buffer_, num_buffer_bytes_valid_, &bytes_consumed);
      num_payload_bytes_remaining_ -= bytes_consumed;
      if (to_return != kTvmErrorNoError) {
        // rewind input, skipping escape bytes.
        size_t buffer_bytes_consumed;
        const uint8_t* input = input_ - (prev_input_size_bytes - input_size_bytes_);
        for (buffer_bytes_consumed = 0; bytes_consumed > 0; ++buffer_bytes_consumed) {
          if (input[buffer_bytes_consumed] != uint8_t(Escape::kEscapeStart)) {
            bytes_consumed--;
          }
        }

        size_t bytes_to_rewind = prev_input_size_bytes - buffer_bytes_consumed;
        input_ -= bytes_to_rewind;
        input_size_bytes_ += bytes_to_rewind;

        // must not have seen escape, since AddToBuffer won't stop in the middle.
        saw_escape_start_ = false;

        return to_return;
      }
    }

    ClearBuffer();
  }

  if (num_payload_bytes_remaining_ == 0) {
    state_ = State::kFindCrcEnd;
  }

  return kTvmErrorNoError;
}

tvm_crt_error_t Unframer::FindCrcEnd() {
  tvm_crt_error_t to_return = AddToBuffer(PacketFieldSizeBytes::kCrc, false);
  if (to_return != kTvmErrorNoError) {
    return to_return;
  }

  if (!IsBufferFull(PacketFieldSizeBytes::kCrc)) {
    return kTvmErrorNoError;
  }

  // TODO(areusch): Handle endianness.
  stream_->PacketDone(crc_ == *reinterpret_cast<uint16_t*>(buffer_));
  ClearBuffer();
  state_ = State::kFindPacketStart;
  return kTvmErrorNoError;
}

void Framer::Reset() { state_ = State::kReset; }

tvm_crt_error_t Framer::Write(const uint8_t* payload, size_t payload_size_bytes) {
  tvm_crt_error_t to_return;
  to_return = StartPacket(payload_size_bytes);
  if (to_return != kTvmErrorNoError) {
    return to_return;
  }

  to_return = WritePayloadChunk(payload, payload_size_bytes);
  if (to_return != 0) {
    return to_return;
  }

  to_return = FinishPacket();
  return to_return;
}

tvm_crt_error_t Framer::StartPacket(size_t payload_size_bytes) {
  uint8_t packet_header[sizeof(uint32_t)];
  size_t ptr = 0;
  if (state_ == State::kReset) {
    packet_header[ptr] = to_integral(Escape::kEscapeNop);
    ptr++;
    tvm_crt_error_t to_return =
        WriteAndCrc(packet_header, ptr, false /* escape */, false /* update_crc */);
    if (to_return != kTvmErrorNoError) {
      return to_return;
    }

    ptr = 0;
  }

  packet_header[ptr] = to_integral(Escape::kEscapeStart);
  ptr++;
  packet_header[ptr] = to_integral(Escape::kPacketStart);
  ptr++;

  crc_ = 0xffff;
  tvm_crt_error_t to_return =
      WriteAndCrc(packet_header, ptr, false /* escape */, true /* update_crc */);
  if (to_return != kTvmErrorNoError) {
    return to_return;
  }

  uint32_t payload_size_wire = payload_size_bytes;
  to_return = WriteAndCrc(reinterpret_cast<uint8_t*>(&payload_size_wire), sizeof(payload_size_wire),
                          true /* escape */, true /* update_crc */);
  if (to_return == kTvmErrorNoError) {
    state_ = State::kTransmitPacketPayload;
    num_payload_bytes_remaining_ = payload_size_bytes;
  }

  return to_return;
}

tvm_crt_error_t Framer::WriteAndCrc(const uint8_t* data, size_t data_size_bytes, bool escape,
                                    bool update_crc) {
  while (data_size_bytes > 0) {
    uint8_t buffer[kMaxStackBufferSizeBytes];
    size_t buffer_ptr = 0;
    size_t i;
    for (i = 0; i < data_size_bytes && buffer_ptr != kMaxStackBufferSizeBytes; ++i) {
      uint8_t c = data[i];
      if (!escape || c != to_integral(Escape::kEscapeStart)) {
        buffer[buffer_ptr] = c;
        buffer_ptr++;
        continue;
      }

      if (buffer_ptr == kMaxStackBufferSizeBytes - 1) {
        break;
      }

      buffer[buffer_ptr] = to_integral(Escape::kEscapeStart);
      buffer_ptr++;

      buffer[buffer_ptr] = to_integral(Escape::kEscapeStart);
      buffer_ptr++;
    }

    size_t bytes_consumed;
    tvm_crt_error_t to_return = stream_->WriteAll(buffer, buffer_ptr, &bytes_consumed);
    if (to_return != kTvmErrorNoError) {
      return to_return;
    }

    if (update_crc) {
      crc_ = crc16_compute(buffer, buffer_ptr, &crc_);
    }

    data_size_bytes -= i;
    data += i;
  }

  return kTvmErrorNoError;
}

tvm_crt_error_t Framer::WritePayloadChunk(const uint8_t* payload_chunk,
                                          size_t payload_chunk_size_bytes) {
  if (state_ != State::kTransmitPacketPayload) {
    return kTvmErrorFramingInvalidState;
  } else if (payload_chunk_size_bytes > num_payload_bytes_remaining_) {
    return kTvmErrorFramingPayloadOverflow;
  }

  TVM_FRAMER_DEBUG_LOG("write payload chunk: %" PRIuMAX " bytes", payload_chunk_size_bytes);
  tvm_crt_error_t to_return = WriteAndCrc(payload_chunk, payload_chunk_size_bytes,
                                          true /* escape */, true /* update_crc */);
  if (to_return != kTvmErrorNoError) {
    state_ = State::kReset;
    return to_return;
  }

  num_payload_bytes_remaining_ -= payload_chunk_size_bytes;
  return kTvmErrorNoError;
}

tvm_crt_error_t Framer::FinishPacket() {
  if (state_ != State::kTransmitPacketPayload) {
    return kTvmErrorFramingInvalidState;
  } else if (num_payload_bytes_remaining_ != 0) {
    return kTvmErrorFramingPayloadIncomplete;
  }

  tvm_crt_error_t to_return = WriteAndCrc(reinterpret_cast<uint8_t*>(&crc_), sizeof(crc_),
                                          true /* escape */, false /* update_crc */);
  if (to_return != kTvmErrorNoError) {
    TVM_FRAMER_DEBUG_LOG("write and crc returned: %02x", to_return);
    state_ = State::kReset;
  } else {
    state_ = State::kIdle;
  }
  return to_return;
}

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm
