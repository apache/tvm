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
 * \file framing.h
 * \brief Framing for RPC.
 */

#ifndef TVM_RUNTIME_CRT_RPC_COMMON_FRAMING_H_
#define TVM_RUNTIME_CRT_RPC_COMMON_FRAMING_H_

#include <inttypes.h>
#include <stddef.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/rpc_common/write_stream.h>

namespace tvm {
namespace runtime {
namespace micro_rpc {

uint16_t crc16_compute(const uint8_t* data, size_t data_size_bytes, uint16_t* previous_crc);

enum class Escape : uint8_t { kEscapeStart = 0xff, kEscapeNop = 0xfe, kPacketStart = 0xfd };

class PacketFieldSizeBytes {
 public:
  static constexpr const size_t kPayloadLength = sizeof(uint32_t);
  static constexpr const size_t kCrc = sizeof(uint16_t);
};

class Unframer {
 public:
  explicit Unframer(WriteStream* stream)
      : stream_{stream},
        state_{State::kFindPacketStart},
        saw_escape_start_{false},
        num_buffer_bytes_valid_{0} {}

  /*!
   * \brief Push data into unframer and try to decode one packet.
   *
   * This function will return when exactly one packet has been decoded. It may not consume all of
   * `data` in this case, and valid bytes may remain at the end of data.
   *
   * \param data The new data to unframe and send downstream.
   * \param data_size_bytes The number of valid bytes in data.
   * \param bytes_consumed Pointer written with the number of bytes consumed from data.
   * \return
   *     - kTvmErrorNoError when successful -- continue writing data.
   *     - kTvmErrorFramingInvalidState when the Unframer was in or enters an invalid state
   *       (probably indicates memory corruption).
   *     - kTvmErrorFramingShortPacket when a new packet started before the current one ended.
   *     - kTvmErrorFramingInvalidEscape when an invalid escape sequence was seen
   */
  tvm_crt_error_t Write(const uint8_t* data, size_t data_size_bytes, size_t* bytes_consumed);

  /*! \brief Reset unframer to initial state. */
  void Reset();

  /*! \brief Return an underestimate of the number of bytes needed from the wire. */
  size_t BytesNeeded();

 private:
  tvm_crt_error_t FindPacketStart();
  tvm_crt_error_t FindPacketLength();
  tvm_crt_error_t FindPacketCrc();
  tvm_crt_error_t FindCrcEnd();

  bool IsBufferFull(size_t buffer_full_bytes) {
    return num_buffer_bytes_valid_ >= buffer_full_bytes;
  }

  /*! \brief Consume input into buffer_ until buffer_ has buffer_full_bytes. */
  tvm_crt_error_t AddToBuffer(size_t buffer_full_bytes, bool update_crc);

  void ClearBuffer();

  /*! \brief Unescape and consume input bytes, storing into buffer.
   *
   * \param buffer A buffer to fill with consumed, unescaped bytes.
   * \param buffer_size_bytes Size of buffer, in bytes.
   * \param bytes_filled A pointer to an accumulator to which is added the number of bytes written
   *      to `buffer`.
   * \param update_crc true when the CRC should be updated with the escaped bytes.
   * \return
   *     - kTvmErrorNoError if successful
   *     - kTvmErrorFramingShortPacket if a start-of-packet escape code was encountered. If so,
   *       *bytes_filled indicates the number of bytes before the Escape::kEscapeStart byte.
   *     - kTvmErrorFramingInvalidEscape if an invalid escape sequence was seen.
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns 0.
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns an invalid positive number.
   *     - Any negative value (i.e. with bits in kTvmErrorSystemErrorMask set) returned by the
   *       WriteStream's Write() function.
   */
  tvm_crt_error_t ConsumeInput(uint8_t* buffer, size_t buffer_size_bytes, size_t* bytes_filled,
                               bool update_crc);

  WriteStream* stream_;

  enum class State : uint8_t {
    kFindPacketStart = 0,
    kFindPacketLength = 1,
    kFindPacketCrc = 2,
    kFindCrcEnd = 3,
  };
  State state_;

  const uint8_t* input_;
  size_t input_size_bytes_;

  bool saw_escape_start_;

  /*! \brief unframe buffer, sized to the longest framing field. */
  uint8_t buffer_[128];

  /*! \brief number of bytes in buffer that are currently valid. */
  size_t num_buffer_bytes_valid_;

  /*! \brief number of payload bytes left to receive before the CRC begins. */
  size_t num_payload_bytes_remaining_;

  /*! \brief Running CRC value. */
  uint16_t crc_;
};

class Framer {
 public:
  typedef ssize_t (*WriteFunc)(const uint8_t* data, size_t data_size_bytes);

  explicit Framer(WriteStream* stream)
      : stream_{stream}, state_{State::kReset}, num_payload_bytes_remaining_{0} {}

  /*! \brief Frame and write a full packet.
   * \param payload The entire packet payload.
   * \param payload_size_bytes Number of bytes in the packet.
   * \return
   *     - kTvmErrorNoError when no error occurs
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns 0.
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns an invalid positive number.
   *     - Any negative value (i.e. with bits in kTvmErrorSystemErrorMask set) returned by the
   *       WriteStream's Write() function.
   */
  tvm_crt_error_t Write(const uint8_t* payload, size_t payload_size_bytes);

  /*! \brief Start framing and writing a new packet to the wire.
   *
   * When transmitting payloads that are too large to be buffered, call this function first to send
   * the packet header and length fields.
   *
   * \param payload_size_bytes Number of payload bytes included as part of this packet.
   * \return
   *     - kTvmErrorNoError when no error occurs
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns 0.
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns an invalid positive number.
   *     - Any negative value (i.e. with bits in kTvmErrorSystemErrorMask set) returned by the
   *       WriteStream's Write() function.
   */
  tvm_crt_error_t StartPacket(size_t payload_size_bytes);

  /*! \brief Write payload data to the wire.
   *
   * When transmitting payloads that are too large to be buffered, call this function after calling
   * StartPacket to escape and transmit framed payloads. This function can be called multiple times
   * for a single packet.
   *
   * \param payload_chunk A piece of the packet payload.
   * \param payload_chunk_size_bytes Number of valid bytes in payload_chunk.
   * \return
   *     - kTvmErrorNoError when no error occurs
   *     - kTvmErrorFramingInvalidState when StartPacket() has not been called.
   *     - kTvmErrorFramingPayloadOverflow when more bytes were requested to be written than were
   *       declared in the payload_size_bytes parameter given to StartPacket().
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns 0.
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns an invalid positive number.
   *     - Any negative value (i.e. with bits in kTvmErrorSystemErrorMask set) returned by the
   *       WriteStream's Write() function.
   */
  tvm_crt_error_t WritePayloadChunk(const uint8_t* payload_chunk, size_t payload_chunk_size_bytes);

  /* \brief Finish writing one packet by sending the CRC.
   *
   * When transmitting paylaods that are too large to be buffered, call this function after sending
   * the entire payload using WritePayloadChunk.
   *
   * \return
   *     - kTvmErrorNoError when no error occurs
   *     - kTvmErrorFramingInvalidState when StartPacket() has not been called.
   *     - kTvmErrorFramingPayloadIncomplete when less bytes were written using WritePayloadChunk()
   *       than were declared in the payload_size_bytes parameter given to StartPacket().
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns 0.
   *     - kTvmErrorWriteStreamShortWrite if the WriteStream passed to constructor's Write()
   *       function returns an invalid positive number.
   *     - Any negative value (i.e. with bits in kTvmErrorSystemErrorMask set) returned by the
   *       WriteStream's Write() function.
   */
  tvm_crt_error_t FinishPacket();

  /* \brief Reset state of the Framer. */
  void Reset();

 private:
  /*! \brief Maximum size of stack-based buffer. */
  static constexpr const size_t kMaxStackBufferSizeBytes = 128;

  enum class State : uint8_t {
    /*! \brief State entered at construction time or after write error, before first packet sent. */
    kReset = 0,

    /*! \brief State entered after a packet has successfully finished transmitting. */
    kIdle = 1,

    /*! \brief State entered when a packet payload or CRC needs to be transmitted. */
    kTransmitPacketPayload = 2,
  };

  /*!
   * \brief Escape data and write the result to wire, and update crc_.
   *
   * \param data Unescaped data to write.
   * \param data_size_bytes Number of valid bytes in data.
   * \param escape true if escaping should be applied.
   * \param update_crc true if escaping should be applied.
   * \return kTvmErrorNoError on success, negative value on error.
   */
  tvm_crt_error_t WriteAndCrc(const uint8_t* data, size_t data_size_bytes, bool escape,
                              bool update_crc);

  /*! \brief Called to write framed data to the transport. */
  WriteStream* stream_;

  /*! \brief State fo the Framer. */
  State state_;

  /*! \brief When state_ == kTransmitPacketPayload, number of payload bytes left to transmit. */
  size_t num_payload_bytes_remaining_;

  /*! \brief Running CRC value. */
  uint16_t crc_;
};

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CRT_RPC_COMMON_FRAMING_H_
