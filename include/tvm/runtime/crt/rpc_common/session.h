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

#ifndef TVM_RUNTIME_CRT_RPC_COMMON_SESSION_H_
#define TVM_RUNTIME_CRT_RPC_COMMON_SESSION_H_

#include <inttypes.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/rpc_common/frame_buffer.h>
#include <tvm/runtime/crt/rpc_common/framing.h>
#include <tvm/runtime/crt/rpc_common/write_stream.h>

namespace tvm {
namespace runtime {
namespace micro_rpc {

enum class MessageType : uint8_t {
  kStartSessionInit = 0x00,
  kStartSessionReply = 0x01,
  kTerminateSession = 0x02,
  kLog = 0x03,
  kNormal = 0x10,
};

#if defined(_MSC_VER)

#pragma pack(push, 1)
typedef struct SessionHeader {
  uint16_t session_id;
  MessageType message_type;
} SessionHeader;
#pragma pack(pop)

#else

typedef struct SessionHeader {
  uint16_t session_id;
  MessageType message_type;
} __attribute__((packed)) SessionHeader;

#endif

/*!
 * \brief CRT communication session management class.
 * Assumes the following properties provided by the underlying transport:
 *  - in-order delivery.
 *  - reliable delivery.
 *
 * Specifically, designed for use with UARTs. Will probably work over semihosting, USB, and TCP;
 * will probably not work reliably enough over UDP.
 */
class Session {
 public:
  /*! \brief Callback invoked when a full message is received.
   *
   * This function is called in the following situations:
   *  - When a new session is established (this typically indicates the remote end reset).
   *    In this case, buf is NULL.
   *  - When a log message or normal traffic is received. In this case, buf points to a
   *    valid buffer containing the message content.
   *
   * \param context The value of `message_received_func_context` passed to the constructor.
   * \param message_type The type of session message received. Currently, this is always
   *      either kNormal or kLog.
   * \param buf When message_type is not kStartSessionMessage, a FrameBuffer whose read cursor is
   *      at the first byte of the message payload. Otherwise, NULL.
   */
  typedef void (*MessageReceivedFunc)(void* context, MessageType message_type, FrameBuffer* buf);

  /*! \brief An invalid nonce value that typically indicates an unknown nonce. */
  static constexpr const uint8_t kInvalidNonce = 0;

  Session(Framer* framer, FrameBuffer* receive_buffer, MessageReceivedFunc message_received_func,
          void* message_received_func_context)
      : local_nonce_{kInvalidNonce},
        session_id_{0},
        state_{State::kReset},
        receiver_{this},
        framer_{framer},
        receive_buffer_{receive_buffer},
        receive_buffer_has_complete_message_{false},
        message_received_func_{message_received_func},
        message_received_func_context_{message_received_func_context} {
    // Session can be used for system startup logging, before the RPC server is instantiated. In
    // this case, allow receive_buffer_ to be nullptr. The instantiator agrees not to use
    // Receiver().
    if (receive_buffer_ != nullptr) {
      receive_buffer_->Clear();
    }
  }

  /*!
   * \brief Send a session terminate message, usually done at startup to interrupt a hanging remote.
   * \param initial_session_nonce Initial nonce that should be used on the first session start
   *      message. Callers should ensure this is different across device resets.
   * \return kTvmErrorNoError on success, or an error code otherwise.
   */
  tvm_crt_error_t Initialize(uint8_t initial_session_nonce);

  /*!
   * \brief Terminate any previously-established session.
   * \return kTvmErrorNoError on success, or an error code otherwise.
   */
  tvm_crt_error_t TerminateSession();

  /*!
   * \brief Start a new session regardless of state. Sends kStartSessionMessage.
   *
   * Generally speaking, this function should be called once per device reset by exactly one side
   * in the system. No traffic can flow until this function is called.
   *
   * \return kTvmErrorNoError on success, or an error code otherwise.
   */
  tvm_crt_error_t StartSession();

  /*!
   * \brief Obtain a WriteStream implementation for use by the framing layer.
   * \return A WriteStream to which received data should be written. Owned by this class.
   */
  WriteStream* Receiver() { return &receiver_; }

  /*!
   * \brief Send a full message including header, payload, and CRC footer.
   * \param message_type One of MessageType; distinguishes the type of traffic at the session layer.
   * \param message_data The data contained in the message.
   * \param message_size_bytes The number of valid bytes in message_data.
   * \return kTvmErrorNoError on success, or an error code otherwise.
   */
  tvm_crt_error_t SendMessage(MessageType message_type, const uint8_t* message_data,
                              size_t message_size_bytes);

  /*!
   * \brief Send the framing and session layer headers.
   *
   * This function allows messages to be sent in pieces.
   *
   * \param message_type One of MessageType; distinguishes the type of traffic at the session layer.
   * \param message_size_bytes The size of the message body, in bytes. Excludes the framing and
   * session layer headers. \return 0 on success, negative error code on failure.
   * \return kTvmErrorNoError on success, or an error code otherwise.
   */
  tvm_crt_error_t StartMessage(MessageType message_type, size_t message_size_bytes);

  /*!
   * \brief Send a part of the message body.
   *
   * This function allows messages to be sent in pieces.
   *
   * \param chunk_data The data contained in this message body chunk.
   * \param chunk_size_bytes The number of valid bytes in chunk_data.
   * \return kTvmErrorNoError on success, or an error code otherwise.
   */
  tvm_crt_error_t SendBodyChunk(const uint8_t* chunk_data, size_t chunk_size_bytes);

  /*!
   * \brief Finish sending the message by sending the framing layer footer.
   * \return kTvmErrorNoError on success, or an error code otherwise.
   */
  tvm_crt_error_t FinishMessage();

  /*! \brief Returns true if the session is in the established state. */
  bool IsEstablished() const { return state_ == State::kSessionEstablished; }

  /*!
   * \brief Clear the receive buffer and prepare to receive next message.
   *
   * Call this function after MessageReceivedFunc is invoked. Any SessionReceiver::Write() calls
   * made will return errors until this function is called to prevent them from corrupting the
   * valid message in the receive buffer.
   */
  void ClearReceiveBuffer();

  /*! \brief A version number used to check compatibility of the remote session implementation. */
  static const constexpr uint8_t kVersion = 0x01;

 private:
  class SessionReceiver : public WriteStream {
   public:
    explicit SessionReceiver(Session* session) : session_{session} {}
    virtual ~SessionReceiver() {}

    ssize_t Write(const uint8_t* data, size_t data_size_bytes) override;
    void PacketDone(bool is_valid) override;

   private:
    void operator delete(void*) noexcept {}  // NOLINT(readability/casting)
    Session* session_;
  };

  enum class State : uint8_t {
    kReset = 0,
    kNoSessionEstablished = 1,
    kStartSessionSent = 2,
    kSessionEstablished = 3,
  };

  void RegenerateNonce();

  tvm_crt_error_t SendInternal(MessageType message_type, const uint8_t* message_data,
                               size_t message_size_bytes);

  void SendSessionStartReply(const SessionHeader& header);

  void ProcessStartSessionInit(const SessionHeader& header);

  void ProcessStartSessionReply(const SessionHeader& header);

  void OnSessionEstablishedMessage();

  void OnSessionTerminatedMessage();

  void SetSessionId(uint8_t initiator_nonce, uint8_t responder_nonce) {
    session_id_ = initiator_nonce | (((uint16_t)responder_nonce) << 8);
  }

  uint8_t InitiatorNonce(uint16_t session_id) { return session_id & 0xff; }

  uint8_t ResponderNonce(uint16_t session_id) { return (session_id >> 8) & 0xff; }

  uint8_t local_nonce_;
  uint16_t session_id_;
  State state_;
  SessionReceiver receiver_;
  Framer* framer_;
  FrameBuffer* receive_buffer_;
  bool receive_buffer_has_complete_message_;
  MessageReceivedFunc message_received_func_;
  void* message_received_func_context_;
};

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CRT_RPC_COMMON_SESSION_H_
