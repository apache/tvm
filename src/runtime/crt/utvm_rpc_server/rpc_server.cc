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
 * \file utvm_rpc_server.cc
 * \brief MicroTVM RPC Server
 */

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

// NOTE: dmlc/base.h contains some declarations that are incompatible with some C embedded
// toolchains. Just pull the bits we need for this file.
#define DMLC_CMAKE_LITTLE_ENDIAN DMLC_IO_USE_LITTLE_ENDIAN
#define DMLC_LITTLE_ENDIAN true
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/internal/common/memory.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/memory.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/rpc_common/frame_buffer.h>
#include <tvm/runtime/crt/rpc_common/framing.h>
#include <tvm/runtime/crt/rpc_common/session.h>
#include <tvm/runtime/crt/utvm_rpc_server.h>

#include "../../minrpc/minrpc_server.h"
#include "crt_config.h"

namespace tvm {
namespace runtime {
namespace micro_rpc {

class MicroIOHandler {
 public:
  MicroIOHandler(Session* session, FrameBuffer* receive_buffer)
      : session_{session}, receive_buffer_{receive_buffer} {}

  void MessageStart(size_t message_size_bytes) {
    session_->StartMessage(MessageType::kNormal, message_size_bytes + 8);
  }

  ssize_t PosixWrite(const uint8_t* buf, size_t buf_size_bytes) {
    int to_return = session_->SendBodyChunk(buf, buf_size_bytes);
    if (to_return < 0) {
      return to_return;
    }
    return buf_size_bytes;
  }

  void MessageDone() { CHECK_EQ(session_->FinishMessage(), kTvmErrorNoError, "FinishMessage"); }

  ssize_t PosixRead(uint8_t* buf, size_t buf_size_bytes) {
    return receive_buffer_->Read(buf, buf_size_bytes);
  }

  void Close() {}

  void Exit(int code) {
    for (;;) {
    }
  }

 private:
  Session* session_;
  FrameBuffer* receive_buffer_;
};

namespace {
// Stored as globals so that they can be used to report initialization errors.
utvm_rpc_channel_write_t g_write_func = nullptr;
void* g_write_func_ctx = nullptr;
}  // namespace

class SerialWriteStream : public WriteStream {
 public:
  SerialWriteStream() {}
  virtual ~SerialWriteStream() {}

  ssize_t Write(const uint8_t* data, size_t data_size_bytes) override {
    return g_write_func(g_write_func_ctx, data, data_size_bytes);
  }

  void PacketDone(bool is_valid) override {}

 private:
  void operator delete(void*) noexcept {}  // NOLINT(readability/casting)
};

class MicroRPCServer {
 public:
  MicroRPCServer(uint8_t* receive_storage, size_t receive_storage_size_bytes,
                 utvm_rpc_channel_write_t write_func, void* write_func_ctx)
      : receive_buffer_{receive_storage, receive_storage_size_bytes},
        framer_{&send_stream_},
        session_{0xa5, &framer_, &receive_buffer_, &HandleCompleteMessageCb, this},
        io_{&session_, &receive_buffer_},
        unframer_{session_.Receiver()},
        rpc_server_{&io_},
        has_pending_byte_{false},
        is_running_{true} {}

  void* operator new(size_t count, void* ptr) { return ptr; }

  void Initialize() { CHECK_EQ(kTvmErrorNoError, session_.Initialize(), "rpc server init"); }

  /*! \brief Process one message from the receive buffer, if possible.
   *
   * \return true if additional messages could be processed. false if the server shutdown request
   * has been received.
   */
  bool Loop() {
    if (has_pending_byte_) {
      size_t bytes_consumed;
      CHECK_EQ(unframer_.Write(&pending_byte_, 1, &bytes_consumed), kTvmErrorNoError,
               "unframer_.Write");
      CHECK_EQ(bytes_consumed, 1, "bytes_consumed");
      has_pending_byte_ = false;
    }

    return is_running_;
  }

  void HandleReceivedByte(uint8_t byte) {
    CHECK(!has_pending_byte_);
    has_pending_byte_ = true;
    pending_byte_ = byte;
  }

  void Log(const uint8_t* message, size_t message_size_bytes) {
    tvm_crt_error_t to_return =
        session_.SendMessage(MessageType::kLog, message, message_size_bytes);
    if (to_return != 0) {
      TVMPlatformAbort(to_return);
    }
  }

 private:
  FrameBuffer receive_buffer_;
  SerialWriteStream send_stream_;
  Framer framer_;
  Session session_;
  MicroIOHandler io_;
  Unframer unframer_;
  MinRPCServer<MicroIOHandler> rpc_server_;

  bool has_pending_byte_;
  uint8_t pending_byte_;
  bool is_running_;

  void HandleCompleteMessage(MessageType message_type, FrameBuffer* buf) {
    if (message_type != MessageType::kNormal) {
      return;
    }

    is_running_ = rpc_server_.ProcessOnePacket();
    session_.ClearReceiveBuffer();
  }

  static void HandleCompleteMessageCb(void* context, MessageType message_type, FrameBuffer* buf) {
    static_cast<MicroRPCServer*>(context)->HandleCompleteMessage(message_type, buf);
  }
};

}  // namespace micro_rpc
}  // namespace runtime
}  // namespace tvm

void* operator new[](size_t count, void* ptr) noexcept { return ptr; }

extern "C" {

static utvm_rpc_server_t g_rpc_server = nullptr;

utvm_rpc_server_t UTvmRpcServerInit(uint8_t* memory, size_t memory_size_bytes,
                                    size_t page_size_bytes_log2,
                                    utvm_rpc_channel_write_t write_func, void* write_func_ctx) {
  tvm::runtime::micro_rpc::g_write_func = write_func;
  tvm::runtime::micro_rpc::g_write_func_ctx = write_func_ctx;

  tvm_crt_error_t err = TVMInitializeRuntime(memory, memory_size_bytes, page_size_bytes_log2);
  if (err != kTvmErrorNoError) {
    TVMPlatformAbort(err);
  }

  auto receive_buffer =
      new (vmalloc(TVM_CRT_MAX_PACKET_SIZE_BYTES)) uint8_t[TVM_CRT_MAX_PACKET_SIZE_BYTES];
  auto rpc_server = new (vmalloc(sizeof(tvm::runtime::micro_rpc::MicroRPCServer)))
      tvm::runtime::micro_rpc::MicroRPCServer(receive_buffer, TVM_CRT_MAX_PACKET_SIZE_BYTES,
                                              write_func, write_func_ctx);
  g_rpc_server = static_cast<utvm_rpc_server_t>(rpc_server);
  rpc_server->Initialize();
  return g_rpc_server;
}

void TVMLogf(const char* format, ...) {
  va_list args;
  char log_buffer[256];
  va_start(args, format);
  size_t num_bytes_logged = vsnprintf(log_buffer, sizeof(log_buffer), format, args);
  va_end(args);

  // Most header-based logging frameworks tend to insert '\n' at the end of the log message.
  // Remove that for remote logging, since the remote logger will do the same.
  if (num_bytes_logged > 0 && log_buffer[num_bytes_logged - 1] == '\n') {
    log_buffer[num_bytes_logged - 1] = 0;
    num_bytes_logged--;
  }

  if (g_rpc_server != nullptr) {
    static_cast<tvm::runtime::micro_rpc::MicroRPCServer*>(g_rpc_server)
        ->Log(reinterpret_cast<uint8_t*>(log_buffer), num_bytes_logged);
  } else {
    tvm::runtime::micro_rpc::SerialWriteStream write_stream;
    tvm::runtime::micro_rpc::Framer framer{&write_stream};
    tvm::runtime::micro_rpc::Session session{0xa5, &framer, nullptr, nullptr, nullptr};
    tvm_crt_error_t err =
        session.SendMessage(tvm::runtime::micro_rpc::MessageType::kLog,
                            reinterpret_cast<uint8_t*>(log_buffer), num_bytes_logged);
    if (err != kTvmErrorNoError) {
      TVMPlatformAbort(err);
    }
  }
}

size_t UTvmRpcServerReceiveByte(utvm_rpc_server_t server_ptr, uint8_t byte) {
  // NOTE(areusch): In the future, this function is intended to work from an IRQ context. That's not
  // needed at present.
  tvm::runtime::micro_rpc::MicroRPCServer* server =
      static_cast<tvm::runtime::micro_rpc::MicroRPCServer*>(server_ptr);
  server->HandleReceivedByte(byte);
  return 1;
}

bool UTvmRpcServerLoop(utvm_rpc_server_t server_ptr) {
  tvm::runtime::micro_rpc::MicroRPCServer* server =
      static_cast<tvm::runtime::micro_rpc::MicroRPCServer*>(server_ptr);
  return server->Loop();
}

}  // extern "C"
