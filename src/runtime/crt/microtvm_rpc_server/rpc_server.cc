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
 * \file rpc_server.cc
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
#define DMLC_LITTLE_ENDIAN 1
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/aot_executor_module.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/microtvm_rpc_server.h>
#include <tvm/runtime/crt/module.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/rpc_common/frame_buffer.h>
#include <tvm/runtime/crt/rpc_common/framing.h>
#include <tvm/runtime/crt/rpc_common/session.h>

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
microtvm_rpc_channel_write_t g_write_func = nullptr;
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
                 microtvm_rpc_channel_write_t write_func, void* write_func_ctx)
      : receive_buffer_{receive_storage, receive_storage_size_bytes},
        framer_{&send_stream_},
        session_{&framer_, &receive_buffer_, &HandleCompleteMessageCb, this},
        io_{&session_, &receive_buffer_},
        unframer_{session_.Receiver()},
        rpc_server_{&io_},
        is_running_{true} {}

  void Initialize() {
    uint8_t initial_session_nonce = Session::kInvalidNonce;
    tvm_crt_error_t error =
        TVMPlatformGenerateRandom(&initial_session_nonce, sizeof(initial_session_nonce));
    CHECK_EQ(kTvmErrorNoError, error, "generating random session id");
    CHECK_EQ(kTvmErrorNoError, session_.Initialize(initial_session_nonce), "rpc server init");
  }

  /*! \brief Process one message from the receive buffer, if possible.
   *
   * \param new_data If not nullptr, a pointer to a buffer pointer, which should point at new input
   *     data to process. On return, updated to point past data that has been consumed.
   * \param new_data_size_bytes Points to the number of valid bytes in `new_data`. On return,
   *     updated to the number of unprocessed bytes remaining in `new_data` (usually 0).
   * \return an error code indicating the outcome of the processing loop.
   */
  tvm_crt_error_t Loop(uint8_t** new_data, size_t* new_data_size_bytes) {
    if (!is_running_) {
      return kTvmErrorPlatformShutdown;
    }

    tvm_crt_error_t err = kTvmErrorNoError;
    if (new_data != nullptr && new_data_size_bytes != nullptr && *new_data_size_bytes > 0) {
      size_t bytes_consumed;
      err = unframer_.Write(*new_data, *new_data_size_bytes, &bytes_consumed);
      *new_data += bytes_consumed;
      *new_data_size_bytes -= bytes_consumed;
    }

    if (err == kTvmErrorNoError && !is_running_) {
      err = kTvmErrorPlatformShutdown;
    }

    return err;
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

extern "C" {

static microtvm_rpc_server_t g_rpc_server = nullptr;

microtvm_rpc_server_t MicroTVMRpcServerInit(microtvm_rpc_channel_write_t write_func,
                                            void* write_func_ctx) {
  tvm::runtime::micro_rpc::g_write_func = write_func;
  tvm::runtime::micro_rpc::g_write_func_ctx = write_func_ctx;

  tvm_crt_error_t err = TVMInitializeRuntime();
  if (err != kTvmErrorNoError) {
    TVMPlatformAbort(err);
  }

  err = TVMAotExecutorModule_Register();
  if (err != kTvmErrorNoError) {
    TVMPlatformAbort(err);
  }

  DLDevice dev = {kDLCPU, 0};
  void* receive_buffer_memory;
  err = TVMPlatformMemoryAllocate(TVM_CRT_MAX_PACKET_SIZE_BYTES, dev, &receive_buffer_memory);
  if (err != kTvmErrorNoError) {
    TVMPlatformAbort(err);
  }
  auto receive_buffer = new (receive_buffer_memory) uint8_t[TVM_CRT_MAX_PACKET_SIZE_BYTES];
  void* rpc_server_memory;
  err = TVMPlatformMemoryAllocate(sizeof(tvm::runtime::micro_rpc::MicroRPCServer), dev,
                                  &rpc_server_memory);
  if (err != kTvmErrorNoError) {
    TVMPlatformAbort(err);
  }
  auto rpc_server = new (rpc_server_memory) tvm::runtime::micro_rpc::MicroRPCServer(
      receive_buffer, TVM_CRT_MAX_PACKET_SIZE_BYTES, write_func, write_func_ctx);
  g_rpc_server = static_cast<microtvm_rpc_server_t>(rpc_server);
  rpc_server->Initialize();
  return g_rpc_server;
}

void TVMLogf(const char* format, ...) {
  va_list args;
  char log_buffer[256];
  va_start(args, format);
  size_t num_bytes_logged = TVMPlatformFormatMessage(log_buffer, sizeof(log_buffer), format, args);
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
    tvm::runtime::micro_rpc::Session session{&framer, nullptr, nullptr, nullptr};
    tvm_crt_error_t err =
        session.SendMessage(tvm::runtime::micro_rpc::MessageType::kLog,
                            reinterpret_cast<uint8_t*>(log_buffer), num_bytes_logged);
    if (err != kTvmErrorNoError) {
      TVMPlatformAbort(err);
    }
  }
}

tvm_crt_error_t MicroTVMRpcServerLoop(microtvm_rpc_server_t server_ptr, uint8_t** new_data,
                                      size_t* new_data_size_bytes) {
  tvm::runtime::micro_rpc::MicroRPCServer* server =
      static_cast<tvm::runtime::micro_rpc::MicroRPCServer*>(server_ptr);
  return server->Loop(new_data, new_data_size_bytes);
}

}  // extern "C"
