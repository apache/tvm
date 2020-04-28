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
 * \file rpc_session.cc
 * \brief RPC session for remote function call.
 */
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>
#include <memory>
#include <array>
#include <string>
#include <chrono>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

#include "rpc_endpoint.h"
#include "rpc_local_session.h"
#include "../object_internal.h"
#include "../../support/ring_buffer.h"
#include "../../support/arena.h"

namespace tvm {
namespace runtime {

/*!
 * Event-driven state-machine based handlers for RPCEndpoint.
 *
 * Key functions:
 *
 * - SendPackedSeq: send the arguments over to the peer
 * - HandleNextEvent: handle the next request from the peer(RPCCode followed by per code protocol).
 */
class RPCEndpoint::EventHandler : public dmlc::Stream {
 public:
  EventHandler(support::RingBuffer* reader,
               support::RingBuffer* writer,
               std::string name,
               std::string* remote_key)
      : reader_(reader),
        writer_(writer),
        name_(name),
        remote_key_(remote_key) {
    this->Clear();

    if (*remote_key == "%toinit") {
      state_ = kInitHeader;
      remote_key_->resize(0);
      pending_request_bytes_ = sizeof(int32_t);
    }
  }

  /*!
   * \brief Bytes needed to fulfill current request
   */
  size_t BytesNeeded() const {
    if (reader_->bytes_available() < pending_request_bytes_) {
      return pending_request_bytes_ - reader_->bytes_available();
    } else {
      return 0;
    }
  }

  /*!
   * \brief Request number of bytes from the reader.
   * \param nbytes The number of bytes
   */
  void RequestBytes(size_t nbytes) {
    pending_request_bytes_ += nbytes;
    reader_->Reserve(pending_request_bytes_);
  }

  /*! \return Whether we are ready to handle next request. */
  bool Ready() const {
    return reader_->bytes_available() >= pending_request_bytes_;
  }

  /*! \return Whether we can perform a clean shutdown */
  bool CanCleanShutdown() const {
    return state_ == kRecvPacketNumBytes;
  }

  /*! \brief Finish the copy ack stage. */
  void FinishCopyAck() {
    this->SwitchToState(kRecvPacketNumBytes);
  }

  /*!
   * \brief Enter the io loop until the next event.
   * \param client_mode Whether we are in the client.
   * \param setreturn The function to set the return value encoding.
   * \return The function to set return values when there is a return event.
   */
  RPCCode HandleNextEvent(bool client_mode, RPCSession::FEncodeReturn setreturn) {
    std::swap(client_mode_, client_mode);

    while (this->Ready()) {
      switch (state_) {
        case kInitHeader: HandleInitHeader(); break;
        case kRecvPacketNumBytes: {
          uint64_t packet_nbytes;
          CHECK(this->Read(&packet_nbytes));
          if (packet_nbytes != 0) {
            this->SwitchToState(kProcessPacket);
            this->RequestBytes(packet_nbytes);
          } else {
            this->SwitchToState(kRecvPacketNumBytes);
          }
          break;
        }
        case kProcessPacket: {
          this->HandleProcessPacket(setreturn);
          break;
        }
        case kReturnReceived: {
          this->SwitchToState(kRecvPacketNumBytes);
          std::swap(client_mode_, client_mode);
          return RPCCode::kReturn;
        }
        case kCopyAckReceived: {
          std::swap(client_mode_, client_mode);
          return RPCCode::kCopyAck;
        }
        case kShutdownReceived: {
          std::swap(client_mode_, client_mode);
          return RPCCode::kShutdown;
        }
      }
    }
    std::swap(client_mode_, client_mode);
    return RPCCode::kNone;
  }

  /*! \brief Clear all the states in the Handler.*/
  void Clear() {
    state_ = kRecvPacketNumBytes;
    pending_request_bytes_ = sizeof(uint64_t);
  }

  /*!
   * \brief Validate that the arguments can be sent through RPC.
   * \param arg_values The argument values.
   * \param type_codes The type codes.
   */
  void ValidateArguments(const TVMValue* arg_values,
                         const int* type_codes,
                         int num_args) {
    TVMArgs args(arg_values, type_codes, num_args);
    for (int i = 0; i < num_args; ++i) {
      int tcode = type_codes[i];
      if (tcode == kTVMObjectHandle || tcode == kTVMObjectRValueRefArg) {
        LOG(FATAL) << "ValueError: Cannot pass argument " << i
                   << ", type " << args[i].AsObjectRef<ObjectRef>()->GetTypeKey()
                   << " is not supported by RPC";
      } else if (tcode == kTVMContext) {
        DLContext ctx = args[i];
        CHECK_LT(static_cast<int>(ctx.device_type), kRPCSessMask)
            << "InternalError: cannot pass RPC context in the channel";
      }
    }
  }

  void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone) {
    LOG(FATAL) << "RPCServerError:" << RPCServerStatusToString(code);
  }

  uint64_t PackedSeqGetNumBytes(const TVMValue* arg_values,
                                const int* type_codes,
                                int num_args,
                                bool client_mode) {
    return RPCReference::PackedSeqGetNumBytes(
        arg_values, type_codes, num_args, client_mode, this);
  }

  void SendPackedSeq(const TVMValue* arg_values,
                     const int* type_codes,
                     int num_args,
                     bool client_mode) {
    RPCReference::SendPackedSeq(
        arg_values, type_codes, num_args, client_mode, this);
  }

  // Endian aware IO handling
  using Stream::Read;
  using Stream::Write;
  using Stream::ReadArray;
  using Stream::WriteArray;

  bool Read(RPCCode* code) {
    int32_t cdata;
    if (!this->Read(&cdata)) return false;
    *code = static_cast<RPCCode>(cdata);
    return true;
  }
  void Write(RPCCode code) {
    int32_t cdata = static_cast<int>(code);
    this->Write(cdata);
  }

  template<typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return arena_.template allocate_<T>(count);
  }

 protected:
  enum State {
    kInitHeader,
    kRecvPacketNumBytes,
    kProcessPacket,
    kReturnReceived,
    kCopyAckReceived,
    kShutdownReceived
  };
  // Current state;
  State state_;
  // Initialize remote header
  bool init_header_step_{0};
  // Whether current handler is client or server mode.
  bool client_mode_{false};
  // Internal arena
  support::Arena arena_;

  // State switcher
  void SwitchToState(State state) {
    // invariant
    if (state != kCopyAckReceived) {
      CHECK_EQ(pending_request_bytes_, 0U)
          << "state=" << state;
    }
    state_ = state;
    CHECK(state != kInitHeader)
        << "cannot switch to init header";
    if (state == kRecvPacketNumBytes) {
      this->RequestBytes(sizeof(uint64_t));
      // recycle arena for the next session.
      arena_.RecycleAll();
    }
  }

  // handler for initial header read
  void HandleInitHeader() {
    if (init_header_step_ == 0) {
      int32_t len;
      this->Read(&len);
      remote_key_->resize(len);
      init_header_step_ = 1;
      this->RequestBytes(len);
      return;
    } else {
      CHECK_EQ(init_header_step_, 1);
      this->ReadArray(dmlc::BeginPtr(*remote_key_), remote_key_->length());
      this->SwitchToState(kRecvPacketNumBytes);
    }
  }

  // Handler for read code.
  void HandleProcessPacket(RPCSession::FEncodeReturn setreturn) {
    RPCCode code = RPCCode::kNone;
    this->Read(&code);

    if (code >= RPCCode::kSyscallCodeStart) {
      this->HandleSyscall(code);
    } else {
        switch (code) {
          case RPCCode::kInitServer: {
            this->HandleInitServer();
            break;
          }
          case RPCCode::kCallFunc: {
            this->HandleNormalCallFunc();
            break;
          }
          case RPCCode::kCopyFromRemote: {
            this->HandleCopyFromRemote();
            break;
          }
          case RPCCode::kCopyToRemote: {
            this->HandleCopyToRemote();
            break;
          }
          case RPCCode::kException:
          case RPCCode::kReturn: {
            this->HandleReturn(code, setreturn);
            break;
          }
          case RPCCode::kCopyAck: {
            this->SwitchToState(kCopyAckReceived);
            break;
          }
          case RPCCode::kShutdown: {
            this->SwitchToState(kShutdownReceived);
            break;
          }
          default: LOG(FATAL) << "Unknown event "  << static_cast<int>(code);
        }
    }
  }

  /*!
   * \brief Recive incoming packed seq from the stream.
   * \return The received argments.
   * \note The TVMArgs is available until we switchstate.
   */
  TVMArgs RecvPackedSeq() {
    TVMValue* values;
    int* tcodes;
    int num_args;
    RPCReference::RecvPackedSeq(&values, &tcodes, &num_args, this);
    return TVMArgs(values, tcodes, num_args);
  }

  /*!
   * \brief Return exception to the remote.
   * \param err_msg The error message.
   */
  void ReturnException(const char* err_msg) {
    RPCReference::ReturnException(err_msg, this);
  }

  /*!
   * \brief Return nullptr to the remote.
   * \param err_msg The error message.
   */
  void ReturnVoid() {
    RPCReference::ReturnVoid(this);
  }

  /*!
   * \brief Return a packed sequence to the remote.
   * \param args The arguments.
   */
  void ReturnPackedSeq(TVMArgs args) {
    RPCReference::ReturnPackedSeq(args.values, args.type_codes, args.size(), this);
  }

  /*!
   * \brief Handle the case when return/exception value is received.
   * \param code The RPC code.
   * \param setreturn The function to encode return.
   */
  void HandleReturn(RPCCode code, RPCSession::FEncodeReturn setreturn) {
    TVMArgs args = RecvPackedSeq();

    if (code == RPCCode::kException) {
      // switch to the state before sending exception.
      this->SwitchToState(kRecvPacketNumBytes);
      std::string msg = args[0];
      LOG(FATAL) << "RPCError: Error caught from RPC call:\n" <<  msg;
    }

    CHECK(setreturn != nullptr) << "fsetreturn not available";
    setreturn(args);

    this->SwitchToState(kReturnReceived);
  }

  void HandleSyscall(RPCCode code);

  void HandleCopyFromRemote() {
    uint64_t handle, offset, num_bytes;
    TVMContext ctx;
    DLDataType type_hint;
    this->Read(&handle);
    this->Read(&offset);
    this->Read(&num_bytes);
    this->Read(&ctx);
    this->Read(&type_hint);
    size_t elem_bytes = (type_hint.bits * type_hint.lanes + 7) / 8;

    char* data_ptr;

    if (ctx.device_type == kDLCPU) {
      data_ptr = reinterpret_cast<char*>(handle) + offset;
      // endian aware handling
      if (!DMLC_IO_NO_ENDIAN_SWAP) {
        char* temp = this->ArenaAlloc<char>(num_bytes);
        std::memcpy(temp, data_ptr, num_bytes);
        dmlc::ByteSwap(temp, elem_bytes, num_bytes / elem_bytes);
        data_ptr = temp;
      }
    } else {
      try {
        data_ptr = this->ArenaAlloc<char>(num_bytes);
        GetServingSession()->CopyFromRemote(
            reinterpret_cast<void*>(handle), offset,
            data_ptr, 0,
            num_bytes, ctx, type_hint);
        // endian aware handling
        if (!DMLC_IO_NO_ENDIAN_SWAP) {
          dmlc::ByteSwap(data_ptr, elem_bytes, num_bytes / elem_bytes);
        }
      } catch (const std::runtime_error &e) {
        this->ReturnException(e.what());
        this->SwitchToState(kRecvPacketNumBytes);
        return;
      }
    }
    RPCCode code = RPCCode::kCopyAck;
    uint64_t packet_nbytes = sizeof(code) + num_bytes;

    // Return Copy Ack
    this->Write(packet_nbytes);
    this->Write(code);
    this->WriteArray(data_ptr, num_bytes);

    this->SwitchToState(kRecvPacketNumBytes);
  }

  void HandleCopyToRemote() {
    uint64_t handle, offset, num_bytes;
    TVMContext ctx;
    DLDataType type_hint;

    this->Read(&handle);
    this->Read(&offset);
    this->Read(&num_bytes);
    this->Read(&ctx);
    this->Read(&type_hint);

    size_t elem_bytes = (type_hint.bits * type_hint.lanes + 7) / 8;

    if (ctx.device_type == kDLCPU) {
       char* dptr = reinterpret_cast<char*>(handle) + offset;
       this->ReadArray(dptr, num_bytes);

        if (!DMLC_IO_NO_ENDIAN_SWAP) {
          dmlc::ByteSwap(dptr, elem_bytes, num_bytes / elem_bytes);
        }
    } else {
      char* temp_data = this->ArenaAlloc<char>(num_bytes);
      this->ReadArray(temp_data, num_bytes);

      if (!DMLC_IO_NO_ENDIAN_SWAP) {
        dmlc::ByteSwap(temp_data, elem_bytes, num_bytes / elem_bytes);
      }

      try {
        GetServingSession()->CopyToRemote(
            temp_data, 0,
            reinterpret_cast<void*>(handle), offset,
            num_bytes, ctx, type_hint);
      } catch (const std::runtime_error &e) {
        this->ReturnException(e.what());
        this->SwitchToState(kRecvPacketNumBytes);
        return;
      }
    }

    this->ReturnVoid();
    this->SwitchToState(kRecvPacketNumBytes);
  }

  // Handle for packed call.
  void HandleNormalCallFunc() {
    uint64_t call_handle;

    this->Read(&call_handle);
    TVMArgs args = RecvPackedSeq();

    try {
      GetServingSession()->CallFunc(
          reinterpret_cast<void*>(call_handle),
          args.values, args.type_codes, args.size(),
          [this](TVMArgs ret) { this->ReturnPackedSeq(ret); });
    } catch (const std::runtime_error& e) {
      this->ReturnException(e.what());
    }

    this->SwitchToState(kRecvPacketNumBytes);
  }

  void HandleInitServer() {
    std::string client_protocol_ver;

    uint64_t len;
    this->Read(&len);
    client_protocol_ver.resize(len);
    this->Read(dmlc::BeginPtr(client_protocol_ver), len);

    TVMArgs args = RecvPackedSeq();

    try {
      CHECK(serving_session_ == nullptr)
          << "Server has already been initialized";

      std::string server_protocol_ver = kRPCProtocolVer;
      CHECK_EQ(client_protocol_ver, server_protocol_ver)
          << "Server[" << name_ << "]: Client protocol version mismatch with the server "
          << " server protocol=" << server_protocol_ver
          << ", client protocol=" << client_protocol_ver;

      if (args.size() == 0) {
        serving_session_ = std::make_shared<LocalSession>();
      } else {
        std::string constructor_name = args[0];
        auto* fconstructor = Registry::Get(constructor_name);
        CHECK(fconstructor != nullptr)
            << " Cannot find session constructor " << constructor_name;
        TVMRetValue con_ret;

        try {
          fconstructor->CallPacked(
              TVMArgs(args.values + 1, args.type_codes + 1, args.size() - 1), &con_ret);
        } catch (const dmlc::Error& e) {
          LOG(FATAL) << "Server[" << name_ << "]:"
                     << " Error caught from session constructor " << constructor_name
                     << ":\n" << e.what();
        }

        CHECK_EQ(con_ret.type_code(), kTVMModuleHandle)
            << "Server[" << name_ << "]:"
            << " Constructor " << constructor_name
            << " need to return an RPCModule";
        Module mod = con_ret;
        std::string tkey = mod->type_key();
        CHECK_EQ(tkey, "rpc")
            << "Constructor " << constructor_name << " to return an RPCModule";
        serving_session_ = RPCModuleGetSession(mod);
      }

      this->ReturnVoid();
    } catch (const std::runtime_error &e) {
      this->ReturnException(e.what());
    }

    this->SwitchToState(kRecvPacketNumBytes);
  }

  // Handler for special syscalls that have a specific RPCCode.
  template<typename F>
  void SysCallHandler(F f) {
    TVMArgs args = RecvPackedSeq();
    try {
      TVMRetValue rv;
      f(GetServingSession(), args, &rv);
      TVMValue ret_value;
      int ret_tcode;
      TVMArgsSetter setter(&ret_value, &ret_tcode);
      setter(0, rv);

      this->ReturnPackedSeq(TVMArgs(&ret_value, &ret_tcode, 1));
    } catch (const std::runtime_error& e) {
      this->ReturnException(e.what());
    }
    this->SwitchToState(kRecvPacketNumBytes);
  }

 private:
  RPCSession* GetServingSession() const {
    CHECK(serving_session_ != nullptr)
        << "Need to call InitRemoteSession first before any further actions";
    return serving_session_.get();
  }
  // Utility functions
  // Internal read function, update pending_request_bytes_
  size_t Read(void* data, size_t size) final {
    CHECK_LE(size, pending_request_bytes_);
    reader_->Read(data, size);
    pending_request_bytes_ -= size;
    return size;
  }
  // wriite the data to the channel.
  void Write(const void* data, size_t size) final {
    writer_->Write(data, size);
  }
  // Number of pending bytes requests
  size_t pending_request_bytes_{0};
  // The ring buffer to read data from.
  support::RingBuffer* reader_;
  // The ringr buffer to write reply to.
  support::RingBuffer* writer_;
  // The session used to serve the RPC requests.
  std::shared_ptr<RPCSession> serving_session_;
  // Name of endpoint.
  std::string name_;
  // remote key
  std::string* remote_key_;
};

RPCCode RPCEndpoint::HandleUntilReturnEvent(
    bool client_mode, RPCSession::FEncodeReturn setreturn) {
  RPCCode code = RPCCode::kCallFunc;
  while (code != RPCCode::kReturn &&
         code != RPCCode::kShutdown &&
         code != RPCCode::kCopyAck) {
    while (writer_.bytes_available() != 0) {
      writer_.ReadWithCallback([this](const void *data, size_t size) {
          return channel_->Send(data, size);
        }, writer_.bytes_available());
    }
    size_t bytes_needed = handler_->BytesNeeded();
    if (bytes_needed != 0) {
      size_t n = reader_.WriteWithCallback([this](void* data, size_t size) {
          return channel_->Recv(data, size);
        }, bytes_needed);
      if (n == 0) {
        if (handler_->CanCleanShutdown()) {
          return RPCCode::kShutdown;
        } else {
          LOG(FATAL) << "Channel closes before we get neded bytes";
        }
      }
    }
    code = handler_->HandleNextEvent(client_mode, setreturn);
  }
  return code;
}

void RPCEndpoint::Init() {
  // Event handler
  handler_ = std::make_shared<EventHandler>(
      &reader_, &writer_, name_, &remote_key_);
  // Quick function to for syscall remote.
  syscall_remote_ = PackedFunc([this](TVMArgs all_args, TVMRetValue* rv) {
    std::lock_guard<std::mutex> lock(mutex_);
    RPCCode code = static_cast<RPCCode>(all_args[0].operator int());
    TVMArgs args(all_args.values + 1, all_args.type_codes +1, all_args.num_args -1);

    uint64_t packet_nbytes =
        sizeof(code) +
        handler_->PackedSeqGetNumBytes(
            args.values, args.type_codes, args.num_args, true);

    // All packet begins with packet nbytes
    handler_->Write(packet_nbytes);
    handler_->Write(code);
    handler_->SendPackedSeq(args.values, args.type_codes, args.num_args, true);

    code = HandleUntilReturnEvent(true, [rv](TVMArgs args) {
      CHECK_EQ(args.size(), 1);
      *rv = args[0];
    });
    CHECK(code == RPCCode::kReturn) << "code=" << static_cast<int>(code);
  });
}

std::shared_ptr<RPCEndpoint> RPCEndpoint::Create(
    std::unique_ptr<RPCChannel> channel,
    std::string name,
    std::string remote_key) {
  std::shared_ptr<RPCEndpoint> endpt = std::make_shared<RPCEndpoint>();
  endpt->channel_ = std::move(channel);
  endpt->name_ = std::move(name);
  endpt->remote_key_ = std::move(remote_key);
  endpt->Init();
  return endpt;
}

RPCEndpoint::~RPCEndpoint() {
  this->Shutdown();
}

void RPCEndpoint::Shutdown() {
  if (channel_ != nullptr) {
    RPCCode code = RPCCode::kShutdown;
    uint64_t packet_nbytes = sizeof(code);

    handler_->Write(packet_nbytes);
    handler_->Write(code);

    // flush all writing buffer to output channel.
    try {
      while (writer_.bytes_available() != 0) {
        size_t n = writer_.ReadWithCallback([this](const void *data, size_t size) {
            return channel_->Send(data, size);
          }, writer_.bytes_available());
        if (n == 0) break;
      }
    } catch (const dmlc::Error& e) {
    }
    channel_.reset(nullptr);
  }
}

void RPCEndpoint::ServerLoop() {
  if (const auto* f = Registry::Get("tvm.rpc.server.start")) {
    (*f)();
  }
  TVMRetValue rv;
  CHECK(HandleUntilReturnEvent(false, [](TVMArgs) {}) == RPCCode::kShutdown);
  if (const auto* f = Registry::Get("tvm.rpc.server.shutdown")) {
    (*f)();
  }
  channel_.reset(nullptr);
}

int RPCEndpoint::ServerAsyncIOEventHandler(const std::string& in_bytes, int event_flag) {
  RPCCode code = RPCCode::kNone;
  if (in_bytes.length() != 0) {
    reader_.Write(in_bytes.c_str(), in_bytes.length());
    code = handler_->HandleNextEvent(false, [](TVMArgs) {});
  }
  if ((event_flag & 2) != 0 && writer_.bytes_available() != 0) {
    writer_.ReadWithCallback([this](const void *data, size_t size) {
        return channel_->Send(data, size);
      }, writer_.bytes_available());
  }
  CHECK(code != RPCCode::kReturn && code != RPCCode::kCopyAck);
  if (code == RPCCode::kShutdown) return 0;
  if (writer_.bytes_available() != 0) return 2;
  return 1;
}

void RPCEndpoint::InitRemoteSession(TVMArgs args) {
  std::lock_guard<std::mutex> lock(mutex_);
  RPCCode code = RPCCode::kInitServer;
  std::string protocol_ver = kRPCProtocolVer;
  uint64_t length = protocol_ver.length();

  uint64_t packet_nbytes =
      sizeof(code) +
      sizeof(length) +
      length +
      handler_->PackedSeqGetNumBytes(
          args.values, args.type_codes, args.num_args, true);

  // All packet begins with packet nbytes
  handler_->Write(packet_nbytes);
  handler_->Write(code);
  handler_->Write(length);
  handler_->WriteArray(protocol_ver.data(), length);
  handler_->SendPackedSeq(args.values, args.type_codes, args.num_args, true);

  code = HandleUntilReturnEvent(true, [](TVMArgs args) {});
  CHECK(code == RPCCode::kReturn) << "code=" << static_cast<int>(code);
}

// Get remote function with name
void RPCEndpoint::CallFunc(RPCSession::PackedFuncHandle h,
                           const TVMValue* arg_values,
                           const int* arg_type_codes,
                           int num_args,
                           RPCSession::FEncodeReturn encode_return) {
  std::lock_guard<std::mutex> lock(mutex_);

  handler_->ValidateArguments(arg_values, arg_type_codes, num_args);
  RPCCode code = RPCCode::kCallFunc;
  uint64_t handle = reinterpret_cast<uint64_t>(h);

  uint64_t packet_nbytes =
      sizeof(code) +
      sizeof(handle) +
      handler_->PackedSeqGetNumBytes(
          arg_values, arg_type_codes, num_args, true);

  handler_->Write(packet_nbytes);
  handler_->Write(code);
  handler_->Write(handle);
  handler_->SendPackedSeq(
      arg_values, arg_type_codes, num_args, true);

  code = HandleUntilReturnEvent(true, encode_return);
  CHECK(code == RPCCode::kReturn) << "code=" << static_cast<int>(code);
}

void RPCEndpoint::CopyToRemote(void* from,
                               size_t from_offset,
                               void* to,
                               size_t to_offset,
                               size_t data_size,
                               TVMContext ctx_to,
                               DLDataType type_hint) {
  std::lock_guard<std::mutex> lock(mutex_);
  RPCCode code = RPCCode::kCopyToRemote;
  uint64_t handle = reinterpret_cast<uint64_t>(to);
  uint64_t offset = static_cast<uint64_t>(to_offset);
  uint64_t size = static_cast<uint64_t>(data_size);

  uint64_t packet_nbytes =
      sizeof(code) +
      sizeof(handle) +
      sizeof(offset) +
      sizeof(size) +
      sizeof(ctx_to) +
      sizeof(type_hint) +
      data_size;

  handler_->Write(packet_nbytes);
  handler_->Write(code);
  handler_->Write(handle);
  handler_->Write(offset);
  handler_->Write(size);
  handler_->Write(ctx_to);
  handler_->Write(type_hint);
  handler_->WriteArray(reinterpret_cast<char*>(from) + from_offset, data_size);

  CHECK(HandleUntilReturnEvent(true, [](TVMArgs){}) == RPCCode::kReturn);
}

void RPCEndpoint::CopyFromRemote(void* from,
                                size_t from_offset,
                                void* to,
                                size_t to_offset,
                                size_t data_size,
                                TVMContext ctx_from,
                                DLDataType type_hint) {
  std::lock_guard<std::mutex> lock(mutex_);
  RPCCode code = RPCCode::kCopyFromRemote;
  uint64_t handle = reinterpret_cast<uint64_t>(from);
  uint64_t offset = static_cast<uint64_t>(from_offset);
  uint64_t size = static_cast<uint64_t>(data_size);

  uint64_t packet_nbytes =
      sizeof(code) +
      sizeof(handle) +
      sizeof(offset) +
      sizeof(size) +
      sizeof(ctx_from) +
      sizeof(type_hint);

  handler_->Write(packet_nbytes);
  handler_->Write(code);
  handler_->Write(handle);
  handler_->Write(offset);
  handler_->Write(size);
  handler_->Write(ctx_from);
  handler_->Write(type_hint);

  TVMRetValue rv;
  CHECK(HandleUntilReturnEvent(true, [](TVMArgs){}) == RPCCode::kCopyAck);
  handler_->ReadArray(reinterpret_cast<char*>(to) + to_offset, data_size);
  handler_->FinishCopyAck();
}

// SysCallEventHandler functions
void RPCGetGlobalFunc(RPCSession* handler, TVMArgs args, TVMRetValue* rv) {
  std::string name = args[0];
  *rv = handler->GetFunction(name);
}

void RPCFreeHandle(RPCSession* handler, TVMArgs args, TVMRetValue *rv) {
  void* handle = args[0];
  int type_code = args[1];
  handler->FreeHandle(handle, type_code);
}

void RPCDevSetDevice(RPCSession* handler, TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  handler->GetDeviceAPI(ctx)->SetDevice(ctx);
}

void RPCDevGetAttr(RPCSession* handler, TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[1].operator int());
  if (kind == kExist) {
    DeviceAPI* api = handler->GetDeviceAPI(ctx, true);
    if (api != nullptr) {
      api->GetAttr(ctx, kind, rv);
    } else {
      *rv = 0;
    }
  } else {
    handler->GetDeviceAPI(ctx)->GetAttr(
        ctx, static_cast<DeviceAttrKind>(kind), rv);
  }
}

void RPCDevAllocData(RPCSession* handler, TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  uint64_t nbytes = args[1];
  uint64_t alignment = args[2];
  DLDataType type_hint = args[3];
  void* data = handler->GetDeviceAPI(ctx)->AllocDataSpace(
      ctx, nbytes, alignment, type_hint);
  *rv = data;
}

void RPCDevFreeData(RPCSession* handler, TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  void* ptr = args[1];
  handler->GetDeviceAPI(ctx)->FreeDataSpace(ctx, ptr);
}

void RPCDevStreamSync(RPCSession* handler, TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  TVMStreamHandle handle = args[1];
  handler->GetDeviceAPI(ctx)->StreamSync(ctx, handle);
}

void RPCCopyAmongRemote(RPCSession* handler, TVMArgs args, TVMRetValue *rv) {
  void* from = args[0];
  uint64_t from_offset = args[1];
  void* to = args[2];
  uint64_t to_offset = args[3];
  uint64_t size = args[4];
  TVMContext ctx_from = args[5];
  TVMContext ctx_to = args[6];
  DLDataType type_hint = args[7];
  TVMStreamHandle stream = args[8];
  TVMContext ctx = ctx_from;

  if (ctx.device_type == kDLCPU) {
    ctx = ctx_to;
  } else {
    CHECK(ctx_to.device_type == kDLCPU ||
          ctx_to.device_type == ctx_from.device_type)
        << "Can not copy across different ctx types directly";
  }
  handler->GetDeviceAPI(ctx)->CopyDataFromTo(
      from, from_offset,
      to, to_offset,
      size, ctx_from, ctx_to, type_hint, stream);
}

void RPCEndpoint::EventHandler::HandleSyscall(RPCCode code) {
  // Event handler sit at clean state at this point.
  switch (code) {
    // system functions
    case RPCCode::kFreeHandle: SysCallHandler(RPCFreeHandle); break;
    case RPCCode::kGetGlobalFunc: SysCallHandler(RPCGetGlobalFunc); break;
    case RPCCode::kDevSetDevice: SysCallHandler(RPCDevSetDevice); break;
    case RPCCode::kDevGetAttr: SysCallHandler(RPCDevGetAttr); break;
    case RPCCode::kDevAllocData: SysCallHandler(RPCDevAllocData); break;
    case RPCCode::kDevFreeData: SysCallHandler(RPCDevFreeData); break;
    case RPCCode::kDevStreamSync: SysCallHandler(RPCDevStreamSync); break;
    case RPCCode::kCopyAmongRemote: SysCallHandler(RPCCopyAmongRemote); break;
    default: LOG(FATAL) << "Unknown event " << static_cast<int>(code);
  }

  CHECK_EQ(state_, kRecvPacketNumBytes);
}

/*!
 * \brief RPC client session that proxies all calls to an endpoint.
 */
class RPCClientSession : public RPCSession,
                         public DeviceAPI {
 public:
  /*!
   * \brief param endpoint The client endpoint of the session.
   */
  explicit RPCClientSession(std::shared_ptr<RPCEndpoint> endpoint)
      : endpoint_(endpoint) {}

  // function overrides
  PackedFuncHandle GetFunction(const std::string& name) final {
    return endpoint_->SysCallRemote(RPCCode::kGetGlobalFunc, name);
  }

  void CallFunc(PackedFuncHandle func,
                const TVMValue* arg_values,
                const int* arg_type_codes,
                int num_args,
                const FEncodeReturn& fencode_return) final {
    endpoint_->CallFunc(
        func, arg_values, arg_type_codes, num_args, fencode_return);
  }

  void CopyToRemote(void* from,
                    size_t from_offset,
                    void* to,
                    size_t to_offset,
                    size_t nbytes,
                    TVMContext ctx_to,
                    DLDataType type_hint) final {
    endpoint_->CopyToRemote(
        from, from_offset, to, to_offset, nbytes, ctx_to, type_hint);
  }

  void CopyFromRemote(void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t nbytes,
                      TVMContext ctx_from,
                      DLDataType type_hint) final {
    endpoint_->CopyFromRemote(
        from, from_offset, to, to_offset, nbytes, ctx_from, type_hint);
  }

  void FreeHandle(void* handle, int type_code) final {
    endpoint_->SysCallRemote(RPCCode::kFreeHandle, handle, type_code);
  }


  void SetDevice(TVMContext ctx) final {
    endpoint_->SysCallRemote(RPCCode::kDevSetDevice, ctx);
  }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (ctx.device_type == kDLCPU && kind == kExist) {
      // cpu always exists.
      *rv = 1;
    } else {
      *rv = endpoint_->SysCallRemote(RPCCode::kDevGetAttr, ctx, static_cast<int>(kind));
    }
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       DLDataType type_hint) final {
    return endpoint_->SysCallRemote(
        RPCCode::kDevAllocData, ctx, nbytes, alignment, type_hint);
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    endpoint_->SysCallRemote(RPCCode::kDevFreeData, ctx, ptr);
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      DLDataType type_hint,
                      TVMStreamHandle stream) final {
    endpoint_->SysCallRemote(
        RPCCode::kCopyAmongRemote,
        const_cast<void*>(from), from_offset,
        to, to_offset,
        size,
        ctx_from, ctx_to,
        type_hint, stream);
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    endpoint_->SysCallRemote(RPCCode::kDevStreamSync, ctx, stream);
  }

  DeviceAPI* GetDeviceAPI(TVMContext ctx, bool allow_missing) final {
    return this;
  }

 private:
  std::shared_ptr<RPCEndpoint> endpoint_;
};

std::shared_ptr<RPCSession>
CreateClientSession(std::shared_ptr<RPCEndpoint> endpoint) {
  return std::make_shared<RPCClientSession>(endpoint);
}

}  // namespace runtime
}  // namespace tvm
