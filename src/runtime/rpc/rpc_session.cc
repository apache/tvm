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
 *  Copyright (c) 2017 by Contributors
 * \file rpc_session.cc
 * \brief RPC session for remote function call.
 */
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
#include "rpc_session.h"
#include "../../common/ring_buffer.h"

namespace tvm {
namespace runtime {
// Temp buffer for data array
struct RPCByteArrayBuffer {
  TVMByteArray arr;
  std::string data;
};
// Temp buffer for data array
struct RPCDataArrayBuffer {
  DLTensor tensor;
  std::vector<int64_t> shape;
};
/*!
 * \brief Temporal argument buffer.
 */
struct RPCArgBuffer {
  // The argument values
  std::vector<TVMValue> value;
  // The type codes.
  std::vector<int> tcode;
  // Temporal resources.
  std::vector<std::unique_ptr<RPCByteArrayBuffer> > temp_bytes;
  // Temporal array
  std::vector<std::unique_ptr<RPCDataArrayBuffer> > temp_array;
  // convert buffer as TVMArgs
  TVMArgs AsTVMArgs() const {
    return TVMArgs(value.data(), tcode.data(), static_cast<int>(value.size()));
  }
};

// Event handler for RPC events.
class RPCSession::EventHandler : public dmlc::Stream {
 public:
  EventHandler(common::RingBuffer* reader,
               common::RingBuffer* writer,
               int rpc_sess_table_index,
               std::string name,
               std::string* remote_key)
      : reader_(reader),
        writer_(writer),
        rpc_sess_table_index_(rpc_sess_table_index),
        name_(name),
        remote_key_(remote_key) {
    this->Clear();
    if (*remote_key == "%toinit") {
      state_ = kInitHeader;
      remote_key_->resize(0);
      pending_request_bytes_ = sizeof(int32_t);
    }
  }
  // Bytes needed to fulfill current request
  size_t BytesNeeded() {
    if (reader_->bytes_available() < pending_request_bytes_) {
      return pending_request_bytes_ - reader_->bytes_available();
    } else {
      return 0;
    }
  }
  // Request number of bytes from reader.
  void RequestBytes(size_t nbytes) {
    pending_request_bytes_ += nbytes;
    reader_->Reserve(pending_request_bytes_);
  }
  // Whether we are ready to handle next request.
  bool Ready() {
    return reader_->bytes_available() >= pending_request_bytes_;
  }
  bool CanCleanShutdown() const {
    return state_ == kRecvCode;
  }
  void FinishCopyAck() {
    this->SwitchToState(kRecvCode);
  }
  RPCCode HandleNextEvent(TVMRetValue* rv,
                          bool client_mode,
                          const PackedFunc* fwrap) {
    std::swap(client_mode_, client_mode);
    while (this->Ready()) {
      switch (state_) {
        case kInitHeader: HandleInitHeader(); break;
        case kRecvCode: HandleRecvCode(); break;
        case kRecvCallHandle: {
          CHECK(this->Read(&call_handle_));
          this->SwitchToState(kRecvPackedSeqNumArgs);
          break;
        }
        case kRecvPackedSeqNumArgs: {
          CHECK(this->Read(&num_packed_args_));
          arg_buf_.reset(new RPCArgBuffer());
          arg_buf_->value.resize(num_packed_args_);
          arg_buf_->tcode.resize(num_packed_args_);
          this->SwitchToState(kRecvPackedSeqTypeCode);
          break;
        }
        case kRecvPackedSeqTypeCode: {
          if (num_packed_args_ != 0) {
            this->ReadArray(arg_buf_->tcode.data(), num_packed_args_);
          }
          arg_index_ = 0;
          arg_recv_stage_ = 0;
          this->SwitchToState(kRecvPackedSeqArg);
          break;
        }
        case kRecvPackedSeqArg: {
          this->HandleRecvPackedSeqArg();
          break;
        }
        case kDoCopyFromRemote: {
          this->HandleCopyFromRemote();
          break;
        }
        case kDoCopyToRemote: {
          this->HandleCopyToRemote();
          break;
        }
        case kReturnReceived: {
          CHECK_GE(arg_buf_->value.size(), 1U);

          TVMArgValue argv = arg_buf_->AsTVMArgs()[0];
          if (argv.type_code() == kFuncHandle ||
              argv.type_code() == kModuleHandle ||
              argv.type_code() == kArrayHandle) {
            CHECK(fwrap != nullptr) << "function/module wrapper not available";
            fwrap->CallPacked(arg_buf_->AsTVMArgs(), rv);
          } else {
            CHECK_EQ(arg_buf_->value.size(), 1U);
            *rv = argv;
          }
          arg_buf_.reset();
          this->SwitchToState(kRecvCode);
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
  // Reset and clear all states.
  void Clear() {
    state_ = kRecvCode;
    pending_request_bytes_ = sizeof(RPCCode);
    arg_recv_stage_ = 0;
    arg_buf_.reset();
  }
  // strip session on mask
  TVMContext StripSessMask(TVMContext ctx) {
    int dev_type = ctx.device_type;
    CHECK_EQ(dev_type / kRPCSessMask, rpc_sess_table_index_ + 1)
        << "Can not pass in local context or context with a different remote session";
    ctx.device_type = static_cast<DLDeviceType>(dev_type % kRPCSessMask);
    return ctx;
  }
  // Send Packed sequence to writer.
  // return_ndarray is a special flag to handle returning of ndarray
  //    In this case, we return the shape, context and data of the array,
  //    as well as a customized PackedFunc that handles deletion of
  //    the array in the remote.
  void SendPackedSeq(const TVMValue* arg_values,
                     const int* type_codes,
                     int n,
                     bool return_ndarray = false) {
    this->Write(n);
    for (int i = 0; i < n; ++i) {
      int tcode = type_codes[i];
      if (tcode == kNDArrayContainer) tcode = kArrayHandle;
      this->Write(tcode);
    }

    // Argument packing.
    for (int i = 0; i < n; ++i) {
      int tcode = type_codes[i];
      TVMValue value = arg_values[i];
      switch (tcode) {
        case kDLInt:
        case kDLUInt:
        case kDLFloat: {
          this->Write<int64_t>(value.v_int64);
          break;
        }
        case kTVMType: {
          this->Write(value.v_type);
          // padding
          int32_t padding = 0;
          this->Write<int32_t>(padding);
          break;
        }
        case kTVMContext: {
          value.v_ctx = StripSessMask(value.v_ctx);
          this->Write(value.v_ctx);
          break;
        }
        case kFuncHandle:
        case kModuleHandle:
        case kHandle: {
          // always send handle in 64 bit.
          uint64_t handle = reinterpret_cast<uint64_t>(value.v_handle);
          this->Write(handle);
          break;
        }
        case kNDArrayContainer:
        case kArrayHandle: {
          DLTensor* arr = static_cast<DLTensor*>(value.v_handle);
          TVMContext ctx;
          uint64_t data;
          if (!return_ndarray) {
            // in the client mode
            // ctx contains the remote table index
            // the space is wrapped by an RemoteSpace
            // that holds reference to the session.
            ctx = StripSessMask(arr->ctx);
            data = reinterpret_cast<uint64_t>(
                static_cast<RemoteSpace*>(arr->data)->data);
          } else {
            // When we return NDArray, we directly return
            // the space and the context
            // The client will be further wrapping
            ctx = arr->ctx;
            data = reinterpret_cast<uint64_t>(arr->data);
          }
          this->Write(data);
          this->Write(ctx);
          this->Write(arr->ndim);
          this->Write(arr->dtype);
          this->WriteArray(arr->shape, arr->ndim);
          CHECK(arr->strides == nullptr)
              << "Do not support strided remote array";
          CHECK_EQ(arr->byte_offset, 0)
              << "Do not support send byte offset";
          break;
        }
        case kNull: break;
        case kStr: {
          const char* s = value.v_str;
          uint64_t len = strlen(s);
          this->Write(len);
          this->WriteArray(s, len);
          break;
        }
        case kBytes: {
          TVMByteArray* bytes = static_cast<TVMByteArray*>(arg_values[i].v_handle);
          uint64_t len = bytes->size;
          this->Write(len);
          this->WriteArray(bytes->data, len);
          break;
        }
        default: {
          LOG(FATAL) << "RPC cannot handle type " << TypeCode2Str(tcode);
          break;
        }
      }
    }
  }

  // Endian aware IO handling
  using Stream::Read;
  using Stream::Write;
  using Stream::ReadArray;
  using Stream::WriteArray;

  inline bool Read(RPCCode* code) {
    int cdata;
    if (!this->Read(&cdata)) return false;
    *code = static_cast<RPCCode>(cdata);
    return true;
  }
  inline void Write(RPCCode code) {
    int cdata = static_cast<int>(code);
    this->Write(cdata);
  }

 protected:
  enum State {
    kInitHeader,
    kRecvCode,
    kRecvCallHandle,
    kRecvPackedSeqNumArgs,
    kRecvPackedSeqTypeCode,
    kRecvPackedSeqArg,
    kDoCopyFromRemote,
    kDoCopyToRemote,
    kReturnReceived,
    kCopyAckReceived,
    kShutdownReceived
  };
  // Current state;
  State state_;
  // The RPCCode to be read.
  RPCCode code_;
  // Handle for the remote function call.
  uint64_t call_handle_;
  // Initialize remote header
  bool init_header_step_{0};
  // Number of packed arguments.
  int num_packed_args_;
  // Current argument index.
  int arg_index_;
  // The stage of each argument receiver.
  int arg_recv_stage_;
  // Whether current handler is client or server mode.
  bool client_mode_{false};
  // Argument buffer
  std::unique_ptr<RPCArgBuffer> arg_buf_;
  // Temp byte buffer.
  std::unique_ptr<RPCByteArrayBuffer> temp_bytes_;
  // Temp array buffer.
  std::unique_ptr<RPCDataArrayBuffer> temp_array_;
  // Internal temporal data space.
  std::string temp_data_;
  // Temp variables for copy request state.
  TVMContext copy_ctx_;
  TVMType copy_dtype_;
  uint64_t copy_handle_, copy_offset_, copy_size_;
  // State switcher
  void SwitchToState(State state) {
    // invariant
    CHECK_EQ(pending_request_bytes_, 0U)
        << "state=" << state;
    state_ = state;
    switch (state) {
      case kInitHeader: {
        LOG(FATAL) << "cannot switch to init header";
        break;
      }
      case kRecvCode: {
        this->RequestBytes(sizeof(RPCCode));
        break;
      }
      case kRecvCallHandle: {
        this->RequestBytes(sizeof(call_handle_));
        break;
      }
      case kRecvPackedSeqNumArgs: {
        this->RequestBytes(sizeof(num_packed_args_));
        break;
      }
      case kRecvPackedSeqTypeCode: {
        this->RequestBytes(sizeof(int) * num_packed_args_);
        break;
      }
      case kRecvPackedSeqArg: {
        CHECK_LE(arg_index_, num_packed_args_);
        if (arg_index_ == num_packed_args_) {
          // The function can change state_ again.
          HandlePackedCall();
        } else {
          RequestRecvPackedSeqArg();
        }
        break;
      }
      case kDoCopyFromRemote: {
        this->RequestBytes(sizeof(uint64_t) * 3);
        this->RequestBytes(sizeof(TVMContext));
        this->RequestBytes(sizeof(TVMType));
        break;
      }
      case kDoCopyToRemote: {
        this->RequestBytes(sizeof(uint64_t) * 3);
        this->RequestBytes(sizeof(TVMContext));
        this->RequestBytes(sizeof(TVMType));
        break;
      }
      case kCopyAckReceived:
      case kReturnReceived:
      case kShutdownReceived: {
        break;
      }
    }
  }
  // Requets bytes needed for next computation.
  void RequestRecvPackedSeqArg() {
    CHECK_EQ(arg_recv_stage_, 0);
    int tcode = arg_buf_->tcode[arg_index_];
    static_assert(sizeof(TVMValue) == sizeof(uint64_t), "invariant");
    switch (tcode) {
      case kDLInt:
      case kDLUInt:
      case kDLFloat:
      case kTVMType:
      case kHandle:
      case kStr:
      case kBytes:
      case kTVMContext: {
        this->RequestBytes(sizeof(TVMValue)); break;
      }
      case kFuncHandle:
      case kModuleHandle: {
        CHECK(client_mode_)
            << "Only client can receive remote functions";
        this->RequestBytes(sizeof(TVMValue)); break;
      }
      case kNull: break;
      case kArrayHandle: {
        this->RequestBytes(sizeof(uint64_t));
        this->RequestBytes(sizeof(TVMContext));
        this->RequestBytes(sizeof(int));
        this->RequestBytes(sizeof(DLDataType));
        break;
      }
      default: {
        LOG(FATAL) << "RPC cannot handle type " << TypeCode2Str(tcode);
        break;
      }
    }
  }
  // Handler for packed sequence argument receive.
  void HandleRecvPackedSeqArg() {
    CHECK_LT(arg_index_, num_packed_args_);
    int tcode = arg_buf_->tcode[arg_index_];
    TVMValue& value = arg_buf_->value[arg_index_];
    if (arg_recv_stage_ == 0) {
      switch (tcode) {
        case kDLInt:
        case kDLUInt:
        case kDLFloat: {
          this->Read<int64_t>(&(value.v_int64));
          ++arg_index_;
          this->SwitchToState(kRecvPackedSeqArg);
          break;
        }
        case kTVMType: {
          this->Read(&(value.v_type));
          int32_t padding = 0;
          this->Read<int32_t>(&padding);
          ++arg_index_;
          this->SwitchToState(kRecvPackedSeqArg);
          break;
        }
        case kTVMContext: {
          this->Read(&(value.v_ctx));
          ++arg_index_;
          this->SwitchToState(kRecvPackedSeqArg);
          break;
        }
        case kFuncHandle:
        case kModuleHandle:
        case kHandle: {
          // always send handle in 64 bit.
          uint64_t handle;
          this->Read(&handle);
          value.v_handle = reinterpret_cast<void*>(handle);
          ++arg_index_;
          this->SwitchToState(kRecvPackedSeqArg);
          break;
        }
        case kNull: {
          value.v_handle = nullptr;
          ++arg_index_;
          this->SwitchToState(kRecvPackedSeqArg);
          break;
        }
        case kStr:
        case kBytes: {
          uint64_t len;
          this->Read(&len);
          temp_bytes_.reset( new RPCByteArrayBuffer());
          temp_bytes_->data.resize(len);
          arg_recv_stage_ = 1;
          this->RequestBytes(len);
          break;
        }
        case kArrayHandle: {
          temp_array_.reset(new RPCDataArrayBuffer());
          uint64_t handle;
          this->Read(&handle);
          DLTensor& tensor = temp_array_->tensor;
          tensor.data = reinterpret_cast<void*>(handle);
          this->Read(&(tensor.ctx));
          this->Read(&(tensor.ndim));
          this->Read(&(tensor.dtype));
          temp_array_->shape.resize(tensor.ndim);
          tensor.shape = temp_array_->shape.data();
          arg_recv_stage_ = 1;
          tensor.strides = nullptr;
          tensor.byte_offset = 0;
          this->RequestBytes(sizeof(int64_t) * tensor.ndim);
          break;
        }
        default: {
          LOG(FATAL) << "RPC cannot handle type " << TypeCode2Str(tcode);
          break;
        }
      }
    } else {
      CHECK_EQ(arg_recv_stage_, 1);
      if (tcode == kStr || tcode == kBytes) {
        if (temp_bytes_->data.size() != 0) {
          this->ReadArray(&(temp_bytes_->data[0]), temp_bytes_->data.size());
        }
        if (tcode == kStr) {
          value.v_str = temp_bytes_->data.c_str();
        } else {
          temp_bytes_->arr.size = static_cast<size_t>(temp_bytes_->data.size());
          temp_bytes_->arr.data = dmlc::BeginPtr(temp_bytes_->data);
          value.v_handle = &(temp_bytes_->arr);
        }
        arg_buf_->temp_bytes.emplace_back(std::move(temp_bytes_));
      } else {
        CHECK_EQ(tcode, kArrayHandle);
        DLTensor& tensor = temp_array_->tensor;
        this->ReadArray(tensor.shape, tensor.ndim);
        value.v_handle = &tensor;
        arg_buf_->temp_array.emplace_back(std::move(temp_array_));
      }
      ++arg_index_;
      arg_recv_stage_ = 0;
      this->SwitchToState(kRecvPackedSeqArg);
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
      this->SwitchToState(kRecvCode);
    }
  }
  // Handler for read code.
  void HandleRecvCode() {
    this->Read(&code_);
    if (code_ > RPCCode::kSystemFuncStart) {
      SwitchToState(kRecvPackedSeqNumArgs);
      return;
    }
    // invariant.
    CHECK_EQ(arg_recv_stage_, 0);
    switch (code_) {
      case RPCCode::kCallFunc: {
        SwitchToState(kRecvCallHandle);
        break;
      }
      case RPCCode::kException:
      case RPCCode::kReturn: {
        SwitchToState(kRecvPackedSeqNumArgs);
        break;
      }
      case RPCCode::kCopyFromRemote: {
        SwitchToState(kDoCopyFromRemote);
        break;
      }
      case RPCCode::kCopyToRemote: {
        SwitchToState(kDoCopyToRemote);
        break;
      }
      case RPCCode::kShutdown: {
        SwitchToState(kShutdownReceived);
        break;
      }
      case RPCCode::kCopyAck: {
        SwitchToState(kCopyAckReceived);
        break;
      }
      default: LOG(FATAL) << "Unknown event "  << static_cast<int>(code_);
    }
  }

  void HandleCopyFromRemote() {
    uint64_t handle, offset, num_bytes;
    TVMContext ctx;
    TVMType type_hint;
    this->Read(&handle);
    this->Read(&offset);
    this->Read(&num_bytes);
    this->Read(&ctx);
    this->Read(&type_hint);
    size_t elem_bytes = (type_hint.bits * type_hint.lanes + 7) / 8;

    if (ctx.device_type == kDLCPU) {
      RPCCode code = RPCCode::kCopyAck;
      this->Write(code);
      char* dptr = reinterpret_cast<char*>(handle) + offset;
      if (!DMLC_IO_NO_ENDIAN_SWAP) {
        temp_data_.resize(0);
        temp_data_.insert(temp_data_.end(), dptr, dptr + num_bytes);
        dmlc::ByteSwap(dmlc::BeginPtr(temp_data_), elem_bytes, num_bytes / elem_bytes);
        this->WriteArray(temp_data_.data(), num_bytes);
      } else {
        this->WriteArray(dptr, num_bytes);
      }
    } else {
      temp_data_.resize(num_bytes + 1);
      try {
        TVMContext cpu_ctx;
        cpu_ctx.device_type = kDLCPU;
        cpu_ctx.device_id = 0;
        DeviceAPI::Get(ctx)->CopyDataFromTo(
            reinterpret_cast<void*>(handle), offset,
            dmlc::BeginPtr(temp_data_), 0,
            num_bytes, ctx, cpu_ctx, type_hint, nullptr);
        RPCCode code = RPCCode::kCopyAck;
        this->Write(code);
        if (!DMLC_IO_NO_ENDIAN_SWAP) {
          dmlc::ByteSwap(dmlc::BeginPtr(temp_data_), elem_bytes, num_bytes / elem_bytes);
        }
        this->WriteArray(&temp_data_[0], num_bytes);
      } catch (const std::runtime_error &e) {
        RPCCode code = RPCCode::kException;
        this->Write(code);
        TVMValue ret_value;
        ret_value.v_str = e.what();
        int ret_tcode = kStr;
        SendPackedSeq(&ret_value, &ret_tcode, 1);
      }
    }
    this->SwitchToState(kRecvCode);
  }

  void HandleCopyToRemote() {
    // use static variable to persist state.
    // This only works if next stage is immediately after this.
    if (arg_recv_stage_ == 0) {
      CHECK(this->Read(&copy_handle_));
      CHECK(this->Read(&copy_offset_));
      CHECK(this->Read(&copy_size_));
      CHECK(this->Read(&copy_ctx_));
      CHECK(this->Read(&copy_dtype_));
      arg_recv_stage_ = 1;
      CHECK_EQ(pending_request_bytes_, 0U);
      this->RequestBytes(copy_size_);
    } else {
      CHECK_EQ(arg_recv_stage_, 1);
      TVMValue ret_value;
      ret_value.v_handle = nullptr;
      int ret_tcode = kNull;
      RPCCode code = RPCCode::kReturn;
      std::string errmsg;

      size_t elem_bytes = (copy_dtype_.bits * copy_dtype_.lanes + 7) / 8;
      if (copy_ctx_.device_type == kDLCPU) {
        char* dptr = reinterpret_cast<char*>(copy_handle_) + copy_offset_;
        this->ReadArray(dptr, copy_size_);
        if (!DMLC_IO_NO_ENDIAN_SWAP) {
          dmlc::ByteSwap(dptr, elem_bytes, copy_size_ / elem_bytes);
        }
      } else {
        temp_data_.resize(copy_size_ + 1);
        this->ReadArray(&temp_data_[0], copy_size_);
        if (!DMLC_IO_NO_ENDIAN_SWAP) {
          dmlc::ByteSwap(dmlc::BeginPtr(temp_data_), elem_bytes, copy_size_ / elem_bytes);
        }
        try {
          TVMContext cpu_ctx;
          cpu_ctx.device_type = kDLCPU;
          cpu_ctx.device_id = 0;
          DeviceAPI::Get(copy_ctx_)->CopyDataFromTo(
              temp_data_.data(), 0,
              reinterpret_cast<void*>(copy_handle_), copy_offset_,
              copy_size_, cpu_ctx, copy_ctx_, copy_dtype_, nullptr);
        } catch (const std::runtime_error &e) {
          code = RPCCode::kException;
          errmsg = e.what();
          ret_value.v_str = errmsg.c_str();
          ret_tcode = kStr;
        }
      }
      this->Write(code);
      SendPackedSeq(&ret_value, &ret_tcode, 1);
      arg_recv_stage_ = 0;
      this->SwitchToState(kRecvCode);
    }
  }
  // Handle for packed call.
  void HandlePackedCall();

  template<typename F>
  void CallHandler(F f) {
    TVMRetValue rv;
    TVMValue ret_value;
    int ret_tcode;
    try {
      // Need to move out, in case f itself need to call RecvPackedSeq
      // Which will override argbuf again.
      std::unique_ptr<RPCArgBuffer> args = std::move(arg_buf_);
      f(args->AsTVMArgs(), &rv);
      RPCCode code = RPCCode::kReturn;
      this->Write(code);
      if (rv.type_code() == kStr) {
        ret_value.v_str = rv.ptr<std::string>()->c_str();
        ret_tcode = kStr;
        SendPackedSeq(&ret_value, &ret_tcode, 1);
      } else if (rv.type_code() == kBytes) {
        std::string* bytes = rv.ptr<std::string>();
        TVMByteArray arr;
        arr.data = bytes->c_str();
        arr.size = bytes->length();
        ret_value.v_handle = &arr;
        ret_tcode = kBytes;
        SendPackedSeq(&ret_value, &ret_tcode, 1);
      } else if (rv.type_code() == kFuncHandle ||
                 rv.type_code() == kModuleHandle) {
        // always send handle in 64 bit.
        CHECK(!client_mode_)
              << "Only server can send function and module handle back.";
        rv.MoveToCHost(&ret_value, &ret_tcode);
        SendPackedSeq(&ret_value, &ret_tcode, 1);
      } else if (rv.type_code() == kNDArrayContainer) {
        // always send handle in 64 bit.
        CHECK(!client_mode_)
            << "Only server can send NDArray back";
        // We follow a special protocol to return NDArray to client side
        // The first pack value is the NDArray handle as DLTensor
        // The second pack value is a customized deleter that deletes the NDArray.
        TVMValue ret_value_pack[2];
        int ret_tcode_pack[2];
        rv.MoveToCHost(&ret_value_pack[0], &ret_tcode_pack[0]);

        NDArray::Container* nd = static_cast<NDArray::Container*>(ret_value_pack[0].v_handle);
        ret_value_pack[1].v_handle = nd;
        ret_tcode_pack[1] = kHandle;
        SendPackedSeq(ret_value_pack, ret_tcode_pack, 2, true);
      } else {
        ret_value = rv.value();
        ret_tcode = rv.type_code();
        SendPackedSeq(&ret_value, &ret_tcode, 1);
      }
    } catch (const std::runtime_error& e) {
      RPCCode code = RPCCode::kException;
      this->Write(code);
      ret_value.v_str = e.what();
      ret_tcode = kStr;
      SendPackedSeq(&ret_value, &ret_tcode, 1);
    }
  }

 private:
  // Utility functions
  // Internal read function, update pending_request_bytes_
  size_t Read(void* data, size_t size) final {
    CHECK_LE(size, pending_request_bytes_);
    reader_->Read(data, size);
    pending_request_bytes_ -= size;
    return size;
  }
  void Write(const void* data, size_t size) final {
    writer_->Write(data, size);
  }
  // Number of pending bytes requests
  size_t pending_request_bytes_;
  // The ring buffer to read data from.
  common::RingBuffer* reader_;
  // The ringr buffer to write reply to.
  common::RingBuffer* writer_;
  // Session table index.
  int rpc_sess_table_index_;
  // Name of session.
  std::string name_;
  // remote key
  std::string* remote_key_;
};

struct RPCSessTable {
 public:
  static constexpr int kMaxRPCSession = 32;
  // Get global singleton
  static RPCSessTable* Global() {
    static RPCSessTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<RPCSession> Get(int index) {
    CHECK(index >= 0 && index < kMaxRPCSession);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<RPCSession> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxRPCSession; ++i) {
      if (tbl_[i].lock() == nullptr) {
        tbl_[i] = ptr; return i;
      }
    }
    LOG(FATAL) << "maximum number of RPC session reached";
    return 0;
  }

 private:
  // The mutex
  std::mutex mutex_;
  // Use weak_ptr intentionally
  // If the RPCSession get released, the pointer session will be released
  std::array<std::weak_ptr<RPCSession>, kMaxRPCSession> tbl_;
};

RPCCode RPCSession::HandleUntilReturnEvent(
    TVMRetValue* rv,  bool client_mode, const PackedFunc* fwrap) {
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
    code = handler_->HandleNextEvent(rv, client_mode, fwrap);
  }
  return code;
}

void RPCSession::Init() {
  // Event handler
  handler_ = std::make_shared<EventHandler>(
      &reader_, &writer_, table_index_, name_, &remote_key_);
  // Quick function to call remote.
  call_remote_ = PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
      handler_->SendPackedSeq(args.values, args.type_codes, args.num_args);
      RPCCode code = HandleUntilReturnEvent(rv, true, nullptr);
      CHECK(code == RPCCode::kReturn) << "code=" << static_cast<int>(code);
    });
}

std::shared_ptr<RPCSession> RPCSession::Create(
    std::unique_ptr<RPCChannel> channel,
    std::string name,
    std::string remote_key) {
  std::shared_ptr<RPCSession> sess = std::make_shared<RPCSession>();
  sess->channel_ = std::move(channel);
  sess->name_ = std::move(name);
  sess->remote_key_ = std::move(remote_key);
  sess->table_index_ = RPCSessTable::Global()->Insert(sess);
  sess->Init();
  return sess;
}

std::shared_ptr<RPCSession> RPCSession::Get(int table_index) {
  return RPCSessTable::Global()->Get(table_index);
}

RPCSession::~RPCSession() {
  this->Shutdown();
}

void RPCSession::Shutdown() {
  if (channel_ != nullptr) {
    RPCCode code = RPCCode::kShutdown;
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

void RPCSession::ServerLoop() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (const auto* f = Registry::Get("tvm.rpc.server.start")) {
    (*f)();
  }
  TVMRetValue rv;
  CHECK(HandleUntilReturnEvent(&rv, false, nullptr) == RPCCode::kShutdown);
  if (const auto* f = Registry::Get("tvm.rpc.server.shutdown")) {
    (*f)();
  }
  channel_.reset(nullptr);
}

int RPCSession::ServerEventHandler(const std::string& bytes, int event_flag) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  RPCCode code = RPCCode::kNone;
  if (bytes.length() != 0) {
    reader_.Write(bytes.c_str(), bytes.length());
    TVMRetValue rv;
    code = handler_->HandleNextEvent(&rv, false, nullptr);
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

// Get remote function with name
void RPCSession::CallFunc(void* h,
                          TVMArgs args,
                          TVMRetValue* rv,
                          const PackedFunc* fwrap) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  RPCCode code = RPCCode::kCallFunc;
  handler_->Write(code);
  uint64_t handle = reinterpret_cast<uint64_t>(h);
  handler_->Write(handle);
  handler_->SendPackedSeq(args.values, args.type_codes, args.num_args);
  code = HandleUntilReturnEvent(rv, true, fwrap);
  CHECK(code == RPCCode::kReturn) << "code=" << static_cast<int>(code);
}

void RPCSession::CopyToRemote(void* from,
                              size_t from_offset,
                              void* to,
                              size_t to_offset,
                              size_t data_size,
                              TVMContext ctx_to,
                              TVMType type_hint) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  ctx_to = handler_->StripSessMask(ctx_to);
  RPCCode code = RPCCode::kCopyToRemote;
  handler_->Write(code);
  uint64_t handle = reinterpret_cast<uint64_t>(to);
  handler_->Write(handle);
  uint64_t offset = static_cast<uint64_t>(to_offset);
  handler_->Write(offset);
  uint64_t size = static_cast<uint64_t>(data_size);
  handler_->Write(size);
  handler_->Write(ctx_to);
  handler_->Write(type_hint);
  handler_->WriteArray(reinterpret_cast<char*>(from) + from_offset, data_size);
  TVMRetValue rv;
  CHECK(HandleUntilReturnEvent(&rv, true, nullptr) == RPCCode::kReturn);
}

void RPCSession::CopyFromRemote(void* from,
                                size_t from_offset,
                                void* to,
                                size_t to_offset,
                                size_t data_size,
                                TVMContext ctx_from,
                                TVMType type_hint) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  ctx_from = handler_->StripSessMask(ctx_from);
  RPCCode code = RPCCode::kCopyFromRemote;
  handler_->Write(code);
  uint64_t handle = reinterpret_cast<uint64_t>(from);
  handler_->Write(handle);
  uint64_t offset = static_cast<uint64_t>(from_offset);
  handler_->Write(offset);
  uint64_t size = static_cast<uint64_t>(data_size);
  handler_->Write(size);
  handler_->Write(ctx_from);
  handler_->Write(type_hint);
  TVMRetValue rv;
  CHECK(HandleUntilReturnEvent(&rv, true, nullptr) == RPCCode::kCopyAck);
  reader_.Reserve(data_size);
  handler_->RequestBytes(data_size);
  while (!handler_->Ready()) {
    size_t bytes_needed = handler_->BytesNeeded();
    reader_.WriteWithCallback([this](void* data, size_t size) {
        size_t n = channel_->Recv(data, size);
        CHECK_NE(n, 0U) << "Channel closes before we get neded bytes";
        return n;
      }, bytes_needed);
  }
  handler_->ReadArray(reinterpret_cast<char*>(to) + to_offset, data_size);
  handler_->FinishCopyAck();
}

RPCFuncHandle RPCSession::GetTimeEvaluator(
    RPCFuncHandle fhandle, TVMContext ctx, int number, int repeat, int min_repeat_ms) {
  return this->CallRemote(
      RPCCode::kGetTimeEvaluator, fhandle, ctx, number, repeat, min_repeat_ms);
}

// Event handler functions
void RPCGetGlobalFunc(TVMArgs args, TVMRetValue* rv) {
  std::string name = args[0];
  auto *fp = tvm::runtime::Registry::Get(name);
  if (fp != nullptr) {
    *rv = static_cast<void*>(new tvm::runtime::PackedFunc(*fp));
  } else {
    *rv = nullptr;
  }
}

void RPCFreeFunc(TVMArgs args, TVMRetValue *rv) {
  void* handle = args[0];
  delete static_cast<PackedFunc*>(handle);
}

void RPCDevSetDevice(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  DeviceAPI::Get(ctx)->SetDevice(ctx);
}

void RPCDevGetAttr(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[1].operator int());
  if (kind == kExist) {
    DeviceAPI* api = DeviceAPI::Get(ctx, true);
    if (api != nullptr) {
      api->GetAttr(ctx, kind, rv);
    } else {
      *rv = 0;
    }
  } else {
    DeviceAPI::Get(ctx)->GetAttr(
        ctx, static_cast<DeviceAttrKind>(kind), rv);
  }
}

void RPCDevAllocData(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  uint64_t nbytes = args[1];
  uint64_t alignment = args[2];
  TVMType type_hint = args[3];
  void* data = DeviceAPI::Get(ctx)->AllocDataSpace(
      ctx, nbytes, alignment, type_hint);
  *rv = data;
}

void RPCDevFreeData(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  void* ptr = args[1];
  DeviceAPI::Get(ctx)->FreeDataSpace(ctx, ptr);
}

void RPCDevStreamSync(TVMArgs args, TVMRetValue *rv) {
  TVMContext ctx = args[0];
  TVMStreamHandle handle = args[1];
  DeviceAPI::Get(ctx)->StreamSync(ctx, handle);
}

void RPCCopyAmongRemote(TVMArgs args, TVMRetValue *rv) {
  void* from = args[0];
  uint64_t from_offset = args[1];
  void* to = args[2];
  uint64_t to_offset = args[3];
  uint64_t size = args[4];
  TVMContext ctx_from = args[5];
  TVMContext ctx_to = args[6];
  TVMType type_hint = args[7];
  TVMStreamHandle stream = args[8];
  TVMContext ctx = ctx_from;
  if (ctx.device_type == kDLCPU) {
    ctx = ctx_to;
  } else {
    CHECK(ctx_to.device_type == kDLCPU ||
          ctx_to.device_type == ctx_from.device_type)
        << "Can not copy across different ctx types directly";
  }
  DeviceAPI::Get(ctx)->CopyDataFromTo(
      from, from_offset,
      to, to_offset,
      size, ctx_from, ctx_to, type_hint, stream);
}

void RPCModuleLoad(TVMArgs args, TVMRetValue *rv) {
  static const PackedFunc* fsys_load_ = nullptr;
  if (fsys_load_ == nullptr) {
    fsys_load_ = runtime::Registry::Get("tvm.rpc.server.load_module");
    CHECK(fsys_load_ != nullptr);
  }
  std::string file_name = args[0];
  TVMRetValue ret = (*fsys_load_)(file_name);
  Module m = ret;
  *rv = static_cast<void*>(new Module(m));
}

void RPCModuleImport(TVMArgs args, TVMRetValue *rv) {
  void* pmod = args[0];
  void* cmod = args[1];
  static_cast<Module*>(pmod)->Import(
      *static_cast<Module*>(cmod));
}

void RPCModuleFree(TVMArgs args, TVMRetValue *rv) {
  void* mhandle = args[0];
  delete static_cast<Module*>(mhandle);
}

void RPCModuleGetFunc(TVMArgs args, TVMRetValue *rv) {
  void* mhandle = args[0];
  PackedFunc pf = static_cast<Module*>(mhandle)->GetFunction(
      args[1], false);
  if (pf != nullptr) {
    *rv = static_cast<void*>(new PackedFunc(pf));
  } else {
    *rv = nullptr;
  }
}

void RPCModuleGetSource(TVMArgs args, TVMRetValue *rv) {
  void* mhandle = args[0];
  std::string fmt = args[1];
  *rv = (*static_cast<Module*>(mhandle))->GetSource(fmt);
}

void RPCNDArrayFree(TVMArgs args, TVMRetValue *rv) {
  void* handle = args[0];
  static_cast<NDArray::Container*>(handle)->DecRef();
}

void RPCGetTimeEvaluator(TVMArgs args, TVMRetValue *rv) {
  PackedFunc *pf = static_cast<PackedFunc*>(args[0].operator void*());
  void *fhandle = new PackedFunc(WrapTimeEvaluator(*pf, args[1], args[2], args[3], args[4]));
  delete pf;
  *rv = fhandle;
}

void RPCSession::EventHandler::HandlePackedCall() {
  CHECK_EQ(pending_request_bytes_, 0U);
  if (code_ == RPCCode::kReturn) {
    state_ = kReturnReceived; return;
  }
  // reset state to clean init state
  state_ = kRecvCode;
  this->RequestBytes(sizeof(RPCCode));
  // Event handler sit at clean state at this point.
  switch (code_) {
    case RPCCode::kCallFunc: {
      PackedFunc* pf = reinterpret_cast<PackedFunc*>(call_handle_);
      CallHandler([pf](TVMArgs args, TVMRetValue* rv) {
          pf->CallPacked(args, rv);
        });
      break;
    }
    case RPCCode::kException: {
      CHECK_EQ(arg_buf_->value.size(), 1U);
      CHECK_EQ(arg_buf_->tcode[0], kStr);
      std::ostringstream os;
      os << "Except caught from RPC call: " << arg_buf_->value[0].v_str;
      arg_buf_.reset();
      throw dmlc::Error(os.str());
      break;
    }
    // system functions
    case RPCCode::kGetTimeEvaluator: CallHandler(RPCGetTimeEvaluator); break;
    case RPCCode::kFreeFunc: CallHandler(RPCFreeFunc); break;
    case RPCCode::kGetGlobalFunc: CallHandler(RPCGetGlobalFunc); break;
    case RPCCode::kDevSetDevice: CallHandler(RPCDevSetDevice); break;
    case RPCCode::kDevGetAttr: CallHandler(RPCDevGetAttr); break;
    case RPCCode::kDevAllocData: CallHandler(RPCDevAllocData); break;
    case RPCCode::kDevFreeData: CallHandler(RPCDevFreeData); break;
    case RPCCode::kDevStreamSync: CallHandler(RPCDevStreamSync); break;
    case RPCCode::kCopyAmongRemote: CallHandler(RPCCopyAmongRemote); break;
    case RPCCode::kModuleLoad: CallHandler(RPCModuleLoad); break;
    case RPCCode::kModuleImport: CallHandler(RPCModuleImport); break;
    case RPCCode::kModuleFree: CallHandler(RPCModuleFree); break;
    case RPCCode::kModuleGetFunc: CallHandler(RPCModuleGetFunc); break;
    case RPCCode::kModuleGetSource: CallHandler(RPCModuleGetSource); break;
    case RPCCode::kNDArrayFree: CallHandler(RPCNDArrayFree); break;
    default: LOG(FATAL) << "Unknown event " << static_cast<int>(code_);
  }
  CHECK_EQ(state_, kRecvCode);
}

PackedFunc WrapTimeEvaluator(PackedFunc pf,
                             TVMContext ctx,
                             int number,
                             int repeat,
                             int min_repeat_ms) {
  auto ftimer = [pf, ctx, number, repeat, min_repeat_ms](TVMArgs args, TVMRetValue *rv) mutable {
    TVMRetValue temp;
    std::ostringstream os;
    // skip first time call, to activate lazy compilation components.
    pf.CallPacked(args, &temp);
    DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);

    for (int i = 0; i < repeat; ++i) {
      std::chrono::time_point<
        std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin, tend;
      double duration_ms = 0.0;

      do {
        if (duration_ms > 0.0) {
          number = static_cast<int>(
              std::max((min_repeat_ms / (duration_ms / number) + 1),
                       number * 1.618));   // 1.618 is chosen by random
        }

        tbegin = std::chrono::high_resolution_clock::now();
        // start timing
        for (int i = 0; i < number; ++i) {
          pf.CallPacked(args, &temp);
        }
        DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
        tend = std::chrono::high_resolution_clock::now();

        duration_ms = std::chrono::duration_cast<std::chrono::duration<double> >
            (tend - tbegin).count() * 1000;
      } while (duration_ms < min_repeat_ms);

      double speed = std::chrono::duration_cast<std::chrono::duration<double> >(
          tend - tbegin).count() / number;
      os.write(reinterpret_cast<char*>(&speed), sizeof(speed));
    }
    std::string blob = os.str();
    TVMByteArray arr;
    arr.size = blob.length();
    arr.data = blob.data();
    // return the time.
    *rv = arr;
  };
  return PackedFunc(ftimer);
}

}  // namespace runtime
}  // namespace tvm
