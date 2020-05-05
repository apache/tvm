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
 * \file rpc_procotol.h
 * \brief Common header defining the communication code used in the RPC protocol.
 */
#ifndef TVM_RUNTIME_RPC_RPC_PROTOCOL_H_
#define TVM_RUNTIME_RPC_RPC_PROTOCOL_H_

namespace tvm {
namespace runtime {

/*! \brief The current RPC procotol version. */
constexpr const char* kRPCProtocolVer = "0.7.0";

/*! \brief The RPC code */
enum class RPCCode : int {
  kNone,
  kShutdown,
  kInitServer,
  kCallFunc,
  kReturn,
  kException,
  kCopyFromRemote,
  kCopyToRemote,
  kCopyAck,
  // The following are syscall code that can send over CallRemote
  kSyscallCodeStart,
  kGetGlobalFunc = kSyscallCodeStart,
  kFreeHandle,
  kDevSetDevice,
  kDevGetAttr,
  kDevAllocData,
  kDevFreeData,
  kDevStreamSync,
  kCopyAmongRemote,
};

/*!
 * \brief List of potential error status during rpc communication.
 */
enum class RPCServerStatus : int {
  kSuccess = 0,
  kInvalidTypeCodeObject,
  kInvalidTypeCodeNDArray,
  kInvalidDLTensorFieldStride,
  kInvalidDLTensorFieldByteOffset,
  kUnknownTypeCode,
  kUnknownRPCCode,
  kRPCCodeNotSupported,
  kUnknownRPCSyscall,
  kCheckError,
  kReadError,
  kWriteError,
  kAllocError
};

/*!
 * \brief Convert RPC server status to string.
 * \param status The status.
 * \return The corresponding string.
 */
inline const char* RPCServerStatusToString(RPCServerStatus status) {
  switch (status) {
    case RPCServerStatus::kSuccess: return "kSuccess";
    case RPCServerStatus::kInvalidTypeCodeObject: return "kInvalidTypeCodeObject";
    case RPCServerStatus::kInvalidTypeCodeNDArray: return "kInvalidTypeCodeNDArray";
    case RPCServerStatus::kInvalidDLTensorFieldStride: return "kInvalidDLTensorFieldStride";
    case RPCServerStatus::kInvalidDLTensorFieldByteOffset: {
      return "kInvalidDLTensorFieldByteOffset";
    }
    case RPCServerStatus::kUnknownTypeCode: return "kUnknownTypeCode";
    case RPCServerStatus::kUnknownRPCCode: return "kUnknownRPCCode";
    case RPCServerStatus::kRPCCodeNotSupported: return "RPCCodeNotSupported";
    case RPCServerStatus::kUnknownRPCSyscall: return "kUnknownRPCSyscall";
    case RPCServerStatus::kCheckError: return "kCheckError";
    case RPCServerStatus::kReadError: return "kReadError";
    case RPCServerStatus::kWriteError: return "kWriteError";
    case RPCServerStatus::kAllocError: return "kAllocError";
    default: return "";
  }
}

/*!
 * \brief Reference implementation of the communication protocol.
 *
 * \note The implementation is intentionally written via template
 *       so it can be used in a dependency free setting.
 *
 * \sa src/runtime/rpc/device/min_rpc_server.h
 */
struct RPCReference {
  /*!
   * \brief Auxiliary class to get the packed sequence.
   * \tparam TChannel The channel to throw errror.
   */
  template<typename TChannel>
  struct PackedSeqNumBytesGetter {
   public:
    explicit PackedSeqNumBytesGetter(TChannel* channel)
        : channel_(channel) {}

    template <typename T>
    void Write(const T& value) {
      num_bytes_ += sizeof(T);
    }

    template <typename T>
    void WriteArray(const T* value, size_t num) {
      num_bytes_ += sizeof(T) * num;
    }

    void ThrowError(RPCServerStatus status) {
      channel_->ThrowError(status);
    }

    uint64_t num_bytes() const {
      return num_bytes_;
    }

   private:
    TChannel* channel_;
    uint64_t num_bytes_{0};
  };

  /*!
   * \return the length of the str.
   * \param str the string.
   * \return The length.
   */
  static uint64_t StrLength(const char* str) {
    uint64_t len = 0;
    while (str[len] != '\0') ++len;
    return len;
  }

  /*!
   * \brief Get the total nbytes to be sent in the packed sequence.
   *
   * \param arg_values The values to be sent over.
   * \param type_codes The type codes to be sent over.
   * \param num_args Number of argument.
   * \param client_mode Whether it is a client to server call.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   * \return The total number of bytes.
   */
  template<typename TChannel>
  static uint64_t PackedSeqGetNumBytes(const TVMValue* arg_values,
                                       const int* type_codes,
                                       int num_args,
                                       bool client_mode,
                                       TChannel* channel) {
    PackedSeqNumBytesGetter<TChannel> getter(channel);
    SendPackedSeq(arg_values, type_codes, num_args, client_mode, &getter);
    return getter.num_bytes();
  }

  /*!
   * \brief Send packed argument sequnce to the other peer.
   *
   * This function serves as the foundational communication primitive between peers.
   *
   * TVMValue sequence encoding protocol(according to the type):
   *
   * - int/float/uint/bytes/str: Serialize all content.
   * - DLTensor: send meta-data, send data handle as opaque handle(via uint64_t)
   * - OpaqueHandle: send as uint64_t
   * - ModuleHandle, PackedFuncHandle: send as uint64_t,
   *   The support to Module/PackedFuncHandle are reserved for arguments
   *   in the CallFunc from a client to server only.
   *   Note that we cannot simply take these argument out(as the handle)
   *   refers to a value on the remote(instead of local).
   *
   * \param arg_values The values to be sent over.
   * \param type_codes The type codes to be sent over.
   * \param num_args Number of argument.
   * \param client_mode Whether it is a client to server call.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template<typename TChannel>
  static void SendPackedSeq(const TVMValue* arg_values,
                            const int* type_codes,
                            int num_args,
                            bool client_mode,
                            TChannel* channel) {
    channel->Write(num_args);
    channel->WriteArray(type_codes, num_args);

    // Argument packing.
    for (int i = 0; i < num_args; ++i) {
      int tcode = type_codes[i];
      TVMValue value = arg_values[i];
      switch (tcode) {
        case kDLInt:
        case kDLUInt:
        case kDLFloat: {
          channel->template Write<int64_t>(value.v_int64);
          break;
        }
        case kTVMDataType: {
          channel->Write(value.v_type);
          // padding
          int32_t padding = 0;
          channel->template Write<int32_t>(padding);
          break;
        }
        case kTVMContext: {
          channel->Write(value.v_ctx);
          break;
        }

        case kTVMPackedFuncHandle:
        case kTVMModuleHandle: {
          if (!client_mode) {
            channel->ThrowError(RPCServerStatus::kInvalidTypeCodeObject);
          }
          // always send handle in 64 bit.
          uint64_t handle = reinterpret_cast<uint64_t>(value.v_handle);
          channel->Write(handle);
          break;
        }
        case kTVMOpaqueHandle: {
          // always send handle in 64 bit.
          uint64_t handle = reinterpret_cast<uint64_t>(value.v_handle);
          channel->Write(handle);
          break;
        }
        case kTVMNDArrayHandle: {
          channel->ThrowError(RPCServerStatus::kInvalidTypeCodeNDArray);
          break;
        }
        case kTVMDLTensorHandle: {
          DLTensor* arr = static_cast<DLTensor*>(value.v_handle);
          TVMContext ctx;
          uint64_t data;
          // When we return NDArray, we directly return
          // the space and the context
          // The client will be further wrapping
          ctx = arr->ctx;
          data = reinterpret_cast<uint64_t>(arr->data);
          channel->Write(data);
          channel->Write(ctx);
          channel->Write(arr->ndim);
          channel->Write(arr->dtype);
          channel->WriteArray(arr->shape, arr->ndim);
          if (arr->strides != nullptr) {
            channel->ThrowError(RPCServerStatus::kInvalidDLTensorFieldStride);
          }
          if (arr->byte_offset != 0) {
            channel->ThrowError(RPCServerStatus::kInvalidDLTensorFieldByteOffset);
          }
          break;
        }
        case kTVMNullptr: break;
        case kTVMStr: {
          const char* s = value.v_str;
          uint64_t len = StrLength(s);
          channel->Write(len);
          channel->WriteArray(s, len);
          break;
        }
        case kTVMBytes: {
          TVMByteArray* bytes = static_cast<TVMByteArray*>(arg_values[i].v_handle);
          uint64_t len = bytes->size;
          channel->Write(len);
          channel->WriteArray(bytes->data, len);
          break;
        }
        default: {
          channel->ThrowError(RPCServerStatus::kUnknownTypeCode);
          break;
        }
      }
    }
  }

  /*!
   * \brief Receive packed seq from the channel.
   *
   * \param out_arg_values The values to be received.
   * \param out_tcodes The type codes to be received.
   * \param out_num_args Number of argument.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   * \note The temporary space are populated via an arena inside channel.
   */
  template<typename TChannel>
  static void RecvPackedSeq(TVMValue** out_values,
                            int** out_tcodes,
                            int* out_num_args,
                            TChannel* channel) {
    // receive number of args
    int num_args;
    channel->Read(&num_args);
    *out_num_args = num_args;

    if (num_args == 0) {
      *out_values = nullptr;
      *out_tcodes = nullptr;
      return;
    }

    TVMValue* values = channel->template ArenaAlloc<TVMValue>(num_args);
    int* tcodes = channel->template ArenaAlloc<int>(num_args);
    *out_values = values;
    *out_tcodes = tcodes;

    // receive type code.
    channel->ReadArray(tcodes, num_args);

    // receive arguments
    for (int i = 0; i < num_args; ++i) {
      auto& value = values[i];
      switch (tcodes[i]) {
        case kDLInt:
        case kDLUInt:
        case kDLFloat: {
          channel->template Read<int64_t>(&(value.v_int64));
          break;
        }
        case kTVMDataType: {
          channel->Read(&(value.v_type));
          int32_t padding = 0;
          channel->template Read<int32_t>(&padding);
          break;
        }
        case kTVMContext: {
          channel->Read(&(value.v_ctx));
          break;
        }
        case kTVMPackedFuncHandle:
        case kTVMModuleHandle:
        case kTVMOpaqueHandle: {
          // always send handle in 64 bit.
          uint64_t handle;
          channel->Read(&handle);
          value.v_handle = reinterpret_cast<void*>(handle);
          break;
        }
        case kTVMNullptr: {
          value.v_handle = nullptr;
          break;
        }
        case kTVMStr: {
          uint64_t len;
          channel->Read(&len);
          char* str = channel->template ArenaAlloc<char>(len + 1);
          str[len] = '\0';
          channel->ReadArray(str, len);
          value.v_str = str;
          break;
        }
        case kTVMBytes: {
          uint64_t len;
          channel->Read(&len);
          TVMByteArray* arr = channel->template ArenaAlloc<TVMByteArray>(1);
          char* data = channel->template ArenaAlloc<char>(len);
          arr->size = len;
          arr->data = data;
          channel->ReadArray(data, len);
          value.v_handle = arr;
          break;
        }
        case kTVMDLTensorHandle: {
          uint64_t handle;
          channel->Read(&handle);
          DLTensor* arr = channel->template ArenaAlloc<DLTensor>(1);
          DLTensor& tensor = *arr;
          tensor.data = reinterpret_cast<void*>(handle);
          channel->Read(&(tensor.ctx));
          channel->Read(&(tensor.ndim));
          channel->Read(&(tensor.dtype));
          tensor.shape = channel->template ArenaAlloc<int64_t>(tensor.ndim);
          channel->ReadArray(tensor.shape, tensor.ndim);
          tensor.strides = nullptr;
          tensor.byte_offset = 0;
          value.v_handle = arr;
          break;
        }
        default: {
          channel->ThrowError(RPCServerStatus::kUnknownTypeCode);
          break;
        }
      }
    }
  }

  /*!
   * \brief Return an exception packet.
   *
   * \param msg The error message.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template<typename TChannel>
  static void ReturnException(const char* msg, TChannel* channel) {
    RPCCode code = RPCCode::kException;
    int32_t num_args = 1;
    int32_t tcode = kTVMStr;
    uint64_t len = StrLength(msg);

    uint64_t packet_nbytes =
        sizeof(code) +
        sizeof(num_args) +
        sizeof(tcode) +
        sizeof(len) +
        len;

    channel->Write(packet_nbytes);
    channel->Write(code);
    channel->Write(num_args);
    channel->Write(tcode);
    channel->Write(len);
    channel->WriteArray(msg, len);
  }

  /*!
   * \brief Return a normal packed sequence packet.
   *
   * \param msg The error message.
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template<typename TChannel>
  static void ReturnPackedSeq(const TVMValue* arg_values,
                              const int* type_codes,
                              int num_args,
                              TChannel* channel) {
    RPCCode code = RPCCode::kReturn;

    uint64_t packet_nbytes =
        sizeof(code) +
        PackedSeqGetNumBytes(
            arg_values, type_codes, num_args, false, channel);

    channel->Write(packet_nbytes);
    channel->Write(code);
    SendPackedSeq(
        arg_values, type_codes, num_args, false, channel);
  }

  /*!
   * \brief Return a null(void) packet.
   *
   * \param channel The communication channel handler.
   * \tparam TChannel The type of the communication channel.
   */
  template<typename TChannel>
  static void ReturnVoid(TChannel* channel) {
    int32_t num_args = 1;
    int32_t tcode = kTVMNullptr;
    RPCCode code = RPCCode::kReturn;

    uint64_t packet_nbytes =
        sizeof(code) +
        sizeof(num_args) +
        sizeof(tcode);

    channel->Write(packet_nbytes);
    channel->Write(code);
    channel->Write(num_args);
    channel->Write(tcode);
  }
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_PROTOCOL_H_
